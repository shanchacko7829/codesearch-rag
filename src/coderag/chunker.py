"""Split a repository into retrievable chunks.

The default strategy is **syntax-aware**: Python files are parsed with the standard
library ``ast`` module and cut at function / method / class boundaries, so a search
hit is a whole, self-contained definition with exact line numbers rather than an
arbitrary 40-line slice. Files that cannot be parsed (syntax errors, other
languages, docs) fall back to overlapping line windows.

``hide_docstrings=True`` produces a second, docstring-free view of each chunk in
``Chunk.body``. The evaluation harness indexes that view and uses the docstring as
the *query*, which is the standard way to build a code-search benchmark without
hand-labelling (the same idea as CodeSearchNet).
"""
from __future__ import annotations

import ast
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import (CLASS_HEADER_CAP, CODE_EXTENSIONS, DEFAULT_EXCLUDE_DIRS, DOC_EXTENSIONS,
                     DOC_WINDOW_LINES, DOC_WINDOW_OVERLAP, MAX_FILE_BYTES, MAX_FUNCTION_LINES,
                     MODULE_CHUNK_CAP, PART_LINES, PART_OVERLAP, TEST_DIRS, WINDOW_LINES,
                     WINDOW_OVERLAP)

_FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)
_COMPOUND = tuple(t for t in (ast.If, ast.Try, getattr(ast, "TryStar", None), ast.With, ast.AsyncWith,
                              ast.For, ast.AsyncFor, ast.While) if t is not None)


@dataclass
class Chunk:
    id: int
    path: str          # repo-relative, forward slashes
    start: int         # 1-based inclusive line range
    end: int
    kind: str          # function | method | class | module | *_part | window | doc
    name: str          # qualified name ("Flask.route"); '' for windows
    signature: str
    doc: str           # docstring text ('' if none)
    code: str          # original source of the chunk
    body: str | None = None   # docstring-masked view, only set when hide_docstrings=True

    @property
    def index_text(self) -> str:
        return self.code if self.body is None else self.body

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start}-{self.end}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("body")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


# ----------------------------------------------------------------------------- helpers
def _docstring_stmt(node):
    body = getattr(node, "body", None)
    if body:
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            return first
    return None


def docstring_spans(tree: ast.AST) -> list[tuple[int, int]]:
    """Line spans (1-based, inclusive) of every module/class/function docstring."""
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, *_FUNC)):
            st = _docstring_stmt(node)
            if st is None:
                continue
            if not isinstance(node, ast.Module) and st.lineno <= node.lineno:
                continue        # docstring shares the "def" line; masking it would delete the signature
            spans.append((st.lineno, st.end_lineno))
    return spans


def _mask(lines: list[str], spans: list[tuple[int, int]]) -> list[str]:
    out = list(lines)
    for a, b in spans:
        for i in range(a - 1, min(b, len(out))):
            out[i] = ""
    return out


def _start(node) -> int:
    decs = getattr(node, "decorator_list", [])
    return min([node.lineno] + [d.lineno for d in decs])


def _signature(node, lines: list[str]) -> str:
    first_body = node.body[0].lineno if node.body else node.lineno + 1
    end = max(node.lineno, first_body - 1)
    return " ".join(s.strip() for s in lines[node.lineno - 1:end])[:300]


# ----------------------------------------------------------------------------- python (AST)
def chunk_python(source: str, path: str, *, hide_docstrings: bool = False,
                 max_function_lines: int = MAX_FUNCTION_LINES) -> list[Chunk] | None:
    """Function/method/class/module chunks. Returns None if the file cannot be parsed."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    lines = source.splitlines()
    masked = _mask(lines, docstring_spans(tree)) if hide_docstrings else None
    out: list[Chunk] = []

    def emit(start, end, kind, name, sig, doc):
        end = min(end, len(lines))
        if end < start:
            return
        code = "\n".join(lines[start - 1:end])
        body = "\n".join(masked[start - 1:end]) if masked is not None else None
        out.append(Chunk(-1, path, start, end, kind, name, sig, doc, code, body))

    def visit(nodes, prefix, in_class):
        for node in nodes:
            if isinstance(node, _FUNC):
                start, end = _start(node), node.end_lineno
                name, sig = prefix + node.name, _signature(node, lines)
                doc = ast.get_docstring(node) or ""
                kind = "method" if in_class else "function"
                if end - start + 1 <= max_function_lines:
                    emit(start, end, kind, name, sig, doc)
                else:                                   # very long function: overlapping parts
                    s, first = start, True
                    while s <= end:
                        e = min(s + PART_LINES - 1, end)
                        emit(s, e, kind + "_part", name, sig, doc if first else "")
                        first = False
                        if e >= end:
                            break
                        s = e - PART_OVERLAP + 1
            elif isinstance(node, ast.ClassDef):
                start, end = _start(node), node.end_lineno
                name = prefix + node.name
                member_starts = [_start(c) for c in node.body if isinstance(c, (*_FUNC, ast.ClassDef))]
                header_end = (min(member_starts) - 1) if member_starts else end
                header_end = min(max(header_end, node.lineno), start + CLASS_HEADER_CAP - 1)
                emit(start, header_end, "class", name, _signature(node, lines), ast.get_docstring(node) or "")
                visit(node.body, name + ".", True)
            elif isinstance(node, _COMPOUND):           # defs hidden under "if"/"try"/"with" blocks
                for field in ("body", "orelse", "finalbody"):
                    visit(getattr(node, field, []) or [], prefix, in_class)
                for handler in getattr(node, "handlers", []) or []:
                    visit(handler.body, prefix, in_class)

    visit(tree.body, "", False)

    # Module-level code between definitions (constants, registries, top-level logic).
    run: list[ast.stmt] = []

    module_doc_stmt = _docstring_stmt(tree)

    def flush():
        if run:
            # A run of nothing but imports / the module docstring is not worth a chunk.
            trivial = all(isinstance(n, (ast.Import, ast.ImportFrom)) or n is module_doc_stmt for n in run)
            s, e = run[0].lineno, run[-1].end_lineno
            if not trivial and e - s + 1 >= 3:
                doc = (ast.get_docstring(tree) or "") if run[0] is module_doc_stmt else ""
                emit(s, min(e, s + MODULE_CHUNK_CAP - 1), "module", "<module>", "", doc)
        run.clear()

    for node in tree.body:
        if isinstance(node, (*_FUNC, ast.ClassDef)):
            flush()
        else:
            run.append(node)
    flush()

    out.sort(key=lambda c: (c.start, c.end))
    return out


# ----------------------------------------------------------------------------- windows (fallback / baseline)
def chunk_windows(source: str, path: str, *, size: int = WINDOW_LINES, overlap: int = WINDOW_OVERLAP,
                  kind: str = "window", hide_docstrings: bool = False) -> list[Chunk]:
    lines = source.splitlines()
    masked = None
    if hide_docstrings and path.endswith(".py"):
        try:
            masked = _mask(lines, docstring_spans(ast.parse(source)))
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            masked = None
    step = max(1, size - overlap)
    out: list[Chunk] = []
    for s in range(1, len(lines) + 1, step):
        e = min(s + size - 1, len(lines))
        code = "\n".join(lines[s - 1:e])
        if code.strip():
            body = "\n".join(masked[s - 1:e]) if masked is not None else None
            out.append(Chunk(-1, path, s, e, kind, "", "", "", code, body))
        if e >= len(lines):
            break
    return out


def chunk_file(text: str, path: str, *, strategy: str = "ast", hide_docstrings: bool = False,
               window_lines: int = WINDOW_LINES, window_overlap: int = WINDOW_OVERLAP,
               max_function_lines: int = MAX_FUNCTION_LINES) -> list[Chunk]:
    if path.endswith(".py"):
        if strategy == "ast":
            parsed = chunk_python(text, path, hide_docstrings=hide_docstrings, max_function_lines=max_function_lines)
            if parsed is not None:
                return parsed
        return chunk_windows(text, path, size=window_lines, overlap=window_overlap, hide_docstrings=hide_docstrings)
    return chunk_windows(text, path, size=DOC_WINDOW_LINES, overlap=DOC_WINDOW_OVERLAP, kind="doc")


# ----------------------------------------------------------------------------- repository walk
def iter_source_files(root, subdir: str = "", exts=CODE_EXTENSIONS, exclude_dirs=DEFAULT_EXCLUDE_DIRS,
                      max_bytes: int = MAX_FILE_BYTES):
    """Yield (repo-relative path, absolute path) in a deterministic order."""
    root = os.path.abspath(root)
    base = os.path.join(root, subdir) if subdir else root
    if not os.path.isdir(base):
        raise FileNotFoundError(f"not a directory: {base}")
    exts = tuple(exts)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in exclude_dirs and not d.startswith("."))
        for fn in sorted(filenames):
            if not fn.endswith(exts):
                continue
            full = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(full) > max_bytes:
                    continue
            except OSError:
                continue
            yield os.path.relpath(full, root).replace(os.sep, "/"), full


def chunk_repo(root, subdir: str = "", *, strategy: str = "ast", hide_docstrings: bool = False,
               include_docs: bool = False, exclude_tests: bool = False, **kw) -> list[Chunk]:
    exclude = set(DEFAULT_EXCLUDE_DIRS) | (set(TEST_DIRS) if exclude_tests else set())
    exts = CODE_EXTENSIONS + (DOC_EXTENSIONS if include_docs else ())
    chunks: list[Chunk] = []
    for rel, full in iter_source_files(root, subdir, exts, exclude):
        text = Path(full).read_text(encoding="utf-8", errors="replace")
        chunks.extend(chunk_file(text, rel, strategy=strategy, hide_docstrings=hide_docstrings, **kw))
    for i, c in enumerate(chunks):
        c.id = i
    return chunks
