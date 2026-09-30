"""Benchmark construction and metrics.

Ground truth without hand labelling: for every documented function or method, the first
paragraph of its docstring is the *query* and the function's own location is the *answer*.
The index is built from a view of the code with docstrings removed, so a retriever cannot
win by matching the query text against the docstring it came from - it has to connect an
English description to identifiers and code (this is how CodeSearchNet is built).

Design choices that keep the numbers honest:
* queries whose text occurs on more than one function are dropped (ambiguous ground truth);
* train/test are split **by file**, so the reranker never sees a test file's functions;
* a retrieved chunk counts as a hit when it overlaps the target by at least half of the
  smaller of the two spans, which lets AST chunks and fixed windows be compared fairly;
* confidence intervals come from a **cluster bootstrap over files**, because functions in
  the same file are not independent.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .text import tokenize

_ROLE = re.compile(r":\w+(?::\w+)?:`~?([^`]*)`")
_DIRECTIVE = re.compile(r"\.\. \w+::[^\n]*")


@dataclass(frozen=True)
class Query:
    text: str
    path: str
    start: int
    end: int
    name: str
    name_overlap: bool      # does the query share a word with the function's own name?


def clean_docstring_query(doc: str, min_words: int = 6, max_words: int = 40) -> str | None:
    """First paragraph of a docstring with reST markup stripped, or None if unusable."""
    if not doc:
        return None
    para = doc.strip().split("\n\n")[0]
    para = _DIRECTIVE.sub(" ", para)
    para = _ROLE.sub(r"\1", para).replace("`", "")
    text = re.sub(r"\s+", " ", para).strip()
    if text.startswith((">>>", "..")):
        return None
    words = text.split(" ")
    if len(words) < min_words:
        return None
    return " ".join(words[:max_words])


def build_benchmark(chunks, min_words: int = 6, max_words: int = 40) -> list[Query]:
    cands = []
    for c in chunks:
        if c.kind in ("function", "method") and c.doc:
            text = clean_docstring_query(c.doc, min_words, max_words)
            if text:
                cands.append((c, text))
    counts = Counter(t.lower() for _, t in cands)
    out = []
    for c, t in cands:
        if counts[t.lower()] > 1:
            continue
        overlap = bool(set(tokenize(t)) & set(tokenize(c.name.split(".")[-1])))
        out.append(Query(t, c.path, c.start, c.end, c.name, overlap))
    return out


def _bucket(path: str) -> float:
    return int(hashlib.md5(path.encode()).hexdigest(), 16) % 1000 / 1000.0


def split_by_file(queries, test_frac: float = 0.25):
    """Deterministic file-level split (a file is entirely train or entirely test)."""
    train = [q for q in queries if _bucket(q.path) >= test_frac]
    test = [q for q in queries if _bucket(q.path) < test_frac]
    return train, test


def is_hit(chunk, q: Query) -> bool:
    """Chunk overlaps the target by >= half of the smaller span."""
    if chunk.path != q.path:
        return False
    overlap = min(chunk.end, q.end) - max(chunk.start, q.start) + 1
    if overlap <= 0:
        return False
    return overlap >= 0.5 * min(chunk.end - chunk.start + 1, q.end - q.start + 1)


def rank_of_first_hit(hits, q: Query) -> int:
    """1-based rank of the first correct result, 0 if none was retrieved."""
    for h in hits:
        if is_hit(h.chunk, q):
            return h.rank
    return 0


def per_query_metrics(ranks) -> dict[str, np.ndarray]:
    r = np.asarray(ranks)
    found = r >= 1
    return {
        "hit@1": (r == 1).astype(float),
        "hit@5": (found & (r <= 5)).astype(float),
        "hit@10": (found & (r <= 10)).astype(float),
        "mrr@10": np.where(found & (r <= 10), 1.0 / np.maximum(r, 1), 0.0),
        "recall@50": found.astype(float),
    }


def bootstrap_ci(values, groups, n_boot: int = 1000, seed: int = 0):
    """Mean and 95% CI of ``values``, resampling whole groups (files) with replacement."""
    values = np.asarray(values, dtype=float)
    _, gid = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = int(gid.max()) + 1
    sums = np.bincount(gid, weights=values, minlength=n_groups)
    cnts = np.bincount(gid, minlength=n_groups).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, n_groups, size=(n_boot, n_groups))
    means = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    return float(values.mean()), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_diff_ci(a, b, groups, n_boot: int = 1000, seed: int = 0):
    """Mean difference a-b on the same queries with a cluster-bootstrap 95% CI."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), groups, n_boot, seed)


def latency_summary(ms) -> dict[str, float]:
    a = np.asarray(ms, dtype=float)
    return {"mean": round(float(a.mean()), 3), "p50": round(float(np.percentile(a, 50)), 3),
            "p95": round(float(np.percentile(a, 95)), 3), "p99": round(float(np.percentile(a, 99)), 3)}


def summarize(ranks, latencies_ms, groups, n_boot: int = 1000) -> dict:
    out = {}
    for name, vals in per_query_metrics(ranks).items():
        v, (lo, hi) = bootstrap_ci(vals, groups, n_boot)
        out[name] = {"value": round(v, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    out["latency_ms"] = latency_summary(latencies_ms)
    return out
