"""Grounded question answering on top of retrieval.

The answer generator is pluggable:

* ``ExtractiveGenerator`` - no language model: lists the most relevant definitions with their
  location, signature and first docstring sentence. Every statement is taken from the code,
  so it cannot hallucinate.
* ``OllamaGenerator`` - sends the retrieved excerpts to a local model served by Ollama.

Whatever model is used, the answer is **checked** afterwards: every ``[n]`` citation must
refer to an excerpt that was actually retrieved; invented citations are reported.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

SYSTEM_PROMPT = (
    "You are a code assistant. Answer the question using ONLY the numbered code excerpts provided. "
    "Cite the excerpts you rely on as [1], [2], ... after the statements they support. "
    "If the excerpts do not contain the answer, say you cannot tell from the provided code. "
    "Do not invent files, functions or behaviour."
)


class GeneratorError(RuntimeError):
    pass


def first_sentence(text: str, limit: int = 160) -> str:
    text = " ".join((text or "").split())
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    s = m.group(1) if m else text
    return s if len(s) <= limit else s[: limit - 1] + "…"


def format_excerpt(n: int, hit, max_chars: int = 1800) -> str:
    c = hit.chunk
    code = c.code if len(c.code) <= max_chars else c.code[:max_chars] + "\n# ... (truncated)"
    label = f"{c.kind} {c.name}".strip()
    return f"[{n}] {c.location} ({label})\n{code}"


def build_prompt(question: str, hits, max_chars: int = 1800) -> str:
    excerpts = "\n\n".join(format_excerpt(i, h, max_chars) for i, h in enumerate(hits, 1))
    return f"Code excerpts:\n\n{excerpts}\n\nQuestion: {question}\n\nAnswer (cite excerpts like [1]):"


class ExtractiveGenerator:
    name = "extractive"

    def generate(self, question: str, hits) -> str:
        lines = [f"Most relevant code for: {question}"]
        for i, h in enumerate(hits, 1):
            c = h.chunk
            what = f"{c.kind} {c.name}".strip()
            summary = first_sentence(c.doc) if c.doc else (c.signature[:120] or "(no docstring)")
            lines.append(f"[{i}] {c.location} - {what}: {summary}")
        return "\n".join(lines)


class OllamaGenerator:
    """Local LLM via Ollama's HTTP API (https://ollama.com). Start it with `ollama run <model>`."""

    def __init__(self, model: str = "qwen2.5-coder:7b", host: str = "http://localhost:11434", timeout: float = 180.0):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.name = f"ollama:{model}"

    def generate(self, question: str, hits) -> str:
        payload = {"model": self.model, "system": SYSTEM_PROMPT, "prompt": build_prompt(question, hits),
                   "stream": False, "options": {"temperature": 0}}
        req = urllib.request.Request(f"{self.host}/api/generate", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise GeneratorError(f"could not get an answer from Ollama at {self.host} "
                                 f"(is it running, and is the model '{self.model}' pulled?): {exc}") from exc
        if "response" not in body:
            raise GeneratorError(f"unexpected reply from Ollama: {str(body)[:200]}")
        return body["response"].strip()


def validate_citations(answer: str, n_sources: int) -> dict:
    cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    valid = [c for c in cited if 1 <= c <= n_sources]
    invalid = [c for c in cited if c not in valid]
    return {"cited": valid, "invalid": invalid, "grounded": bool(valid) and not invalid}


def answer_question(index, question: str, generator=None, k: int = 5, mode: str = "auto") -> dict:
    generator = generator or ExtractiveGenerator()
    t0 = time.perf_counter()
    hits = index.search(question, k=k, mode=mode)
    t1 = time.perf_counter()
    if not hits:
        return {"question": question, "answer": "No relevant code found.", "sources": [],
                "citations": {"cited": [], "invalid": [], "grounded": False}, "generator": generator.name,
                "latency_ms": {"retrieval": round((t1 - t0) * 1000, 2), "generation": 0.0}}
    answer = generator.generate(question, hits)
    t2 = time.perf_counter()
    sources = [{"n": i, "location": h.chunk.location, "kind": h.chunk.kind, "name": h.chunk.name,
                "score": round(h.score, 4), "snippet": h.chunk.code[:400]} for i, h in enumerate(hits, 1)]
    return {"question": question, "answer": answer, "sources": sources,
            "citations": validate_citations(answer, len(hits)), "generator": generator.name,
            "latency_ms": {"retrieval": round((t1 - t0) * 1000, 2), "generation": round((t2 - t1) * 1000, 2)}}
