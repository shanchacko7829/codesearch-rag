"""The searchable index: chunks + BM25 + dense embeddings (+ optional reranker).

Search modes: ``bm25``, ``dense``, ``hybrid`` (reciprocal rank fusion of the two) and
``rerank`` (hybrid candidates re-ordered by the learned reranker). The index is saved
as plain files (JSON lines, JSON, npz) - there is no pickle, so an index from someone
else's repository can be loaded without executing code.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict, Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from .bm25 import BM25, Vocabulary, build_counts
from .chunker import Chunk
from .config import NAME_BOOST, RERANK_DEPTH, RETRIEVAL_DEPTH, RRF_K
from .dense import BACKENDS, LSABackend
from .rerank import Reranker, extract_features
from .text import tokenize

INDEX_VERSION = 1
MODES = ("bm25", "dense", "hybrid", "rerank", "auto")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int
    bm25: float
    dense: float


@dataclass
class Retrieval:
    tokens: list
    ids: np.ndarray
    cnts: np.ndarray
    bm25: np.ndarray
    dense: np.ndarray | None
    bm25_top: np.ndarray
    dense_top: np.ndarray
    fused: list            # [(chunk index, rrf score)] best first


def chunk_tokens(chunk: Chunk, name_boost: int = NAME_BOOST) -> list[str]:
    """Document tokens: symbol name (boosted by repetition) + file path + the code itself."""
    return tokenize(chunk.name) * name_boost + tokenize(chunk.path) + tokenize(chunk.index_text)


def _top(scores: np.ndarray, n: int, positive_only: bool = False) -> np.ndarray:
    n = min(n, len(scores))
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    idx = np.argpartition(-scores, n - 1)[:n]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return idx[scores[idx] > 0] if positive_only else idx


def rrf(rank_lists, k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank)."""
    acc: dict[int, float] = {}
    for lst in rank_lists:
        for r, i in enumerate(lst):
            acc[int(i)] = acc.get(int(i), 0.0) + 1.0 / (k + r + 1)
    return sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))


class CodeIndex:
    def __init__(self, chunks, vocab: Vocabulary, bm25: BM25, backend, meta=None, cache_size: int = 1024):
        self.chunks, self.vocab, self.bm25, self.backend = chunks, vocab, bm25, backend
        self.meta = meta or {}
        self.reranker: Reranker | None = None
        self._tok_sets: dict[int, tuple] = {}
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self.cache_hits = self.cache_misses = 0

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, chunks, backend=None, name_boost: int = NAME_BOOST, verbose: bool = False) -> "CodeIndex":
        log = print if verbose else (lambda *a, **k: None)
        t0 = time.perf_counter()
        docs = [chunk_tokens(c, name_boost) for c in chunks]
        t1 = time.perf_counter()
        vocab, counts = build_counts(docs)
        bm25 = BM25(counts)
        t2 = time.perf_counter()
        backend = backend or LSABackend()
        backend.fit(chunks, counts, bm25.df)
        t3 = time.perf_counter()
        meta = {"version": INDEX_VERSION, "n_chunks": len(chunks), "n_terms": len(vocab), "name_boost": name_boost,
                "kinds": dict(Counter(c.kind for c in chunks)), "backend": backend.name,
                "timings_s": {"tokenize": round(t1 - t0, 2), "bm25": round(t2 - t1, 2), "dense": round(t3 - t2, 2)}}
        log(f"indexed {len(chunks):,} chunks, {len(vocab):,} terms: {meta['timings_s']}")
        return cls(chunks, vocab, bm25, backend, meta)

    # ------------------------------------------------------------------ retrieval
    def token_sets(self, i: int):
        """(name tokens, path tokens, signature tokens) of chunk i - used by the reranker."""
        s = self._tok_sets.get(i)
        if s is None:
            c = self.chunks[i]
            s = (frozenset(tokenize(c.name)), frozenset(tokenize(c.path)), frozenset(tokenize(c.signature)))
            self._tok_sets[i] = s
        return s

    def retrieve(self, query: str, depth: int = RETRIEVAL_DEPTH, need_dense: bool = True) -> Retrieval:
        tokens = tokenize(query)
        ids, cnts = self.vocab.encode(tokens)
        bm = self.bm25.scores(ids, cnts)
        dn = None
        if need_dense:
            qv = self.backend.encode_query(query, ids, cnts)
            if qv is not None:
                dn = self.backend.doc_emb @ qv
        bm_top = _top(bm, depth, positive_only=True)
        dn_top = _top(dn, depth) if dn is not None else np.empty(0, dtype=np.int64)
        fused = rrf([bm_top, dn_top])
        return Retrieval(tokens, ids, cnts, bm, dn, bm_top, dn_top, fused)

    def search(self, query: str, k: int = 10, mode: str = "hybrid", use_cache: bool = True) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "auto":
            mode = "rerank" if self.reranker is not None else "hybrid"
        if mode == "rerank" and self.reranker is None:
            raise ValueError("no reranker attached (train one with `coderag eval` or pass --reranker)")
        key = (mode, k, query)
        if use_cache:
            with self._lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    self.cache_hits += 1
                    return self._cache[key]
                self.cache_misses += 1

        ret = self.retrieve(query, need_dense=mode != "bm25")
        if mode == "bm25":
            order, scores = ret.bm25_top[:k], ret.bm25
        elif mode == "dense":
            order, scores = ret.dense_top[:k], ret.dense
        elif mode == "hybrid":
            fused = ret.fused[:k]
            order, scores = np.array([i for i, _ in fused], dtype=np.int64), dict(fused)
        else:
            cand = np.array([i for i, _ in ret.fused[:max(k, RERANK_DEPTH)]], dtype=np.int64)
            if len(cand) == 0:
                order, scores = cand, {}
            else:
                s = self.reranker.score(extract_features(self, ret, cand))
                pick = np.argsort(-s, kind="stable")[:k]
                order, scores = cand[pick], {int(cand[j]): float(s[j]) for j in pick}

        hits = []
        for r, i in enumerate(order):
            i = int(i)
            score = float(scores[i]) if isinstance(scores, dict) else float(scores[i])
            hits.append(Hit(self.chunks[i], score, r + 1, float(ret.bm25[i]),
                            float(ret.dense[i]) if ret.dense is not None else 0.0))
        if use_cache:
            with self._lock:
                self._cache[key] = hits
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return hits

    def cache_info(self) -> dict:
        return {"size": len(self._cache), "hits": self.cache_hits, "misses": self.cache_misses}

    # ------------------------------------------------------------------ persistence
    def save(self, directory) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        arrays, backend_meta = self.backend.state()
        meta = {**self.meta, "backend": {"name": self.backend.name, **backend_meta}}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        (d / "vocab.json").write_text(json.dumps(self.vocab.terms))
        sparse.save_npz(d / "counts.npz", self.bm25.counts, compressed=True)
        np.savez_compressed(d / "dense.npz", **arrays)
        if self.reranker is not None:
            self.reranker.save(d / "reranker.json")

    @classmethod
    def load(cls, directory) -> "CodeIndex":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("version") != INDEX_VERSION:
            raise ValueError(f"unsupported index version {meta.get('version')} (expected {INDEX_VERSION})")
        chunks = []
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        vocab = Vocabulary(json.loads((d / "vocab.json").read_text()))
        bm25 = BM25(sparse.load_npz(d / "counts.npz"))
        bmeta = dict(meta["backend"])
        backend = BACKENDS[bmeta.pop("name")].from_state(dict(np.load(d / "dense.npz")), bmeta)
        idx = cls(chunks, vocab, bm25, backend, {**meta, "backend": backend.name})
        if (d / "reranker.json").exists():
            idx.reranker = Reranker.load(d / "reranker.json")
        return idx
