"""End-to-end experiment: build the benchmark, train the reranker, evaluate every method.

Everything reported in the README comes out of ``run_experiment`` and is written to
``results/eval_<name>.json``; nothing is typed in by hand.
"""
from __future__ import annotations

import random
import subprocess
import time
from pathlib import Path

import numpy as np

from .benchmark import (Query, build_benchmark, is_hit, paired_diff_ci, per_query_metrics, rank_of_first_hit,
                        split_by_file, summarize)
from .chunker import chunk_repo
from .config import LSA_DIMS, RERANK_DEPTH, SEED
from .dense import LSABackend
from .index import CodeIndex
from .rerank import Reranker, extract_features

MODES = ("bm25", "dense", "hybrid", "rerank")
DEPTH = 50


def evaluate_mode(index: CodeIndex, queries, mode: str, warmup: int = 5):
    """(rank of first hit per query, latency in ms per query). The query cache is bypassed."""
    for q in queries[:warmup]:
        index.search(q.text, k=DEPTH, mode=mode, use_cache=False)
    ranks, lat = [], []
    for q in queries:
        t = time.perf_counter()
        hits = index.search(q.text, k=DEPTH, mode=mode, use_cache=False)
        lat.append((time.perf_counter() - t) * 1000.0)
        ranks.append(rank_of_first_hit(hits, q))
    return np.array(ranks), np.array(lat)


def train_reranker(index: CodeIndex, queries, depth: int = RERANK_DEPTH, C: float = 1.0):
    """Fit the reranker on candidate lists of the training queries (label = candidate is the target)."""
    xs, ys, used = [], [], 0
    for q in queries:
        ret = index.retrieve(q.text)
        cand = np.array([i for i, _ in ret.fused[:depth]], dtype=np.int64)
        if len(cand) == 0:
            continue
        y = np.array([is_hit(index.chunks[int(c)], q) for c in cand], dtype=int)
        if y.sum() == 0:          # target not among the candidates: nothing to learn from this query
            continue
        xs.append(extract_features(index, ret, cand))
        ys.append(y)
        used += 1
    X, y = np.vstack(xs), np.concatenate(ys)
    model = Reranker.fit(X, y, C)
    return model, {"queries_used": used, "queries_total": len(queries), "rows": int(len(y)), "positives": int(y.sum())}


def _git_commit(repo) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=15)
        return r.stdout.strip() or None
    except Exception:
        return None


def _examples(index, queries, ranks, limit: int = 4):
    picks = [i for i, r in enumerate(ranks) if r == 1][:2]
    picks += [i for i, r in enumerate(ranks) if 2 <= r <= 10][:1]
    picks += [i for i, r in enumerate(ranks) if r == 0 or r > 10][:1]
    out = []
    for i in picks[:limit]:
        q = queries[i]
        hits = index.search(q.text, k=3, mode="rerank", use_cache=False)
        out.append({"query": q.text, "target": f"{q.path}:{q.start}-{q.end} {q.name}", "rank": int(ranks[i]),
                    "top3": [f"{h.chunk.location} {h.chunk.name}" for h in hits]})
    return out


def run_experiment(repo, subdir: str, name: str, n_queries: int = 800, n_train: int = 1500, dims: int = LSA_DIMS,
                   test_frac: float = 0.25, seed: int = SEED, n_boot: int = 1000, transfer_reranker=None,
                   save_reranker=None, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.perf_counter()
    chunks = chunk_repo(repo, subdir, strategy="ast", hide_docstrings=True, exclude_tests=True)
    all_queries = build_benchmark(chunks)
    train_q, test_q = split_by_file(all_queries, test_frac)
    rng = random.Random(seed)
    test_q = rng.sample(test_q, min(n_queries, len(test_q)))
    train_q = rng.sample(train_q, min(n_train, len(train_q)))
    if len(test_q) < 20:
        raise RuntimeError(f"only {len(test_q)} test queries - repository too small for a meaningful benchmark")
    log(f"[{name}] {len(chunks):,} chunks | queries: {len(all_queries):,} eligible, {len(train_q):,} train, {len(test_q):,} test")

    index = CodeIndex.build(chunks, LSABackend(dims), verbose=verbose)
    reranker, train_stats = train_reranker(index, train_q)
    index.reranker = reranker
    if save_reranker:
        reranker.save(save_reranker)
    log(f"[{name}] reranker trained on {train_stats['queries_used']} queries; weights: {reranker.weights()}")

    groups = [q.path for q in test_q]
    raw, modes = {}, {}
    for mode in MODES:
        raw[mode] = evaluate_mode(index, test_q, mode)
        modes[mode] = summarize(*raw[mode], groups, n_boot)
        log(f"[{name}] {mode:7} MRR@10={modes[mode]['mrr@10']['value']:.3f} Hit@1={modes[mode]['hit@1']['value']:.3f} "
            f"Hit@10={modes[mode]['hit@10']['value']:.3f} p50={modes[mode]['latency_ms']['p50']}ms")
    if transfer_reranker:
        own = index.reranker
        index.reranker = Reranker.load(transfer_reranker)
        raw["rerank_transfer"] = evaluate_mode(index, test_q, "rerank")
        modes["rerank_transfer"] = summarize(*raw["rerank_transfer"], groups, n_boot)
        index.reranker = own
        log(f"[{name}] rerank (trained on another repo) MRR@10={modes['rerank_transfer']['mrr@10']['value']:.3f}")

    mrr = {m: per_query_metrics(raw[m][0])["mrr@10"] for m in raw}

    def diff(a, b):
        d, (lo, hi) = paired_diff_ci(mrr[a], mrr[b], groups, n_boot)
        return {"diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)], "significant": bool(lo > 0 or hi < 0)}

    paired = {"hybrid_minus_bm25": diff("hybrid", "bm25"), "hybrid_minus_dense": diff("hybrid", "dense"),
              "rerank_minus_hybrid": diff("rerank", "hybrid")}
    if "rerank_transfer" in mrr:
        paired["rerank_transfer_minus_hybrid"] = diff("rerank_transfer", "hybrid")

    overlap = np.array([q.name_overlap for q in test_q])
    difficulty = {}
    for label, mask in (("query_shares_a_word_with_name", overlap), ("no_shared_word_with_name", ~overlap)):
        row = {"n": int(mask.sum())}
        for m in MODES:
            pq = per_query_metrics(raw[m][0])
            row[m] = {"mrr@10": round(float(pq["mrr@10"][mask].mean()), 4) if mask.any() else None,
                      "hit@10": round(float(pq["hit@10"][mask].mean()), 4) if mask.any() else None}
        difficulty[label] = row

    # Chunking ablation: same queries, same hit rule, fixed 40-line windows instead of AST chunks.
    win_chunks = chunk_repo(repo, subdir, strategy="windows", hide_docstrings=True, exclude_tests=True)
    win_index = CodeIndex.build(win_chunks, LSABackend(dims))
    chunking = {"ast": {m: modes[m] for m in ("bm25", "hybrid")}, "windows": {}, "n_chunks":
                {"ast": len(chunks), "windows": len(win_chunks)}}
    for mode in ("bm25", "hybrid"):
        chunking["windows"][mode] = summarize(*evaluate_mode(win_index, test_q, mode), groups, n_boot)
        log(f"[{name}] fixed windows {mode:6} MRR@10={chunking['windows'][mode]['mrr@10']['value']:.3f}")

    return {
        "repo": name, "subdir": subdir, "commit": _git_commit(repo), "seed": seed,
        "corpus": {"files": len({c.path for c in chunks}), "chunks": len(chunks), "terms": index.meta["n_terms"],
                   "chunks_by_kind": index.meta["kinds"], "index_build_s": index.meta["timings_s"]},
        "benchmark": {"eligible_queries": len(all_queries), "train_queries": len(train_q), "test_queries": len(test_q),
                      "test_files": len(set(groups)), "share_with_name_overlap": round(float(overlap.mean()), 3)},
        "modes": modes, "paired": paired, "difficulty": difficulty, "chunking": chunking,
        "reranker": {"train": train_stats, "weights": reranker.weights()},
        "examples": _examples(index, test_q, raw["rerank"][0]),
        "seconds": round(time.perf_counter() - t0, 1),
    }
