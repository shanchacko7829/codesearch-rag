"""A small learned reranker over the fused candidate list.

First-stage retrieval (BM25 + dense, fused with reciprocal rank fusion) is cheap and
has high recall; the reranker then re-orders the top candidates using features the
first stage cannot see together: how well the query matches the symbol *name*, the file
*path* and the *signature*, how many query terms a chunk covers, and what kind of chunk
it is. It is a pointwise logistic regression so the learned weights are inspectable, it
is trained on the benchmark's training split only, and it is saved as plain JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

FEATURE_NAMES = [
    "bm25_rel", "dense", "dense_rel", "rrf_rel", "bm25_rank_recip", "dense_rank_recip",
    "name_overlap", "path_overlap", "sig_overlap", "log_len", "is_method", "is_function", "is_class",
    "term_coverage", "idf_coverage",
]


def extract_features(index, ret, cand) -> np.ndarray:
    """Feature matrix (len(cand) x len(FEATURE_NAMES)) for candidate chunk indices of one query."""
    cand = np.asarray(cand, dtype=np.int64)
    m = len(cand)
    bm = ret.bm25[cand].astype(np.float64)
    bm_rel = bm / (bm.max() + 1e-9)
    if ret.dense is not None:
        dn = ret.dense[cand].astype(np.float64)
        dn_rel = dn - dn.max()
    else:
        dn, dn_rel = np.zeros(m), np.zeros(m)
    fused = dict(ret.fused)
    rrf = np.array([fused.get(int(c), 0.0) for c in cand])
    rrf_rel = rrf / (rrf.max() + 1e-9)
    bm_rank = {int(i): r for r, i in enumerate(ret.bm25_top)}
    dn_rank = {int(i): r for r, i in enumerate(ret.dense_top)}
    bm_rr = np.array([1.0 / (1 + bm_rank[int(c)]) if int(c) in bm_rank else 0.0 for c in cand])
    dn_rr = np.array([1.0 / (1 + dn_rank[int(c)]) if int(c) in dn_rank else 0.0 for c in cand])

    q = set(ret.tokens)
    nq = max(len(q), 1)
    name_ov, path_ov, sig_ov = np.zeros(m), np.zeros(m), np.zeros(m)
    is_method, is_function, is_class = np.zeros(m), np.zeros(m), np.zeros(m)
    for j, c in enumerate(cand):
        names, paths, sigs = index.token_sets(int(c))
        name_ov[j], path_ov[j], sig_ov[j] = len(q & names) / nq, len(q & paths) / nq, len(q & sigs) / nq
        kind = index.chunks[int(c)].kind
        is_method[j] = kind.startswith("method")
        is_function[j] = kind.startswith("function")
        is_class[j] = kind == "class"

    if len(ret.ids):
        matched = (index.bm25.counts[cand][:, ret.ids].toarray() > 0).astype(np.float64)
        cov = matched.mean(axis=1)
        w = index.bm25.idf[ret.ids].astype(np.float64)
        idf_cov = matched @ w / max(float(w.sum()), 1e-9)
    else:
        cov, idf_cov = np.zeros(m), np.zeros(m)

    log_len = np.log1p(index.bm25.doc_len[cand].astype(np.float64))
    return np.column_stack([bm_rel, dn, dn_rel, rrf_rel, bm_rr, dn_rr, name_ov, path_ov, sig_ov, log_len,
                            is_method, is_function, is_class, cov, idf_cov])


class Reranker:
    def __init__(self, mean, scale, coef, intercept, feature_names=None):
        self.mean, self.scale = np.asarray(mean, float), np.asarray(scale, float)
        self.coef, self.intercept = np.asarray(coef, float), float(intercept)
        self.feature_names = list(feature_names or FEATURE_NAMES)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, C: float = 1.0) -> "Reranker":
        mean, scale = X.mean(axis=0), X.std(axis=0)
        scale[scale < 1e-9] = 1.0
        lr = LogisticRegression(C=C, max_iter=2000)
        lr.fit((X - mean) / scale, y)
        return cls(mean, scale, lr.coef_[0], lr.intercept_[0])

    def score(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.scale) @ self.coef + self.intercept

    def weights(self) -> dict[str, float]:
        """Standardised coefficients: larger magnitude = more influence on the ranking."""
        return {n: round(float(w), 3) for n, w in sorted(zip(self.feature_names, self.coef), key=lambda t: -abs(t[1]))}

    def to_dict(self) -> dict:
        return {"feature_names": self.feature_names, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept}

    @classmethod
    def from_dict(cls, d: dict) -> "Reranker":
        return cls(d["mean"], d["scale"], d["coef"], d["intercept"], d["feature_names"])

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "Reranker":
        return cls.from_dict(json.loads(Path(path).read_text()))
