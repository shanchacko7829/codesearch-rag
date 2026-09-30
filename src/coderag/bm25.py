"""Okapi BM25 over a sparse term-document matrix (implemented from scratch with scipy).

Scoring a query is one sparse matrix-vector product over the query's terms, so it is
fast enough for tens of thousands of chunks without an external search engine.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse

from .config import BM25_B, BM25_K1


class Vocabulary:
    def __init__(self, terms):
        self.terms = list(terms)
        self.index = {t: i for i, t in enumerate(self.terms)}

    def __len__(self) -> int:
        return len(self.terms)

    def encode(self, tokens) -> tuple[np.ndarray, np.ndarray]:
        """Unique in-vocabulary term ids and their counts (out-of-vocabulary terms are dropped)."""
        c = Counter(t for t in tokens if t in self.index)
        ids = np.fromiter((self.index[t] for t in c), dtype=np.int64, count=len(c))
        cnt = np.fromiter(c.values(), dtype=np.float64, count=len(c))
        return ids, cnt


def build_counts(docs_tokens) -> tuple[Vocabulary, sparse.csr_matrix]:
    """Token lists -> (vocabulary, documents x terms count matrix)."""
    index: dict[str, int] = {}
    indptr, indices, data = [0], [], []
    for toks in docs_tokens:
        for term, n in Counter(toks).items():
            indices.append(index.setdefault(term, len(index)))
            data.append(n)
        indptr.append(len(indices))
    n_docs = len(indptr) - 1
    if n_docs == 0:
        raise ValueError("cannot build an index from zero documents")
    terms = [""] * len(index)
    for term, j in index.items():
        terms[j] = term
    X = sparse.csr_matrix((np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32),
                           np.asarray(indptr, dtype=np.int32)), shape=(n_docs, len(index)))
    X.sort_indices()
    return Vocabulary(terms), X


class BM25:
    def __init__(self, counts: sparse.csr_matrix, k1: float = BM25_K1, b: float = BM25_B):
        self.counts = counts.tocsr()
        c = self.counts
        self.k1, self.b = k1, b
        self.n_docs, n_terms = c.shape
        self.doc_len = np.asarray(c.sum(axis=1)).ravel().astype(np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 1.0
        self.df = np.bincount(c.indices, minlength=n_terms).astype(np.int64)
        self.idf = np.log1p((self.n_docs - self.df + 0.5) / (self.df + 0.5)).astype(np.float32)
        rows = np.repeat(np.arange(self.n_docs), np.diff(c.indptr))
        tf = c.data
        denom = tf + k1 * (1.0 - b + b * self.doc_len[rows] / self.avgdl)
        w = (self.idf[c.indices] * tf * (k1 + 1.0) / denom).astype(np.float32)
        # terms x documents, so a query only touches the rows of its own terms
        self.weights_t = sparse.csr_matrix((w, c.indices, c.indptr), shape=c.shape).T.tocsr()

    def scores(self, ids: np.ndarray, cnts: np.ndarray) -> np.ndarray:
        """BM25 score of every document for a query given as (term ids, term counts)."""
        if len(ids) == 0:
            return np.zeros(self.n_docs, dtype=np.float32)
        q = (1.0 + np.log(cnts)).astype(np.float32)        # dampen repeated query terms
        return np.asarray(self.weights_t[ids].T.dot(q)).ravel()
