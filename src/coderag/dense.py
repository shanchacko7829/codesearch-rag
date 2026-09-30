"""Dense retrieval backends.

* ``LSABackend`` (default): latent semantic analysis - TF-IDF followed by a truncated
  SVD. It needs no model download and no GPU, it is deterministic, and it captures
  co-occurrence ("session" ~ "cookie" ~ "request") that exact-match BM25 cannot.
  Terms that occur in only one chunk are dropped from the dense side (BM25 keeps them,
  which is what makes exact identifier search work).
* ``SentenceTransformerBackend``: optional neural embeddings, enabled with
  ``--backend sbert`` after ``pip install sentence-transformers``.

A backend exposes ``fit``, ``encode_query``, ``doc_emb`` and ``state``/``from_state`` so the
index can be saved and restored without pickle.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from .config import LSA_DIMS, SEED


def _l2(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


class LSABackend:
    name = "lsa"

    def __init__(self, dims: int = LSA_DIMS, min_df: int = 2, seed: int = SEED):
        self.dims, self.min_df, self.seed = dims, min_df, seed
        self.doc_emb = self.components = self.keep = self.idf = self._col = None

    def _set(self, keep, idf, components, doc_emb, n_terms):
        self.keep, self.idf, self.components, self.doc_emb = keep, idf, components, doc_emb
        self._col = np.full(n_terms, -1, dtype=np.int64)
        self._col[keep] = np.arange(len(keep))

    def fit(self, chunks, counts, df) -> None:
        n_docs, n_terms = counts.shape
        keep = np.flatnonzero(df >= self.min_df)
        if keep.size < 2:
            keep = np.arange(n_terms)
        X = counts[:, keep].tocsr().astype(np.float32)
        X.data = 1.0 + np.log(X.data)                                   # sublinear term frequency
        idf = (np.log((1.0 + n_docs) / (1.0 + df[keep])) + 1.0).astype(np.float32)
        X = normalize(X.multiply(idf).tocsr())
        k = max(1, min(self.dims, min(X.shape) - 1))
        svd = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=5, random_state=self.seed)
        emb = svd.fit_transform(X)
        self._set(keep, idf, svd.components_.astype(np.float32), _l2(emb).astype(np.float32), n_terms)

    def encode_query(self, query: str, ids: np.ndarray, cnts: np.ndarray):
        if len(ids) == 0:
            return None
        pos = self._col[ids]
        sel = pos >= 0
        if not sel.any():
            return None
        v = (1.0 + np.log(cnts[sel])).astype(np.float32) * self.idf[pos[sel]]
        v = v / np.linalg.norm(v)
        q = self.components[:, pos[sel]] @ v
        norm = np.linalg.norm(q)
        return None if norm == 0 else (q / norm).astype(np.float32)

    def state(self):
        arrays = {"doc_emb": self.doc_emb, "components": self.components, "keep": self.keep, "idf": self.idf}
        return arrays, {"dims": int(self.components.shape[0]), "min_df": self.min_df, "seed": self.seed,
                        "n_terms": int(len(self._col))}

    @classmethod
    def from_state(cls, arrays, meta):
        b = cls(meta["dims"], meta["min_df"], meta["seed"])
        b._set(arrays["keep"], arrays["idf"], arrays["components"], arrays["doc_emb"], meta["n_terms"])
        return b


class SentenceTransformerBackend:
    name = "sbert"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64,
                 max_chars: int = 2000):
        self.model_name, self.batch_size, self.max_chars = model_name, batch_size, max_chars
        self.doc_emb = None
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("the 'sbert' backend needs: pip install sentence-transformers") from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, chunks, counts, df) -> None:
        texts = [f"{c.name}\n{c.index_text}"[: self.max_chars] for c in chunks]
        emb = self._load_model().encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                        show_progress_bar=False)
        self.doc_emb = np.asarray(emb, dtype=np.float32)

    def encode_query(self, query: str, ids, cnts):
        v = self._load_model().encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(v, dtype=np.float32)

    def state(self):
        return {"doc_emb": self.doc_emb}, {"model_name": self.model_name, "batch_size": self.batch_size,
                                           "max_chars": self.max_chars}

    @classmethod
    def from_state(cls, arrays, meta):
        b = cls(meta["model_name"], meta["batch_size"], meta["max_chars"])
        b.doc_emb = arrays["doc_emb"]
        return b


BACKENDS = {"lsa": LSABackend, "sbert": SentenceTransformerBackend}
