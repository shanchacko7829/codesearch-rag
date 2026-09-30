import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from coderag.benchmark import (Query, bootstrap_ci, build_benchmark, clean_docstring_query, is_hit, paired_diff_ci,
                               per_query_metrics, split_by_file)
from coderag.bm25 import BM25, build_counts
from coderag.chunker import chunk_repo
from coderag.fetch import extract_zip_archive
from coderag.index import CodeIndex, rrf
from coderag.qa import ExtractiveGenerator, answer_question, validate_citations
from coderag.rerank import FEATURE_NAMES, Reranker
from coderag.text import split_identifier, stem, tokenize
from tests.helpers import make_repo


class TextTests(unittest.TestCase):
    def test_split_identifier(self):
        self.assertEqual(split_identifier("make_response"), ["make", "response"])
        self.assertEqual(split_identifier("RequestContext"), ["request", "context"])

    def test_stemming_merges_variants(self):
        self.assertEqual(stem("returning"), stem("returned"))
        self.assertEqual(stem("queries"), "query")

    def test_tokenize_keeps_full_identifier(self):
        self.assertIn("make_response", tokenize("make_response"))


class ChunkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = make_repo()

    def test_function_line_ranges(self):
        cs = {c.name: c for c in chunk_repo(self.repo, "", hide_docstrings=False, exclude_tests=True)}
        c = cs["hash_password"]
        self.assertEqual((c.start, c.end, c.kind), (5, 7, "function"))
        self.assertEqual(cs["SessionStore.drop_session"].kind, "method")

    def test_tests_excluded(self):
        paths = {c.path for c in chunk_repo(self.repo, "", exclude_tests=True)}
        self.assertFalse(any(p.startswith("tests/") for p in paths))

    def test_docstring_hidden_from_body_but_kept_in_code(self):
        cs = {c.name: c for c in chunk_repo(self.repo, "", hide_docstrings=True, exclude_tests=True)}
        c = cs["hash_password"]
        self.assertNotIn("salted sha256", c.body)
        self.assertIn("salted sha256", c.code)
        self.assertIn("salted sha256", c.doc)

    def test_syntax_error_file_falls_back_to_windows(self):
        cs = [c for c in chunk_repo(self.repo, "", exclude_tests=True) if c.path.endswith("broken.py")]
        self.assertEqual([c.kind for c in cs], ["window"])


class Bm25Tests(unittest.TestCase):
    def test_matching_document_ranks_first(self):
        docs = [["cache", "user", "session"], ["parse", "json", "file"], ["render", "html", "template"]]
        vocab, counts = build_counts(docs)
        bm = BM25(counts)
        ids, cnts = vocab.encode(["json", "parse"])
        s = bm.scores(ids, cnts)
        self.assertEqual(int(np.argmax(s)), 1)
        self.assertEqual(s[0], 0.0)


class FusionTests(unittest.TestCase):
    def test_rrf_prefers_items_high_in_both_lists(self):
        fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
        self.assertEqual([i for i, _ in fused][:2], [1, 3])

    def test_rrf_scores_are_reciprocal_ranks(self):
        fused = dict(rrf([[7]], k=60))
        self.assertAlmostEqual(fused[7], 1 / 61)


class BenchmarkTests(unittest.TestCase):
    def test_query_cleaning(self):
        self.assertIsNone(clean_docstring_query("Too short."))
        q = clean_docstring_query("Return the :class:`Foo` object that `bar` makes for you today.\n\nMore.")
        self.assertEqual(q, "Return the Foo object that bar makes for you today.")

    def test_hit_rule(self):
        q = Query("x", "a.py", 10, 30, "f", True)

        class C:
            def __init__(s, p, a, b):
                s.path, s.start, s.end = p, a, b
        self.assertTrue(is_hit(C("a.py", 10, 30), q))
        self.assertTrue(is_hit(C("a.py", 5, 100), q))      # big window containing target
        self.assertFalse(is_hit(C("a.py", 40, 60), q))
        self.assertFalse(is_hit(C("b.py", 10, 30), q))
        self.assertFalse(is_hit(C("a.py", 25, 200), q))    # overlaps only a sliver of the target

    def test_split_is_by_file_and_deterministic(self):
        qs = [Query("t", f"f{i % 12}.py", i, i + 1, "n", True) for i in range(120)]
        tr, te = split_by_file(qs)
        self.assertFalse({q.path for q in tr} & {q.path for q in te})
        self.assertEqual(split_by_file(qs)[1], te)

    def test_metrics(self):
        m = per_query_metrics([1, 2, 0, 11])
        self.assertEqual(list(m["hit@1"]), [1, 0, 0, 0])
        self.assertEqual(list(m["hit@10"]), [1, 1, 0, 0])
        self.assertEqual(list(m["mrr@10"]), [1.0, 0.5, 0.0, 0.0])
        self.assertEqual(list(m["recall@50"]), [1, 1, 0, 1])

    def test_bootstrap_ci_contains_mean_and_shrinks(self):
        rng = np.random.default_rng(0)
        vals = rng.random(400)
        groups = np.repeat(np.arange(40), 10)
        mean, (lo, hi) = bootstrap_ci(vals, groups, 500)
        self.assertLessEqual(lo, mean)
        self.assertGreaterEqual(hi, mean)
        self.assertLess(hi - lo, 0.2)
        d, (dlo, dhi) = paired_diff_ci(vals, vals, groups, 200)
        self.assertEqual((d, dlo, dhi), (0.0, 0.0, 0.0))

    def test_benchmark_drops_duplicate_queries(self):
        chunks = chunk_repo(make_repo(), "", hide_docstrings=True, exclude_tests=True)
        qs = build_benchmark(chunks, min_words=4)
        self.assertEqual(len({q.text.lower() for q in qs}), len(qs))
        self.assertTrue(any(q.name == "read_lines" for q in qs))


class IndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = make_repo()
        cls.chunks = chunk_repo(cls.repo, "", hide_docstrings=False, exclude_tests=True)
        cls.index = CodeIndex.build(cls.chunks)

    def test_search_finds_the_right_function(self):
        for mode in ("bm25", "hybrid", "auto"):
            hits = self.index.search("check whether a password matches the stored digest", k=3, mode=mode)
            self.assertEqual(hits[0].chunk.name, "verify_password", mode)

    def test_bad_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.index.search("anything", mode="nope")

    def test_cache(self):
        idx = CodeIndex.build(self.chunks)
        idx.search("walk a directory for python files")
        idx.search("walk a directory for python files")
        self.assertEqual(idx.cache_info()["hits"], 1)

    def test_save_load_roundtrip_is_identical(self):
        with tempfile.TemporaryDirectory() as d:
            self.index.save(d)
            self.assertFalse(any(p.suffix == ".pkl" for p in Path(d).iterdir()))
            loaded = CodeIndex.load(d)
        q = "create a new session token for a user"
        a = [(h.chunk.id, round(h.score, 6)) for h in self.index.search(q, k=5, mode="hybrid", use_cache=False)]
        b = [(h.chunk.id, round(h.score, 6)) for h in loaded.search(q, k=5, mode="hybrid", use_cache=False)]
        self.assertEqual(a, b)


class RerankerTests(unittest.TestCase):
    def test_learns_and_roundtrips(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, len(FEATURE_NAMES)))
        y = (X[:, 0] + 0.1 * rng.normal(size=300) > 0).astype(int)
        r = Reranker.fit(X, y)
        self.assertEqual(next(iter(r.weights())), FEATURE_NAMES[0])      # biggest weight first
        with tempfile.TemporaryDirectory() as d:
            r.save(Path(d) / "r.json")
            r2 = Reranker.load(Path(d) / "r.json")
        np.testing.assert_allclose(r.score(X), r2.score(X))

    def test_rerank_mode_runs_after_training(self):
        repo = make_repo()
        idx = CodeIndex.build(chunk_repo(repo, "", hide_docstrings=False, exclude_tests=True))
        rng = np.random.default_rng(2)
        X = rng.normal(size=(60, len(FEATURE_NAMES)))
        idx.reranker = Reranker.fit(X, (X[:, 0] > 0).astype(int))
        self.assertTrue(idx.search("read a file into lines", k=3, mode="rerank"))


class QaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = CodeIndex.build(chunk_repo(make_repo(), "", hide_docstrings=False, exclude_tests=True))

    def test_citation_validation(self):
        self.assertEqual(validate_citations("see [1] and [3]", 3)["grounded"], True)
        v = validate_citations("see [1] and [9]", 3)
        self.assertEqual(v["invalid"], [9])
        self.assertFalse(v["grounded"])
        self.assertFalse(validate_citations("no citations", 3)["grounded"])

    def test_extractive_answer_is_grounded(self):
        r = answer_question(self.index, "how are passwords hashed", ExtractiveGenerator(), k=3)
        self.assertTrue(r["citations"]["grounded"])
        self.assertIn("pkg/auth.py", r["answer"])
        self.assertEqual(len(r["sources"]), 3)


class ZipTests(unittest.TestCase):
    def _zip(self, names):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for n in names:
                zf.writestr(n, "x")
        return buf.getvalue()

    def test_strips_top_folder(self):
        with tempfile.TemporaryDirectory() as d:
            extract_zip_archive(self._zip(["repo-1/", "repo-1/a/b.py"]), Path(d) / "out")
            self.assertTrue((Path(d) / "out/a/b.py").exists())

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                extract_zip_archive(self._zip(["repo-1/../../evil.py"]), Path(d) / "out")


if __name__ == "__main__":
    unittest.main()
