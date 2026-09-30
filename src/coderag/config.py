"""Defaults shared across the package."""
from __future__ import annotations

SEED = 42

# Directories never worth indexing (VCS metadata, vendored deps, build output, generated code).
DEFAULT_EXCLUDE_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env", "__pycache__", "build", "dist",
    "site-packages", ".tox", ".eggs", ".mypy_cache", ".pytest_cache", "migrations", "locale",
})
# Excluded additionally when benchmarking, so the benchmark measures library code, not its test-suite.
TEST_DIRS = frozenset({"tests", "test", "testing"})

CODE_EXTENSIONS = (".py",)
DOC_EXTENSIONS = (".md", ".rst", ".txt")
MAX_FILE_BYTES = 1_000_000

# Chunking
MAX_FUNCTION_LINES = 150     # longer functions are split into overlapping parts
PART_LINES = 80
PART_OVERLAP = 10
CLASS_HEADER_CAP = 60
MODULE_CHUNK_CAP = 60
WINDOW_LINES = 40            # baseline "fixed-size window" chunker
WINDOW_OVERLAP = 10
DOC_WINDOW_LINES = 30
DOC_WINDOW_OVERLAP = 5

# Retrieval
BM25_K1 = 1.2
BM25_B = 0.75
NAME_BOOST = 2               # symbol-name tokens are repeated this many times in the document
LSA_DIMS = 256
RRF_K = 60
RETRIEVAL_DEPTH = 100        # candidates taken from each retriever before fusion
RERANK_DEPTH = 50            # candidates the reranker re-orders

# Pinned versions used for the benchmark (a tag AND the commit it must resolve to).
PRESETS = {
    "flask": {
        "url": "https://github.com/pallets/flask", "tag": "3.1.3", "subdir": "src/flask",
        "commit": "22d924701a6ae2e4cd01e9a15bbaf3946094af65",
    },
    "requests": {
        "url": "https://github.com/psf/requests", "tag": "v2.32.5", "subdir": "src/requests",
        "commit": "b25c87d7cb8d6a18a37fa12442b5f883f9e41741",
    },
    "django": {
        "url": "https://github.com/django/django", "tag": "5.2.6", "subdir": "django",
        "commit": "75c4403f07b8ad25893f7832dbe8fc6814b53b2d",
    },
}
