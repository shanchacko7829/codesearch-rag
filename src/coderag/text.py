"""Code-aware tokenisation shared by indexing and querying.

Identifiers are split on snake_case and camelCase boundaries so that the query
"parse json body" can match ``get_json`` and ``parseJsonBody``. The full
lower-cased identifier is kept as an additional token, so an exact identifier in
a query (``get_json``) still gets an exact-match boost.

Both sides go through the *same* pipeline, so the light stemmer below only has to
be consistent, not linguistically perfect.
"""
from __future__ import annotations

import re

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# English function words + Python keywords that appear in nearly every chunk.
STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does for from had has have how i if in into is it its
may might must of on or our shall should so than that the their them then there these they this those to us
was we were what when where which while who whom why will with would you your
self cls def class return import pass else elif try except raise lambda yield del global nonlocal assert
async await break continue finally py none true false
""".split())


def split_identifier(ident: str) -> list[str]:
    """``getJSONBody`` -> ['get', 'json', 'body'];  ``_cache_key`` -> ['cache', 'key']."""
    parts: list[str] = []
    for piece in ident.split("_"):
        if piece:
            parts.extend(p for p in _CAMEL.split(piece) if p)
    return [p.lower() for p in parts]


def stem(w: str) -> str:
    """Very light suffix stripper: returns/returned/returning -> 'return', handle/handling -> 'handl'."""
    if len(w) <= 2 or not w.isalpha():
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ing") and len(w) >= 5:
        w = w[:-3]
    elif w.endswith("ed") and len(w) >= 4:
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if w.endswith("e") and len(w) >= 3:
        w = w[:-1]
    return w


def tokenize(text: str, keep_full: bool = True) -> list[str]:
    """Tokens in document order (duplicates preserved; term frequency matters for BM25)."""
    out: list[str] = []
    for m in _IDENT.finditer(text):
        ident = m.group()
        parts = split_identifier(ident)
        if keep_full and len(parts) > 1:
            full = ident.strip("_").lower()
            if full and full not in STOPWORDS:
                out.append(full)
        for p in parts:
            if len(p) < 2 or p in STOPWORDS or p.isdigit():
                continue
            out.append(stem(p))
    return out
