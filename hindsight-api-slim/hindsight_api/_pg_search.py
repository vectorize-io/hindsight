"""Helpers for ParadeDB pg_search index configuration."""

from __future__ import annotations

import re
from collections.abc import Sequence

PG_SEARCH_TOKENIZER_ENV = "HINDSIGHT_API_TEXT_SEARCH_EXTENSION_PG_SEARCH_TOKENIZER"

_SIMPLE_TOKENIZERS = {
    "unicode_words",
    "simple",
    "whitespace",
    "literal",
    "literal_normalized",
    "chinese_compatible",
    "icu",
    "jieba",
    "source_code",
}

_TOKENIZER_ALIASES = {
    "chinese_lindera": "lindera(chinese)",
    "japanese_lindera": "lindera(japanese)",
    "korean_lindera": "lindera(korean)",
    "lindera_chinese": "lindera(chinese)",
    "lindera_japanese": "lindera(japanese)",
    "lindera_korean": "lindera(korean)",
}


def normalize_pg_search_tokenizer(value: str | None) -> str:
    """Validate and normalize a ParadeDB pg_search tokenizer setting.

    Returns an empty string when unset. The returned value is safe to embed after
    ``pdb.`` in a CREATE INDEX expression.
    """

    tokenizer = (value or "").strip().lower()
    if not tokenizer:
        return ""

    if tokenizer in _TOKENIZER_ALIASES:
        return _TOKENIZER_ALIASES[tokenizer]

    if tokenizer in _SIMPLE_TOKENIZERS:
        return tokenizer

    lindera_match = re.fullmatch(r"lindera\((chinese|japanese|korean)\)", tokenizer)
    if lindera_match:
        return tokenizer

    ngram_match = re.fullmatch(r"(ngram|edge_ngram)\((\d{1,3}),\s*(\d{1,3})\)", tokenizer)
    if ngram_match:
        kind, min_gram, max_gram = ngram_match.groups()
        min_value = int(min_gram)
        max_value = int(max_gram)
        if min_value <= 0 or min_value > max_value:
            raise ValueError(
                f"Invalid {PG_SEARCH_TOKENIZER_ENV}: {value!r}. "
                "ngram and edge_ngram require positive min/max gram sizes with min <= max."
            )
        return f"{kind}({min_value},{max_value})"

    raise ValueError(
        f"Invalid {PG_SEARCH_TOKENIZER_ENV}: {value!r}. "
        "Supported values are: unicode_words, simple, whitespace, literal, "
        "literal_normalized, chinese_compatible, icu, jieba, source_code, "
        "chinese_lindera, japanese_lindera, korean_lindera, or "
        "lindera(chinese|japanese|korean), ngram(min,max), or edge_ngram(min,max)."
    )


def pg_search_bm25_columns(
    key_field: str,
    text_fields: Sequence[str],
    tokenizer: str | None,
) -> str:
    """Build a ParadeDB BM25 column list for CREATE INDEX."""

    normalized = normalize_pg_search_tokenizer(tokenizer)
    if not normalized:
        return ", ".join([key_field, *text_fields])

    return ", ".join([key_field, *(f"({field}::pdb.{normalized})" for field in text_fields)])
