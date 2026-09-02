"""IDF-aware BM25 query-term selection (native tsvector long-query timeout fix).

Two layers:
- Deterministic unit tests over the selection/fallback logic with a fake
  connection (no DB) — these always run.
- One pg0 integration test that reads real ``pg_stats`` document-frequency
  statistics, proving the catalog SQL keeps the rare (selective) term and drops
  the ubiquitous one.
"""

from __future__ import annotations

from uuid import uuid4

import asyncpg
import pytest

from hindsight_api.engine.search.bm25_term_selection import select_selective_bm25_tokens


class _RaisingConn:
    """Connection whose fetch must never be called."""

    async def fetch(self, *args, **kwargs):  # pragma: no cover - asserts if reached
        raise AssertionError("stats lookup should have been skipped")


class _StubConn:
    def __init__(self, rows=None, error=None):
        self._rows = rows or []
        self._error = error
        self.calls = 0

    async def fetch(self, *args, **kwargs):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._rows


def _row(ord_: int, df: float, has_lexeme: bool = True) -> dict:
    return {"ord": ord_, "df": df, "has_lexeme": has_lexeme}


# --------------------------------------------------------------------------
# Fast paths — no stats lookup
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_under_cap_returns_unchanged_without_query():
    tokens = ["alpha", "beta", "gamma"]
    result = await select_selective_bm25_tokens(
        _RaisingConn(), tokens, schema="public", table="memory_units", language="english", max_terms=16
    )
    assert result is tokens


@pytest.mark.asyncio
async def test_zero_cap_disables_selection():
    tokens = ["alpha"] * 50
    result = await select_selective_bm25_tokens(
        _RaisingConn(), tokens, schema="public", table="memory_units", language="english", max_terms=0
    )
    assert result is tokens


# --------------------------------------------------------------------------
# Selection logic
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keeps_lowest_df_terms_preserving_order():
    tokens = ["common", "rare", "mid", "everywhere"]
    # df: everywhere(0.9) > common(0.5) > mid(0.2) > rare(0.01)
    rows = [_row(1, 0.5), _row(2, 0.01), _row(3, 0.2), _row(4, 0.9)]
    result = await select_selective_bm25_tokens(
        _StubConn(rows), tokens, schema="public", table="memory_units", language="english", max_terms=2
    )
    # Two lowest-df terms are "rare" and "mid"; returned in original order.
    assert result == ["rare", "mid"]


@pytest.mark.asyncio
async def test_drops_stopwords_before_counting():
    tokens = ["the", "rare", "common", "and"]
    rows = [
        _row(1, 0.0, has_lexeme=False),  # stopword -> no lexeme
        _row(2, 0.01),
        _row(3, 0.6),
        _row(4, 0.0, has_lexeme=False),  # stopword
    ]
    result = await select_selective_bm25_tokens(
        _StubConn(rows), tokens, schema="public", table="memory_units", language="english", max_terms=3
    )
    # Stopwords excluded entirely; only the two real terms survive.
    assert result == ["rare", "common"]


# --------------------------------------------------------------------------
# Fallbacks — never block recall
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_falls_back_to_first_n_on_lookup_error():
    tokens = ["a", "b", "c", "d"]
    result = await select_selective_bm25_tokens(
        _StubConn(error=RuntimeError("boom")),
        tokens,
        schema="public",
        table="memory_units",
        language="english",
        max_terms=2,
    )
    assert result == ["a", "b"]


@pytest.mark.asyncio
async def test_falls_back_when_no_stats_rows():
    tokens = ["a", "b", "c", "d"]
    result = await select_selective_bm25_tokens(
        _StubConn(rows=[]), tokens, schema="public", table="memory_units", language="english", max_terms=2
    )
    assert result == ["a", "b"]


@pytest.mark.asyncio
async def test_all_stopwords_falls_back_to_first_n():
    tokens = ["the", "an", "of", "and"]
    rows = [_row(i + 1, 0.0, has_lexeme=False) for i in range(4)]
    result = await select_selective_bm25_tokens(
        _StubConn(rows), tokens, schema="public", table="memory_units", language="english", max_terms=2
    )
    assert result == ["the", "an"]


# --------------------------------------------------------------------------
# Integration — real pg_stats document frequencies
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_selects_rare_term_from_real_pg_stats(pg0_db_url):
    """End-to-end: ANALYZE populates per-lexeme frequencies; selection keeps the
    rare term and drops the ubiquitous one, using the same catalog SQL as prod."""
    table = f"mu_dfsel_{uuid4().hex[:8]}"
    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(f'CREATE TABLE public."{table}" (id serial primary key, search_vector tsvector)')
        # "ubiquitous" in every row; alpha/beta/gamma spread; "zzrareterm" in a few.
        for i in range(600):
            words = "ubiquitous alpha beta gamma" if i % 2 == 0 else "ubiquitous alpha delta epsilon"
            if i % 60 == 0:
                words += " zzrareterm"
            await conn.execute(
                f'INSERT INTO public."{table}" (search_vector) VALUES (to_tsvector($1::regconfig, $2))',
                "english",
                words,
            )
        await conn.execute(f'ANALYZE public."{table}"')

        tokens = ["ubiquitous", "alpha", "beta", "gamma", "delta", "epsilon", "zzrareterm"]
        kept = await select_selective_bm25_tokens(
            conn, tokens, schema="public", table=table, language="english", max_terms=2
        )

        assert "zzrareterm" in kept  # rarest term is kept
        assert "ubiquitous" not in kept  # most common term is dropped
        assert len(kept) == 2
    finally:
        await conn.execute(f'DROP TABLE IF EXISTS public."{table}"')
        await conn.close()
