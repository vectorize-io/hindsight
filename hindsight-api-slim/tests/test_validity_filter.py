"""Default-retrieval filtering of temporally superseded facts.

Superseded facts keep their row in ``memory_units`` (``valid_until`` set) so
as-of queries can reach them, but every default retrieval arm must hide them.
The predicate is centralized in ``sql.base.validity_clause`` and injected into
the dialect-level SQL builders; these tests pin both layers:

* construction: every arm/CTE builder emits the clause on both dialects, so a
  refactor can't silently drop the filter from one arm;
* behavior: a fact disappears from recall when superseded and reappears when
  the supersession is cleared (proves the SQL actually filters, end to end).
"""

from datetime import UTC, datetime, timedelta

import asyncpg
import pytest

from hindsight_api.engine.db.ops_oracle import OracleOps
from hindsight_api.engine.db.ops_postgresql import PostgreSQLOps
from hindsight_api.engine.memory_engine import Budget
from hindsight_api.engine.sql.base import validity_clause
from hindsight_api.engine.sql.oracle import OracleDialect
from hindsight_api.engine.sql.postgresql import PostgreSQLDialect

CLAUSE = "valid_until IS NULL"


def test_validity_clause_alias():
    assert validity_clause() == "AND valid_until IS NULL"
    assert validity_clause("mu") == "AND mu.valid_until IS NULL"


def test_validity_clause_as_of_variant():
    clause = validity_clause("mu", as_of_param="$7")
    # Undated facts stay visible (no occurred_start -> presumed valid at any instant),
    # superseded rows reappear when they were still true at the instant.
    assert clause == (
        "AND (mu.occurred_start IS NULL OR mu.occurred_start <= $7) AND (mu.valid_until IS NULL OR mu.valid_until > $7)"
    )


@pytest.mark.parametrize("dialect", [PostgreSQLDialect(), OracleDialect()], ids=["pg", "oracle"])
def test_arm_builders_accept_validity_override(dialect):
    override = validity_clause(as_of_param="$9")
    semantic = dialect.build_semantic_arm(
        table="memory_units",
        cols="id, text",
        fact_type="world",
        embedding_param="$1",
        bank_id_param="$2",
        fetch_limit=100,
        min_similarity=0.1,
        validity_sql=override,
    )
    # The as-of predicate must REPLACE the default filter, not stack on it —
    # stacking would keep hiding superseded rows that were valid at the instant.
    assert override in semantic
    assert CLAUSE not in semantic.replace(override, "")


@pytest.mark.parametrize("dialect", [PostgreSQLDialect(), OracleDialect()], ids=["pg", "oracle"])
def test_semantic_and_bm25_arms_filter_superseded(dialect):
    semantic = dialect.build_semantic_arm(
        table="memory_units",
        cols="id, text",
        fact_type="world",
        embedding_param="$1",
        bank_id_param="$2",
        fetch_limit=100,
        min_similarity=0.1,
    )
    bm25 = dialect.build_bm25_arm(
        table="memory_units",
        cols="id, text",
        fact_type="world",
        bank_id_param="$2",
        limit_param="$3",
        text_param="$4",
    )
    assert CLAUSE in semantic
    assert CLAUSE in bm25


@pytest.mark.parametrize("ops", [PostgreSQLOps(), OracleOps()], ids=["pg", "oracle"])
def test_graph_expansion_ctes_filter_superseded(ops):
    entity_cte = ops.build_entity_expansion_cte("memory_units", "unit_entities", 5)
    sem_causal_cte = ops.build_semantic_causal_cte("memory_links", "memory_units")
    assert f"mu.{CLAUSE}" in entity_cte
    # Both semantic UNION branches and the causal branch join mu independently;
    # each needs its own filter or superseded rows leak through that branch.
    assert sem_causal_cte.count(f"mu.{CLAUSE}") >= 3


async def test_superseded_fact_hidden_from_recall(memory, pg0_db_url, request_context):
    """End-to-end: supersession hides a fact from recall; clearing restores it."""
    bank_id = f"test_validity_{datetime.now(UTC).timestamp()}"

    await memory.retain_async(
        bank_id=bank_id,
        content="Alice works at Acme Corporation as a software engineer.",
        event_date=datetime(2024, 1, 15, tzinfo=UTC),
        request_context=request_context,
    )

    async def _recall_texts() -> list[str]:
        result = await memory.recall_async(
            bank_id=bank_id,
            query="Where does Alice work?",
            budget=Budget.LOW,
            max_tokens=500,
            fact_type=["world"],
            request_context=request_context,
        )
        return [r.text for r in result.results]

    assert _texts_mention_acme(await _recall_texts()), "fact must be recallable before supersession"

    # Supersede every fact in the bank directly (the supersession worker is a
    # later PR; here we only verify the retrieval-side contract). occurred_start
    # is backfilled first to satisfy chk_mu_supersession_needs_occurred.
    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(
            """
            UPDATE memory_units
            SET occurred_start = COALESCE(occurred_start, mentioned_at, now()),
                valid_until = COALESCE(occurred_start, mentioned_at, now()) + $2::interval,
                superseded_at = now()
            WHERE bank_id = $1
            """,
            bank_id,
            timedelta(days=1),
        )
        assert not _texts_mention_acme(await _recall_texts()), "superseded fact must be hidden from recall"

        await conn.execute(
            "UPDATE memory_units SET valid_until = NULL, superseded_at = NULL WHERE bank_id = $1",
            bank_id,
        )
    finally:
        await conn.close()

    assert _texts_mention_acme(await _recall_texts()), "clearing supersession must restore recall"


def _texts_mention_acme(texts: list[str]) -> bool:
    return any("acme" in t.lower() for t in texts)


async def test_as_of_recall_returns_superseded_fact(memory, pg0_db_url, request_context):
    """Point-in-time recall re-exposes superseded facts valid at that instant.

    Timeline: fact occurs 2024-01-15, superseded effective 2024-06-01.
      as_of 2024-01-01 -> hidden (did not exist yet)
      as_of 2024-03-01 -> visible, with valid_until/superseded_by populated
      as_of 2024-07-01 -> hidden (no longer true)
      default          -> hidden
    """
    bank_id = f"test_asof_{datetime.now(UTC).timestamp()}"

    await memory.retain_async(
        bank_id=bank_id,
        content="Alice works at Acme Corporation as a software engineer.",
        event_date=datetime(2024, 1, 15, tzinfo=UTC),
        request_context=request_context,
    )
    await memory.retain_async(
        bank_id=bank_id,
        content="Alice now works at Beta Industries as a staff engineer.",
        event_date=datetime(2024, 6, 1, tzinfo=UTC),
        request_context=request_context,
    )

    conn = await asyncpg.connect(pg0_db_url)
    try:
        superseder_id = await conn.fetchval(
            "SELECT id FROM memory_units WHERE bank_id = $1 AND lower(text) LIKE '%beta%' LIMIT 1",
            bank_id,
        )
        assert superseder_id is not None
        await conn.execute(
            """
            UPDATE memory_units
            SET occurred_start = $2, valid_until = $3, superseded_at = now(), superseded_by = $4
            WHERE bank_id = $1 AND lower(text) LIKE '%acme%'
            """,
            bank_id,
            datetime(2024, 1, 15, tzinfo=UTC),
            datetime(2024, 6, 1, tzinfo=UTC),
            superseder_id,
        )
    finally:
        await conn.close()

    async def _recall(as_of: datetime | None):
        result = await memory.recall_async(
            bank_id=bank_id,
            query="Where does Alice work?",
            budget=Budget.LOW,
            max_tokens=500,
            fact_type=["world"],
            request_context=request_context,
            as_of=as_of,
        )
        return result.results

    assert not _texts_mention_acme([r.text for r in await _recall(None)])
    assert not _texts_mention_acme([r.text for r in await _recall(datetime(2024, 1, 1, tzinfo=UTC))])
    assert not _texts_mention_acme([r.text for r in await _recall(datetime(2024, 7, 1, tzinfo=UTC))])

    mid_results = await _recall(datetime(2024, 3, 1, tzinfo=UTC))
    acme = [r for r in mid_results if "acme" in r.text.lower()]
    assert acme, "fact valid at as_of must be returned"
    assert acme[0].valid_until is not None and acme[0].valid_until.startswith("2024-06-01")
    assert acme[0].superseded_by == str(superseder_id)
