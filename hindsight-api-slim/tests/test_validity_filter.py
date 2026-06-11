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
