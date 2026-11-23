"""Oracle hot-query shapes from #6771: shared cursors, one-pass vector search, index hints.

The SQL itself runs against a real Oracle in ``test-api-oracle``; these pin the shapes
without one, and that the PostgreSQL SQL is left exactly as it was.
"""

import json
import uuid

import pytest

from hindsight_api.engine.db.oracle import OracleConnection, _rewrite_pg_to_oracle
from hindsight_api.engine.db.ops_oracle import OracleOps
from hindsight_api.engine.memories.pg.banks import bank_page_rows, list_bank_rows
from hindsight_api.engine.sql.oracle import OracleDialect
from hindsight_api.engine.sql.postgresql import PostgreSQLDialect

_ARM = dict(
    table="memory_units",
    cols="id, text, fact_type",
    embedding_param=":1",
    bank_id_param=":2",
    fetch_limit=50,
    min_similarity=0.3,
    tags_clause="AND tags IS NOT NULL",
)


class _FakeConn:
    def __init__(self, backend_type: str, rows: list | None = None) -> None:
        self.backend_type = backend_type
        self.rows = rows or []
        self.calls: list[tuple[str, tuple]] = []

    async def fetch(self, query: str, *args: object) -> list:
        self.calls.append((query, args))
        return self.rows


def test_rewrite_is_cached_and_shares_no_mutable_state() -> None:
    sql = "UPDATE banks SET name = $1 WHERE bank_id = $2 RETURNING bank_id, name"
    first = _rewrite_pg_to_oracle(sql)
    assert _rewrite_pg_to_oracle(sql) is first
    assert first.returning_cols == ("bank_id", "name")


def test_in_lists_pad_to_a_power_of_two_with_per_query_names() -> None:
    query, _, _ = _rewrite_pg_to_oracle("SELECT id FROM t WHERE id = ANY($1) AND kind != ALL($2)")

    params: dict = {"1": ["a", "b", "c"], "2": ["x"]}
    three, _ = OracleConnection._expand_any_lists(query, params)
    four, _ = OracleConnection._expand_any_lists(query, {"1": ["a", "b", "c", "d"], "2": ["y"]})
    # Same SQL text for 3 and 4 items, and across calls: Oracle reuses the cursor.
    assert three == four
    assert "id IN (:any0_0, :any0_1, :any0_2, :any0_3)" in three
    assert "kind NOT IN (:any1_0)" in three
    # Padding repeats the last item, so IN / NOT IN mean the same thing.
    assert [params[f"any0_{i}"] for i in range(4)] == ["a", "b", "c", "c"]


def test_in_lists_past_the_last_bucket_stay_exact() -> None:
    query, _, _ = _rewrite_pg_to_oracle("SELECT id FROM t WHERE id = ANY($1)")
    ids = [str(i) for i in range(1001)]
    expanded, params = OracleConnection._expand_any_lists(query, {"1": json.dumps(ids)})
    assert params is not None
    assert expanded.count(":any0_") == 1001


def test_oracle_ranks_several_fact_types_in_one_scan() -> None:
    arms = OracleDialect().build_semantic_arms(fact_types=["world", "experience"], **_ARM)
    assert len(arms) == 1
    (arm,) = arms
    assert "fact_type IN ('world', 'experience')" in arm
    assert "ROW_NUMBER() OVER (PARTITION BY fact_type ORDER BY VECTOR_DISTANCE(embedding, :1, COSINE))" in arm
    assert "WHERE rn_ <= 50" in arm
    # Recall keeps the first rows it reads per type, so they must arrive nearest first.
    assert "ORDER BY fact_type, rn_" in arm
    assert "AND tags IS NOT NULL" in arm


def test_oracle_single_fact_type_keeps_its_arm() -> None:
    dialect = OracleDialect()
    assert dialect.build_semantic_arms(fact_types=["world"], **_ARM) == [
        dialect.build_semantic_arm(fact_type="world", **_ARM)
    ]


def test_postgres_keeps_one_arm_per_fact_type() -> None:
    dialect = PostgreSQLDialect()
    assert dialect.build_semantic_arms(fact_types=["world", "experience"], **_ARM) == [
        dialect.build_semantic_arm(fact_type="world", **_ARM),
        dialect.build_semantic_arm(fact_type="experience", **_ARM),
    ]


def test_oracle_bm25_arm_pins_the_text_index() -> None:
    arm = OracleDialect().build_bm25_arm(
        table="memory_units",
        cols="id",
        fact_type="world",
        bank_id_param=":2",
        limit_param=":3",
        text_param=":4",
    )
    assert "/*+ INDEX(m idx_mu_content_text) */" in arm
    assert "FROM memory_units m" in arm


@pytest.mark.asyncio
async def test_fetch_unit_dates_batches_ids_per_thousand() -> None:
    conn = _FakeConn("oracle", rows=[{"id": "u"}])
    ids = [str(uuid.uuid4()) for _ in range(1500)]
    rows = await OracleOps().fetch_unit_dates(conn, "memory_units", ids)  # type: ignore[arg-type]
    assert [len(args[0]) for _, args in conn.calls] == [1000, 500]
    assert all("WHERE id = ANY($1)" in q for q, _ in conn.calls)
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_oracle_bank_watermark_reads_the_utc_index() -> None:
    conn = _FakeConn("oracle")
    await list_bank_rows(conn=conn, fq_table=lambda t: t, where_clause="", params=[])
    await bank_page_rows(conn=conn, fq_table=lambda t: t, bank_ids=["b"], sql_owned=["b"])
    for query, _ in conn.calls:
        assert "/*+ INDEX(m idx_mu_bank_updated_at) */" in query
        assert "ORDER BY SYS_EXTRACT_UTC(m.updated_at) DESC FETCH FIRST 1 ROW ONLY" in query
        assert "MAX(m.updated_at)" not in query


@pytest.mark.asyncio
async def test_postgres_bank_watermark_sql_is_unchanged() -> None:
    conn = _FakeConn("postgresql")
    await list_bank_rows(conn=conn, fq_table=lambda t: t, where_clause="", params=[])
    await bank_page_rows(conn=conn, fq_table=lambda t: t, bank_ids=["b"], sql_owned=["b"])
    listed, paged = (q for q, _ in conn.calls)
    assert "(SELECT MAX(m.updated_at) FROM memory_units m WHERE m.bank_id = b.bank_id) AS last_fact_at" in listed
    assert "SELECT bank_id, MAX(updated_at) AS last_fact_at" in paged
    assert "/*+" not in listed + paged
