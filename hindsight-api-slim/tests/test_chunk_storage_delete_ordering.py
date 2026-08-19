"""Regression coverage for deterministic chunk deletion ordering."""

import uuid

import pytest

from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain import chunk_storage
from hindsight_api.models import RequestContext


class RecordingConn:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def execute(self, sql: str, *args: object) -> None:
        self.calls.append((sql, args))


@pytest.mark.asyncio
async def test_delete_chunks_by_ids_predeletes_links_before_chunks():
    conn = RecordingConn()
    chunk_ids = ["chunk-b", "chunk-a"]

    await chunk_storage.delete_chunks_by_ids(conn, chunk_ids)

    assert len(conn.calls) == 2
    link_sql, link_args = conn.calls[0]
    chunk_sql, chunk_args = conn.calls[1]

    assert link_args == (chunk_ids,)
    assert chunk_args == (chunk_ids,)

    assert "DELETE FROM" in link_sql
    assert "memory_links" in link_sql
    assert "target_units AS MATERIALIZED" in link_sql
    assert "ordered_links AS MATERIALIZED" in link_sql
    assert "ORDER BY" in link_sql
    assert "FOR UPDATE OF ml" in link_sql

    assert "DELETE FROM" in chunk_sql
    assert "chunks" in chunk_sql
    assert "ordered_chunks AS MATERIALIZED" in chunk_sql
    assert "ORDER BY chunk_id" in chunk_sql
    assert "FOR UPDATE" in chunk_sql


@pytest.mark.asyncio
async def test_delete_chunks_by_ids_matches_link_endpoints_through_indexable_joins():
    """The endpoint match must stay two single-column joins, never an OR across columns.

    ``tu.id = ml.from_unit_id OR tu.id = ml.to_unit_id`` cannot be driven from either
    endpoint index, so PostgreSQL made memory_links the outer relation of a semi join
    and sequentially scanned the entire table on every delete — 20s at 1.5M links,
    over the asyncpg command timeout at production sizes (issue #3387).
    """
    conn = RecordingConn()

    await chunk_storage.delete_chunks_by_ids(conn, ["chunk-a"])

    link_sql = conn.calls[0][0]
    normalized = " ".join(link_sql.split())
    assert "tu.id = ml.from_unit_id OR tu.id = ml.to_unit_id" not in normalized, (
        "the endpoint match must not reintroduce an OR across from_unit_id/to_unit_id"
    )
    assert "JOIN target_units tu ON tu.id = ml.from_unit_id" in link_sql
    assert "JOIN target_units tu ON tu.id = ml.to_unit_id" in link_sql
    assert "UNION" in link_sql


@pytest.mark.asyncio
async def test_delete_chunks_by_ids_noops_without_chunks():
    conn = RecordingConn()

    await chunk_storage.delete_chunks_by_ids(conn, [])

    assert conn.calls == []


@pytest.mark.asyncio
async def test_delete_chunks_by_ids_sweeps_links_on_both_endpoints(
    memory: MemoryEngine, request_context: RequestContext
):
    """Both endpoints of the deleted chunk's units lose their links; other links survive.

    Equivalence guard for the #3387 rewrite: the OR-across-columns predicate became a
    UNION of two single-column joins, so a link is now matched twice — once from each
    side — and a rewrite that dropped one arm would silently leave half the links behind
    for the FK cascade to delete in executor-chosen order, reopening the #2570 deadlock.
    """
    bank_id = f"test-link-sweep-{uuid.uuid4().hex[:8]}"
    if not get_memories().writes_memory_rows_in_sql_for(bank_id):
        pytest.skip("memory_links reference memory_units rows, which this store keeps outside SQL")

    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    pool = await memory._get_pool()

    doc_id = str(uuid.uuid4())
    doomed_chunk = f"{doc_id}_0"
    kept_chunk = f"{doc_id}_1"

    async def _unit(conn, chunk_id: str | None) -> uuid.UUID:
        unit_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO memory_units (
                id, bank_id, text, fact_type, event_date, document_id, chunk_id, created_at, updated_at
            ) VALUES ($1, $2, $3, 'experience', NOW(), $4, $5, NOW(), NOW())
            """,
            unit_id,
            bank_id,
            f"fact {unit_id}",
            doc_id if chunk_id else None,
            chunk_id,
        )
        return unit_id

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO documents (id, bank_id, original_text, content_hash, created_at, updated_at)
            VALUES ($1, $2, 'doomed text\n\nkept text', 'hash-old', NOW(), NOW())
            """,
            doc_id,
            bank_id,
        )
        for idx, chunk_id in enumerate((doomed_chunk, kept_chunk)):
            await conn.execute(
                """
                INSERT INTO chunks (chunk_id, document_id, bank_id, chunk_index, chunk_text, content_hash)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                chunk_id,
                doc_id,
                bank_id,
                idx,
                f"text {idx}",
                f"hash-{idx}",
            )

        doomed = await _unit(conn, doomed_chunk)
        kept = await _unit(conn, kept_chunk)
        unrelated = await _unit(conn, None)

        # One link per direction on the doomed unit, plus one that must survive.
        for from_id, to_id in ((doomed, kept), (unrelated, doomed), (kept, unrelated)):
            await conn.execute(
                """
                INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, bank_id)
                VALUES ($1, $2, 'semantic', 0.8, $3)
                """,
                from_id,
                to_id,
                bank_id,
            )

    async with pool.acquire() as conn:
        async with conn.transaction():
            await chunk_storage.delete_chunks_by_ids(conn, [doomed_chunk], bank_id, ops=memory._backend.ops)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT from_unit_id, to_unit_id FROM memory_links WHERE bank_id = $1",
            bank_id,
        )
        queued = await conn.fetch(
            "SELECT unit_id FROM graph_maintenance_queue WHERE bank_id = $1",
            bank_id,
        )

    assert {(str(r["from_unit_id"]), str(r["to_unit_id"])) for r in rows} == {(str(kept), str(unrelated))}, (
        "links on both endpoints of the deleted chunk's unit must go; the survivors' link must stay"
    )
    assert {str(r["unit_id"]) for r in queued} == {str(unrelated)}, (
        "a surviving unit whose outgoing link targeted the deleted chunk must be queued for relinking"
    )

    await memory.delete_bank(bank_id, request_context=request_context)
