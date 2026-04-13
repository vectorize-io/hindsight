"""
Regression tests for chunk_storage.store_chunks_batch idempotency.

Covers vectorize-io/hindsight#977: re-submitting a retain under the same
document_id must not fail with ``UniqueViolationError`` on ``pk_chunks``.
The upstream retain paths (cascade delete on first batch, delta retain)
should usually prevent a chunk_id collision, but any bug in those paths
used to surface as a raw Postgres constraint violation. ``store_chunks_batch``
is now idempotent: inserting the same ``chunk_id`` twice overwrites the
existing row rather than raising.
"""

from datetime import datetime, timezone

import pytest

from hindsight_api.engine.retain import chunk_storage
from hindsight_api.engine.retain.types import ChunkMetadata


def _ts() -> float:
    return datetime.now(timezone.utc).timestamp()


async def _seed_bank_and_document(conn, bank_id: str, document_id: str) -> None:
    """Insert the minimum rows required for the chunks FK to pass."""
    await conn.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )
    await conn.execute(
        """
        INSERT INTO documents (id, bank_id, original_text, content_hash)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (id, bank_id) DO NOTHING
        """,
        document_id,
        bank_id,
        "seed",
        "seed-hash",
    )


@pytest.mark.asyncio
async def test_store_chunks_batch_is_idempotent_for_same_chunk_id(memory):
    """
    Regression for #977.

    Directly exercises the chunk insert path: inserting a ChunkMetadata with
    a chunk_index that already exists (i.e., the same chunk_id) must not
    raise. The new content should overwrite the old one.
    """
    bank_id = f"test_chunk_upsert_{_ts()}"
    document_id = "doc-upsert-regression"

    pool = await memory._get_pool()
    try:
        async with pool.acquire() as conn:
            await _seed_bank_and_document(conn, bank_id, document_id)

            # First insert — fresh chunks at indices 0, 1, 2.
            v1 = [
                ChunkMetadata(chunk_text="alpha", fact_count=1, content_index=0, chunk_index=0),
                ChunkMetadata(chunk_text="beta", fact_count=1, content_index=0, chunk_index=1),
                ChunkMetadata(chunk_text="gamma", fact_count=1, content_index=0, chunk_index=2),
            ]
            v1_map = await chunk_storage.store_chunks_batch(conn, bank_id, document_id, v1)
            assert set(v1_map.keys()) == {0, 1, 2}

            # Second insert — overlapping chunk_index (1 and 2) with new text,
            # plus a fresh chunk at index 3. Before the fix this raised
            # asyncpg.exceptions.UniqueViolationError on pk_chunks; after the
            # fix the conflicting rows are overwritten and the new one is
            # inserted.
            v2 = [
                ChunkMetadata(chunk_text="beta-updated", fact_count=1, content_index=0, chunk_index=1),
                ChunkMetadata(chunk_text="gamma-updated", fact_count=1, content_index=0, chunk_index=2),
                ChunkMetadata(chunk_text="delta", fact_count=1, content_index=0, chunk_index=3),
            ]
            v2_map = await chunk_storage.store_chunks_batch(conn, bank_id, document_id, v2)
            assert set(v2_map.keys()) == {1, 2, 3}

            # Verify the stored state matches the upserted content.
            rows = await conn.fetch(
                """
                SELECT chunk_index, chunk_text, content_hash
                FROM chunks
                WHERE document_id = $1 AND bank_id = $2
                ORDER BY chunk_index
                """,
                document_id,
                bank_id,
            )
            by_index = {row["chunk_index"]: row for row in rows}

            assert set(by_index.keys()) == {0, 1, 2, 3}, (
                "Expected four chunks total after upsert (0 untouched, 1-2 overwritten, 3 new)"
            )
            assert by_index[0]["chunk_text"] == "alpha", "Untouched chunk must be preserved"
            assert by_index[1]["chunk_text"] == "beta-updated", "Conflicting chunk must be overwritten"
            assert by_index[2]["chunk_text"] == "gamma-updated", "Conflicting chunk must be overwritten"
            assert by_index[3]["chunk_text"] == "delta", "New chunk must be inserted"

            # content_hash should reflect the new text, not the original.
            assert by_index[1]["content_hash"] == chunk_storage.compute_chunk_hash("beta-updated")
            assert by_index[2]["content_hash"] == chunk_storage.compute_chunk_hash("gamma-updated")
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM chunks WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM documents WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_store_chunks_batch_second_call_with_identical_payload(memory):
    """
    The exact #977 shape: ``store_chunks_batch`` called twice with the same
    chunks must succeed both times (the second call is a no-op in terms of
    stored content, but must not raise).
    """
    bank_id = f"test_chunk_upsert_identical_{_ts()}"
    document_id = "doc-upsert-identical"

    pool = await memory._get_pool()
    try:
        async with pool.acquire() as conn:
            await _seed_bank_and_document(conn, bank_id, document_id)

            chunks = [
                ChunkMetadata(chunk_text=f"chunk-{i}", fact_count=1, content_index=0, chunk_index=i)
                for i in range(5)
            ]

            await chunk_storage.store_chunks_batch(conn, bank_id, document_id, chunks)
            # Second call with identical chunks — must not raise.
            await chunk_storage.store_chunks_batch(conn, bank_id, document_id, chunks)

            count = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE document_id = $1 AND bank_id = $2",
                document_id,
                bank_id,
            )
            assert count == 5, "Second identical insert should not duplicate rows"
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM chunks WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM documents WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)
