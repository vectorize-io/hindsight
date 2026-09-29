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
@pytest.mark.memory_backend_incompatible
async def test_store_chunks_batch_is_idempotent_for_same_chunk_id(memory):
    """
    Regression for #977.

    Directly exercises the chunk insert path: inserting a ChunkMetadata with
    a chunk_index that already exists (i.e., the same chunk_id) must not
    raise. The new content should overwrite the old one.
    """
    bank_id = f"test_chunk_upsert_{_ts()}"
    document_id = "doc-upsert-regression"

    backend = await memory._get_backend()
    ops = backend.ops
    try:
        async with backend.acquire() as conn:
            await _seed_bank_and_document(conn, bank_id, document_id)

            # First insert — fresh chunks at indices 0, 1, 2.
            v1 = [
                ChunkMetadata(chunk_text="alpha", fact_count=1, content_index=0, chunk_index=0),
                ChunkMetadata(chunk_text="beta", fact_count=1, content_index=0, chunk_index=1),
                ChunkMetadata(chunk_text="gamma", fact_count=1, content_index=0, chunk_index=2),
            ]
            v1_map = await chunk_storage.store_chunks_batch(conn, bank_id, document_id, v1, ops=ops)
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
            v2_map = await chunk_storage.store_chunks_batch(conn, bank_id, document_id, v2, ops=ops)
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
        async with backend.acquire() as conn:
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

    backend = await memory._get_backend()
    ops = backend.ops
    try:
        async with backend.acquire() as conn:
            await _seed_bank_and_document(conn, bank_id, document_id)

            chunks = [
                ChunkMetadata(chunk_text=f"chunk-{i}", fact_count=1, content_index=0, chunk_index=i) for i in range(5)
            ]

            await chunk_storage.store_chunks_batch(conn, bank_id, document_id, chunks, ops=ops)
            # Second call with identical chunks — must not raise.
            await chunk_storage.store_chunks_batch(conn, bank_id, document_id, chunks, ops=ops)

            count = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE document_id = $1 AND bank_id = $2",
                document_id,
                bank_id,
            )
            assert count == 5, "Second identical insert should not duplicate rows"
    finally:
        async with backend.acquire() as conn:
            await conn.execute("DELETE FROM chunks WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM documents WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_chunk_ids_do_not_collide_across_banks(memory):
    """
    Regression for #4244.

    ``a`` + ``b_c`` and ``a_b`` + ``c`` used to produce the same chunk id
    (``a_b_c_0``), so a retain in the second bank overwrote the first bank's
    chunk row. Bank ids and document ids are arbitrary user-supplied strings,
    so the collision is reachable from the public API.
    """
    prefix = f"t{int(_ts() * 1000)}"
    bank_a, doc_a = prefix, "b_c"
    bank_b, doc_b = f"{prefix}_b", "c"

    backend = await memory._get_backend()
    ops = backend.ops
    try:
        async with backend.acquire() as conn:
            await _seed_bank_and_document(conn, bank_a, doc_a)
            await _seed_bank_and_document(conn, bank_b, doc_b)

            map_a = await chunk_storage.store_chunks_batch(
                conn,
                bank_a,
                doc_a,
                [ChunkMetadata(chunk_text="bank-a-secret", fact_count=1, content_index=0, chunk_index=0)],
                ops=ops,
            )
            map_b = await chunk_storage.store_chunks_batch(
                conn,
                bank_b,
                doc_b,
                [ChunkMetadata(chunk_text="bank-b-text", fact_count=1, content_index=0, chunk_index=0)],
                ops=ops,
            )

            assert map_a[0] != map_b[0], "Distinct bank/document pairs must not share a chunk id"

            text_a = await conn.fetchval(
                "SELECT chunk_text FROM chunks WHERE bank_id = $1 AND document_id = $2 AND chunk_index = 0",
                bank_a,
                doc_a,
            )
            assert text_a == "bank-a-secret", "A retain in another bank must not modify this bank's chunk"

            rows = await conn.fetch(
                "SELECT bank_id FROM chunks WHERE bank_id = ANY($1::text[])",
                [bank_a, bank_b],
            )
            assert sorted(r["bank_id"] for r in rows) == sorted([bank_a, bank_b])
    finally:
        async with backend.acquire() as conn:
            for bank_id in (bank_a, bank_b):
                await conn.execute("DELETE FROM chunks WHERE bank_id = $1", bank_id)
                await conn.execute("DELETE FROM documents WHERE bank_id = $1", bank_id)
                await conn.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_upsert_refuses_a_chunk_id_owned_by_another_bank(memory):
    """
    Second half of #4244: ids written before the fix can still collide, so the upsert
    itself refuses a conflicting row that belongs to a different bank rather than
    overwriting it.
    """
    from hindsight_api.engine.db.ops import ChunkIdOwnedByAnotherBank

    prefix = f"t{int(_ts() * 1000)}o"
    bank_a, bank_b = prefix, f"{prefix}_other"
    document_id = "doc"
    legacy_chunk_id = f"{bank_a}_legacy_0"

    backend = await memory._get_backend()
    ops = backend.ops
    try:
        async with backend.acquire() as conn:
            await _seed_bank_and_document(conn, bank_a, document_id)
            await _seed_bank_and_document(conn, bank_b, document_id)

            await conn.execute(
                """
                INSERT INTO chunks (chunk_id, document_id, bank_id, chunk_text, chunk_index, content_hash)
                VALUES ($1, $2, $3, 'bank-a-secret', 0, 'h')
                """,
                legacy_chunk_id,
                document_id,
                bank_a,
            )

            with pytest.raises(ChunkIdOwnedByAnotherBank):
                await ops.bulk_upsert_chunks(
                    conn,
                    "chunks",
                    [legacy_chunk_id],
                    [document_id],
                    [bank_b],
                    ["bank-b-text"],
                    [0],
                    ["h2"],
                )

        async with backend.acquire() as conn:
            text = await conn.fetchval("SELECT chunk_text FROM chunks WHERE chunk_id = $1", legacy_chunk_id)
            assert text == "bank-a-secret", "The other bank's chunk text must survive"
    finally:
        async with backend.acquire() as conn:
            for bank_id in (bank_a, bank_b):
                await conn.execute("DELETE FROM chunks WHERE bank_id = $1", bank_id)
                await conn.execute("DELETE FROM documents WHERE bank_id = $1", bank_id)
                await conn.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)
