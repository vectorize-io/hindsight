"""Regression tests for DataAccessOps.lock_document_for_write.

Covers vectorize-io/hindsight#1944: the retain document-ownership gate used a
single ``INSERT ... ON CONFLICT DO UPDATE ... RETURNING`` upsert. PostgreSQL
runs it as-is, but the Oracle adapter rewrites ``ON CONFLICT DO UPDATE`` to a
``MERGE``, which cannot carry a ``RETURNING`` clause — so every retain 500'd
with ``DPY-1003: the executed statement does not return rows``.

The lock-and-read step now lives behind ``ops.lock_document_for_write`` so each
backend implements it natively (PG: one upsert; Oracle: idempotent insert +
``SELECT ... FOR UPDATE``). These tests pin the contract on PG (run in the
default suite) and the Oracle rewrite limitation that motivated the split.
"""

from datetime import datetime, timezone

import pytest

from hindsight_api.engine.memory_engine import fq_table


def _ts() -> float:
    return datetime.now(timezone.utc).timestamp()


async def _seed_bank(conn, bank_id: str) -> None:
    await conn.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


@pytest.mark.asyncio
async def test_lock_document_for_write_returns_pending_then_existing_hash(memory):
    """Fresh row → '__pending__'; existing row → its stored hash."""
    bank_id = f"test_lock_doc_{_ts()}"
    doc_id = "doc-lock-regression"
    documents = fq_table("documents")

    backend = await memory._get_backend()
    ops = backend.ops
    async with backend.acquire() as conn:
        await _seed_bank(conn, bank_id)

        # First call creates the row and reports the placeholder hash.
        async with conn.transaction():
            first = await ops.lock_document_for_write(conn, documents, doc_id, bank_id)
        assert first == "__pending__"

        # Promote the placeholder to a real content hash, as the real retain
        # flow does immediately after taking the lock.
        await conn.execute(
            f"UPDATE {documents} SET content_hash = $1 WHERE id = $2 AND bank_id = $3",
            "real-hash-v1",
            doc_id,
            bank_id,
        )

        # A subsequent writer sees the committed hash (and re-takes the lock).
        async with conn.transaction():
            second = await ops.lock_document_for_write(conn, documents, doc_id, bank_id)
        assert second == "real-hash-v1"


@pytest.mark.asyncio
async def test_lock_document_for_write_isolates_by_bank(memory):
    """The lock/read is scoped to (id, bank_id) — same doc_id in another bank
    is a distinct row and starts at '__pending__'."""
    suffix = _ts()
    bank_a = f"test_lock_doc_a_{suffix}"
    bank_b = f"test_lock_doc_b_{suffix}"
    doc_id = "shared-doc-id"
    documents = fq_table("documents")

    backend = await memory._get_backend()
    ops = backend.ops
    async with backend.acquire() as conn:
        await _seed_bank(conn, bank_a)
        await _seed_bank(conn, bank_b)

        async with conn.transaction():
            assert await ops.lock_document_for_write(conn, documents, doc_id, bank_a) == "__pending__"
        await conn.execute(
            f"UPDATE {documents} SET content_hash = $1 WHERE id = $2 AND bank_id = $3",
            "bank-a-hash",
            doc_id,
            bank_a,
        )

        async with conn.transaction():
            assert await ops.lock_document_for_write(conn, documents, doc_id, bank_b) == "__pending__"


def test_oracle_merge_rewrite_cannot_carry_returning():
    """Root cause of #1944: PG's single-statement upsert-and-lock rewrites to an
    Oracle MERGE, and MERGE can't RETURNING — so a ``fetchval`` on it gets no
    rows back (DPY-1003). The dialect split in lock_document_for_write exists to
    avoid emitting this form on Oracle."""
    from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle

    query, ignore_dup, returning_cols = _rewrite_pg_to_oracle(
        "INSERT INTO documents (id, bank_id, original_text, content_hash) "
        "VALUES ($1, $2, '', '__pending__') "
        "ON CONFLICT (id, bank_id) DO UPDATE SET content_hash = documents.content_hash "
        "RETURNING content_hash"
    )

    assert query.lstrip().upper().startswith("MERGE")
    # The RETURNING clause is silently dropped by the MERGE rewrite — this is
    # exactly why the old single-statement form returned no rows on Oracle.
    assert "RETURNING" not in query.upper()
    assert returning_cols is None
    assert not ignore_dup


def test_oracle_select_for_update_hash_translates_cleanly():
    """The Oracle fallback reads the hash with a plain SELECT ... FOR UPDATE,
    which translates without a MERGE so the scalar fetch returns the column."""
    from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle

    query, ignore_dup, returning_cols = _rewrite_pg_to_oracle(
        "SELECT content_hash FROM documents WHERE id = $1 AND bank_id = $2 FOR UPDATE"
    )

    assert "MERGE" not in query.upper()
    assert "FOR UPDATE" in query.upper()
    assert ":1" in query and ":2" in query
    assert not ignore_dup
    assert returning_cols is None
