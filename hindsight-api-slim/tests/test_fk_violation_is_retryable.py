"""Regression for #4453 — a retain dropped at ``retry_count = 0`` by an FK race.

A retain writing ``unit_entities`` races a delete removing the units it is
writing for: the document-delete endpoint takes only a bank lock, and a
multi-batch retain does not hold its document row between batches. The insert
then lands on a parent that is already gone and raises
``ForeignKeyViolationError``.

Nothing about the retain's data is wrong — the parent vanished *this moment* —
so the retry, running against a settled database, succeeds. But
``_is_non_retryable_task_error`` classified every
``IntegrityConstraintViolationError`` as deterministic (the blanket from #980,
whose motivating case was a genuinely deterministic ``UniqueViolationError`` on
``pk_chunks``), so these retains went terminal on first contact and the
conversations they carried were dropped with no second attempt and no error to
the caller — ``async: true`` retains report success at submit.

The race is reproduced for real here, not simulated: two hand-driven
connections, the actual ``bulk_insert_unit_entities`` write path, and the
resulting exception fed to the classifier. Note the direction — the *inserting*
side is what fails. The deleting side is safe: the FK is ``ON DELETE CASCADE``
and this codebase runs every transaction at READ COMMITTED, so the cascade
re-reads and removes a concurrently committed child rather than erroring.
"""

import uuid

import asyncpg
import pytest

from hindsight_api.engine.db.ops_postgresql import PostgreSQLOps
from hindsight_api.engine.db.postgresql import PostgresConnection
from hindsight_api.engine.memory_engine import _is_non_retryable_task_error

# Drives raw memory_units / unit_entities rows through the Postgres ops layer.
pytestmark = pytest.mark.memory_backend_incompatible


@pytest.mark.asyncio
async def test_retain_fk_violation_from_concurrent_delete_is_retryable(pg0_db_url):
    bank_id = f"fk-retryable-{uuid.uuid4().hex}"
    doc_id = f"doc-{uuid.uuid4().hex}"
    ops = PostgreSQLOps()

    setup = await asyncpg.connect(pg0_db_url)
    retain = await asyncpg.connect(pg0_db_url)
    deleter = await asyncpg.connect(pg0_db_url)
    try:
        await setup.execute(
            "INSERT INTO documents (id, bank_id, original_text, content_hash) VALUES ($1, $2, 'text', 'hash')",
            doc_id,
            bank_id,
        )
        # A unit committed by an earlier retain batch, and the entity the next
        # batch is about to post it against.
        unit_id = await setup.fetchval(
            """
            INSERT INTO memory_units (bank_id, text, event_date, fact_type, document_id)
            VALUES ($1, 'unit', now(), 'world', $2)
            RETURNING id
            """,
            bank_id,
            doc_id,
        )
        entity_id = await setup.fetchval(
            "INSERT INTO entities (bank_id, canonical_name) VALUES ($1, $2) RETURNING id",
            bank_id,
            f"entity-{uuid.uuid4().hex[:8]}",
        )

        # --- The delete lands and commits between the retain's batches ---
        await deleter.execute(
            "DELETE FROM memory_units WHERE document_id = $1 AND bank_id = $2",
            doc_id,
            bank_id,
        )

        # --- The retain's next batch posts the unit it wrote a moment ago ---
        fk_error: asyncpg.ForeignKeyViolationError | None = None
        tx = retain.transaction()
        await tx.start()
        try:
            await ops.bulk_insert_unit_entities(PostgresConnection(retain), "unit_entities", [unit_id], [entity_id])
            await tx.commit()
        except asyncpg.ForeignKeyViolationError as exc:
            fk_error = exc
            await tx.rollback()

        assert fk_error is not None, (
            "Expected the unit_entities insert to raise ForeignKeyViolationError "
            "against the concurrently deleted parent unit — the race in #4453 no "
            "longer reproduces, so this test is asserting nothing."
        )
        assert "fk_unit_entities_unit_id_memory_units" in str(fk_error)

        assert _is_non_retryable_task_error(fk_error) is False, (
            "A foreign-key violation is a concurrency race, not bad data: the "
            "retry succeeds against a settled database. Classifying it "
            "non-retryable drops the retain at retry_count=0 (#4453)."
        )
    finally:
        await setup.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
        await setup.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)
        await setup.execute("DELETE FROM documents WHERE bank_id = $1", bank_id)
        await setup.close()
        await retain.close()
        await deleter.close()


def test_deterministic_integrity_siblings_stay_non_retryable():
    """#980's motivating case, and the rest of the family, keep their behaviour.

    Only the FK class is a race. A unique violation on ``pk_chunks`` — the
    failure #980 introduced the blanket for — is a function of the data, and so
    are the not-null, check, exclusion and restrict violations alongside it.
    """
    assert _is_non_retryable_task_error(asyncpg.exceptions.UniqueViolationError("pk_chunks")) is True
    assert _is_non_retryable_task_error(asyncpg.exceptions.NotNullViolationError("text")) is True
    assert _is_non_retryable_task_error(asyncpg.exceptions.CheckViolationError("ck_units")) is True
    assert _is_non_retryable_task_error(asyncpg.exceptions.ExclusionViolationError("ex_units")) is True
    assert _is_non_retryable_task_error(asyncpg.exceptions.RestrictViolationError("rs_units")) is True
