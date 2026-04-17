"""
Tests for EntityResolver edge cases.
"""

import uuid
from datetime import datetime, timezone

import pytest

from hindsight_api.engine.db import create_database_backend
from hindsight_api.engine.entity_resolver import EntityResolver
from hindsight_api.pg0 import resolve_database_url

# ---------------------------------------------------------------------------
# Unit tests for discard_pending_stats() — no database required
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discard_pending_stats_clears_both_dicts():
    """discard_pending_stats() must remove entries for the current task from
    both _pending_stats and _pending_cooccurrences."""
    resolver = EntityResolver(pool=None)  # type: ignore[arg-type]
    key = resolver._task_key()

    resolver._pending_stats[key] = [object()]  # type: ignore[list-item]
    resolver._pending_cooccurrences[key] = [object()]  # type: ignore[list-item]

    resolver.discard_pending_stats()

    assert key not in resolver._pending_stats
    assert key not in resolver._pending_cooccurrences


@pytest.mark.asyncio
async def test_discard_pending_stats_is_idempotent():
    """Calling discard_pending_stats() when nothing is pending must not raise."""
    resolver = EntityResolver(pool=None)  # type: ignore[arg-type]
    resolver.discard_pending_stats()
    resolver.discard_pending_stats()  # second call — still safe


@pytest.mark.asyncio
async def test_discard_pending_stats_does_not_affect_other_task_keys():
    """discard_pending_stats() must only remove the current task's entries,
    leaving entries keyed under other task IDs untouched."""
    resolver = EntityResolver(pool=None)  # type: ignore[arg-type]
    other_key = -1  # A fake key that can never be a real task id

    resolver._pending_stats[other_key] = [object()]  # type: ignore[list-item]
    resolver._pending_cooccurrences[other_key] = [object()]  # type: ignore[list-item]

    resolver.discard_pending_stats()  # discards current task's key only

    assert other_key in resolver._pending_stats, "other task's stats must be preserved"
    assert other_key in resolver._pending_cooccurrences, "other task's cooccurrences must be preserved"


@pytest.mark.asyncio
async def test_resolve_entities_batch_handles_unicode_lower_conflicts(pg0_db_url):
    """
    Existing entities with PostgreSQL/Python lowercase mismatches should resolve
    to the conflicted row instead of leaving a missing entity_id.
    """
    resolved_url = await resolve_database_url(pg0_db_url)
    backend = create_database_backend("postgresql")
    await backend.initialize(resolved_url, min_size=1, max_size=2, command_timeout=30)
    bank_id = f"test-entity-resolver-{uuid.uuid4().hex[:8]}"
    event_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
    resolver = EntityResolver(pool=backend, entity_lookup="full")

    try:
        async with backend.acquire() as conn:
            existing_entity_id = await conn.fetchval(
                """
                INSERT INTO entities (bank_id, canonical_name, first_seen, last_seen, mention_count)
                VALUES ($1, $2, $3, $3, 1)
                RETURNING id
                """,
                bank_id,
                "İstanbul",
                event_date,
            )

            resolved_ids = await resolver.resolve_entities_batch(
                bank_id=bank_id,
                entities_data=[
                    {
                        "text": "istanbul",
                        "nearby_entities": [],
                        "event_date": event_date,
                    }
                ],
                context="unicode case mismatch",
                unit_event_date=event_date,
                conn=conn,
            )

            entity_rows = await conn.fetch(
                """
                SELECT id, canonical_name
                FROM entities
                WHERE bank_id = $1
                ORDER BY canonical_name
                """,
                bank_id,
            )

        assert resolved_ids == [existing_entity_id]
        assert len(entity_rows) == 1
        assert entity_rows[0]["id"] == existing_entity_id
        assert entity_rows[0]["canonical_name"] == "İstanbul"
    finally:
        async with backend.acquire() as conn:
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)
        await backend.shutdown()
