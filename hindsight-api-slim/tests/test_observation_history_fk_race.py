"""Regression: observation_history write for a since-deleted observation is skipped.

Under parallel (or same-batch delete-then-update) consolidation, one write path
can remove an observation from ``memory_units`` before another writes its
``observation_history`` snapshot. The INSERT then trips
``observation_history_observation_id_fkey``. Because consolidation runs in
autocommit (no enclosing transaction), catching the FK violation and skipping
the best-effort history row is safe — the connection stays usable and the
current observation state remains the source of truth.

Regression for #2597 / #2506: before the fix this raised
``asyncpg.ForeignKeyViolationError`` and failed the whole consolidation task.
"""

import uuid

import pytest

from hindsight_api.engine.consolidation.consolidator import (
    _append_observation_history,
    _ObservationHistorySnapshot,
)
from hindsight_api.engine.db_utils import acquire_with_retry


@pytest.mark.asyncio
async def test_append_history_for_missing_observation_is_skipped(memory, request_context):
    """A history write targeting an absent observation is skipped, not fatal."""
    bank_id = f"test-obs-history-fk-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id, request_context=request_context)

    snapshot = _ObservationHistorySnapshot(
        previous_text="old",
        previous_tags=[],
        previous_occurred_start=None,
        previous_occurred_end=None,
        previous_mentioned_at=None,
        new_source_memory_ids=[],
    )
    # Never inserted into memory_units, so the FK target is absent.
    missing_observation_id = str(uuid.uuid4())

    pool = await memory._get_pool()
    async with acquire_with_retry(pool) as conn:
        # Pre-fix this raised asyncpg.ForeignKeyViolationError; the fix skips it.
        await _append_observation_history(conn, bank_id, missing_observation_id, snapshot, max_entries=10)
        # Autocommit: the failed INSERT did not poison the connection.
        assert await conn.fetchval("SELECT 1") == 1
