"""Tests for which mental models a consolidation run triggers a refresh for.

``_trigger_mental_model_refreshes`` prefilters candidates in SQL and then gates each
one on ``compute_mental_model_is_stale``. The prefilter must key off the model's
*resolved* refresh scope (``_resolve_refresh_tag_filtering``), not its ``tags``
column: ``tags_match`` "any"/"all" and ``trigger.tag_groups`` both let a tagged model
see untagged memories, and gating on the column starved them (#3053).

Deterministic — no LLM, no consolidation run: the trigger function is called directly
and refresh submission is monkeypatched.
"""

import json
import uuid

import pytest

from hindsight_api.engine.consolidation.consolidator import _trigger_mental_model_refreshes
from hindsight_api.engine.memory_engine import MemoryEngine


async def _make_bank(memory: MemoryEngine, request_context) -> str:
    bank_id = f"mmscope-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    return bank_id


async def _insert_mm(
    conn,
    bank_id: str,
    *,
    tags: list[str] | None = None,
    trigger_extra: dict | None = None,
    last_refreshed_offset: str = "1 day",
) -> str:
    """Insert a pinned mental model with refresh_after_consolidation=true.

    ``last_refreshed_offset`` is an interval string applied as ``now() - INTERVAL <offset>``.
    """
    mm_id = f"mm-{uuid.uuid4().hex}"
    trigger: dict = {"refresh_after_consolidation": True, **(trigger_extra or {})}
    await conn.execute(
        f"""
        INSERT INTO mental_models
          (id, bank_id, subtype, name, source_query, content, tags, trigger, last_refreshed_at)
        VALUES ($1, $2, 'pinned', 'scope model', 'what changed', 'body', $3, $4::jsonb,
                now() - INTERVAL '{last_refreshed_offset}')
        """,
        mm_id,
        bank_id,
        tags or [],
        json.dumps(trigger),
    )
    return mm_id


async def _insert_fact(conn, bank_id: str, tags: list[str] | None = None) -> None:
    """Insert a memory newer than any model's last_refreshed_at."""
    await conn.execute(
        "INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at) "
        "VALUES ($1, $2, 'a fresh fact', 'observation', $3, now(), now())",
        uuid.uuid4(),
        bank_id,
        tags or [],
    )


def _patch_submit(memory: MemoryEngine, monkeypatch) -> list[str]:
    submitted: list[str] = []

    async def _record(*, bank_id, mental_model_id, request_context, skip_if_in_flight=False):
        submitted.append(mental_model_id)
        return {"operation_id": str(uuid.uuid4())}

    monkeypatch.setattr(memory, "submit_async_refresh_mental_model", _record)
    return submitted


@pytest.mark.asyncio
async def test_tagged_strict_model_skipped_when_only_untagged_consolidated(
    memory: MemoryEngine, request_context, monkeypatch
):
    """A tagged model on the default all_strict scope is NOT refreshed by an
    untagged-only consolidation — strict matching excludes untagged memories, so a
    refresh would regenerate identical content (and risk clobbering it)."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, tags=["alpha"])
        await _insert_fact(conn, bank)  # untagged -> outside an all_strict scope

    submitted = _patch_submit(memory, monkeypatch)
    await _trigger_mental_model_refreshes(
        memory_engine=memory, bank_id=bank, request_context=request_context, consolidated_tags=None
    )

    assert mm_id not in submitted


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_tagged_non_strict_model_refreshed_when_only_untagged_consolidated(
    memory: MemoryEngine, request_context, monkeypatch
):
    """A tagged model with tags_match="any" IS refreshed by an untagged-only
    consolidation: non-strict matching puts untagged memories in its scope (#3053)."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, tags=["alpha"], trigger_extra={"tags_match": "any"})
        await _insert_fact(conn, bank)  # untagged -> inside a non-strict scope

    submitted = _patch_submit(memory, monkeypatch)
    await _trigger_mental_model_refreshes(
        memory_engine=memory, bank_id=bank, request_context=request_context, consolidated_tags=None
    )

    assert mm_id in submitted


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_tag_groups_model_refreshed_when_only_untagged_consolidated(
    memory: MemoryEngine, request_context, monkeypatch
):
    """trigger.tag_groups overrides the tags column entirely, so a model carrying
    tags can still have untagged memories in scope and must stay a candidate."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(
            conn,
            bank,
            tags=["alpha"],
            trigger_extra={"tag_groups": [{"tags": ["beta"], "match": "any"}]},
        )
        await _insert_fact(conn, bank)  # untagged -> matched by the group's "any"

    submitted = _patch_submit(memory, monkeypatch)
    await _trigger_mental_model_refreshes(
        memory_engine=memory, bank_id=bank, request_context=request_context, consolidated_tags=None
    )

    assert mm_id in submitted


@pytest.mark.asyncio
async def test_non_strict_model_skipped_when_nothing_changed(memory: MemoryEngine, request_context, monkeypatch):
    """Widening the prefilter must not produce spurious refreshes: with no memory
    newer than last_refreshed_at, the staleness gate still skips the model."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, tags=["alpha"], trigger_extra={"tags_match": "any"})
        # No facts inserted -> nothing in scope is newer than last_refreshed_at.

    submitted = _patch_submit(memory, monkeypatch)
    await _trigger_mental_model_refreshes(
        memory_engine=memory, bank_id=bank, request_context=request_context, consolidated_tags=None
    )

    assert mm_id not in submitted


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_tagged_model_refreshed_when_its_tag_was_consolidated(memory: MemoryEngine, request_context, monkeypatch):
    """The overlap path is unchanged: a strict tagged model is refreshed when a memory
    carrying its tag was consolidated."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, tags=["alpha"])
        await _insert_fact(conn, bank, tags=["alpha"])

    submitted = _patch_submit(memory, monkeypatch)
    await _trigger_mental_model_refreshes(
        memory_engine=memory, bank_id=bank, request_context=request_context, consolidated_tags=["alpha"]
    )

    assert mm_id in submitted


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_non_strict_model_refreshed_on_mixed_run_with_foreign_tags(
    memory: MemoryEngine, request_context, monkeypatch
):
    """A run that consolidates both untagged and foreign-tagged memories reports only
    the foreign tags. A non-strict model whose tags do not overlap them is still
    reached by the untagged half, so the overlap branch must widen the same way."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        strict_mm = await _insert_mm(conn, bank, tags=["alpha"])
        loose_mm = await _insert_mm(conn, bank, tags=["alpha"], trigger_extra={"tags_match": "any"})
        await _insert_fact(conn, bank)  # the untagged half of the run

    submitted = _patch_submit(memory, monkeypatch)
    await _trigger_mental_model_refreshes(
        memory_engine=memory, bank_id=bank, request_context=request_context, consolidated_tags=["beta"]
    )

    assert loose_mm in submitted
    assert strict_mm not in submitted
