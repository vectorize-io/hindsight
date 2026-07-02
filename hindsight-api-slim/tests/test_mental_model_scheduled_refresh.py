"""Tests for cron-scheduled mental model refresh.

Covers the discovery routine ``public.mental_models_with_cron()`` and the
maintenance-loop job ``MaintenanceLoop._run_scheduled_mm_refresh``: a model is
refreshed only when its cron schedule is *due* AND it is *stale* (new memories in
its scope since the last refresh). Deterministic — no LLM, refresh submission is
monkeypatched.
"""

import json
import uuid

import pytest

from hindsight_api.api.http import MentalModelTrigger
from hindsight_api.engine.maintenance import MaintenanceLoop
from hindsight_api.engine.memory_engine import MemoryEngine


def test_refresh_cron_and_auto_refresh_are_mutually_exclusive():
    """A trigger cannot set both refresh_after_consolidation and refresh_cron."""
    # Either alone is fine.
    MentalModelTrigger(refresh_after_consolidation=True)
    MentalModelTrigger(refresh_cron="0 3 * * *")
    # Both together is rejected.
    with pytest.raises(ValueError, match="mutually exclusive"):
        MentalModelTrigger(refresh_after_consolidation=True, refresh_cron="0 3 * * *")


def test_full_refresh_interval_validates_duration_values():
    """full_refresh_interval accepts durations but rejects booleans/malformed values."""
    assert MentalModelTrigger(full_refresh_interval="7d").full_refresh_interval == "7d"
    assert MentalModelTrigger(full_refresh_interval=604800).full_refresh_interval == 604800
    assert MentalModelTrigger(full_refresh_interval="").full_refresh_interval is None

    with pytest.raises(ValueError, match="duration"):
        MentalModelTrigger(full_refresh_interval=True)
    with pytest.raises(ValueError, match="finite"):
        MentalModelTrigger(full_refresh_interval="nan")
    with pytest.raises(ValueError):
        MentalModelTrigger(full_refresh_interval="not-a-duration")


async def _make_bank(memory: MemoryEngine, request_context) -> str:
    bank_id = f"mmcron-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    return bank_id


async def _insert_mm(
    conn,
    bank_id: str,
    *,
    refresh_cron: str | None,
    last_refreshed_offset: str,
    tags: list[str] | None = None,
) -> str:
    """Insert a pinned mental model. ``last_refreshed_offset`` is an interval
    string applied as ``now() - INTERVAL <offset>`` (e.g. '1 day', '0 seconds')."""
    mm_id = f"mm-{uuid.uuid4().hex}"
    trigger = {"refresh_after_consolidation": False}
    if refresh_cron is not None:
        trigger["refresh_cron"] = refresh_cron
    await conn.execute(
        f"""
        INSERT INTO mental_models
          (id, bank_id, subtype, name, source_query, content, tags, trigger, last_refreshed_at)
        VALUES ($1, $2, 'pinned', 'sched model', 'what changed', 'body', $3, $4::jsonb,
                now() - INTERVAL '{last_refreshed_offset}')
        """,
        mm_id,
        bank_id,
        tags or [],
        json.dumps(trigger),
    )
    return mm_id


async def _insert_fact(conn, bank_id: str, tags: list[str] | None = None) -> None:
    await conn.execute(
        "INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at) "
        "VALUES ($1, $2, 'a fresh fact', 'experience', $3, now(), now())",
        uuid.uuid4(),
        bank_id,
        tags or [],
    )


def _patch_submit(memory: MemoryEngine, monkeypatch) -> list[str]:
    submitted: list[str] = []

    async def _record(*, bank_id, mental_model_id, request_context):
        submitted.append(mental_model_id)
        return {"operation_id": str(uuid.uuid4())}

    monkeypatch.setattr(memory, "submit_async_refresh_mental_model", _record)
    return submitted


@pytest.mark.asyncio
async def test_refresh_cron_round_trips_through_create_and_get(memory: MemoryEngine, request_context):
    """refresh_cron set on a mental model's trigger persists and reads back."""
    bank = await _make_bank(memory, request_context)
    created = await memory.create_mental_model(
        bank_id=bank,
        name="scheduled model",
        source_query="what changed",
        content="body",
        trigger={"refresh_after_consolidation": False, "refresh_cron": "0 3 * * *"},
        request_context=request_context,
    )
    fetched = await memory.get_mental_model(bank, created["id"], request_context=request_context)
    assert fetched["trigger"]["refresh_cron"] == "0 3 * * *"


@pytest.mark.asyncio
async def test_routine_returns_cron_models_excludes_plain_and_in_flight(memory: MemoryEngine, request_context):
    """The discovery routine returns models with a cron schedule and excludes both
    cron-less models and models with an in-flight refresh operation."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        cron_mm = await _insert_mm(conn, bank, refresh_cron="*/5 * * * *", last_refreshed_offset="1 day")
        plain_mm = await _insert_mm(conn, bank, refresh_cron=None, last_refreshed_offset="1 day")
        in_flight_mm = await _insert_mm(conn, bank, refresh_cron="*/5 * * * *", last_refreshed_offset="1 day")
        await conn.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'refresh_mental_model', 'processing', $3::jsonb)
            """,
            uuid.uuid4(),
            bank,
            json.dumps({"mental_model_id": in_flight_mm}),
        )
        rows = await conn.fetch(
            "SELECT mental_model_id FROM public.mental_models_with_cron() WHERE bank_id = $1",
            bank,
        )

    returned = {r["mental_model_id"] for r in rows}
    assert cron_mm in returned
    assert plain_mm not in returned  # no cron -> not a candidate
    assert in_flight_mm not in returned  # already being refreshed -> excluded


@pytest.mark.asyncio
async def test_due_and_stale_model_is_refreshed(memory: MemoryEngine, request_context, monkeypatch):
    """A model whose cron is due and that has new memories in scope is refreshed."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, refresh_cron="*/5 * * * *", last_refreshed_offset="1 day")
        await _insert_fact(conn, bank)  # newer than last_refreshed_at -> stale

    submitted = _patch_submit(memory, monkeypatch)
    await MaintenanceLoop(memory)._run_scheduled_mm_refresh()

    assert mm_id in submitted


@pytest.mark.asyncio
async def test_due_but_not_stale_model_is_skipped(memory: MemoryEngine, request_context, monkeypatch):
    """A model whose cron is due but whose scope has no new memories is skipped —
    a scheduled refresh must not burn an LLM call to regenerate identical content."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        # Cron is due (last refresh a day ago), but no memory_units in this fresh
        # bank are newer than last_refreshed_at, so the model is not stale.
        mm_id = await _insert_mm(conn, bank, refresh_cron="*/5 * * * *", last_refreshed_offset="1 day")

    submitted = _patch_submit(memory, monkeypatch)
    await MaintenanceLoop(memory)._run_scheduled_mm_refresh()

    assert mm_id not in submitted


@pytest.mark.asyncio
async def test_due_periodic_full_refresh_bypasses_staleness_skip(
    memory: MemoryEngine, request_context, monkeypatch
):
    """A due delta full-refresh interval submits even when no new facts arrived."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, refresh_cron="*/5 * * * *", last_refreshed_offset="1 day")
        await conn.execute(
            """
            UPDATE mental_models
            SET trigger = $1::jsonb,
                reflect_response = $2::jsonb
            WHERE id = $3
            """,
            json.dumps(
                {
                    "refresh_after_consolidation": False,
                    "refresh_cron": "*/5 * * * *",
                    "mode": "delta",
                    "full_refresh_interval": "7d",
                }
            ),
            json.dumps({"last_full_refreshed_at": "2000-01-01T00:00:00+00:00"}),
            mm_id,
        )

    submitted = _patch_submit(memory, monkeypatch)
    await MaintenanceLoop(memory)._run_scheduled_mm_refresh()

    assert mm_id in submitted


@pytest.mark.asyncio
async def test_not_due_model_is_skipped_even_when_stale(memory: MemoryEngine, request_context, monkeypatch):
    """A model whose cron has not elapsed since the last refresh is not refreshed,
    even when new memories exist — the cron gate, not just staleness, must hold."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        # Yearly cron (Jan 1); last refresh seconds ago -> the most recent fire is
        # well before last_refreshed_at, so it is not due.
        mm_id = await _insert_mm(conn, bank, refresh_cron="0 0 1 1 *", last_refreshed_offset="5 seconds")
        await _insert_fact(conn, bank)  # stale, but cron not due

    submitted = _patch_submit(memory, monkeypatch)
    await MaintenanceLoop(memory)._run_scheduled_mm_refresh()

    assert mm_id not in submitted
