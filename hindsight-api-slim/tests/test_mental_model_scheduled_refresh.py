"""Tests for cron-scheduled mental model refresh.

Covers the discovery routine ``public.mental_models_with_cron()`` and the
maintenance-loop job ``MaintenanceLoop._run_scheduled_mm_refresh``: a model is
refreshed only when its cron schedule is *due* AND it is *stale* (new memories in
its scope since the last refresh). Deterministic — no LLM, refresh submission is
monkeypatched.
"""

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from hindsight_api.engine.llm_interface import ProviderContentPolicyError, ProviderRateLimitResetError
from hindsight_api.engine.maintenance import MaintenanceLoop
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.response_models import MentalModelTrigger
from tests.store_seed import store_write_facts


def test_full_refresh_cron_combines_with_either_trigger():
    """full_refresh_cron is a second schedule, not a third exclusive trigger."""
    MentalModelTrigger(mode="delta", refresh_after_consolidation=True, full_refresh_cron="0 3 * * 0")
    MentalModelTrigger(mode="delta", refresh_cron="0 * * * *", full_refresh_cron="0 3 * * 0")
    assert MentalModelTrigger(full_refresh_cron="  ").full_refresh_cron is None
    with pytest.raises(ValueError, match="full_refresh_cron is not a valid cron expression"):
        MentalModelTrigger(full_refresh_cron="every sunday")


def test_refresh_cron_and_auto_refresh_are_mutually_exclusive():
    """A trigger cannot set both refresh_after_consolidation and refresh_cron."""
    # Either alone is fine.
    MentalModelTrigger(refresh_after_consolidation=True)
    MentalModelTrigger(refresh_cron="0 3 * * *")
    # Both together is rejected.
    with pytest.raises(ValueError, match="mutually exclusive"):
        MentalModelTrigger(refresh_after_consolidation=True, refresh_cron="0 3 * * *")


async def _make_bank(memory: MemoryEngine, request_context) -> str:
    bank_id = f"mmcron-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    return bank_id


async def _insert_mm(
    conn,
    bank_id: str,
    *,
    refresh_cron: str | None,
    last_refreshed_offset: str,
    tags: list[str] | None = None,
    full_refresh_cron: str | None = None,
    last_full_refreshed_offset: str | None = None,
) -> str:
    """Insert a pinned mental model. ``last_refreshed_offset`` is an interval
    string applied as ``now() - INTERVAL <offset>`` (e.g. '1 day', '0 seconds');
    ``last_full_refreshed_offset`` likewise, None leaving the column null."""
    mm_id = f"mm-{uuid.uuid4().hex}"
    trigger = {"refresh_after_consolidation": False}
    if refresh_cron is not None:
        trigger["refresh_cron"] = refresh_cron
    if full_refresh_cron is not None:
        trigger["full_refresh_cron"] = full_refresh_cron
    last_full = f"now() - INTERVAL '{last_full_refreshed_offset}'" if last_full_refreshed_offset else "NULL"
    await conn.execute(
        f"""
        INSERT INTO mental_models
          (id, bank_id, subtype, name, source_query, content, tags, trigger, last_refreshed_at,
           last_full_refreshed_at)
        VALUES ($1, $2, 'pinned', 'sched model', 'what changed', 'body', $3, $4::jsonb,
                now() - INTERVAL '{last_refreshed_offset}', {last_full})
        """,
        mm_id,
        bank_id,
        tags or [],
        json.dumps(trigger),
    )
    return mm_id


async def _new_fact(memory: MemoryEngine, bank_id: str) -> None:
    """A fact newer than the model's last refresh, written where the bank's memories live.

    Written directly rather than retained: a retain also runs consolidation, which refreshes
    or re-dates the very model these tests are timing.
    """
    if get_memories().store_owned_for(bank_id):
        await store_write_facts(memory, bank_id, [str(uuid.uuid4())], ["a fresh fact"], fact_type="experience")
        return
    async with memory._pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at) "
            "VALUES ($1, $2, 'a fresh fact', 'experience', $3, now(), now())",
            uuid.uuid4(),
            bank_id,
            [],
        )


def _patch_submit(memory: MemoryEngine, monkeypatch, modes: dict[str, str | None] | None = None) -> list[str]:
    """Record submitted model ids; when ``modes`` is given, also each submit's ``mode``."""
    submitted: list[str] = []

    async def _record(
        *, bank_id, mental_model_id, request_context, skip_if_in_flight=False, automatic=False, mode=None
    ):
        submitted.append(mental_model_id)
        if modes is not None:
            modes[mental_model_id] = mode
        return {"operation_id": str(uuid.uuid4())}

    monkeypatch.setattr(memory, "submit_async_refresh_mental_model", _record)
    return submitted


def _stall_worker(memory: MemoryEngine, monkeypatch) -> None:
    """Queue operations without executing them.

    Reproduces the condition the duplicate waves were observed under (#3210): the
    ops stay pending because completions stalled fleet-wide.
    """

    async def _never_runs(task_dict):
        return None

    monkeypatch.setattr(memory._task_backend, "submit_task", _never_runs)


async def _mark_processing(memory: MemoryEngine, operation_id: str) -> None:
    """Put a queued operation into the state a worker claim leaves it in."""
    async with memory._pool.acquire() as conn:
        await conn.execute(
            "UPDATE async_operations SET status = 'processing' WHERE operation_id = $1",
            uuid.UUID(operation_id),
        )


async def _count_refresh_ops(memory: MemoryEngine, bank_id: str) -> int:
    async with memory._pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM async_operations WHERE bank_id = $1 AND operation_type = 'refresh_mental_model'",
            bank_id,
        )


@pytest.mark.asyncio
async def test_refresh_operations_name_the_model_they_refreshed(memory: MemoryEngine, request_context, monkeypatch):
    """The operations *list* carries `mental_model_id` for a refresh.

    These operations have no `document_id`, and the list carries no `result_metadata`
    either, so without it the log cannot say which model an operation refreshed — a bank
    with hundreds of models is undiagnosable from the list alone. The single-operation
    read already answers it through `result_metadata`, so it gets no duplicate field.
    """
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, refresh_cron="*/5 * * * *", last_refreshed_offset="1 day")
    _stall_worker(memory, monkeypatch)

    submitted = await memory.submit_async_refresh_mental_model(
        bank_id=bank, mental_model_id=mm_id, request_context=request_context
    )

    listed = await memory.list_operations(bank_id=bank, request_context=request_context)
    refreshes = [op for op in listed["operations"] if op["task_type"] == "refresh_mental_model"]
    assert len(refreshes) == 1
    assert refreshes[0]["mental_model_id"] == mm_id
    # The field this used to be the only handle on is still null for these ops, which is
    # exactly why mental_model_id is needed.
    assert refreshes[0]["document_id"] is None

    # The single-operation read answers the same question through result_metadata.
    status = await memory.get_operation_status(
        bank_id=bank, operation_id=submitted["operation_id"], request_context=request_context
    )
    assert status["result_metadata"]["mental_model_id"] == mm_id


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
    await _new_fact(memory, bank)  # newer than last_refreshed_at -> stale

    submitted = _patch_submit(memory, monkeypatch)
    await MaintenanceLoop(memory)._run_scheduled_mm_refresh()

    assert mm_id in submitted


@pytest.mark.asyncio
async def test_scheduled_refresh_is_marked_automatic(memory: MemoryEngine, request_context, monkeypatch):
    """A cron refresh is subject to the minimum-interval floor.

    Nobody asked for this refresh individually, so it is exactly what
    ``min_refresh_interval_seconds`` rate-limits (#3480). The flag lives in the submit's
    task payload, and dropping it here would silently exempt the whole cron path — with
    no failing test anywhere, because the floor simply never engages.
    """
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, refresh_cron="*/5 * * * *", last_refreshed_offset="1 day")
    await _new_fact(memory, bank)

    recorded: dict[str, bool] = {}

    async def _record(
        *, bank_id, mental_model_id, request_context, skip_if_in_flight=False, automatic=False, mode=None
    ):
        recorded[mental_model_id] = automatic
        return {"operation_id": str(uuid.uuid4())}

    monkeypatch.setattr(memory, "submit_async_refresh_mental_model", _record)
    await MaintenanceLoop(memory)._run_scheduled_mm_refresh()

    assert recorded[mm_id] is True


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
async def test_concurrent_scheduled_submits_queue_one_refresh(memory: MemoryEngine, request_context, monkeypatch):
    """Two schedulers enqueueing the same due model at once queue exactly one op.

    Regression for #3210: the maintenance loop runs in every process, and the
    in-flight guard in ``mental_models_with_cron()`` is a *read* — every process saw
    the same "nothing in flight" snapshot and inserted its own operation, so a few
    hundred due models became thousands of queued refreshes. The enqueue now does the
    in-flight check under the bank row lock, inside the inserting transaction.
    """
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, refresh_cron="*/5 * * * *", last_refreshed_offset="1 day")
    _stall_worker(memory, monkeypatch)

    first, second = await asyncio.gather(
        memory.submit_async_refresh_mental_model(
            bank_id=bank, mental_model_id=mm_id, request_context=request_context, skip_if_in_flight=True
        ),
        memory.submit_async_refresh_mental_model(
            bank_id=bank, mental_model_id=mm_id, request_context=request_context, skip_if_in_flight=True
        ),
    )

    assert await _count_refresh_ops(memory, bank) == 1
    # The loser reports the winner's operation rather than a fresh one, so the
    # maintenance loop can count it as "already in flight".
    deduped = second if second.get("deduplicated") else first
    kept = first if second.get("deduplicated") else second
    assert deduped["deduplicated"] is True
    assert deduped["operation_id"] == kept["operation_id"]


@pytest.mark.asyncio
async def test_user_triggered_refresh_still_queues_while_one_is_processing(
    memory: MemoryEngine, request_context, monkeypatch
):
    """An explicit refresh still queues while a scheduled one is *running*.

    Only a running refresh needs a second operation: it may have read the source query
    before the user edited it, so a user asking for a refresh has new intent that the
    in-flight run cannot be assumed to cover. (A merely *queued* refresh does cover it —
    see test_user_triggered_refresh_folds_into_a_queued_one.)
    """
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, refresh_cron="*/5 * * * *", last_refreshed_offset="1 day")
    _stall_worker(memory, monkeypatch)

    scheduled = await memory.submit_async_refresh_mental_model(
        bank_id=bank, mental_model_id=mm_id, request_context=request_context, skip_if_in_flight=True
    )
    await _mark_processing(memory, scheduled["operation_id"])
    manual = await memory.submit_async_refresh_mental_model(
        bank_id=bank, mental_model_id=mm_id, request_context=request_context
    )

    assert manual["operation_id"] != scheduled["operation_id"]
    assert await _count_refresh_ops(memory, bank) == 2


@pytest.mark.asyncio
async def test_not_due_model_is_skipped_even_when_stale(memory: MemoryEngine, request_context, monkeypatch):
    """A model whose cron has not elapsed since the last refresh is not refreshed,
    even when new memories exist — the cron gate, not just staleness, must hold."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        # Yearly cron (Jan 1); last refresh seconds ago -> the most recent fire is
        # well before last_refreshed_at, so it is not due.
        mm_id = await _insert_mm(conn, bank, refresh_cron="0 0 1 1 *", last_refreshed_offset="5 seconds")
    await _new_fact(memory, bank)  # stale, but cron not due

    submitted = _patch_submit(memory, monkeypatch)
    await MaintenanceLoop(memory)._run_scheduled_mm_refresh()

    assert mm_id not in submitted


@pytest.mark.asyncio
async def test_failed_model_is_not_requeued_until_a_refresh_succeeds(
    memory: MemoryEngine, request_context, monkeypatch
):
    """#4532: a failed refresh leaves the model stale, so every scheduler tick used to
    queue it again and pay the LLM for the same failure, forever. After a failure the
    automatic triggers leave it alone; an explicit refresh still runs, and a success
    resumes them."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, refresh_cron="*/5 * * * *", last_refreshed_offset="1 day")
    await _new_fact(memory, bank)
    _stall_worker(memory, monkeypatch)
    await memory._record_mental_model_refresh_failure(
        bank, mm_id, outcome="refresh_failed_error", failure_reason="unexpected_error", error_message="boom"
    )

    loop = MaintenanceLoop(memory)
    await loop._run_scheduled_mm_refresh()
    await loop._run_scheduled_mm_refresh()
    assert await _count_refresh_ops(memory, bank) == 0

    await memory.submit_async_refresh_mental_model(bank_id=bank, mental_model_id=mm_id, request_context=request_context)
    assert await _count_refresh_ops(memory, bank) == 1

    async with memory._pool.acquire() as conn:
        await conn.execute("UPDATE mental_models SET last_refreshed_at = now() WHERE id = $1", mm_id)
    assert await memory._automatic_refresh_paused(bank, mm_id) is False


class _ProviderStatusError(Exception):
    """A provider SDK error carrying an HTTP status, like openai's APIStatusError."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"provider answered {status_code}")
        self.status_code = status_code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        _ProviderStatusError(429),
        _ProviderStatusError(503),
        ProviderRateLimitResetError(retry_at=datetime.now(UTC) + timedelta(hours=1), message="session limit"),
    ],
    ids=["rate-limit", "provider-down", "quota-reset"],
)
async def test_a_temporary_provider_failure_does_not_pause_automatic_refresh(
    memory: MemoryEngine, request_context, monkeypatch, error
):
    """#5394: one refresh that hit a rate or session limit paused the page for good,
    and pages stayed frozen for days after the limit lifted. A temporary failure must
    leave the next automatic trigger free to run it."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, refresh_cron="*/5 * * * *", last_refreshed_offset="1 day")
    await _new_fact(memory, bank)

    async def _fail(**kwargs):
        raise error

    monkeypatch.setattr(memory, "refresh_mental_model", _fail)
    with pytest.raises(type(error)):
        await memory._handle_refresh_mental_model({"bank_id": bank, "mental_model_id": mm_id})
    assert await memory._automatic_refresh_paused(bank, mm_id) is False

    _stall_worker(memory, monkeypatch)
    await MaintenanceLoop(memory)._run_scheduled_mm_refresh()
    assert await _count_refresh_ops(memory, bank) == 1


@pytest.mark.asyncio
async def test_a_deterministic_refresh_failure_still_pauses(memory: MemoryEngine, request_context, monkeypatch):
    """#4532 holds for failures that would repeat on the same prompt."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, refresh_cron="*/5 * * * *", last_refreshed_offset="1 day")

    async def _fail(**kwargs):
        raise ProviderContentPolicyError("declined")

    monkeypatch.setattr(memory, "refresh_mental_model", _fail)
    with pytest.raises(ProviderContentPolicyError):
        await memory._handle_refresh_mental_model({"bank_id": bank, "mental_model_id": mm_id})
    assert await memory._automatic_refresh_paused(bank, mm_id) is True


@pytest.mark.asyncio
async def test_refresh_cut_off_by_the_wall_timeout_counts_as_failed(memory: MemoryEngine, request_context):
    """The wall ceiling cancels the refresh before its own failure handling runs, so
    the poller's notification has to record it, or the scheduler re-queues it."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, refresh_cron="*/5 * * * *", last_refreshed_offset="1 day")

    await memory.on_task_wall_timeout(
        {"type": "refresh_mental_model", "bank_id": bank, "mental_model_id": mm_id},
        None,
        "timed out",
        "llm.codex.tools.attempt=1/2",
    )

    assert await memory._automatic_refresh_paused(bank, mm_id) is True


@pytest.mark.asyncio
async def test_refresh_cut_off_during_a_retry_backoff_does_not_pause(
    memory: MemoryEngine, request_context, monkeypatch
):
    """#5561: a provider outage left the refresh sleeping out a retry backoff when the
    ceiling fired, and the page stayed paused for days after the provider came back.
    The failure is still recorded in history, but the next automatic trigger runs."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, refresh_cron="*/5 * * * *", last_refreshed_offset="1 day")
    await _new_fact(memory, bank)

    await memory.on_task_wall_timeout(
        {"type": "refresh_mental_model", "bank_id": bank, "mental_model_id": mm_id},
        None,
        "timed out",
        "llm.codex.tools.attempt=1/2.backoff",
    )

    assert await memory._automatic_refresh_paused(bank, mm_id) is False
    history = await memory.get_mental_model_history(bank, mm_id, request_context=request_context)
    assert [h["kind"] for h in history] == ["refresh_failed"]
    _stall_worker(memory, monkeypatch)
    await MaintenanceLoop(memory)._run_scheduled_mm_refresh()
    assert await _count_refresh_ops(memory, bank) == 1


@pytest.mark.asyncio
async def test_the_failure_is_visible_on_the_model_and_cleared_by_a_success(memory: MemoryEngine, request_context):
    """`last_refresh_failed_at` is what the UI reads to say a model has stopped
    refreshing itself, so it has to be on the read and gone again after a success."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, refresh_cron=None, last_refreshed_offset="1 day")

    await memory._record_mental_model_refresh_failure(
        bank, mm_id, outcome="refresh_failed_error", failure_reason="unexpected_error", error_message="boom"
    )
    failed = await memory.get_mental_model(bank, mm_id, request_context=request_context)
    assert failed["last_refresh_failed_at"] is not None
    listed = await memory.list_mental_models(bank, request_context=request_context)
    assert listed.items[0]["last_refresh_failed_at"] == failed["last_refresh_failed_at"]

    await memory.update_mental_model(bank, mm_id, content="a fresh answer", request_context=request_context)
    refreshed = await memory.get_mental_model(bank, mm_id, request_context=request_context)
    assert refreshed["last_refresh_failed_at"] is None
    assert await memory._automatic_refresh_paused(bank, mm_id) is False


# ── scheduled full refresh (trigger.full_refresh_cron, #4340) ────────────────


@pytest.mark.asyncio
async def test_routine_returns_models_with_only_a_full_refresh_cron(memory: MemoryEngine, request_context):
    """A delta model on refresh-after-consolidation has no refresh_cron, but its
    full_refresh_cron must still be discovered, with the clock it comes due against."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(
            conn,
            bank,
            refresh_cron=None,
            last_refreshed_offset="1 hour",
            full_refresh_cron="0 3 * * 0",
            last_full_refreshed_offset="3 days",
        )
        rows = await conn.fetch("SELECT * FROM public.mental_models_with_cron() WHERE bank_id = $1", bank)

    assert [r["mental_model_id"] for r in rows] == [mm_id]
    assert rows[0]["refresh_cron"] is None
    assert rows[0]["full_refresh_cron"] == "0 3 * * 0"
    assert rows[0]["last_full_refreshed_at"] is not None


@pytest.mark.asyncio
async def test_due_full_refresh_is_submitted_in_full_mode_even_when_not_stale(
    memory: MemoryEngine, request_context, monkeypatch
):
    """The full refresh exists to clear drift past deltas left, which a bank with no
    new memories still carries — so it is not gated on staleness, and it is submitted
    with mode='full'."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        # Every-5-minutes full refresh last ran a day ago; a delta ran an hour ago.
        mm_id = await _insert_mm(
            conn,
            bank,
            refresh_cron=None,
            last_refreshed_offset="1 hour",
            full_refresh_cron="*/5 * * * *",
            last_full_refreshed_offset="1 day",
        )

    modes: dict[str, str | None] = {}
    submitted = _patch_submit(memory, monkeypatch, modes)
    await MaintenanceLoop(memory)._run_scheduled_mm_refresh()

    # Other tests' models share the database, so only this model's submits are checked.
    assert submitted.count(mm_id) == 1
    assert modes[mm_id] == "full"


@pytest.mark.asyncio
async def test_full_refresh_is_skipped_when_nothing_refreshed_since_the_last_one(
    memory: MemoryEngine, request_context, monkeypatch
):
    """No refresh since the last full one means no drift to clear."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(
            conn,
            bank,
            refresh_cron=None,
            last_refreshed_offset="1 day",
            full_refresh_cron="*/5 * * * *",
            last_full_refreshed_offset="1 day",
        )

    submitted = _patch_submit(memory, monkeypatch)
    await MaintenanceLoop(memory)._run_scheduled_mm_refresh()

    assert mm_id not in submitted


@pytest.mark.asyncio
async def test_full_refresh_comes_due_against_its_own_clock(memory: MemoryEngine, request_context, monkeypatch):
    """A recent delta refresh does not push the full refresh back, and a recent full
    refresh keeps a not-yet-fired yearly schedule quiet."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        # Delta 5 seconds ago — if due-ness keyed off last_refreshed_at this would never fire.
        due_mm = await _insert_mm(
            conn,
            bank,
            refresh_cron=None,
            last_refreshed_offset="5 seconds",
            full_refresh_cron="*/5 * * * *",
            last_full_refreshed_offset="1 day",
        )
        # Yearly full refresh that last ran seconds ago: not due.
        quiet_mm = await _insert_mm(
            conn,
            bank,
            refresh_cron=None,
            last_refreshed_offset="1 second",
            full_refresh_cron="0 0 1 1 *",
            last_full_refreshed_offset="5 seconds",
        )

    submitted = _patch_submit(memory, monkeypatch)
    await MaintenanceLoop(memory)._run_scheduled_mm_refresh()

    assert due_mm in submitted
    assert quiet_mm not in submitted


@pytest.mark.asyncio
async def test_when_both_schedules_are_due_the_full_refresh_wins(memory: MemoryEngine, request_context, monkeypatch):
    """One submit, in full mode: it covers what the plain refresh would have done."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(
            conn,
            bank,
            refresh_cron="*/5 * * * *",
            last_refreshed_offset="1 hour",
            full_refresh_cron="*/5 * * * *",
            last_full_refreshed_offset="1 day",
        )
    await _new_fact(memory, bank)

    modes: dict[str, str | None] = {}
    submitted = _patch_submit(memory, monkeypatch, modes)
    await MaintenanceLoop(memory)._run_scheduled_mm_refresh()

    # Other tests' models share the database, so only this model's submits are checked.
    assert submitted.count(mm_id) == 1
    assert modes[mm_id] == "full"


async def _queued_payload(memory: MemoryEngine, operation_id: str) -> dict:
    # Raw read: the task payload is internal to the queue and no engine method exposes it,
    # and the payload's mode is exactly what the fold is supposed to change.
    async with memory._pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT task_payload FROM async_operations WHERE operation_id = $1", uuid.UUID(operation_id)
        )
    return json.loads(raw) if isinstance(raw, str) else raw


@pytest.mark.asyncio
async def test_a_full_refresh_folded_into_a_queued_one_upgrades_it(memory: MemoryEngine, request_context, monkeypatch):
    """Dedupe keeps one queued refresh per model; a full submit that folds into a queued
    plain one must make that one full, or its 'full' is silently dropped. A plain submit
    folding into a queued full one leaves it full."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, refresh_cron=None, last_refreshed_offset="1 day")
    _stall_worker(memory, monkeypatch)

    plain = await memory.submit_async_refresh_mental_model(
        bank_id=bank, mental_model_id=mm_id, request_context=request_context
    )
    assert "_mode" not in await _queued_payload(memory, plain["operation_id"])

    full = await memory.submit_async_refresh_mental_model(
        bank_id=bank, mental_model_id=mm_id, request_context=request_context, mode="full"
    )
    assert full["deduplicated"] is True
    assert full["operation_id"] == plain["operation_id"]
    assert (await _queued_payload(memory, plain["operation_id"]))["_mode"] == "full"

    await memory.submit_async_refresh_mental_model(bank_id=bank, mental_model_id=mm_id, request_context=request_context)
    assert (await _queued_payload(memory, plain["operation_id"]))["_mode"] == "full"
    assert await _count_refresh_ops(memory, bank) == 1
