"""A floor on how often an *automatic* mental-model refresh may run — issue #3480.

A bank with several models on `refresh_after_consolidation` paid for a full agentic
rebuild of every stale model on every retain, however small: ~250k tokens for a
three-fact retain, ~11.6M tokens a day from ordinary conversational traffic.

`min_refresh_interval_seconds` bounds that. The refresh is *parked*, not skipped:
the operation stays queued with `next_retry_at` set to the end of the window, so
every trigger that fires while it waits folds into it (#3487) and the eventual run
covers all of them. Skipping the submit instead would drop the work — nothing
re-checks the model, so a burst's memories would stay unsynthesised until some
later unrelated trigger happened to land outside the window.

Deterministic — no LLM: `refresh_mental_model` is stubbed on the paths that are
supposed to reach it, and the parked paths never get that far.

The setup writes `mental_models` and `memory_units` rows directly (as
test_mental_model_refresh_pending_dedupe_3487.py does) because the public API cannot
express what these cases need: a model whose *last refresh* landed a chosen number of
seconds ago, which is the one input the floor is computed from. Producing that through
retain + consolidation would mean waiting out real wall-clock time. Assertions are on
`async_operations` — the queue state this feature actually changes — not on the memory
tables.
"""

import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from hindsight_api.engine.memory_engine import _REFRESH_AUTOMATIC_KEY, MemoryEngine
from hindsight_api.worker.exceptions import DeferOperation

INTERVAL = 1800


async def _make_bank(memory: MemoryEngine, request_context) -> str:
    bank_id = f"mminterval-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    return bank_id


async def _insert_mm(conn, bank_id: str, *, trigger: dict, refreshed_seconds_ago: int) -> str:
    mm_id = f"mm-{uuid.uuid4().hex}"
    await conn.execute(
        """
        INSERT INTO mental_models
          (id, bank_id, subtype, name, source_query, content, tags, trigger, last_refreshed_at)
        VALUES ($1, $2, 'pinned', 'interval model', 'what changed', 'body', $3, $4::jsonb, $5)
        """,
        mm_id,
        bank_id,
        [],
        json.dumps(trigger),
        datetime.now(UTC) - timedelta(seconds=refreshed_seconds_ago),
    )
    return mm_id


def _stub_refresh(memory: MemoryEngine, monkeypatch) -> list[str]:
    """Record which models actually reached the refresh, without running one."""
    refreshed: list[str] = []

    async def _fake_refresh(bank_id, mental_model_id, **kwargs):
        refreshed.append(mental_model_id)
        return {"id": mental_model_id, "content": "body", "reflect_response": {}, "source_query": "q"}

    async def _skip_outcome_metadata(operation_id, refreshed_model):
        return None

    monkeypatch.setattr(memory, "refresh_mental_model", _fake_refresh)
    monkeypatch.setattr(memory, "_write_refresh_outcome_metadata", _skip_outcome_metadata)
    return refreshed


def _task(bank_id: str, mm_id: str, *, automatic: bool) -> dict:
    task: dict = {"bank_id": bank_id, "mental_model_id": mm_id, "operation_id": str(uuid.uuid4())}
    if automatic:
        task[_REFRESH_AUTOMATIC_KEY] = True
    return task


# --------------------------------------------------------------------------
# Precedence: model > bank > global. Pure, no DB.
# --------------------------------------------------------------------------


def test_per_model_interval_wins_over_the_bank_setting(memory: MemoryEngine):
    resolved = memory._resolve_min_refresh_interval(
        {"min_refresh_interval_seconds": 60},
        {"mental_model_min_refresh_interval_seconds": 3600},
    )
    assert resolved == 60


def test_per_model_zero_exempts_a_model_from_a_bank_wide_floor(memory: MemoryEngine):
    """An explicit 0 is an override, not an absent value — that is how one hot model
    stays current while the rest of its bank is rate-limited."""
    resolved = memory._resolve_min_refresh_interval(
        {"min_refresh_interval_seconds": 0},
        {"mental_model_min_refresh_interval_seconds": 3600},
    )
    assert resolved == 0


def test_absent_per_model_value_falls_back_to_the_bank_setting(memory: MemoryEngine):
    assert memory._resolve_min_refresh_interval({}, {"mental_model_min_refresh_interval_seconds": 900}) == 900
    assert (
        memory._resolve_min_refresh_interval(
            {"min_refresh_interval_seconds": None},
            {"mental_model_min_refresh_interval_seconds": 900},
        )
        == 900
    )


def test_nothing_configured_anywhere_means_no_floor(memory: MemoryEngine):
    """The default is backwards compatible: every trigger refreshes at once."""
    assert memory._resolve_min_refresh_interval(None, {}) == 0


def test_a_json_string_trigger_is_parsed(memory: MemoryEngine):
    """Backends hand back jsonb as text; the resolver must not silently read 0 from it."""
    assert memory._resolve_min_refresh_interval(json.dumps({"min_refresh_interval_seconds": 120}), {}) == 120


# --------------------------------------------------------------------------
# The park itself.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_automatic_refresh_inside_the_window_is_parked(memory: MemoryEngine, request_context, monkeypatch):
    """No recall, no LLM — and the deferral points at the moment the window closes."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(
            conn,
            bank,
            trigger={"refresh_after_consolidation": True, "min_refresh_interval_seconds": INTERVAL},
            refreshed_seconds_ago=60,
        )
    refreshed = _stub_refresh(memory, monkeypatch)

    with pytest.raises(DeferOperation) as excinfo:
        await memory._handle_refresh_mental_model(_task(bank, mm_id, automatic=True))

    assert refreshed == []
    # ~29 minutes out: 30-minute window, one minute already elapsed.
    remaining = (excinfo.value.exec_date - datetime.now(UTC)).total_seconds()
    assert INTERVAL - 120 < remaining <= INTERVAL - 30
    assert "min_refresh_interval_seconds" in excinfo.value.reason


@pytest.mark.asyncio
async def test_automatic_refresh_outside_the_window_runs(memory: MemoryEngine, request_context, monkeypatch):
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(
            conn,
            bank,
            trigger={"refresh_after_consolidation": True, "min_refresh_interval_seconds": INTERVAL},
            refreshed_seconds_ago=INTERVAL + 60,
        )
    refreshed = _stub_refresh(memory, monkeypatch)

    await memory._handle_refresh_mental_model(_task(bank, mm_id, automatic=True))

    assert refreshed == [mm_id]


@pytest.mark.asyncio
async def test_manual_refresh_ignores_the_floor(memory: MemoryEngine, request_context, monkeypatch):
    """An explicit refresh asked for one now. Same model, same window, no park."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(
            conn,
            bank,
            trigger={"refresh_after_consolidation": True, "min_refresh_interval_seconds": INTERVAL},
            refreshed_seconds_ago=1,
        )
    refreshed = _stub_refresh(memory, monkeypatch)

    await memory._handle_refresh_mental_model(_task(bank, mm_id, automatic=False))

    assert refreshed == [mm_id]


@pytest.mark.asyncio
async def test_no_interval_configured_never_parks(memory: MemoryEngine, request_context, monkeypatch):
    """Regression guard for the default: a model refreshed a second ago still refreshes."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, trigger={"refresh_after_consolidation": True}, refreshed_seconds_ago=1)
    refreshed = _stub_refresh(memory, monkeypatch)

    await memory._handle_refresh_mental_model(_task(bank, mm_id, automatic=True))

    assert refreshed == [mm_id]


@pytest.mark.asyncio
async def test_bank_config_supplies_the_floor_without_a_per_model_value(
    memory: MemoryEngine, request_context, monkeypatch
):
    """The per-bank lever works on models that say nothing about intervals."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(conn, bank, trigger={"refresh_after_consolidation": True}, refreshed_seconds_ago=60)
    await memory.update_bank_config(
        bank_id=bank,
        updates={"mental_model_min_refresh_interval_seconds": INTERVAL},
        request_context=request_context,
    )
    refreshed = _stub_refresh(memory, monkeypatch)

    with pytest.raises(DeferOperation):
        await memory._handle_refresh_mental_model(_task(bank, mm_id, automatic=True))
    assert refreshed == []


# --------------------------------------------------------------------------
# The fold has one field to reconcile: an explicit refresh must not inherit a park.
# --------------------------------------------------------------------------


async def _queued_refresh_op(memory: MemoryEngine, bank_id: str):
    async with memory._pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT operation_id, status, next_retry_at, task_payload FROM async_operations "
            "WHERE bank_id = $1 AND operation_type = 'refresh_mental_model'",
            bank_id,
        )


def _stall_worker(memory: MemoryEngine, monkeypatch) -> None:
    async def _never_runs(task_dict):
        return None

    monkeypatch.setattr(memory._task_backend, "submit_task", _never_runs)


async def _park(memory: MemoryEngine, operation_id, when: datetime) -> None:
    async with memory._pool.acquire() as conn:
        await conn.execute(
            "UPDATE async_operations SET next_retry_at = $2 WHERE operation_id = $1",
            operation_id,
            when,
        )


@pytest.mark.asyncio
async def test_explicit_refresh_releases_a_parked_one_it_folds_into(memory: MemoryEngine, request_context, monkeypatch):
    """Otherwise "refresh now" silently inherits somebody else's rate limit.

    The fold itself is right — one operation does the work either way (#3487) — but the
    automatic marker and the `next_retry_at` have to come off, or the caller waits out a
    window they never asked for, and the handler would re-park the operation anyway.
    """
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(
            conn,
            bank,
            trigger={"refresh_after_consolidation": True, "min_refresh_interval_seconds": INTERVAL},
            refreshed_seconds_ago=60,
        )
    _stall_worker(memory, monkeypatch)

    automatic = await memory.submit_async_refresh_mental_model(
        bank_id=bank, mental_model_id=mm_id, request_context=request_context, automatic=True
    )
    parked_until = datetime.now(UTC) + timedelta(seconds=INTERVAL)
    await _park(memory, uuid.UUID(automatic["operation_id"]), parked_until)

    explicit = await memory.submit_async_refresh_mental_model(
        bank_id=bank, mental_model_id=mm_id, request_context=request_context
    )

    assert explicit["deduplicated"] is True
    assert explicit["operation_id"] == automatic["operation_id"]

    row = await _queued_refresh_op(memory, bank)
    assert row["next_retry_at"] is None
    payload = row["task_payload"]
    payload = json.loads(payload) if isinstance(payload, str) else payload
    assert _REFRESH_AUTOMATIC_KEY not in payload


@pytest.mark.asyncio
async def test_an_automatic_submit_leaves_an_existing_park_in_place(memory: MemoryEngine, request_context, monkeypatch):
    """The whole point of the floor: the next trigger folds in and keeps waiting."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(
            conn,
            bank,
            trigger={"refresh_after_consolidation": True, "min_refresh_interval_seconds": INTERVAL},
            refreshed_seconds_ago=60,
        )
    _stall_worker(memory, monkeypatch)

    first = await memory.submit_async_refresh_mental_model(
        bank_id=bank, mental_model_id=mm_id, request_context=request_context, automatic=True
    )
    parked_until = (datetime.now(UTC) + timedelta(seconds=INTERVAL)).replace(microsecond=0)
    await _park(memory, uuid.UUID(first["operation_id"]), parked_until)

    for _ in range(5):
        folded = await memory.submit_async_refresh_mental_model(
            bank_id=bank, mental_model_id=mm_id, request_context=request_context, automatic=True
        )
        assert folded["operation_id"] == first["operation_id"]

    row = await _queued_refresh_op(memory, bank)
    assert row["next_retry_at"] == parked_until
    payload = row["task_payload"]
    payload = json.loads(payload) if isinstance(payload, str) else payload
    assert payload[_REFRESH_AUTOMATIC_KEY] is True


def test_a_malformed_stored_value_is_ignored_not_raised(memory: MemoryEngine):
    """A bad value must cost the model its rate limit, not every future refresh.

    The API validates this field, so only a direct column write produces one — and
    raising here would fail the operation on every retry forever.
    """
    assert memory._resolve_min_refresh_interval({"min_refresh_interval_seconds": "soon"}, {}) == 0
    assert (
        memory._resolve_min_refresh_interval(
            {"min_refresh_interval_seconds": "soon"},
            {"mental_model_min_refresh_interval_seconds": 300},
        )
        == 300
    )
    assert memory._resolve_min_refresh_interval({}, {"mental_model_min_refresh_interval_seconds": "later"}) == 0


@pytest.mark.asyncio
async def test_the_after_consolidation_trigger_marks_its_submits_automatic(
    memory: MemoryEngine, request_context, monkeypatch
):
    """The other automatic path. Asserted on the queued payload, not the call kwargs.

    If this trigger stopped marking its submits, the floor would silently stop applying
    to the case #3480 was actually about — and no other test would fail, because nothing
    else depends on the flag.
    """
    from hindsight_api.engine.consolidation.consolidator import _trigger_mental_model_refreshes

    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_mm(
            conn,
            bank,
            trigger={"refresh_after_consolidation": True, "min_refresh_interval_seconds": INTERVAL},
            refreshed_seconds_ago=86400,
        )
        # Newer than the model's last refresh, so it reads as stale and is a candidate.
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at) "
            "VALUES ($1, $2, 'a fresh fact', 'experience', $3, now(), now())",
            uuid.uuid4(),
            bank,
            [],
        )
    _stall_worker(memory, monkeypatch)

    assert await _trigger_mental_model_refreshes(memory, bank, request_context) == 1

    row = await _queued_refresh_op(memory, bank)
    payload = row["task_payload"]
    payload = json.loads(payload) if isinstance(payload, str) else payload
    assert payload[_REFRESH_AUTOMATIC_KEY] is True
    assert payload["mental_model_id"] == mm_id


def test_env_var_reads_and_tolerates_an_unfilled_value(monkeypatch):
    """A commented-out template line that gets uncommented but not filled (`VAR=`) must
    fall back to the default rather than crash config load."""
    from hindsight_api.config import ENV_MENTAL_MODEL_MIN_REFRESH_INTERVAL_SECONDS, HindsightConfig

    monkeypatch.setenv(ENV_MENTAL_MODEL_MIN_REFRESH_INTERVAL_SECONDS, "1800")
    assert HindsightConfig.from_env().mental_model_min_refresh_interval_seconds == 1800

    monkeypatch.setenv(ENV_MENTAL_MODEL_MIN_REFRESH_INTERVAL_SECONDS, "")
    assert HindsightConfig.from_env().mental_model_min_refresh_interval_seconds == 0

    monkeypatch.setenv(ENV_MENTAL_MODEL_MIN_REFRESH_INTERVAL_SECONDS, "-60")
    assert HindsightConfig.from_env().mental_model_min_refresh_interval_seconds == 0
