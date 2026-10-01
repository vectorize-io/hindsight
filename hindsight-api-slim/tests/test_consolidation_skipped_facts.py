"""A fact consolidation consumed but wrote nothing for is marked, not just stamped (#5054).

``_consolidate_batch`` stamps ``consolidated_at`` on every fact it hands to the LLM, and that
stamp is what takes a fact out of the pending set. A fact the model deliberately declined
(``skipped`` / ``no_durable_knowledge``) therefore looked exactly like one folded into an
observation: out of rebuild for good, cited by nothing, and no signal anywhere that it happened.

The stamp is unchanged. ``consolidation_skipped_at`` is written beside it, in the same
transaction, only for the facts that ended the final pass without an observation. These tests
pin when it is written and, as importantly, when it is not.

Raw SQL on ``memory_units`` is deliberate here: the marker has no public read surface, so the
column itself is the only thing that can be asserted.
"""

from __future__ import annotations

import json
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api import RequestContext
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation import consolidator as consolidator_module
from hindsight_api.engine.consolidation.consolidator import (
    _ConsolidationBatchResponse,
    _CreateAction,
    run_consolidation_job,
)
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memory_engine import MemoryEngine, fq_table
from hindsight_api.engine.providers.mock_llm import MockLLM
from hindsight_api.metrics import MetricsCollector, NoOpMetricsCollector

pytestmark = pytest.mark.memory_backend_incompatible


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


@asynccontextmanager
async def _bank(memory: MemoryEngine, request_context: RequestContext):
    bank_id = f"skipped-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        yield bank_id
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


def _llm(callback):
    mock_llm = MockLLM(provider="mock", api_key="", base_url="", model="mock-model")
    mock_llm.set_response_callback(callback)
    wrapper = MagicMock()
    wrapper.with_config.return_value = mock_llm
    return wrapper


def _fact_ids(messages) -> list[str]:
    prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
    return re.findall(r"\[([0-9a-f-]{36})\]", prompt)


class _Responses:
    """Scripted consolidation responses: one flag per LLM call, ``True`` = create one observation, ``False`` = decline."""

    def __init__(self, *per_call: bool) -> None:
        self._create_on_call = list(per_call)
        self.created_for: list[str] = []

    def __call__(self, messages, scope):
        if scope != "consolidation":
            return _ConsolidationBatchResponse()
        ids = _fact_ids(messages)
        create = self._create_on_call.pop(0) if self._create_on_call else False
        if not create or not ids:
            return _ConsolidationBatchResponse()
        self.created_for.append(ids[0])
        return _ConsolidationBatchResponse(
            creates=[_CreateAction(text="Alice lives in Berlin", source_fact_ids=[ids[0]])]
        )


def _fail_then_decline(fail_first_n: int):
    """Raise on the first ``fail_first_n`` consolidation calls, then decline every fact."""
    calls = 0

    def callback(messages, scope):
        nonlocal calls
        if scope != "consolidation":
            return _ConsolidationBatchResponse()
        calls += 1
        if calls <= fail_first_n:
            raise ValueError(f"simulated LLM failure (call {calls})")
        return _ConsolidationBatchResponse()

    return callback


async def _insert_memory(conn, bank_id: str, text: str, tags: list[str], scopes=None) -> uuid.UUID:
    mem_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, tags, observation_scopes, created_at)
        VALUES ($1, $2, $3, 'experience', $4, $5::jsonb, now())
        """,
        mem_id,
        bank_id,
        text,
        tags,
        json.dumps(scopes),
    )
    return mem_id


async def _run(memory: MemoryEngine, bank_id: str, request_context, callback, **config_overrides):
    original = memory._consolidation_llm_config
    memory._consolidation_llm_config = _llm(callback)
    raw = _get_raw_config()
    overrides = {"consolidation_llm_batch_size": 4, "consolidation_llm_parallelism": 1, **config_overrides}
    fake = type(raw)(**{**{f: getattr(raw, f) for f in raw.__dataclass_fields__}, **overrides})
    try:
        with (
            patch.object(memory._config_resolver, "resolve_full_config", return_value=fake),
            patch.object(memory, "submit_async_consolidation"),
        ):
            return await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)
    finally:
        memory._consolidation_llm_config = original


class _RecordingCollector(NoOpMetricsCollector):
    """Captures skipped-fact counts; every other metric stays a no-op."""

    def __init__(self) -> None:
        self.skipped: list[int] = []

    def record_consolidation_skipped_facts(self, count: int) -> None:
        self.skipped.append(count)


class _Row:
    def __init__(self, record) -> None:
        self.id = str(record["id"])
        self.stamped = record["consolidated_at"] is not None
        self.skipped = record["consolidation_skipped_at"] is not None


async def _facts(memory: MemoryEngine, bank_id: str) -> dict[str, _Row]:
    async with memory._pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, text, consolidated_at, consolidation_skipped_at FROM memory_units "
            "WHERE bank_id = $1 AND fact_type = 'experience'",
            bank_id,
        )
    return {r["text"]: _Row(r) for r in rows}


@pytest.mark.asyncio
async def test_a_fact_the_model_declines_is_marked_skipped_and_a_consumed_one_is_not(
    memory: MemoryEngine, request_context
):
    async with _bank(memory, request_context) as bank_id:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])
            await _insert_memory(conn, bank_id, "Alice said hello", ["user:alice"])
        responses = _Responses(True)

        await _run(memory, bank_id, request_context, responses)

        facts = await _facts(memory, bank_id)
        consumed = {text for text, row in facts.items() if row.id in responses.created_for}
        assert len(consumed) == 1
        for text, row in facts.items():
            # Both leave the pending set exactly as before; only the declined one is marked.
            assert row.stamped, text
            assert row.skipped == (text not in consumed), text


@pytest.mark.asyncio
async def test_a_fact_an_earlier_scope_pass_consumed_is_not_marked_when_the_final_pass_declines(
    memory: MemoryEngine, request_context
):
    """``per_tag`` runs one LLM call per tag scope and only the last may stamp. Judging "skipped" from
    that last pass alone would mark a fact the first pass already folded into an observation."""
    async with _bank(memory, request_context) as bank_id:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice", "topic:move"], "per_tag")

        await _run(memory, bank_id, request_context, _Responses(True, False))

        (fact,) = (await _facts(memory, bank_id)).values()
        assert fact.stamped
        assert not fact.skipped


@pytest.mark.asyncio
async def test_a_fact_every_scope_pass_declines_is_marked(memory: MemoryEngine, request_context):
    async with _bank(memory, request_context) as bank_id:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice said hello", ["user:alice", "topic:chat"], "per_tag")

        await _run(memory, bank_id, request_context, _Responses(False, False))

        (fact,) = (await _facts(memory, bank_id)).values()
        assert fact.stamped
        assert fact.skipped


@pytest.mark.asyncio
async def test_a_batch_that_rolls_back_after_marking_leaves_no_marker_and_counts_nothing(
    memory: MemoryEngine, request_context
):
    """The marker commits with the stamp and the observations (#3876). The write fails AFTER the real
    marking has run, so a marker written outside the transaction would survive the rollback."""
    collector = _RecordingCollector()
    store = get_memories()
    real_mark = store.mark_consolidation_skipped

    async def mark_then_fail(**kwargs):
        await real_mark(**kwargs)
        raise RuntimeError("write failed after marking")

    async with _bank(memory, request_context) as bank_id:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])
            await _insert_memory(conn, bank_id, "Alice said hello", ["user:alice"])

        with (
            patch.object(consolidator_module, "get_metrics_collector", return_value=collector),
            patch.object(store, "mark_consolidation_skipped", new=mark_then_fail),
        ):
            with pytest.raises(RuntimeError, match="write failed after marking"):
                await _run(memory, bank_id, request_context, _Responses(True))

        for text, row in (await _facts(memory, bank_id)).items():
            assert not row.stamped, text
            assert not row.skipped, text
        assert collector.skipped == []


class _FailingCollector(NoOpMetricsCollector):
    def record_consolidation_skipped_facts(self, count: int) -> None:
        raise RuntimeError("metrics backend down")


@pytest.mark.asyncio
async def test_the_skip_counter_is_recorded_only_after_the_marks_have_committed(memory: MemoryEngine, request_context):
    """Recording inside the transaction would count marks that a later failure rolls back. A recorder that
    raises tells the two apart: after the commit the marks are already durable, inside it they are not."""
    async with _bank(memory, request_context) as bank_id:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice said hello", ["user:alice"])

        with patch.object(consolidator_module, "get_metrics_collector", return_value=_FailingCollector()):
            with pytest.raises(RuntimeError, match="metrics backend down"):
                await _run(memory, bank_id, request_context, _Responses(False))

        (fact,) = (await _facts(memory, bank_id)).values()
        assert fact.stamped
        assert fact.skipped


@pytest.mark.asyncio
async def test_a_bisected_batch_marks_each_fact_once_it_is_finally_consumed(memory: MemoryEngine, request_context):
    """A batch of two fails all its retries, is halved, and each half is then consumed and declined. The
    failed attempts marked nothing; the facts are marked by the attempt that finally consumed them."""
    collector = _RecordingCollector()
    async with _bank(memory, request_context) as bank_id:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])
            await _insert_memory(conn, bank_id, "Alice said hello", ["user:alice"])

        with patch.object(consolidator_module, "get_metrics_collector", return_value=collector):
            # One attempt per call and one failing call: the batch of two fails, is halved, and each
            # half succeeds. Without this the retry loop sleeps 1s + 2s and the test relies on the
            # default of 3 attempts.
            await _run(
                memory, bank_id, request_context, _fail_then_decline(fail_first_n=1), consolidation_max_attempts=1
            )

        facts = await _facts(memory, bank_id)
        assert len(facts) == 2
        assert all(row.stamped and row.skipped for row in facts.values())
        assert sum(collector.skipped) == 2


@pytest.mark.asyncio
async def test_a_fact_whose_llm_calls_all_fail_is_not_marked(memory: MemoryEngine, request_context):
    """Failed, not declined: it gets ``consolidation_failed_at`` and is retried by the operator, so it
    must not also carry a verdict that it was skipped."""
    async with _bank(memory, request_context) as bank_id:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice said hello", ["user:alice"])

        await _run(memory, bank_id, request_context, _fail_then_decline(fail_first_n=999), consolidation_max_attempts=1)

        (fact,) = (await _facts(memory, bank_id)).values()
        assert not fact.stamped
        assert not fact.skipped


@pytest.mark.asyncio
async def test_a_batch_discarded_because_a_source_was_edited_marks_nothing(memory: MemoryEngine, request_context):
    """When a source fact is edited between the read and the write, the whole response is dropped
    and the facts stay pending (#4831). They were not declined, so they must not be marked."""
    async with _bank(memory, request_context) as bank_id:
        async with memory._pool.acquire() as conn:
            fact_id = await _insert_memory(conn, bank_id, "Alice said hello", ["user:alice"])

        with patch.object(
            consolidator_module, "_sources_changed_since_read", new=AsyncMock(return_value=[str(fact_id)])
        ):
            # One memory per round: the fact stays pending after the discard, and the job would
            # otherwise fetch it again until the round limit (100 LLM calls).
            await _run(memory, bank_id, request_context, _Responses(False), consolidation_max_memories_per_round=1)

        (fact,) = (await _facts(memory, bank_id)).values()
        assert not fact.stamped
        assert not fact.skipped


@pytest.mark.asyncio
async def test_clearing_the_bank_requeues_a_skipped_fact_and_clears_its_marker(memory: MemoryEngine, request_context):
    async with _bank(memory, request_context) as bank_id:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice said hello", ["user:alice"])
        await _run(memory, bank_id, request_context, _Responses(False))
        assert next(iter((await _facts(memory, bank_id)).values())).skipped

        await memory.clear_observations(bank_id, request_context=request_context)

        (fact,) = (await _facts(memory, bank_id)).values()
        assert not fact.stamped
        assert not fact.skipped


@pytest.mark.asyncio
async def test_invalidating_then_reverting_a_skipped_fact_clears_its_marker(memory: MemoryEngine, request_context):
    """Invalidate copies every ``memory_units`` column into the archive by name, so the archive needs the
    new column too, and revert re-consolidates the fact from scratch, which includes the marker."""
    async with _bank(memory, request_context) as bank_id:
        async with memory._pool.acquire() as conn:
            fact_id = await _insert_memory(conn, bank_id, "Alice said hello", ["user:alice"])
        await _run(memory, bank_id, request_context, _Responses(False))
        assert next(iter((await _facts(memory, bank_id)).values())).skipped

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            await memory.update_memory_unit(bank_id, str(fact_id), state="invalidated", request_context=request_context)
            async with memory._pool.acquire() as conn:
                archived = await conn.fetchval(
                    "SELECT consolidation_skipped_at FROM invalidated_memory_units WHERE id = $1", fact_id
                )
            assert archived is not None  # the archive carries the column, and the copy kept the marker
            await memory.update_memory_unit(bank_id, str(fact_id), state="valid", request_context=request_context)

        (fact,) = (await _facts(memory, bank_id)).values()
        assert not fact.stamped
        assert not fact.skipped


@pytest.mark.asyncio
async def test_marking_a_fact_skipped_is_scheduler_state_and_leaves_updated_at_alone(
    memory: MemoryEngine, request_context
):
    """Like ``mark_consolidated``: bookkeeping is not an edit, and bumping ``updated_at`` would make
    every consolidation pass look like a write to the staleness check."""
    async with _bank(memory, request_context) as bank_id:
        async with memory._pool.acquire() as conn:
            fact_id = await _insert_memory(conn, bank_id, "Alice said hello", ["user:alice"])
            before = await conn.fetchval("SELECT updated_at FROM memory_units WHERE id = $1", fact_id)

            await get_memories().mark_consolidation_skipped(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                unit_ids=[str(fact_id)],
                when=datetime.now(timezone.utc),
            )

            after = await conn.fetchval("SELECT updated_at FROM memory_units WHERE id = $1", fact_id)
        assert after == before


@pytest.mark.asyncio
async def test_the_skip_counter_counts_the_facts_that_were_marked(memory: MemoryEngine, request_context):
    """A run's skip rate has no other signal: ``pending_consolidation`` reads 0 and nothing failed."""
    collector = _RecordingCollector()
    async with _bank(memory, request_context) as bank_id:
        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Alice moved to Berlin", ["user:alice"])
            await _insert_memory(conn, bank_id, "Alice said hello", ["user:alice"])

        with patch.object(consolidator_module, "get_metrics_collector", return_value=collector):
            await _run(memory, bank_id, request_context, _Responses(True))

        assert sum(row.skipped for row in (await _facts(memory, bank_id)).values()) == 1
        assert collector.skipped == [1]


@pytest.mark.asyncio
async def test_editing_a_skipped_fact_requeues_it_and_clears_its_marker(memory: MemoryEngine, request_context):
    """An edit changes what the fact says, so the earlier verdict on the old text no longer applies."""
    async with _bank(memory, request_context) as bank_id:
        async with memory._pool.acquire() as conn:
            fact_id = await _insert_memory(conn, bank_id, "Alice said hello", ["user:alice"])
        await _run(memory, bank_id, request_context, _Responses(False))
        assert next(iter((await _facts(memory, bank_id)).values())).skipped

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            await memory.update_memory_unit(
                bank_id, str(fact_id), text="Alice moved to Berlin", request_context=request_context
            )

        (fact,) = (await _facts(memory, bank_id)).values()
        assert not fact.stamped
        assert not fact.skipped


def test_the_skip_counter_is_a_real_instrument_that_adds_the_count():
    """The OTel collector registers ``hindsight.consolidation.skipped_facts`` and adds what it is given."""
    meter = MagicMock()
    counters: dict[str, MagicMock] = {}

    def make_counter(**kwargs):
        # One mock per instrument: a shared mock would pass even if the method added to another counter.
        counters[kwargs["name"]] = MagicMock()
        return counters[kwargs["name"]]

    meter.create_counter.side_effect = make_counter
    config = MagicMock()
    config.metrics_include_bank_id = False
    with (
        patch("hindsight_api.metrics.get_meter", return_value=meter),
        patch("hindsight_api.config.get_config", return_value=config),
    ):
        collector = MetricsCollector()

    skipped = counters["hindsight.consolidation.skipped_facts"]

    collector.record_consolidation_skipped_facts(3)

    skipped.add.assert_called_once()
    assert skipped.add.call_args.args[0] == 3
    assert not [name for name, counter in counters.items() if counter is not skipped and counter.add.called]
