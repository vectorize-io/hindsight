"""Deterministic tests for automatic fact supersession.

Covers the pure interval algebra (every truth-table row), defensive verdict
parsing, enqueue filtering, and the worker's database mechanics with the LLM
arbitration stubbed out. The arbitration prompt's model-following behaviour is
judged separately in ``test_supersession_judge.py`` (``hs_llm_core``).
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import asyncpg
import pytest

from hindsight_api.engine.retain import supersession
from hindsight_api.engine.retain.supersession import (
    FactTimeline,
    SupersessionVerdict,
    enqueue_supersession_checks,
    resolve_supersession,
    run_fact_supersession_job,
    validate_verdict_indices,
)
from hindsight_api.models import RequestContext


def _tl(
    fact_id: str = "new",
    start: datetime | None = None,
    end: datetime | None = None,
    seen: datetime | None = None,
) -> FactTimeline:
    return FactTimeline(id=fact_id, occurred_start=start, occurred_end=end, mentioned_at=seen)


JAN = datetime(2024, 1, 1, tzinfo=UTC)
MAR = datetime(2024, 3, 1, tzinfo=UTC)
JUN = datetime(2024, 6, 1, tzinfo=UTC)


class TestIntervalAlgebra:
    def test_missing_occurred_start_returns_none(self):
        assert resolve_supersession(_tl(start=None), _tl("c", start=JAN)) is None
        assert resolve_supersession(_tl(start=JAN), _tl("c", start=None)) is None

    def test_disjoint_intervals_coexist(self):
        # "worked at X until Feb" vs "works at Y since June" — both stay.
        old = _tl("c", start=JAN, end=datetime(2024, 2, 1, tzinfo=UTC))
        new = _tl("new", start=JUN)
        assert resolve_supersession(new, old) is None
        assert resolve_supersession(old, new) is None

    def test_earlier_overlapping_candidate_is_superseded(self):
        action = resolve_supersession(_tl("new", start=JUN), _tl("c", start=JAN))
        assert action is not None
        assert action.loser_id == "c"
        assert action.winner_id == "new"
        assert action.valid_until == JUN

    def test_late_arrival_reverses_onto_new_fact(self):
        # Ingesting an old message: the candidate starts later, so the NEW fact loses.
        action = resolve_supersession(_tl("new", start=JAN), _tl("c", start=JUN))
        assert action is not None
        assert action.loser_id == "new"
        assert action.winner_id == "c"
        assert action.valid_until == JUN

    def test_equal_starts_later_mention_wins(self):
        action = resolve_supersession(
            _tl("new", start=JAN, seen=MAR),
            _tl("c", start=JAN, seen=datetime(2024, 2, 1, tzinfo=UTC)),
        )
        assert action is not None
        assert action.loser_id == "c"
        assert action.winner_id == "new"
        # CHECK constraint demands valid_until strictly after occurred_start.
        assert action.valid_until > JAN

    def test_equal_starts_tied_or_missing_mention_is_undecidable(self):
        assert resolve_supersession(_tl("new", start=JAN, seen=MAR), _tl("c", start=JAN, seen=MAR)) is None
        assert resolve_supersession(_tl("new", start=JAN), _tl("c", start=JAN, seen=MAR)) is None


def test_verdict_out_of_range_indices_dropped():
    verdict = SupersessionVerdict(duplicate_indices=[0, 5, -1], contradicted_indices=[1, 3])
    cleaned = validate_verdict_indices(verdict, candidate_count=3)
    assert cleaned.duplicate_indices == [0]
    assert cleaned.contradicted_indices == [1]


@dataclass
class _StubFact:
    fact_type: str
    occurred_start: datetime | None


async def test_enqueue_filters_to_dated_world_facts(pg0_db_url):
    bank_id = f"test_sq_enqueue_{datetime.now(UTC).timestamp()}"
    ids = [str(uuid.uuid4()) for _ in range(4)]
    facts = [
        _StubFact("world", JAN),  # eligible
        _StubFact("world", None),  # undated -> excluded (no interval algebra)
        _StubFact("experience", JAN),  # experience -> excluded
        _StubFact("world", JUN),  # eligible
    ]
    conn = await asyncpg.connect(pg0_db_url)
    try:
        queued = await enqueue_supersession_checks(conn, bank_id, ids, facts)
        rows = await conn.fetch("SELECT memory_id FROM supersession_queue WHERE bank_id = $1", bank_id)
        assert queued == 2
        assert {str(r["memory_id"]) for r in rows} == {ids[0], ids[3]}
        await conn.execute("DELETE FROM supersession_queue WHERE bank_id = $1", bank_id)
    finally:
        await conn.close()


async def test_retain_with_feature_disabled_enqueues_nothing(memory, pg0_db_url, request_context):
    # Default config ships with enable_fact_supersession=False; retain must not queue.
    bank_id = f"test_sq_gate_{datetime.now(UTC).timestamp()}"
    await memory.retain_async(
        bank_id=bank_id,
        content="Alice works at Acme Corporation.",
        event_date=JAN,
        request_context=request_context,
    )
    conn = await asyncpg.connect(pg0_db_url)
    try:
        count = await conn.fetchval("SELECT COUNT(*) FROM supersession_queue WHERE bank_id = $1", bank_id)
    finally:
        await conn.close()
    assert count == 0


async def test_submit_short_circuits_on_empty_queue(memory, request_context):
    result = await memory.submit_async_fact_supersession(
        bank_id=f"test_sq_empty_{datetime.now(UTC).timestamp()}",
        request_context=request_context,
    )
    assert result == {"operation_id": None, "no_work": True}


async def test_worker_supersedes_contradicted_fact(memory, pg0_db_url, request_context, monkeypatch):
    """Worker mechanics end to end with arbitration stubbed: claim, recall
    candidates, apply interval algebra, idempotent write, consolidated_at reset.
    """
    bank_id = f"test_sq_worker_{datetime.now(UTC).timestamp()}"

    await memory.retain_async(
        bank_id=bank_id,
        content="Alice works at Acme Corporation as a software engineer.",
        event_date=JAN,
        request_context=request_context,
    )
    await memory.retain_async(
        bank_id=bank_id,
        content="Alice now works at Beta Industries as a staff engineer.",
        event_date=JUN,
        request_context=request_context,
    )

    conn = await asyncpg.connect(pg0_db_url)
    try:
        # fact_type filter matters: MockLLM also creates observation rows with
        # the same text; pinning those would leave the world facts undated.
        old_id = await conn.fetchval(
            "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'world' "
            "AND lower(text) LIKE '%acme%' LIMIT 1",
            bank_id,
        )
        new_id = await conn.fetchval(
            "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'world' "
            "AND lower(text) LIKE '%beta%' LIMIT 1",
            bank_id,
        )
        assert old_id and new_id
        # Pin the temporal coordinates the algebra needs (MockLLM extraction
        # does not reliably set occurred_start) and mark the old fact as
        # already consolidated so the reset is observable.
        await conn.execute(
            "UPDATE memory_units SET occurred_start = $2, consolidated_at = now() WHERE id = $1",
            old_id,
            JAN,
        )
        pre_consolidated = await conn.fetchval("SELECT consolidated_at FROM memory_units WHERE id = $1", old_id)
        await conn.execute("UPDATE memory_units SET occurred_start = $2 WHERE id = $1", new_id, JUN)
        await conn.execute(
            "INSERT INTO supersession_queue (bank_id, memory_id) VALUES ($1, $2)",
            bank_id,
            new_id,
        )
    finally:
        await conn.close()

    async def _stub_arbitrate(llm_config, new_fact_text, candidate_texts):
        # Contradict every Acme candidate; deterministic stand-in for the LLM.
        from hindsight_api.engine.response_models import TokenUsage
        from hindsight_api.engine.retain.supersession import ArbitrationResult

        contradicted = [i for i, t in enumerate(candidate_texts) if "acme" in t.lower()]
        return ArbitrationResult(verdict=SupersessionVerdict(contradicted_indices=contradicted), usage=TokenUsage())

    monkeypatch.setattr(supersession, "_arbitrate", _stub_arbitrate)

    internal_ctx = RequestContext(internal=True)
    result = await run_fact_supersession_job(memory, bank_id, internal_ctx)
    assert result["superseded"] == 1

    conn = await asyncpg.connect(pg0_db_url)
    try:
        row = await conn.fetchrow(
            "SELECT valid_until, superseded_at, superseded_by, consolidated_at FROM memory_units WHERE id = $1",
            old_id,
        )
        assert row["valid_until"] == JUN
        assert row["superseded_by"] == new_id
        assert row["superseded_at"] is not None
        # The worker resets consolidated_at; the (synchronous in tests) auto
        # consolidation it submits then re-consumes the superseded fact and
        # stamps a NEWER consolidated_at — evidence of the full reset→re-read
        # cycle that lets the observation layer narrate the belief change.
        assert row["consolidated_at"] is not None and row["consolidated_at"] > pre_consolidated

        # Idempotency: requeue and rerun — the superseded candidate is invisible
        # to the worker's recall (validity filter), so nothing changes.
        await conn.execute("INSERT INTO supersession_queue (bank_id, memory_id) VALUES ($1, $2)", bank_id, new_id)
    finally:
        await conn.close()

    result2 = await run_fact_supersession_job(memory, bank_id, internal_ctx)
    assert result2["superseded"] == 0

    conn = await asyncpg.connect(pg0_db_url)
    try:
        unchanged = await conn.fetchval("SELECT valid_until FROM memory_units WHERE id = $1", old_id)
        assert unchanged == JUN
    finally:
        await conn.close()


async def test_worker_skips_already_superseded_or_missing(memory, pg0_db_url, request_context):
    """Step-0 self-check: queue rows pointing at gone/superseded facts are dropped."""
    bank_id = f"test_sq_skip_{datetime.now(UTC).timestamp()}"
    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(
            "INSERT INTO supersession_queue (bank_id, memory_id) VALUES ($1, $2)",
            bank_id,
            uuid.uuid4(),  # fact never existed
        )
    finally:
        await conn.close()

    result = await run_fact_supersession_job(memory, bank_id, RequestContext(internal=True))
    assert result == {"checked": 0, "superseded": 0, "duplicates_noted": 0, "skipped": 0}

    conn = await asyncpg.connect(pg0_db_url)
    try:
        remaining = await conn.fetchval("SELECT COUNT(*) FROM supersession_queue WHERE bank_id = $1", bank_id)
        assert remaining == 0, "claimed rows must not be re-queued for missing facts"
    finally:
        await conn.close()


def test_equal_start_boundary_respects_check_constraint():
    # Regression guard for the boundary floor: a winner first mentioned BEFORE
    # the shared occurred_start must still produce valid_until > occurred_start.
    action = resolve_supersession(
        _tl("new", start=JAN, seen=JAN - timedelta(days=1)),
        _tl("c", start=JAN, seen=JAN - timedelta(days=2)),
    )
    assert action is not None
    assert action.valid_until > JAN
