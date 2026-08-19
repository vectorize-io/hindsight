"""Observation merges must never narrow an observation's temporal bounds (#3477).

Every path that folds evidence into an existing observation — the ordinary UPDATE, the
CREATE-time semantic dedup fold, the UPDATE-time dedup fold — has to widen
``event_date``/``occurred_start`` to the earliest known value and
``occurred_end``/``mentioned_at`` to the latest. Before the fix the dedup folds rewrote only
text/sources, so an observation could cite dated source facts while reporting no event
interval at all.

The SQL-executing tests here run against a real database on purpose: the merge lives in SQL,
and asserting on a mocked connection cannot tell a working statement from one PostgreSQL
refuses to plan.
"""

import types
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation import consolidator
from hindsight_api.engine.consolidation.consolidator import (
    _aggregate_source_fields,
    _DedupDecision,
    _DedupOutcome,
    _execute_create_action,
    _execute_update_action,
    _TemporalBounds,
    run_consolidation_job,
)
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.response_models import MemoryFact

EARLY = datetime(2020, 3, 1, tzinfo=timezone.utc)
LATE = datetime(2021, 7, 4, 12, 30, tzinfo=timezone.utc)


@pytest.fixture
def observations_enabled():
    """Observations on, and a loose dedup gate so the near-duplicate CREATE reaches the fold."""
    config = _get_raw_config()
    previous_observations = config.enable_observations
    previous_threshold = config.consolidation_dedup_threshold
    config.enable_observations = True
    config.consolidation_dedup_threshold = 0.5
    yield config
    config.enable_observations = previous_observations
    config.consolidation_dedup_threshold = previous_threshold


async def _retain_fact(
    memory: MemoryEngine,
    request_context,
    config,
    bank_id: str,
    content: str,
    marker: str,
    bounds: _TemporalBounds = _TemporalBounds(),
) -> uuid.UUID:
    """Retain ``content`` and return the id of the resulting fact containing ``marker``.

    The mock extractor also emits facts for the extraction prompt's own boilerplate; those are
    marked consolidated so only the fact under test can join a later consolidation job. Retain
    runs with observations off so the caller controls when consolidation happens.
    """
    previous = config.enable_observations
    config.enable_observations = False
    try:
        await memory.retain_async(bank_id=bank_id, content=content, request_context=request_context)
    finally:
        config.enable_observations = previous

    async with memory._pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE memory_units SET consolidated_at = now()
             WHERE bank_id = $1 AND fact_type <> 'observation' AND consolidated_at IS NULL
               AND text NOT LIKE $2
            """,
            bank_id,
            f"%{marker}%",
        )
        # Repeat the marker predicate rather than leaning on the sweep above: if the extractor
        # ever emits two facts for this content, stamping both and returning an arbitrary one
        # would make the caller's source-id assertions flaky rather than fail here.
        stamped = await conn.fetch(
            """
            UPDATE memory_units
               SET event_date = $2, occurred_start = $3, occurred_end = $4, mentioned_at = $5
             WHERE bank_id = $1 AND fact_type <> 'observation' AND consolidated_at IS NULL
               AND text LIKE $6
            RETURNING id
            """,
            bank_id,
            bounds.event_date,
            bounds.occurred_start,
            bounds.occurred_end,
            bounds.mentioned_at,
            f"%{marker}%",
        )
    assert len(stamped) == 1, f"expected exactly one unconsolidated fact matching {marker!r}, got {len(stamped)}"
    return stamped[0]["id"]


async def _seed_undated_observation(
    memory: MemoryEngine, bank_id: str, source_fact_id: uuid.UUID, text: str
) -> uuid.UUID:
    """Create an observation the way undated source facts do: stamped with a consolidation-time
    ``event_date``/``mentioned_at`` and no occurred interval at all."""
    action = await _execute_create_action(
        pool=await memory._get_backend(),
        memory_engine=memory,
        bank_id=bank_id,
        source_memory_ids=[source_fact_id],
        text=text,
    )
    assert action == "created"
    async with memory._pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, occurred_start FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'",
            bank_id,
        )
    assert row["occurred_start"] is None
    return row["id"]


async def _observations(memory: MemoryEngine, bank_id: str) -> list:
    async with memory._pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT id, text, event_date, occurred_start, occurred_end, mentioned_at, source_memory_ids
              FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation' ORDER BY created_at
            """,
            bank_id,
        )


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_dedup_create_fold_widens_bounds_of_the_twin(memory: MemoryEngine, request_context, observations_enabled):
    """#3477: a CREATE folded into a near-twin must hand over its source facts' dates.

    Without this the twin ends up citing a dated source fact while still reporting
    ``occurred_start``/``occurred_end`` as NULL — the exact shape reported in the issue.
    """
    bank_id = f"test-temporal-fold-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    undated_fact = await _retain_fact(
        memory, request_context, observations_enabled, bank_id, "Alice moved to Berlin for work.", "Alice"
    )
    observation_id = await _seed_undated_observation(memory, bank_id, undated_fact, "Alice moved to Berlin for work.")

    dated_fact = await _retain_fact(
        memory,
        request_context,
        observations_enabled,
        bank_id,
        "Alice relocated to Berlin for a new job.",
        "relocated",
        _TemporalBounds(event_date=EARLY, occurred_start=EARLY, occurred_end=LATE, mentioned_at=LATE),
    )

    # The adjudicating LLM votes "merge", so this CREATE folds into the twin instead of inserting.
    with patch.object(
        consolidator,
        "_dedup_decision_from_response",
        lambda _response: _DedupDecision(action="merge", text="Alice moved to Berlin for a new job.", reason="test"),
    ):
        await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

    rows = await _observations(memory, bank_id)
    assert len(rows) == 1, f"the CREATE should have folded into the twin, not inserted: {[dict(r) for r in rows]}"
    folded = rows[0]
    assert folded["id"] == observation_id
    assert dated_fact in folded["source_memory_ids"], "the dated fact must be cited by the survivor"
    assert folded["occurred_start"] == EARLY
    assert folded["occurred_end"] == LATE
    assert folded["event_date"] == EARLY, "the source's earlier event_date must win over the stamped one"
    assert folded["mentioned_at"] > LATE, "mentioned_at keeps the later of the two (the stamped one)"

    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_update_widens_bounds_from_its_source_facts(memory: MemoryEngine, request_context, observations_enabled):
    """The ordinary UPDATE path inherits every temporal field from its sources, event_date included."""
    bank_id = f"test-temporal-update-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    undated_fact = await _retain_fact(
        memory, request_context, observations_enabled, bank_id, "Bob plays the cello.", "cello"
    )
    observation_id = await _seed_undated_observation(memory, bank_id, undated_fact, "Bob plays the cello.")
    stamped = (await _observations(memory, bank_id))[0]

    dated_fact = await _retain_fact(
        memory,
        request_context,
        observations_enabled,
        bank_id,
        "Bob joined a string quartet.",
        "quartet",
        _TemporalBounds(event_date=EARLY, occurred_start=EARLY, occurred_end=LATE, mentioned_at=LATE),
    )

    embedding = await _execute_update_action(
        pool=await memory._get_backend(),
        memory_engine=memory,
        bank_id=bank_id,
        source_memory_ids=[dated_fact],
        observation_id=str(observation_id),
        new_text="Bob plays the cello in a string quartet.",
        observations=[
            MemoryFact(
                id=str(observation_id),
                text=stamped["text"],
                fact_type="observation",
                source_fact_ids=[str(undated_fact)],
                tags=[],
            )
        ],
        source_bounds=_TemporalBounds(event_date=EARLY, occurred_start=EARLY, occurred_end=LATE, mentioned_at=LATE),
    )
    assert embedding is not None, "the update must have landed"

    updated = (await _observations(memory, bank_id))[0]
    assert updated["occurred_start"] == EARLY
    assert updated["occurred_end"] == LATE
    assert updated["event_date"] == EARLY, "event_date must follow the sources, not stay at creation time"
    assert updated["mentioned_at"] == stamped["mentioned_at"], "the stamped mention is later than the source's"

    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_dedup_update_fold_unions_the_bounds_of_both_rows(
    memory: MemoryEngine, request_context, observations_enabled
):
    """The UPDATE-time fold deletes the folded-from row, so the survivor must absorb its dates."""
    bank_id = f"test-temporal-updfold-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    twin_fact = await _retain_fact(
        memory, request_context, observations_enabled, bank_id, "Carla ran the Rome marathon.", "marathon"
    )
    updated_fact = await _retain_fact(
        memory, request_context, observations_enabled, bank_id, "Carla trains every morning.", "trains"
    )

    twin_text = "Carla ran the Rome marathon."
    updated_text = "Carla trains every morning and ran the Rome marathon."
    await _execute_create_action(
        pool=await memory._get_backend(),
        memory_engine=memory,
        bank_id=bank_id,
        source_memory_ids=[twin_fact],
        text=twin_text,
        occurred_start=EARLY,
    )
    await _execute_create_action(
        pool=await memory._get_backend(),
        memory_engine=memory,
        bank_id=bank_id,
        source_memory_ids=[updated_fact],
        text=updated_text,
        occurred_end=LATE,
    )
    rows = {row["text"]: row for row in await _observations(memory, bank_id)}
    twin_id, updated_id = rows[twin_text]["id"], rows[updated_text]["id"]
    assert rows[twin_text]["occurred_end"] is None and rows[updated_text]["occurred_start"] is None

    # The re-embedded UPDATE drifted onto the twin and the LLM votes "merge".
    with patch.object(
        consolidator,
        "_dedup_adjudicate",
        AsyncMock(
            return_value=_DedupOutcome(
                best_id=str(twin_id), merged_text=twin_text, should_merge=True, best_text=twin_text
            )
        ),
    ):
        await consolidator._dedup_reconcile_update(
            await memory._get_backend(),
            memory,
            bank_id,
            observations_enabled,
            None,
            str(updated_id),
            updated_text,
            None,
            [],
        )

    survivors = await _observations(memory, bank_id)
    assert [row["id"] for row in survivors] == [twin_id], "the folded-from row must be gone"
    assert survivors[0]["occurred_start"] == EARLY, "the twin keeps its own start"
    assert survivors[0]["occurred_end"] == LATE, "and absorbs the deleted row's end"
    assert updated_fact in survivors[0]["source_memory_ids"]

    await memory.delete_bank(bank_id, request_context=request_context)


def test_temporal_bounds_merge_keeps_the_widest_known_interval():
    """min/max per field, with a missing value on either side ignored."""
    merged = _TemporalBounds(event_date=LATE, occurred_start=LATE, occurred_end=EARLY).merged_with(
        _TemporalBounds(event_date=EARLY, occurred_end=LATE, mentioned_at=LATE)
    )
    assert merged == _TemporalBounds(event_date=EARLY, occurred_start=LATE, occurred_end=LATE, mentioned_at=LATE)
    assert _TemporalBounds().merged_with(_TemporalBounds()) == _TemporalBounds()


def test_temporal_bounds_of_reads_a_source_aggregation():
    """``_aggregate_source_fields`` is where an observation's dates come from, so its result has
    to hand over all four fields unchanged."""
    aggregation = _aggregate_source_fields(
        [
            {"event_date": LATE, "occurred_start": LATE, "occurred_end": LATE, "mentioned_at": EARLY},
            {"event_date": EARLY, "occurred_start": EARLY, "occurred_end": None, "mentioned_at": LATE},
        ]
    )
    assert _TemporalBounds.of(aggregation) == _TemporalBounds(
        event_date=EARLY, occurred_start=EARLY, occurred_end=LATE, mentioned_at=LATE
    )


@pytest.mark.asyncio
async def test_store_owned_fold_merges_bounds_like_the_sql_path():
    """A memories store that owns its rows re-upserts the whole observation, so the fold has to
    apply the same widening in Python."""
    stored = types.SimpleNamespace(
        source_memory_ids=["existing"],
        tags=["t1"],
        created_at=EARLY,
        event_date=LATE,
        occurred_start=LATE,
        occurred_end=EARLY,
        mentioned_at=EARLY,
    )
    store = types.SimpleNamespace(get_memories=AsyncMock(return_value=[stored]), upsert_observation=AsyncMock())

    with patch.object(consolidator.embedding_utils, "generate_embeddings_batch", AsyncMock(return_value=[[0.1, 0.2]])):
        await consolidator._reconcile_merge_via_store(
            store,
            conn=object(),
            memory_engine=types.SimpleNamespace(embeddings=object()),
            bank_id="bank-1",
            observation_id=str(uuid.uuid4()),
            merged_text="merged",
            add_source_ids=[uuid.uuid4()],
            add_bounds=_TemporalBounds(event_date=EARLY, occurred_end=LATE, mentioned_at=LATE),
        )

    record = store.upsert_observation.await_args.kwargs["record"]
    assert record.event_date == EARLY
    assert record.occurred_start == LATE
    assert record.occurred_end == LATE
    assert record.mentioned_at == LATE
    assert record.created_at == EARLY, "fields the fold does not own are preserved"
