"""Deduplicate observation source IDs on create/update writes and prompt serialization (#4799).

Repeated IDs in ``source_memory_ids`` inflate consolidation prompts (each occurrence expands
into ``source_memories`` text) and inflate ``proof_count``. Deduplication is by ID only —
distinct IDs that share the same text stay distinct — and first-seen order is preserved.
"""

from __future__ import annotations

import json
import types
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation import consolidator as C
from hindsight_api.engine.consolidation.consolidator import (
    _build_observations_for_llm,
    _ConsolidationBatchResponse,
    _CreateAction,
    _source_memory_ids_as_uuids,
    _unique_source_memory_ids,
    _UpdateAction,
    run_consolidation_job,
)
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.providers.mock_llm import MockLLM
from hindsight_api.engine.response_models import MemoryFact
from tests import consolidation_actions


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


def _override_config(memory: MemoryEngine, **overrides):
    raw = _get_raw_config()
    fake = type(raw)(**{**{f: getattr(raw, f) for f in raw.__dataclass_fields__}, **overrides})
    return patch.object(memory._config_resolver, "resolve_full_config", return_value=fake)


def _llm(callback):
    mock_llm = MockLLM(provider="mock", api_key="", base_url="", model="mock-model")
    mock_llm.set_response_callback(callback)
    wrapper = MagicMock()
    wrapper.with_config.return_value = mock_llm
    return wrapper


async def _list_observations(memory: MemoryEngine, bank_id: str, request_context) -> list[dict]:
    page = await memory.list_memory_units(bank_id, fact_type="observation", limit=500, request_context=request_context)
    return page["items"]


async def _retain_facts(memory: MemoryEngine, request_context, bank_id: str, contents: list[str]) -> list[uuid.UUID]:
    """Retain facts with observations off so the caller controls consolidation writes."""
    config = _get_raw_config()
    previous = config.enable_observations
    config.enable_observations = False
    try:
        ids: list[uuid.UUID] = []
        for content in contents:
            created = await memory.retain_async(bank_id=bank_id, content=content, request_context=request_context)
            ids.extend(uuid.UUID(i) for i in created)
        return ids
    finally:
        config.enable_observations = previous


# ---------------------------------------------------------------------------
# Serializer unit tests (dirty-row protection; fixture-only, no DB mutation)
# ---------------------------------------------------------------------------


class TestBuildObservationsForLlmSourceDedupe:
    def test_empty_or_none_sources_still_proof_count_1(self):
        empty = MemoryFact(id="o1", text="t", fact_type="observation", source_fact_ids=[])
        none = MemoryFact(id="o2", text="t", fact_type="observation", source_fact_ids=None)
        out = _build_observations_for_llm([empty, none], {})
        assert out[0]["proof_count"] == 1
        assert out[1]["proof_count"] == 1
        assert "source_memories" not in out[0]
        assert "source_memories" not in out[1]

    def test_proof_count_is_unique_id_count_not_raw_length(self):
        a, b = str(uuid.uuid4()), str(uuid.uuid4())
        obs = MemoryFact(
            id="o",
            text="obs",
            fact_type="observation",
            source_fact_ids=[a, a, b, a, b],
        )
        source_facts = {
            a: MemoryFact(id=a, text="fact-a", fact_type="world"),
            b: MemoryFact(id=b, text="fact-b", fact_type="world"),
        }
        out = _build_observations_for_llm([obs], source_facts)[0]
        assert out["proof_count"] == 2
        assert [sm["text"] for sm in out["source_memories"]] == ["fact-a", "fact-b"]

    def test_expand_each_unique_id_once_preserving_order(self):
        a, b, c = (str(uuid.uuid4()) for _ in range(3))
        obs = MemoryFact(
            id="o",
            text="obs",
            fact_type="observation",
            source_fact_ids=[a, b, a, c, b],
        )
        source_facts = {
            a: MemoryFact(id=a, text="A", fact_type="world"),
            b: MemoryFact(id=b, text="B", fact_type="world"),
            c: MemoryFact(id=c, text="C", fact_type="world"),
        }
        out = _build_observations_for_llm([obs], source_facts)[0]
        assert out["proof_count"] == 3
        assert [sm["text"] for sm in out["source_memories"]] == ["A", "B", "C"]

    def test_missing_from_map_skipped_in_expand_but_not_in_unique_count(self):
        """Example: [A,A,B,C] with map only A,B → proof_count=3, expand 2 entries."""
        a, b, c = (str(uuid.uuid4()) for _ in range(3))
        obs = MemoryFact(
            id="o",
            text="obs",
            fact_type="observation",
            source_fact_ids=[a, a, b, c],
        )
        source_facts = {
            a: MemoryFact(id=a, text="A", fact_type="world"),
            b: MemoryFact(id=b, text="B", fact_type="world"),
        }
        out = _build_observations_for_llm([obs], source_facts)[0]
        assert out["proof_count"] == 3
        assert [sm["text"] for sm in out["source_memories"]] == ["A", "B"]

    def test_three_distinct_ids_same_text_all_kept(self):
        ids = [str(uuid.uuid4()) for _ in range(3)]
        obs = MemoryFact(id="o", text="obs", fact_type="observation", source_fact_ids=ids)
        source_facts = {sid: MemoryFact(id=sid, text="same text", fact_type="world") for sid in ids}
        out = _build_observations_for_llm([obs], source_facts)[0]
        assert out["proof_count"] == 3
        assert len(out["source_memories"]) == 3
        assert [sm["text"] for sm in out["source_memories"]] == ["same text"] * 3

    def test_dirty_array_with_thousands_of_repeats_matches_unique_input(self):
        a, b, c = (str(uuid.uuid4()) for _ in range(3))
        dirty = [a, b, c] * 1200  # 3600 entries, 3 distinct
        unique = [a, b, c]
        source_facts = {
            a: MemoryFact(id=a, text="A", fact_type="world", context="ca"),
            b: MemoryFact(id=b, text="B", fact_type="world", context="cb"),
            c: MemoryFact(id=c, text="C", fact_type="world", context="cc"),
        }
        dirty_obs = MemoryFact(id="o", text="obs", fact_type="observation", source_fact_ids=dirty)
        unique_obs = MemoryFact(id="o", text="obs", fact_type="observation", source_fact_ids=unique)
        dirty_out = _build_observations_for_llm([dirty_obs], source_facts)
        unique_out = _build_observations_for_llm([unique_obs], source_facts)
        assert dirty_out == unique_out
        # Structure/content identity — not merely a shorter character count.
        assert json.dumps(dirty_out) == json.dumps(unique_out)


# ---------------------------------------------------------------------------
# Helper unit tests: string dedupe + UUID conversion for SQL binds
# ---------------------------------------------------------------------------


class TestSourceMemoryIdHelpers:
    def test_unique_preserves_order_and_collapses_str_uuid(self):
        a, b = uuid.uuid4(), uuid.uuid4()
        assert _unique_source_memory_ids([a, str(a), b, str(b), a]) == [str(a), str(b)]

    def test_as_uuids_converts_normalized_strings(self):
        a, b = uuid.uuid4(), uuid.uuid4()
        ids = _unique_source_memory_ids([a, a, b])
        converted = _source_memory_ids_as_uuids(ids)
        assert converted == [a, b]
        assert all(isinstance(x, uuid.UUID) for x in converted)

    def test_as_uuids_empty(self):
        assert _source_memory_ids_as_uuids([]) == []


# ---------------------------------------------------------------------------
# SQL write path: create / update via consolidation_actions + list_memory_units
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_create_with_duplicate_source_ids_dedupes_array_and_proof_count(memory: MemoryEngine, request_context):
    bank_id = f"test-src-dedupe-create-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        fact_ids = await _retain_facts(
            memory,
            request_context,
            bank_id,
            [
                "Alice loves hiking in the mountains every weekend.",
                "Bob prefers quiet evenings with a book.",
            ],
        )
        assert len(fact_ids) >= 2
        a, b = fact_ids[0], fact_ids[1]
        # Duplicate A twice, then B, then A again — expect [A, B] order and proof_count 2.
        action = await consolidation_actions.create_observation(
            pool=await memory._get_backend(),
            memory_engine=memory,
            bank_id=bank_id,
            source_memory_ids=[a, a, b, a],
            observation_text="Alice and Bob have different weekend habits.",
        )
        assert action["action"] == "created"

        observations = await _list_observations(memory, bank_id, request_context)
        assert len(observations) == 1
        obs = observations[0]
        assert obs["source_memory_ids"] == [str(a), str(b)]
        assert obs["proof_count"] == 2
        assert obs["proof_count"] != 1, "create must write unique length, not the hardcoded 1"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_create_keeps_three_distinct_ids_with_same_text(memory: MemoryEngine, request_context):
    bank_id = f"test-src-dedupe-same-text-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        # Three retains of identical wording still yield three distinct fact IDs.
        fact_ids = await _retain_facts(
            memory,
            request_context,
            bank_id,
            [
                "The lake freezes in January.",
                "The lake freezes in January.",
                "The lake freezes in January.",
            ],
        )
        # Retain may coalesce identical content; ensure we have three distinct unit ids
        # by falling back to listing if the mock extractor returned fewer.
        if len(fact_ids) < 3:
            page = await memory.list_memory_units(
                bank_id, fact_type=["world", "experience"], limit=500, request_context=request_context
            )
            fact_ids = [uuid.UUID(item["id"]) for item in page["items"]]
        assert len(fact_ids) >= 3
        a, b, c = fact_ids[0], fact_ids[1], fact_ids[2]

        action = await consolidation_actions.create_observation(
            pool=await memory._get_backend(),
            memory_engine=memory,
            bank_id=bank_id,
            source_memory_ids=[a, b, c],
            observation_text="The lake freezes in January.",
        )
        assert action["action"] == "created"
        obs = (await _list_observations(memory, bank_id, request_context))[0]
        assert obs["source_memory_ids"] == [str(a), str(b), str(c)]
        assert obs["proof_count"] == 3
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_create_live_filter_drops_dead_then_dedupes(memory: MemoryEngine, request_context):
    bank_id = f"test-src-dedupe-live-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        fact_ids = await _retain_facts(
            memory,
            request_context,
            bank_id,
            [
                "Carol sails on Sundays.",
                "Dan climbs on Saturdays.",
            ],
        )
        a, b = fact_ids[0], fact_ids[1]
        dead = uuid.uuid4()  # never inserted

        action = await consolidation_actions.create_observation(
            pool=await memory._get_backend(),
            memory_engine=memory,
            bank_id=bank_id,
            source_memory_ids=[a, dead, a, b, dead],
            observation_text="Carol and Dan have weekend hobbies.",
        )
        assert action["action"] == "created"
        obs = (await _list_observations(memory, bank_id, request_context))[0]
        assert obs["source_memory_ids"] == [str(a), str(b)]
        assert obs["proof_count"] == 2
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_update_old_union_new_overlap_dedupes_old_then_new(memory: MemoryEngine, request_context):
    bank_id = f"test-src-dedupe-update-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        fact_ids = await _retain_facts(
            memory,
            request_context,
            bank_id,
            [
                "Eve paints landscapes.",
                "Frank plays the violin.",
                "Grace writes poetry.",
            ],
        )
        assert len(fact_ids) >= 3
        a, b, c = fact_ids[0], fact_ids[1], fact_ids[2]

        created = await consolidation_actions.create_observation(
            pool=await memory._get_backend(),
            memory_engine=memory,
            bank_id=bank_id,
            source_memory_ids=[a, b],
            observation_text="Eve and Frank are artists.",
        )
        obs_id = created["observation_id"]

        # Model carries old sources as str (as recall does); new live_ids are UUID.
        # Overlap on A must collapse; order: old first (A, B), then new C.
        model = MemoryFact(
            id=obs_id,
            text="Eve and Frank are artists.",
            fact_type="observation",
            source_fact_ids=[str(a), str(b)],
        )
        emb = await consolidation_actions.execute_update_action(
            pool=await memory._get_backend(),
            memory_engine=memory,
            bank_id=bank_id,
            source_memory_ids=[a, a, c, a],  # overlap + dups among new
            observation_id=obs_id,
            new_text="Eve, Frank, and Grace are artists.",
            observations=[model],
        )
        assert emb is not None

        obs = (await _list_observations(memory, bank_id, request_context))[0]
        assert obs["source_memory_ids"] == [str(a), str(b), str(c)]
        assert obs["proof_count"] == 3

        # Same sources updated again — set and count must not grow.
        model2 = MemoryFact(
            id=obs_id,
            text="Eve, Frank, and Grace are artists.",
            fact_type="observation",
            source_fact_ids=[str(a), str(b), str(c)],
        )
        emb2 = await consolidation_actions.execute_update_action(
            pool=await memory._get_backend(),
            memory_engine=memory,
            bank_id=bank_id,
            source_memory_ids=[a, b, c, a, b],
            observation_id=obs_id,
            new_text="Eve, Frank, and Grace are artists.",
            observations=[model2],
        )
        assert emb2 is not None
        obs2 = (await _list_observations(memory, bank_id, request_context))[0]
        assert obs2["source_memory_ids"] == [str(a), str(b), str(c)]
        assert obs2["proof_count"] == 3
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_update_live_filter_on_new_sources_then_dedupe(memory: MemoryEngine, request_context):
    bank_id = f"test-src-dedupe-upd-live-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        fact_ids = await _retain_facts(
            memory,
            request_context,
            bank_id,
            [
                "Hank runs marathons.",
                "Iris cycles to work.",
            ],
        )
        a, b = fact_ids[0], fact_ids[1]
        created = await consolidation_actions.create_observation(
            pool=await memory._get_backend(),
            memory_engine=memory,
            bank_id=bank_id,
            source_memory_ids=[a],
            observation_text="Hank runs marathons.",
        )
        obs_id = created["observation_id"]
        dead = uuid.uuid4()
        model = MemoryFact(
            id=obs_id,
            text="Hank runs marathons.",
            fact_type="observation",
            source_fact_ids=[str(a)],
        )
        emb = await consolidation_actions.execute_update_action(
            pool=await memory._get_backend(),
            memory_engine=memory,
            bank_id=bank_id,
            source_memory_ids=[b, dead, b, dead],
            observation_id=obs_id,
            new_text="Hank runs; Iris cycles.",
            observations=[model],
        )
        assert emb is not None
        obs = (await _list_observations(memory, bank_id, request_context))[0]
        assert obs["source_memory_ids"] == [str(a), str(b)]
        assert obs["proof_count"] == 2
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# External store (store-owned) write path — unit-level, no SQL insert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_owned_create_writes_unique_list_and_proof_count():
    a, b = uuid.uuid4(), uuid.uuid4()
    captured: dict = {}

    async def upsert_observation(*, conn, bank_id, record):
        captured["record"] = record

    store = types.SimpleNamespace(
        store_owned_for=lambda bank_id: True,
        upsert_observation=upsert_observation,
    )
    conn = AsyncMock()
    # Live filter returns all (with duplicates preserved until our dedupe).
    live = [a, a, b, a]

    with (
        patch("hindsight_api.engine.consolidation.consolidator.get_memories", lambda: store),
        patch.object(C, "_filter_live_source_memories", AsyncMock(return_value=live)),
    ):
        result = await C._apply_create_observation(
            conn=conn,
            memory_engine=types.SimpleNamespace(
                _backend=types.SimpleNamespace(ops=types.SimpleNamespace(uses_observation_sources_table=False))
            ),
            bank_id="bank-1",
            source_memory_ids=live,
            observation_text="obs",
            embedding_str="[0.1,0.2]",
            tags=[],
        )

    assert result["action"] == "created"
    record = captured["record"]
    assert record.source_memory_ids == [str(a), str(b)]
    assert record.proof_count == 2


@pytest.mark.asyncio
async def test_store_owned_update_normalizes_str_and_uuid_before_dedupe():
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    obs_id = str(uuid.uuid4())
    captured: dict = {}

    async def upsert_observation(*, conn, bank_id, record):
        captured["record"] = record

    stored = types.SimpleNamespace(
        source_memory_ids=[str(a), str(b)],
        tags=[],
        created_at=None,
        event_date=None,
        occurred_start=None,
        occurred_end=None,
        mentioned_at=None,
    )
    store = types.SimpleNamespace(
        store_owned_for=lambda bank_id: True,
        get_memories=AsyncMock(return_value=[stored]),
        upsert_observation=upsert_observation,
    )
    # New live ids overlap A (as UUID) while model carries A/B as str — must collapse.
    live_new = [a, a, c]
    model = MemoryFact(
        id=obs_id,
        text="old",
        fact_type="observation",
        source_fact_ids=[str(a), str(b)],
    )
    prepared = C._PreparedUpdate(
        update=_UpdateAction(text="new", observation_id=obs_id, source_fact_ids=[str(a), str(c)]),
        model=model,
        source_mems=[],
        source_memory_ids=live_new,
        source_fact_tags=[],
        source_bounds=C._TemporalBounds(),
        embedding_str="[0.1]",
    )

    with (
        patch("hindsight_api.engine.consolidation.consolidator.get_memories", lambda: store),
        patch.object(C, "_filter_live_source_memories", AsyncMock(return_value=live_new)),
    ):
        emb = await C._apply_update_action(
            conn=AsyncMock(),
            memory_engine=types.SimpleNamespace(
                _backend=types.SimpleNamespace(ops=types.SimpleNamespace(uses_observation_sources_table=False))
            ),
            bank_id="bank-1",
            prepared=prepared,
        )

    assert emb is not None
    record = captured["record"]
    assert record.source_memory_ids == [str(a), str(b), str(c)]
    assert record.proof_count == 3


# ---------------------------------------------------------------------------
# api-slim integration: run_consolidation_job CREATE with repeated source IDs
# (NOT a hindsight-system-tests story; dirty-row serializer coverage is the
# TestBuildObservationsForLlmSourceDedupe unit class above.)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_consolidation_job_create_with_repeated_source_ids_persists_unique(memory: MemoryEngine, request_context):
    """Mock LLM CREATE with duplicate source_fact_ids; job persists unique IDs + matching proof_count."""
    bank_id = f"test-src-dedupe-job-create-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        fact_ids = await _retain_facts(
            memory,
            request_context,
            bank_id,
            [
                "Julia teaches mathematics at the local college.",
                "Kevin tutors students in physics after class.",
            ],
        )
        assert len(fact_ids) >= 2
        a, b = str(fact_ids[0]), str(fact_ids[1])
        # LLM emits repeats; raw length must exceed unique so we document write-path intent.
        llm_emitted_source_ids = [a, a, b, a, b, a]
        unique_source_ids = [a, b]
        assert len(llm_emitted_source_ids) > len(unique_source_ids)

        def callback(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            return _ConsolidationBatchResponse(
                creates=[
                    _CreateAction(
                        text="Julia and Kevin teach STEM subjects.",
                        source_fact_ids=llm_emitted_source_ids,
                    )
                ]
            )

        original_llm = memory._consolidation_llm_config
        memory._consolidation_llm_config = _llm(callback)
        try:
            with (
                _override_config(memory, consolidation_llm_batch_size=8, consolidation_llm_parallelism=1),
                patch.object(memory, "submit_async_consolidation"),
            ):
                await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

            observations = await _list_observations(memory, bank_id, request_context)
            assert len(observations) >= 1
            obs = next(o for o in observations if "Julia" in o["text"] or "STEM" in o["text"])
            assert obs["source_memory_ids"] == unique_source_ids
            assert obs["proof_count"] == len(unique_source_ids)
            assert obs["proof_count"] == len(obs["source_memory_ids"])
            assert len(llm_emitted_source_ids) > len(obs["source_memory_ids"])
        finally:
            memory._consolidation_llm_config = original_llm
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
