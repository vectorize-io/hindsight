"""The consolidation DELETE contract (#4152).

``deletes[].observation_id`` names the observation to remove. It has no defensible
fallback — a delete with no target cannot be guessed at without risking the wrong
row — so it stays required and a delete that omits it still fails closed.

What made that expensive is that rejecting one delete entry rejects the ENTIRE
``_ConsolidationBatchResponse``: the batch's good creates and updates go with it,
the batch is reported failed, and the caller bisects. A model that reliably omits
the field therefore drains the whole supersession path while every run-level
counter still reads healthy.

Two things narrow it, and these tests pin both:

1. The prompt now shows the shape of a delete entry — a worked example with a
   populated ``deletes`` array, and a field rule that says the id is mandatory.
   Before this, both examples ended in ``"deletes": []`` and the ``deletes`` rule
   only said *when* to delete.
2. ``_DeleteAction`` accepts ``id`` as an alias, the name a model copying the
   observation's own field emits. Unambiguous — a delete entry carries exactly one
   identifier — and it keeps a near-miss from discarding the creates beside it.
"""

from __future__ import annotations

import json
import re
import uuid
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation.consolidator import (
    _ConsolidationBatchResponse,
    _CreateAction,
    _DeleteAction,
    run_consolidation_job,
)
from hindsight_api.engine.consolidation.prompts import build_consolidation_system_prompt
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.providers.mock_llm import MockLLM


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


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_delete_accepts_the_id_alias():
    """``id`` is what a model copying the observation's own field name emits."""
    obs_id = str(uuid.uuid4())
    response = _ConsolidationBatchResponse.model_validate(
        {"deletes": [{"id": obs_id, "reason": "superseded by the new fact"}]}
    )
    assert [d.observation_id for d in response.deletes] == [obs_id]


def test_delete_still_accepts_the_canonical_name():
    obs_id = str(uuid.uuid4())
    response = _ConsolidationBatchResponse.model_validate(
        {"deletes": [{"observation_id": obs_id, "reason": "superseded"}]}
    )
    assert [d.observation_id for d in response.deletes] == [obs_id]
    # populate_by_name keeps in-process construction working too.
    assert _DeleteAction(observation_id=obs_id).observation_id == obs_id


def test_delete_naming_no_target_still_fails_closed():
    """A delete that identifies nothing is rejected — the alias widens what counts
    as an identifier, it does not invent one from ``reason`` prose."""
    with pytest.raises(ValidationError) as exc_info:
        _ConsolidationBatchResponse.model_validate({"deletes": [{"reason": "the Civic observation is superseded"}]})

    errors = exc_info.value.errors()
    assert [e["type"] for e in errors] == ["missing"]


def test_alias_saves_the_creates_and_updates_in_the_same_response():
    """The point of the alias: one near-miss delete no longer discards the batch.

    Everything in a rejected response is lost, not just its deletes — so an
    ``id``-shaped delete used to cost the good creates beside it as well.
    """
    obs_id = str(uuid.uuid4())
    payload = {
        "creates": [{"text": "Alice works nights.", "source_fact_ids": [str(uuid.uuid4())], "reason": "new facet"}],
        "updates": [],
        "deletes": [{"id": obs_id, "reason": "superseded"}],
    }
    response = _ConsolidationBatchResponse.model_validate(payload)
    assert len(response.creates) == 1
    assert response.deletes[0].observation_id == obs_id


def test_json_schema_still_advertises_only_the_canonical_name():
    """The alias must not change what a grammar-constrained provider is told to
    emit — it only widens what a free-form one gets away with."""
    schema = _ConsolidationBatchResponse.model_json_schema()["$defs"]["_DeleteAction"]
    assert schema["required"] == ["observation_id"]
    assert set(schema["properties"]) == {"observation_id", "reason"}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_prompt_states_that_deletes_need_an_observation_id():
    prompt = build_consolidation_system_prompt()
    deletes_rule = next(line for line in prompt.splitlines() if line.startswith("- `deletes`"))
    assert "observation_id" in deletes_rule


def test_prompt_shows_a_worked_example_of_a_populated_deletes_array():
    """Both prior examples ended in ``"deletes": []``, so the model had never been
    shown the shape of a delete entry."""
    prompt = build_consolidation_system_prompt()
    match = re.search(r'"deletes": \[\{[^}]*\}\]', prompt)
    assert match is not None, "no worked example with a populated deletes array"
    assert "observation_id" in match.group(0)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


async def _insert_memory(conn, bank_id: str, text: str, tags: list[str]) -> uuid.UUID:
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
        json.dumps(None),
    )
    return mem_id


async def _observations(memory: MemoryEngine, bank_id: str, request_context) -> list[dict]:
    return (
        await memory.list_memory_units(bank_id, fact_type="observation", limit=1000, request_context=request_context)
    )["items"]


def _install_llm(memory: MemoryEngine, callback):
    mock_llm = MockLLM(provider="mock", api_key="", base_url="", model="mock-model")
    mock_llm.set_response_callback(callback)
    wrapper = MagicMock()
    wrapper.with_config.return_value = mock_llm
    memory._consolidation_llm_config = wrapper


def _fact_ids(messages) -> list[str]:
    prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
    return re.findall(r"\[([0-9a-f-]{36})\]", prompt)


async def _seed_one_observation(memory: MemoryEngine, bank_id: str, request_context, text: str, tags: list[str]) -> str:
    """Consolidate one fact into one observation with a well-behaved model."""
    async with memory._pool.acquire() as conn:
        await _insert_memory(conn, bank_id, text, tags)

    def seed_callback(messages, scope):
        if scope != "consolidation":
            return _ConsolidationBatchResponse()
        return _ConsolidationBatchResponse(
            creates=[_CreateAction(text=text, source_fact_ids=[fid]) for fid in _fact_ids(messages)]
        )

    _install_llm(memory, seed_callback)
    with (
        _override_config(memory, consolidation_llm_batch_size=4, consolidation_llm_parallelism=1),
        patch.object(memory, "submit_async_consolidation"),
    ):
        await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

    seeded = await _observations(memory, bank_id, request_context)
    assert len(seeded) == 1
    return str(seeded[0]["id"])


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
@pytest.mark.parametrize("id_field", ["observation_id", "id"])
async def test_supersession_delete_lands_under_either_field_name(memory: MemoryEngine, request_context, id_field: str):
    """End to end, an ``id``-shaped delete removes the superseded observation just
    as the canonical one does — and the run reports it."""
    bank_id = f"test-del-{id_field}-{uuid.uuid4().hex[:8]}"
    tags = [f"user:{uuid.uuid4().hex[:6]}"]
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    original_llm = memory._consolidation_llm_config
    try:
        stale_id = await _seed_one_observation(memory, bank_id, request_context, "Bob is on the beta waitlist.", tags)

        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "The beta waitlist was shut down", tags)

        def deleting_callback(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            return _ConsolidationBatchResponse.model_validate(
                {
                    "creates": [
                        {
                            "text": "Bob uses the product through open signup.",
                            "source_fact_ids": [fid],
                            "reason": "replacement for the superseded waitlist observation",
                        }
                        for fid in _fact_ids(messages)
                    ],
                    "deletes": [{id_field: stale_id, "reason": "the waitlist no longer exists"}],
                }
            )

        _install_llm(memory, deleting_callback)
        with (
            _override_config(memory, consolidation_llm_batch_size=4, consolidation_llm_parallelism=1),
            patch.object(memory, "submit_async_consolidation"),
        ):
            result = await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

        assert result["observations_deleted"] == 1
        assert result["llm_batch_failures"] == 0
        surviving = {str(o["id"]) for o in await _observations(memory, bank_id, request_context)}
        assert stale_id not in surviving
    finally:
        memory._consolidation_llm_config = original_llm
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_delete_with_no_identifier_fails_the_batch_and_is_reported(memory: MemoryEngine, request_context):
    """Fail-closed is still the behaviour for a delete naming nothing — but the run
    now says the responses were discarded instead of reading as healthy (#4151)."""
    bank_id = f"test-del-none-{uuid.uuid4().hex[:8]}"
    tags = ["user:carol"]
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    original_llm = memory._consolidation_llm_config
    try:
        stale_id = await _seed_one_observation(memory, bank_id, request_context, "Carol owns a 2019 Honda Civic.", tags)

        async with memory._pool.acquire() as conn:
            await _insert_memory(conn, bank_id, "Carol sold the Civic in March 2025", tags)
            await _insert_memory(conn, bank_id, "Carol now drives a leased EV", tags)

        def malformed_callback(messages, scope):
            if scope != "consolidation":
                return _ConsolidationBatchResponse()
            ids = _fact_ids(messages)
            creates = [_CreateAction(text=f"Observation about {fid[:8]}", source_fact_ids=[fid]) for fid in ids]
            if len(ids) > 1:
                # Names the observation in prose only — no identifier at all.
                return _ConsolidationBatchResponse.model_validate(
                    {
                        "creates": [c.model_dump() for c in creates],
                        "deletes": [{"reason": "the Civic observation is superseded — Carol sold it"}],
                    }
                )
            return _ConsolidationBatchResponse(creates=creates)

        _install_llm(memory, malformed_callback)
        with (
            _override_config(memory, consolidation_llm_batch_size=2, consolidation_llm_parallelism=1),
            patch.object(memory, "submit_async_consolidation"),
        ):
            result = await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

        # Bisection rescued the facts, so the stuck-fact gauge stays 0 ...
        stats = await memory.get_bank_stats(bank_id, request_context=request_context)
        assert stats["failed_consolidation"] == 0
        # ... and the delete never landed, as fail-closed requires.
        assert result["observations_deleted"] == 0
        surviving = {str(o["id"]) for o in await _observations(memory, bank_id, request_context)}
        assert stale_id in surviving
        # But the run no longer looks clean: the discarded responses are counted.
        assert result["llm_batch_failures"] > 0
    finally:
        memory._consolidation_llm_config = original_llm
        await memory.delete_bank(bank_id, request_context=request_context)
