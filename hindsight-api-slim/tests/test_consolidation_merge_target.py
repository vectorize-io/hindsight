"""A merge delete moves its evidence only onto an observation the model was shown.

The model names a merge target by id. It is honoured only when that observation is in the
unioned recall pool the model read (#6324 rewrote that lookup); any other target downgrades the
merge to a plain delete, so the deleted observation's evidence is not folded into an
observation the model never saw.
"""

import uuid
from unittest.mock import patch

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.consolidation import consolidator
from hindsight_api.engine.consolidation.consolidator import _BatchLLMResult, _DeleteAction, run_consolidation_job
from hindsight_api.engine.memory_engine import MemoryEngine
from tests.consolidation_actions import execute_create_action
from tests.test_consolidation_temporal_merge import (  # noqa: F401 — fixture
    _retain_fact,
    observations_enabled,
)


async def _observation_sources(
    memory: MemoryEngine, bank_id: str, request_context: RequestContext
) -> dict[str, list[str]]:
    """Each observation's id → the ids of the facts it cites, read through the engine."""
    listed = await memory.list_memory_units(bank_id, fact_type="observation", request_context=request_context)
    sources: dict[str, list[str]] = {}
    for item in listed["items"]:
        unit = await memory.get_memory_unit(bank_id, item["id"], request_context)
        sources[str(item["id"])] = unit["source_memory_ids"]
    return sources


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
@pytest.mark.parametrize("target", ["shown", "hidden"])
async def test_merge_folds_evidence_only_into_a_shown_observation(
    memory: MemoryEngine, request_context, observations_enabled, target: str
):
    bank_id = f"test-merge-target-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)

    texts = {
        "gone": "The user likes React for frontend work.",
        "shown": "The user likes React for UI work.",
        "hidden": "The user likes React for side projects.",
    }
    markers = {"gone": "frontend", "shown": "UI work", "hidden": "side projects"}
    facts = {
        key: await _retain_fact(memory, request_context, observations_enabled, bank_id, text, markers[key])
        for key, text in texts.items()
    }
    for key, text in texts.items():
        action = await execute_create_action(
            pool=await memory._get_backend(),
            memory_engine=memory,
            bank_id=bank_id,
            source_memory_ids=[facts[key]],
            text=text,
        )
        assert action == "created"
    listed = await memory.list_memory_units(bank_id, fact_type="observation", request_context=request_context)
    by_text = {o["text"]: str(o["id"]) for o in listed["items"]}
    ids = {key: by_text[text] for key, text in texts.items()}
    await _retain_fact(memory, request_context, observations_enabled, bank_id, "The user likes React a lot.", "a lot")

    find_related = consolidator._find_related_observations

    async def recall_without_hidden(**kwargs):
        result = await find_related(**kwargs)
        return result.model_copy(update={"results": [o for o in result.results if str(o.id) != ids["hidden"]]})

    async def scripted_llm(**_kwargs) -> _BatchLLMResult:
        return _BatchLLMResult(deletes=[_DeleteAction(observation_id=ids["gone"], merged_into=ids[target])])

    with (
        patch.object(consolidator, "_find_related_observations", recall_without_hidden),
        patch.object(consolidator, "_consolidate_batch_with_llm", scripted_llm),
    ):
        await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

    sources = await _observation_sources(memory, bank_id, request_context)
    gone_fact = str(facts["gone"])
    assert ids["gone"] not in sources, "the merged-away observation is deleted"
    assert (gone_fact in sources[ids["shown"]]) is (target == "shown")
    assert gone_fact not in sources[ids["hidden"]], "never folded into an observation the model was not shown"

    await memory.delete_bank(bank_id, request_context=request_context)
