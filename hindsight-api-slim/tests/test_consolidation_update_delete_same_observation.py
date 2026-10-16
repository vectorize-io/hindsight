"""An LLM response that both UPDATEs and DELETEs one observation must keep the update.

Seen while reproducing #2550: a fact superseded an observation and the model rewrote it to the
new state AND deleted it. Deletes applied first, the update found no row, and the new state was
lost while its source fact was stamped consolidated.
"""

import uuid
from unittest.mock import patch

import pytest

from hindsight_api.engine.consolidation import consolidator
from hindsight_api.engine.consolidation.consolidator import (
    _BatchLLMResult,
    _DeleteAction,
    _UpdateAction,
    run_consolidation_job,
)
from hindsight_api.engine.memory_engine import MemoryEngine
from tests.test_consolidation_temporal_merge import (  # noqa: F401 — fixture
    _retain_fact,
    _seed_undated_observation,
    observations_enabled,
)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_update_wins_over_delete_of_the_same_observation(
    memory: MemoryEngine, request_context, observations_enabled
):
    bank_id = f"test-upd-del-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)

    old_fact = await _retain_fact(
        memory, request_context, observations_enabled, bank_id, "React is the user's favorite framework.", "React"
    )
    observation_id = await _seed_undated_observation(
        memory, bank_id, old_fact, "React is the user's favorite framework."
    )
    new_fact = await _retain_fact(
        memory, request_context, observations_enabled, bank_id, "Svelte is the user's favorite framework.", "Svelte"
    )

    async def scripted_llm(**_kwargs) -> _BatchLLMResult:
        return _BatchLLMResult(
            updates=[
                _UpdateAction(
                    text="Svelte is the user's favorite framework; React was before.",
                    observation_id=str(observation_id),
                    source_fact_ids=[str(new_fact)],
                )
            ],
            deletes=[_DeleteAction(observation_id=str(observation_id), reason="superseded")],
        )

    with patch.object(consolidator, "_consolidate_batch_with_llm", scripted_llm):
        await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

    observations = (await memory.list_memory_units(bank_id, fact_type="observation", request_context=request_context))[
        "items"
    ]
    assert [o["text"] for o in observations] == ["Svelte is the user's favorite framework; React was before."]

    await memory.delete_bank(bank_id, request_context=request_context)
