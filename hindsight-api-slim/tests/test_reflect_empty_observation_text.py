"""Reflect must not 500 when it cites an observation whose text is empty (#5840).

``tool_search_observations`` serialises each observation through ``_prune_nulls``, which drops
empty strings, so an observation with ``text == ""`` reaches ``based_on`` with no ``text`` key.
Building the ``MemoryFact`` from that dict used to fail validation after the answer was written.
"""

import uuid

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.reflect.models import ReflectAgentResult, ToolCall


@pytest.mark.asyncio
async def test_reflect_citing_an_observation_with_empty_text_returns_the_answer(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch
):
    bank_id = f"test-empty-obs-{uuid.uuid4().hex[:8]}"
    observation_id = str(uuid.uuid4())

    async def fake_agent(**kwargs):
        return ReflectAgentResult(
            text="The answer.",
            tool_trace=[
                ToolCall(
                    tool="search_observations",
                    input={"query": "q"},
                    # The shape _prune_nulls leaves for an observation whose text is "".
                    output={"observations": [{"id": observation_id, "fact_type": "observation"}]},
                    duration_ms=1,
                )
            ],
            used_observation_ids=[observation_id],
        )

    monkeypatch.setattr("hindsight_api.engine.memory_engine.run_reflect_agent", fake_agent)
    try:
        result = await memory.reflect_async(
            bank_id=bank_id, query="q", request_context=request_context, exclude_mental_models=True
        )
        assert result.text == "The answer."
        assert [(fact.id, fact.text) for fact in result.based_on["observation"]] == [(observation_id, "")]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
