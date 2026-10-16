"""The decision-model gate in front of a delta knowledge-page refresh.

With ``mental_model_refresh_delta_gate`` on and a decision model set up, a delta refresh
reads the memories new since its last run, asks the decision model which could change the
page, and hands only those to the delta LLM call — reflect never runs. None kept means no
LLM call at all, and the watermark still moves past them.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.config import ENV_MENTAL_MODEL_REFRESH_DELTA_GATE, clear_config_cache
from hindsight_api.engine.reflect.delta_ops import DeltaOperationList
from hindsight_api.engine.response_models import LLMCallResult, ReflectResult, TokenUsage

TOPIC = "What are the user's durable collaboration preferences?"
RELEVANT = "The user prefers bullet points over prose."
UNRELATED = "The build server runs Linux."


async def _page_with_new_memories(memory: MemoryEngine, request_context: RequestContext) -> tuple[str, str]:
    bank_id = f"test-delta-gate-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    mm = await memory.create_mental_model(
        bank_id=bank_id,
        name="User Preferences",
        source_query=TOPIC,
        content="# Preferences\n\nThe user prefers concise answers.\n",
        trigger={"mode": "delta"},
        request_context=request_context,
    )
    assert memory._pool is not None
    async with memory._pool.acquire() as conn:
        await conn.execute(
            "UPDATE mental_models SET last_refreshed_at = NOW() - INTERVAL '1 day', "
            "last_refreshed_source_query = source_query WHERE bank_id = $1 AND id = $2",
            bank_id,
            mm["id"],
        )
        for text in (RELEVANT, UNRELATED):
            await conn.execute(
                "INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at) "
                "VALUES ($1, $2, $3, 'world', ARRAY[]::varchar[], NOW() - INTERVAL '2 minutes', "
                "NOW() - INTERVAL '2 minutes')",
                uuid.uuid4(),
                bank_id,
                text,
            )
    return bank_id, mm["id"]


def _set_gate(monkeypatch, on: bool) -> None:
    monkeypatch.setenv(ENV_MENTAL_MODEL_REFRESH_DELTA_GATE, str(on).lower())
    clear_config_cache()


@pytest.fixture
def gated(memory: MemoryEngine, monkeypatch):
    """Gate on, a decision model that keeps only the memory about preferences, and
    reflect plus the delta LLM call recorded."""
    _set_gate(monkeypatch, True)
    memory._reflect_decision_model_client = SimpleNamespace(  # type: ignore[assignment]
        could_change_page=AsyncMock(side_effect=lambda topic, page, text: "prefers" in text)
    )
    calls = SimpleNamespace(reflect=[], llm=[])

    async def fake_reflect_async(**kwargs):
        calls.reflect.append(kwargs)
        return ReflectResult(text="", based_on={})

    async def fake_call(*, messages, **kwargs):
        calls.llm.append(messages)
        ops = DeltaOperationList.model_validate(
            {"operations": [{"op": "append_block", "section_id": "preferences", "text": "Bullet points."}]}
        )
        return LLMCallResult(content=ops, usage=TokenUsage())

    monkeypatch.setattr(memory, "reflect_async", fake_reflect_async)
    monkeypatch.setattr(memory._mental_model_refresh_llm_config, "call", fake_call)
    yield calls
    monkeypatch.undo()
    clear_config_cache()


async def test_only_the_kept_memories_reach_the_llm(memory, request_context, gated):
    bank_id, mm_id = await _page_with_new_memories(memory, request_context)

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )

    assert gated.reflect == [], "the gate replaces reflect"
    assert len(gated.llm) == 1
    prompt = str(gated.llm[0])
    assert RELEVANT in prompt and UNRELATED not in prompt
    assert "Bullet points." in refreshed["content"]
    assert refreshed["reflect_response"]["delta_gate"] == {"memories": 2, "kept": 1}


async def test_nothing_kept_means_no_llm_call(memory, request_context, gated):
    memory._reflect_decision_model_client.could_change_page.side_effect = None
    memory._reflect_decision_model_client.could_change_page.return_value = False
    bank_id, mm_id = await _page_with_new_memories(memory, request_context)

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )

    assert gated.reflect == [] and gated.llm == []
    assert refreshed["content"] == "# Preferences\n\nThe user prefers concise answers.\n"
    assert refreshed["reflect_response"]["delta_skipped_reason"] == "no_new_facts"
    assert refreshed["reflect_response"]["delta_gate"] == {"memories": 2, "kept": 0}
    async with memory._pool.acquire() as conn:
        assert (
            await memory.compute_mental_model_is_stale(
                conn,
                bank_id,
                await conn.fetchrow(
                    "SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at FROM mental_models "
                    "WHERE bank_id = $1 AND id = $2",
                    bank_id,
                    mm_id,
                ),
            )
            is False
        ), "the watermark moves past memories the gate dropped"


@pytest.mark.parametrize("why", ["switched_off", "over_the_cap", "decision_model_failed"])
async def test_falls_back_to_reflect(memory, request_context, gated, monkeypatch, why):
    if why == "switched_off":
        _set_gate(monkeypatch, False)
    elif why == "over_the_cap":
        monkeypatch.setattr("hindsight_api.engine.memory_engine._DELTA_GATE_MAX_MEMORIES", 1)
    else:
        memory._reflect_decision_model_client.could_change_page.side_effect = RuntimeError("down")
    bank_id, mm_id = await _page_with_new_memories(memory, request_context)

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm_id, request_context=request_context
    )

    assert len(gated.reflect) == 1
    assert "delta_gate" not in refreshed["reflect_response"]
