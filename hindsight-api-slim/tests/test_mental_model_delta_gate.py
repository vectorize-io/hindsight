"""The decision-model gate in front of a delta knowledge-page refresh.

With a decision model set up, a delta refresh reads the memories new since its last run,
asks the decision model which could change the page, and hands only those to the delta LLM
call — reflect never runs. None kept means no LLM call at all, and the watermark still moves
past them.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memory_engine import MentalModelRefreshError
from hindsight_api.engine.reflect.delta_ops import DeltaOperationList
from hindsight_api.engine.response_models import LLMCallResult, ReflectResult, TokenUsage
from hindsight_api.engine.schema import fq_table

TOPIC = "What are the user's durable collaboration preferences?"
RELEVANT = "The user prefers bullet points over prose."
UNRELATED = "The build server runs Linux."


@dataclass(frozen=True)
class _Page:
    bank_id: str
    mm_id: str


async def _page_with_new_memories(memory: MemoryEngine, request_context: RequestContext) -> _Page:
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
    # Direct SQL because the window is the thing under test: the page must look
    # refreshed a day ago and the memories must land after that, and neither the
    # refresh nor retain lets a caller back-date ``last_refreshed_at``/``updated_at``.
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
    return _Page(bank_id=bank_id, mm_id=mm["id"])


@pytest.fixture
def gated(memory: MemoryEngine, monkeypatch):
    """A decision model that keeps only the memory about preferences, and reflect plus
    the delta LLM call recorded."""
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


async def test_only_the_kept_memories_reach_the_llm(memory, request_context, gated):
    page = await _page_with_new_memories(memory, request_context)

    refreshed = await memory.refresh_mental_model(
        bank_id=page.bank_id, mental_model_id=page.mm_id, request_context=request_context
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
    page = await _page_with_new_memories(memory, request_context)

    refreshed = await memory.refresh_mental_model(
        bank_id=page.bank_id, mental_model_id=page.mm_id, request_context=request_context
    )

    assert gated.reflect == [] and gated.llm == []
    assert refreshed["content"] == "# Preferences\n\nThe user prefers concise answers.\n"
    assert refreshed["reflect_response"]["delta_skipped_reason"] == "no_new_facts"
    assert refreshed["reflect_response"]["delta_gate"] == {"memories": 2, "kept": 0}
    async with memory._pool.acquire() as conn:
        assert (
            await memory.compute_mental_model_is_stale(
                conn,
                page.bank_id,
                await conn.fetchrow(
                    "SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at FROM mental_models "
                    "WHERE bank_id = $1 AND id = $2",
                    page.bank_id,
                    page.mm_id,
                ),
            )
            is False
        ), "the watermark moves past memories the gate dropped"


@pytest.mark.parametrize("why", ["no_decision_model", "bank_opted_out", "over_the_cap", "decision_model_failed"])
async def test_falls_back_to_reflect(memory, request_context, gated, monkeypatch, why):
    if why == "no_decision_model":
        monkeypatch.setattr(memory, "_reflect_decision_model", lambda: None)
    elif why == "over_the_cap":
        monkeypatch.setattr("hindsight_api.engine.memory_engine._DELTA_GATE_MAX_MEMORIES", 1)
    elif why == "decision_model_failed":
        memory._reflect_decision_model_client.could_change_page.side_effect = RuntimeError("down")
    page = await _page_with_new_memories(memory, request_context)
    if why == "bank_opted_out":
        await memory._config_resolver.update_bank_config(page.bank_id, {"mental_model_delta_use_decision_model": False})

    refreshed = await memory.refresh_mental_model(
        bank_id=page.bank_id, mental_model_id=page.mm_id, request_context=request_context
    )

    assert len(gated.reflect) == 1
    assert "delta_gate" not in refreshed["reflect_response"]


@pytest.mark.parametrize("switch", ["mental_model_refresh_use_decision_model", "reflect_use_decision_model"])
async def test_other_decision_switches_leave_the_gate_on(memory, request_context, gated, switch):
    """Each decision-model path has its own opt-out: turning off reflect's leaves the gate on."""
    page = await _page_with_new_memories(memory, request_context)
    await memory._config_resolver.update_bank_config(page.bank_id, {switch: False})

    refreshed = await memory.refresh_mental_model(
        bank_id=page.bank_id, mental_model_id=page.mm_id, request_context=request_context
    )

    assert gated.reflect == []
    assert refreshed["reflect_response"]["delta_gate"] == {"memories": 2, "kept": 1}


async def test_directives_reach_the_gated_edit(memory, request_context, gated):
    """Reflect hands the delta edit the bank's directives; the gated path must too."""
    page = await _page_with_new_memories(memory, request_context)
    await memory.create_directive(
        page.bank_id, "Tone", "Always write dates as YYYY-MM-DD.", request_context=request_context
    )

    refreshed = await memory.refresh_mental_model(
        bank_id=page.bank_id, mental_model_id=page.mm_id, request_context=request_context
    )

    assert "Always write dates as YYYY-MM-DD." in str(gated.llm[0])
    assert refreshed["reflect_response"]["delta_gate"] == {"memories": 2, "kept": 1}, "directives are not memories"


async def test_a_failed_gated_edit_says_why(memory, request_context, gated, monkeypatch):
    """The gate leaves no reflect candidate, so a failed edit must not read as "empty content"."""
    page = await _page_with_new_memories(memory, request_context)

    async def boom(*, messages, **kwargs):
        raise RuntimeError("simulated provider 500")

    monkeypatch.setattr(memory._mental_model_refresh_llm_config, "call", boom)
    with pytest.raises(MentalModelRefreshError):
        await memory.refresh_mental_model(
            bank_id=page.bank_id, mental_model_id=page.mm_id, request_context=request_context
        )

    preserved = await memory.get_mental_model(
        bank_id=page.bank_id, mental_model_id=page.mm_id, request_context=request_context
    )
    assert preserved["content"] == "# Preferences\n\nThe user prefers concise answers.\n"
    assert preserved["reflect_response"]["refresh_skipped"] == "delta_ops_failed"


async def test_the_store_lists_new_memories_in_the_page_scope(memory: MemoryEngine, request_context):
    """The gate's read: the watermark's scope and window, oldest first, capped."""
    bank_id = f"test-delta-gate-scope-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    now = datetime.now(UTC)
    since, until = now - timedelta(hours=1), now
    rows = [
        # (text, fact_type, tags, minutes ago)
        ("before the window", "world", ["a"], 120),
        ("on the lower bound", "world", ["a"], 60),
        ("oldest in window", "world", ["a"], 50),
        ("newer in window", "experience", ["a"], 40),
        ("wrong tag", "world", ["b"], 30),
        ("observation", "observation", ["a"], 20),
        ("newest in window", "world", ["a"], 10),
    ]
    # Direct SQL to place each row at an exact ``updated_at``, which retain cannot do.
    async with memory._pool.acquire() as conn:
        for text, fact_type, tags, minutes in rows:
            at = since if text == "on the lower bound" else now - timedelta(minutes=minutes)
            await conn.execute(
                "INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $6)",
                uuid.uuid4(),
                bank_id,
                text,
                fact_type,
                tags,
                at,
            )

        async def listed(limit: int = 10) -> list[str]:
            found = await get_memories().memories_updated_between(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                until=until,
                since=since,
                limit=limit,
                fact_types=["world", "experience"],
                tags=["a"],
                tags_match="any_strict",
            )
            assert found is not None
            return [m.text for m in found]

        assert await listed() == ["oldest in window", "newer in window", "newest in window"]
        assert await listed(limit=2) == ["oldest in window", "newer in window"]
