"""Reflect's own time limit excludes queue time for a knowledge-page refresh only (#6140).

A burst of refreshes shares a small per-operation LLM cap, so most of each one's time
can be spent queued behind the others. Charging that to the limit cancelled refreshes
that finished well inside it when run alone. An interactive reflect keeps the plain
limit: a caller is waiting on it.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.permit_wait import waiting_for_permit
from hindsight_api.models import RequestContext


@pytest.fixture
def engine(monkeypatch):
    """A MemoryEngine with everything reflect_async touches before the agent stubbed,
    a 1s reflect limit, and an agent whose one LLM call queues 1.5s for a permit."""
    pytest.importorskip("sentence_transformers", reason="MemoryEngine construction needs the local-ml extra")
    from hindsight_api import MemoryEngine
    from hindsight_api.engine.memory_engine import DirectivePage
    from hindsight_api.engine.reflect.models import ReflectAgentResult

    eng = MemoryEngine(memory_llm_provider="mock", memory_llm_model="mock-model", skip_llm_verification=True)
    eng._authenticate_tenant = AsyncMock()  # type: ignore[method-assign]
    eng.ensure_bank_profile = AsyncMock(return_value={"name": "Test", "mission": ""})  # type: ignore[method-assign]
    eng._bank_freshness = AsyncMock(  # type: ignore[method-assign]
        return_value={"last_consolidated_at": None, "pending_consolidation": 0, "last_memory_write_at": None}
    )
    eng.list_directives = AsyncMock(return_value=DirectivePage(items=[], total=0))  # type: ignore[method-assign]
    eng._get_backend = AsyncMock(return_value=SimpleNamespace())  # type: ignore[method-assign]
    eng._config_resolver = SimpleNamespace(
        resolve_full_config=AsyncMock(
            return_value=SimpleNamespace(
                llm_gemini_safety_settings=None,
                reflect_use_decision_model=False,
                mental_model_refresh_use_decision_model=False,
            )
        ),
        get_bank_config=AsyncMock(return_value={}),
    )

    async def queued_agent(**_kwargs):
        with waiting_for_permit():
            await asyncio.sleep(1.5)
        return ReflectAgentResult(text="ok")

    monkeypatch.setenv("HINDSIGHT_API_REFLECT_WALL_TIMEOUT", "1")
    clear_config_cache()
    monkeypatch.setattr("hindsight_api.engine.memory_engine.run_reflect_agent", queued_agent)
    yield eng
    clear_config_cache()


async def _reflect(engine, operation_label: str):
    return await engine.reflect_async(
        bank_id="bank-1",
        query="test",
        request_context=RequestContext(),
        exclude_mental_models=True,
        _operation_label=operation_label,
    )


@pytest.mark.asyncio
async def test_a_refresh_queued_past_the_limit_finishes(engine):
    result = await _reflect(engine, "refresh_mental_model")
    assert result.text == "ok"


@pytest.mark.asyncio
async def test_an_interactive_reflect_still_counts_the_queue(engine):
    with pytest.raises(TimeoutError, match="timed out after 1 seconds"):
        await _reflect(engine, "reflect")
