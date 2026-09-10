"""The reflect agent's tool token budgets come from config, not from constants (#4239).

``reflect_async`` resolves ``recall_max_tokens`` / ``recall_chunks_max_tokens`` from
env -> tenant -> bank config, and mental-model refresh layers its own
``trigger.recall_*`` overrides on top. Those values used to reach the agent only as
*closure defaults* on ``recall_fn``, while ``_execute_tool`` always passed the token
arguments positionally from constants of its own -- so none of the configured values
ever applied on the agent path. This asserts they now travel explicitly.

The per-argument clamping those limits drive is covered in ``test_reflect_tools.py``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hindsight_api.models import RequestContext


@pytest.fixture
def engine():
    """A MemoryEngine with everything reflect_async touches before the agent stubbed."""
    pytest.importorskip("sentence_transformers", reason="MemoryEngine construction needs the local-ml extra")
    from hindsight_api import MemoryEngine
    from hindsight_api.engine.memory_engine import DirectivePage

    eng = MemoryEngine(
        memory_llm_provider="mock",
        memory_llm_model="mock-model",
        skip_llm_verification=True,
    )
    eng._authenticate_tenant = AsyncMock()  # type: ignore[method-assign]
    eng.get_bank_profile = AsyncMock(return_value={"name": "Test", "mission": ""})  # type: ignore[method-assign]
    eng.get_bank_freshness = AsyncMock(  # type: ignore[method-assign]
        return_value={"last_consolidated_at": None, "pending_consolidation": 0, "last_memory_write_at": None}
    )
    eng.list_directives = AsyncMock(return_value=DirectivePage(items=[], total=0))  # type: ignore[method-assign]
    eng._get_backend = AsyncMock(return_value=SimpleNamespace())  # type: ignore[method-assign]
    return eng


def _with_bank_config(engine, **overrides):
    engine._config_resolver = SimpleNamespace(
        resolve_full_config=AsyncMock(return_value=SimpleNamespace(llm_gemini_safety_settings=None)),
        get_bank_config=AsyncMock(return_value=dict(overrides)),
    )


async def _capture_limits(engine, monkeypatch, **reflect_kwargs):
    from hindsight_api.engine.reflect.models import ReflectAgentResult

    captured: dict = {}

    async def fake_run_reflect_agent(**kwargs):
        captured["limits"] = kwargs["tool_token_limits"]
        return ReflectAgentResult(text="ok")

    monkeypatch.setattr("hindsight_api.engine.memory_engine.run_reflect_agent", fake_run_reflect_agent)

    await engine.reflect_async(
        bank_id="bank-1",
        query="test",
        request_context=RequestContext(),
        exclude_mental_models=True,
        **reflect_kwargs,
    )
    return captured["limits"]


@pytest.mark.asyncio
async def test_bank_config_recall_budgets_reach_the_agent(engine, monkeypatch):
    """A per-bank recall budget must be what the agent's recall tool defaults to."""
    _with_bank_config(engine, recall_max_tokens=9000, recall_chunks_max_tokens=4500)

    limits = await _capture_limits(engine, monkeypatch)

    assert limits.recall_max_tokens == 9000
    assert limits.recall_chunk_max_tokens == 4500


@pytest.mark.asyncio
async def test_mental_model_trigger_overrides_reach_the_agent(engine, monkeypatch):
    """The per-mental-model trigger overrides win over bank config, as documented."""
    _with_bank_config(engine, recall_max_tokens=9000, recall_chunks_max_tokens=4500)

    limits = await _capture_limits(
        engine,
        monkeypatch,
        recall_max_tokens_override=3000,
        recall_chunks_max_tokens_override=1500,
    )

    assert limits.recall_max_tokens == 3000
    assert limits.recall_chunk_max_tokens == 1500


@pytest.mark.asyncio
async def test_unconfigured_bank_falls_back_to_server_defaults(engine, monkeypatch):
    """An empty bank config resolves to the env-backed server defaults, not to zero."""
    from hindsight_api.config import DEFAULT_RECALL_CHUNKS_MAX_TOKENS, DEFAULT_RECALL_MAX_TOKENS

    _with_bank_config(engine)

    limits = await _capture_limits(engine, monkeypatch)

    assert limits.recall_max_tokens == DEFAULT_RECALL_MAX_TOKENS
    assert limits.recall_chunk_max_tokens == DEFAULT_RECALL_CHUNKS_MAX_TOKENS
