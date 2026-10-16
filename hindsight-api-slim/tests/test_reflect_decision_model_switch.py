"""Which reflects the decision model orchestrates: the server's setup, then each bank's switch.

``reflect_async`` hands the agent the decision model's sufficiency check (which is what makes a
reflect decision-orchestrated) only when one is set up for reflect and the bank keeps the switch
for that operation on: ``reflect_use_decision_model`` for a reflect, and
``mental_model_refresh_use_decision_model`` for the reflect behind a knowledge-page refresh. A
misread field name would silently send every refresh down the wrong path, so each pairing is
pinned here.
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
    eng.ensure_bank_profile = AsyncMock(return_value={"name": "Test", "mission": ""})  # type: ignore[method-assign]
    eng._bank_freshness = AsyncMock(  # type: ignore[method-assign]
        return_value={"last_consolidated_at": None, "pending_consolidation": 0, "last_memory_write_at": None}
    )
    eng.list_directives = AsyncMock(return_value=DirectivePage(items=[], total=0))  # type: ignore[method-assign]
    eng._get_backend = AsyncMock(return_value=SimpleNamespace())  # type: ignore[method-assign]
    return eng


def _decision_model_set_up(engine) -> SimpleNamespace:
    decision_model = SimpleNamespace(model="jev-test", evidence_is_sufficient=AsyncMock(return_value=True))
    engine._reflect_decision_model_client = decision_model
    return decision_model


async def _orchestrated(engine, monkeypatch, operation_label: str, **switches: bool) -> bool:
    """Whether the agent was handed the decision model, for a bank with ``switches``."""
    from hindsight_api.engine.reflect.models import ReflectAgentResult

    engine._config_resolver = SimpleNamespace(
        resolve_full_config=AsyncMock(
            return_value=SimpleNamespace(
                llm_gemini_safety_settings=None,
                reflect_use_decision_model=switches.get("reflect", True),
                mental_model_refresh_use_decision_model=switches.get("refresh", True),
            )
        ),
        get_bank_config=AsyncMock(return_value={}),
    )
    captured: dict = {}

    async def fake_run_reflect_agent(**kwargs):
        captured.update(kwargs)
        return ReflectAgentResult(text="ok")

    monkeypatch.setattr("hindsight_api.engine.memory_engine.run_reflect_agent", fake_run_reflect_agent)
    await engine.reflect_async(
        bank_id="bank-1",
        query="test",
        request_context=RequestContext(),
        exclude_mental_models=True,
        _operation_label=operation_label,
    )
    sufficient = captured["evidence_is_sufficient_fn"]
    assert (sufficient is None) == (captured["prune_evidence_fn"] is None), "the check and the pruning go together"
    return sufficient is not None


@pytest.mark.asyncio
async def test_without_a_decision_model_reflect_runs_agent_turns(engine, monkeypatch):
    assert not await _orchestrated(engine, monkeypatch, "reflect")


@pytest.mark.asyncio
async def test_a_set_up_decision_model_orchestrates_reflect(engine, monkeypatch):
    _decision_model_set_up(engine)
    assert await _orchestrated(engine, monkeypatch, "reflect")


@pytest.mark.asyncio
async def test_a_bank_can_keep_reflect_on_agent_turns(engine, monkeypatch):
    _decision_model_set_up(engine)
    assert not await _orchestrated(engine, monkeypatch, "reflect", reflect=False)
    assert await _orchestrated(engine, monkeypatch, "reflect", refresh=False), "the refresh switch is not reflect's"


@pytest.mark.asyncio
@pytest.mark.parametrize("label", ["refresh_mental_model", "dry_run_refresh_mental_model"])
async def test_a_refresh_follows_its_own_switch(engine, monkeypatch, label):
    _decision_model_set_up(engine)
    assert await _orchestrated(engine, monkeypatch, label, reflect=False), "the reflect switch is not the refresh's"
    assert not await _orchestrated(engine, monkeypatch, label, refresh=False)


@pytest.mark.parametrize("provider", ["cloudflare", "openai"])
def test_a_provider_reflect_cannot_speak_leaves_agent_turns(engine, monkeypatch, provider):
    """Reflect's client speaks only System One: another provider would fail every first round."""
    from hindsight_api.config import DecisionModelConfig

    setup = DecisionModelConfig(
        provider=provider,
        api_key="k",
        model="m",
        base_url="https://example.invalid",
        timeout=5.0,
        max_concurrent=1,
        max_request_tokens=8192,
    )
    monkeypatch.setattr(
        "hindsight_api.engine.memory_engine.get_config",
        lambda: SimpleNamespace(decision_model_config=lambda operation: setup),
    )

    assert engine._reflect_decision_model() is None
