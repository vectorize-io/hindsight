"""Which reflects the decision model orchestrates: the server's setup, then each bank's switch.

``reflect_async`` hands the agent the decision model's sufficiency check (which is what makes a
reflect decision-orchestrated) only when one is set up for reflect and the bank keeps the switch
for that operation on: ``reflect_use_decision_model`` for a reflect, and
``mental_model_refresh_use_decision_model`` for the reflect behind a page refresh. A
misread field name would silently send every refresh down the wrong path, so each pairing is
pinned here.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.memories import ReflectFreshness
from hindsight_api.models import RequestContext


def _make_engine(decision_engine_for=None):
    """A MemoryEngine with everything reflect_async touches before the agent stubbed."""
    pytest.importorskip("sentence_transformers", reason="MemoryEngine construction needs the local-ml extra")
    from hindsight_api import MemoryEngine
    from hindsight_api.engine.memory_engine import DirectivePage

    eng = MemoryEngine(
        memory_llm_provider="mock",
        memory_llm_model="mock-model",
        skip_llm_verification=True,
        embeddings=object(),
        decision_engine_for=decision_engine_for,
    )
    eng._authenticate_tenant = AsyncMock()  # type: ignore[method-assign]
    eng.ensure_bank_profile = AsyncMock(return_value={"name": "Test", "mission": ""})  # type: ignore[method-assign]
    eng._reflect_freshness = AsyncMock(  # type: ignore[method-assign]
        return_value=ReflectFreshness(pending=0, last_memory_write_at=None)
    )
    eng.list_directives = AsyncMock(return_value=DirectivePage(items=[], total=0))  # type: ignore[method-assign]
    eng._get_backend = AsyncMock(return_value=SimpleNamespace())  # type: ignore[method-assign]
    return eng


@pytest.fixture
def engine():
    return _make_engine()


def _decision_model_set_up(engine) -> SimpleNamespace:
    """Inject a decision engine for reflect, as the constructor's decision_engine_for would."""
    decision_model = SimpleNamespace(
        provider="typesafe", model="jev-test", evidence_is_sufficient=AsyncMock(return_value=True)
    )
    engine._decision_engine_for = lambda operation: decision_model if operation == "reflect" else None
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
    if sufficient is not None:
        # Ask it, so a test can tell which engine was handed over, not only that one was.
        await sufficient("test", "evidence")
    return sufficient is not None


@pytest.mark.asyncio
async def test_without_a_decision_model_reflect_runs_agent_turns(engine, monkeypatch):
    assert not await _orchestrated(engine, monkeypatch, "reflect")


@pytest.mark.asyncio
async def test_a_set_up_decision_model_orchestrates_reflect(engine, monkeypatch):
    _decision_model_set_up(engine)
    assert await _orchestrated(engine, monkeypatch, "reflect")


class _InjectedReflect:
    """A decision engine whose sufficiency check records that reflect asked it."""

    provider = "typesafe"
    model = "injected"

    def __init__(self) -> None:
        self.operations: list[str] = []

    async def evidence_is_sufficient(self, query: str, evidence: str, *, pictures=()) -> bool:
        self.operations.append("reflect")
        return True


@pytest.mark.asyncio
async def test_an_injected_engine_is_the_one_reflect_asks(monkeypatch):
    """MemoryEngine(decision_engine_for=) is the engine reflect asks, not one assigned afterwards."""
    fake = _InjectedReflect()
    seen: list[str] = []

    def engine_for(operation: str):
        seen.append(operation)
        # Truthy for recall so construction does not build a local cross-encoder. Recall is not run.
        return fake if operation in ("reflect", "recall") else None

    engine = _make_engine(engine_for)
    assert await _orchestrated(engine, monkeypatch, "reflect")
    assert seen == ["recall", "reflect"]
    assert fake.operations == ["reflect"]


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


def _set_up_for_reflect(monkeypatch, setup) -> None:
    monkeypatch.setattr(
        "hindsight_api.engine.decision_model.hindsight_config.get_config",
        lambda: SimpleNamespace(decision_model_config=lambda operation: setup),
    )


@pytest.mark.parametrize("provider", ["cloudflare", "openai"])
def test_a_provider_reflect_cannot_speak_leaves_agent_turns(engine, monkeypatch, provider):
    """Reflect runs on System One only: its first round is tuned to it."""
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
    _set_up_for_reflect(monkeypatch, setup)

    assert engine._reflect_decision_model() is None


@pytest.mark.parametrize("provider", ["typesafe", "hindsight"])
def test_a_system_one_provider_drives_reflect(engine, monkeypatch, provider):
    """Hindsight Inference serves System One too, so reflect and pages run on it."""
    from hindsight_api.config import DecisionModelConfig
    from hindsight_api.engine.decision_model import DecisionModel, decision_engine

    setup = DecisionModelConfig(
        provider=provider,
        api_key="k",
        model="m",
        base_url="https://example.invalid",
        timeout=5.0,
        max_concurrent=1,
        max_request_tokens=8192,
        tokenizer="jev",
    )
    _set_up_for_reflect(monkeypatch, setup)

    client = engine._reflect_decision_model()
    assert isinstance(client, DecisionModel) and client.provider == provider
    assert client is decision_engine("reflect"), "reflect asks the shared engine, not a client of its own"
    # Its cut and verdict are retried like the shared decision model's calls: a single 429 used
    # to drop them, and that round answered like agent mode.
    assert client.retry_policy is not None
    # Its questions are sized to the configured window, counted in the model's vocabulary (#6659):
    # the class default (26k) refused every first request on a smaller window.
    assert client.question_tokens == 8192
    assert client.tokenizer == "jev"
