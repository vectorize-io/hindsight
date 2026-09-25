"""Tests for the mental-model-refresh LLM config group (issue #4463).

The automatic refresh drives the same agent as interactive reflect, so it used to
be pinned to the reflect LLM. `HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_*` lets it
diverge; unset, it must stay *literally* the reflect config so nothing changes.
"""

import pytest

from hindsight_api import MemoryEngine
from hindsight_api.config import clear_config_cache, get_config
from hindsight_api.engine.llm_wrapper import _scope_to_operation

BASE_ENV = {
    "HINDSIGHT_API_SKIP_LLM_VERIFICATION": "true",
    "HINDSIGHT_API_LLM_PROVIDER": "mock",
    "HINDSIGHT_API_LLM_MODEL": "default-model",
    "HINDSIGHT_API_REFLECT_LLM_PROVIDER": "mock",
    "HINDSIGHT_API_REFLECT_LLM_MODEL": "reflect-model",
}


@pytest.fixture
def engine_env(monkeypatch):
    """Apply BASE_ENV plus any extra vars, then build an engine."""

    def _build(**extra: str) -> MemoryEngine:
        for key, value in {**BASE_ENV, **extra}.items():
            monkeypatch.setenv(key, value)
        clear_config_cache()
        return MemoryEngine(skip_llm_verification=True)

    yield _build
    clear_config_cache()


def test_unset_reuses_the_reflect_config_object(engine_env):
    """The backwards-compatible path: same object, so no second provider is built.

    Only when reflect and refresh end up on the same timeout — here both on the
    global one; see the next tests.
    """
    engine = engine_env(HINDSIGHT_API_LLM_TIMEOUT="60")

    assert engine._mental_model_refresh_llm_config is engine._reflect_llm_config
    assert get_config().has_mental_model_refresh_llm_override() is False


def test_unset_refresh_does_not_inherit_reflects_interactive_30s_default(engine_env):
    """#4532: reflect's 30s default is for a waiting caller. A background refresh's
    final answer over a large prompt timed out on it every time, so refresh takes the
    global LLM timeout instead — everything else still comes from reflect."""
    engine = engine_env()

    refresh = engine._mental_model_refresh_llm_config
    assert engine._reflect_llm_config.timeout == 30.0
    assert refresh.timeout == get_config().llm_timeout == 120.0
    assert refresh.model == "reflect-model"


def test_unset_refresh_does_not_inherit_an_explicit_reflect_timeout(engine_env):
    """Refresh never inherits reflect's timeout, not even one the operator set: that
    one is sized for interactive reflect too."""
    engine = engine_env(HINDSIGHT_API_REFLECT_LLM_TIMEOUT="45")

    assert engine._reflect_llm_config.timeout == 45.0
    assert engine._mental_model_refresh_llm_config.timeout == 120.0


def test_unset_follows_a_reflect_config_swapped_in_after_init(engine_env):
    """Regression: the refresh must track later writes to `_reflect_llm_config`.

    Real-LLM eval fixtures build the engine, then replace `_reflect_llm_config`
    with a live provider. An alias captured in __init__ kept serving the
    construction-time mock, so those evals got 'mock response' back.
    """
    engine = engine_env()
    swapped = object()

    engine._reflect_llm_config = swapped

    assert engine._mental_model_refresh_llm_config is swapped
    assert engine._llm_for_reflect_operation("refresh_mental_model") is swapped


def test_override_still_wins_over_a_later_reflect_swap(engine_env):
    """The other direction: an explicit override is not clobbered by the swap."""
    engine = engine_env(
        HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_PROVIDER="mock",
        HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_MODEL="refresh-model",
    )
    configured = engine._mental_model_refresh_llm_config

    engine._reflect_llm_config = object()

    assert engine._mental_model_refresh_llm_config is configured


def test_override_gives_the_refresh_its_own_model(engine_env):
    engine = engine_env(
        HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_PROVIDER="mock",
        HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_MODEL="refresh-model",
    )

    assert engine._mental_model_refresh_llm_config.model == "refresh-model"
    # Interactive reflect is untouched.
    assert engine._reflect_llm_config.model == "reflect-model"


def test_unset_fields_fall_back_to_reflect_not_global(engine_env):
    """Only MODEL is overridden; reasoning comes from the reflect group. The timeout
    does not: refresh never inherits reflect's, it takes the global one."""
    engine = engine_env(
        HINDSIGHT_API_REFLECT_LLM_TIMEOUT="45",
        HINDSIGHT_API_REFLECT_LLM_REASONING_EFFORT="high",
        HINDSIGHT_API_LLM_TIMEOUT="10",
        HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_MODEL="refresh-model",
    )

    refresh = engine._mental_model_refresh_llm_config
    assert refresh.model == "refresh-model"
    assert refresh.provider == "mock"  # inherited from reflect
    assert refresh.reasoning_effort == "high"
    assert refresh.timeout == 10.0


def test_own_timeout_wins_over_reflect(engine_env):
    """The headline case: a long background budget next to a short interactive one."""
    engine = engine_env(
        HINDSIGHT_API_REFLECT_LLM_TIMEOUT="30",
        HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_MODEL="refresh-model",
        HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_TIMEOUT="2700",
    )

    assert engine._mental_model_refresh_llm_config.timeout == 2700.0
    assert engine._reflect_llm_config.timeout == 30.0


@pytest.mark.parametrize(
    "label, expect_refresh",
    [
        ("refresh_mental_model", True),
        ("dry_run_refresh_mental_model", True),
        ("reflect", False),
    ],
)
def test_operation_label_selects_the_config(engine_env, label, expect_refresh):
    """`reflect_async` routes by its operation label — the only seam refresh has."""
    engine = engine_env(
        HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_PROVIDER="mock",
        HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_MODEL="refresh-model",
    )

    picked = engine._llm_for_reflect_operation(label)
    expected = engine._mental_model_refresh_llm_config if expect_refresh else engine._reflect_llm_config
    assert picked is expected


def test_refresh_scopes_get_their_own_concurrency_bucket():
    """Capping the refresh must not cap interactive reflect, and vice versa."""
    for scope in ("refresh_mental_model", "dry_run_refresh_mental_model", "mental_model_delta_ops"):
        assert _scope_to_operation(scope) == "mental_model_refresh"
    assert _scope_to_operation("reflect_tool_call") == "reflect"
