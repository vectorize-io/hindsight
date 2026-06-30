"""End-to-end checks that per-operation temperature reaches the LLM call.

These drive the real pipeline with the mock LLM provider (which records the
``temperature`` it receives) and assert that each operation forwards the
configured value -- including ``None``, which omits the parameter for models
that reject explicit temperatures (issue #2459).

Config resolution itself is unit-tested in ``test_llm_temperature_env.py``;
here we verify the value is actually threaded through to ``provider.call()``.
"""

from datetime import datetime, timezone

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.search import think_utils


def _calls_for_scope(memory, scope: str) -> list[dict]:
    """Collect mock call records for a scope across the engine's LLM configs.

    retain/reflect/consolidation each wrap a distinct provider instance, so a
    given scope only lands on one of them; gather from all and filter.
    """
    seen_impls: dict[int, object] = {}
    for config in (
        memory._llm_config,
        memory._retain_llm_config,
        memory._reflect_llm_config,
        memory._consolidation_llm_config,
    ):
        impl = config._provider_impl
        seen_impls[id(impl)] = impl

    calls: list[dict] = []
    for impl in seen_impls.values():
        calls.extend(impl.get_mock_calls())
    return [c for c in calls if c.get("scope") == scope]


@pytest.mark.asyncio
async def test_retain_forwards_configured_temperature(memory, request_context):
    """Retain's fact extraction must call the LLM with the retain temperature (0.1 default)."""
    bank_id = f"test_temp_retain_{datetime.now(timezone.utc).timestamp()}"
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice is a senior engineer at TechCorp. She works on distributed systems.",
            context="team overview",
            event_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            request_context=request_context,
        )

        extract_calls = _calls_for_scope(memory, "retain_extract_facts")
        assert extract_calls, "retain should have made a fact-extraction LLM call"
        assert all(c["temperature"] == 0.1 for c in extract_calls)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_reflect_think_forwards_configured_temperature(memory):
    """The reflect 'thinking' path must call the LLM with the reflect temperature (0.9 default)."""
    reflect_config = memory._reflect_llm_config
    reflect_config._provider_impl.clear_mock_calls()

    await think_utils.reflect(
        llm_config=reflect_config,
        query="What does Alice work on?",
        world_facts=["Alice works on distributed systems."],
    )

    think_calls = [c for c in reflect_config._provider_impl.get_mock_calls() if c["scope"] == "memory_think"]
    assert think_calls, "reflect should have made a memory_think LLM call"
    assert all(c["temperature"] == 0.9 for c in think_calls)


@pytest.mark.asyncio
async def test_global_none_omits_temperature_on_real_call(memory, monkeypatch):
    """HINDSIGHT_API_LLM_TEMPERATURE=none must omit (None) the temperature on a live call."""
    monkeypatch.setenv("HINDSIGHT_API_LLM_TEMPERATURE", "none")
    clear_config_cache()
    try:
        reflect_config = memory._reflect_llm_config
        reflect_config._provider_impl.clear_mock_calls()

        await think_utils.reflect(
            llm_config=reflect_config,
            query="What does Alice work on?",
            world_facts=["Alice works on distributed systems."],
        )

        think_calls = [c for c in reflect_config._provider_impl.get_mock_calls() if c["scope"] == "memory_think"]
        assert think_calls, "reflect should have made a memory_think LLM call"
        assert all(c["temperature"] is None for c in think_calls), "temperature should be omitted"
    finally:
        # Restore the cached config so later tests see default temperatures.
        clear_config_cache()
