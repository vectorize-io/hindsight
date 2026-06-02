"""Tests for per-operation LLM concurrency caps.

These tests exercise the dispatch logic in `llm_wrapper` that gates calls on
per-operation semaphores when `HINDSIGHT_API_{RETAIN,REFLECT,CONSOLIDATION}_LLM_MAX_CONCURRENT`
is set. They patch the module-level semaphore registry so they can run without
needing to re-import the module with custom env vars.
"""
import asyncio
from contextlib import AsyncExitStack
from unittest.mock import patch

import pytest

from hindsight_api.engine import llm_wrapper
from hindsight_api.engine.llm_wrapper import (
    LLMProvider,
    _scope_to_operation,
    _semaphores_for_scope,
)


class TestScopeToOperation:
    """Map call-site scope strings to per-operation buckets."""

    @pytest.mark.parametrize(
        "scope, expected",
        [
            ("retain", "retain"),
            ("retain_extract_facts", "retain"),
            ("reflect", "reflect"),
            ("reflect_structured", "reflect"),
            ("reflect_tool_call", "reflect"),
            ("consolidation", "consolidation"),
            # Out-of-bucket scopes — only the global cap applies.
            ("memory_think", None),
            ("bank_mission", None),
            ("mental_model_delta_ops", None),
            ("verification", None),
            ("", None),
        ],
    )
    def test_scope_dispatch(self, scope, expected):
        assert _scope_to_operation(scope) == expected


class TestSemaphoresForScope:
    """`_semaphores_for_scope` always returns the global semaphore, plus the
    per-op one when configured."""

    def test_no_per_op_configured(self):
        with patch.object(llm_wrapper, "_per_op_llm_semaphores", {}):
            sems = _semaphores_for_scope("retain_extract_facts")
            assert sems == [llm_wrapper._global_llm_semaphore]

    def test_per_op_configured_for_matching_scope(self):
        retain_sem = asyncio.Semaphore(2)
        with patch.object(llm_wrapper, "_per_op_llm_semaphores", {"retain": retain_sem}):
            sems = _semaphores_for_scope("retain_extract_facts")
            # Per-op acquired first so contention queues on the narrower cap.
            assert sems == [retain_sem, llm_wrapper._global_llm_semaphore]

    def test_per_op_configured_for_other_scope(self):
        retain_sem = asyncio.Semaphore(2)
        with patch.object(llm_wrapper, "_per_op_llm_semaphores", {"retain": retain_sem}):
            sems = _semaphores_for_scope("reflect")
            # Reflect call does not see retain's per-op semaphore.
            assert sems == [llm_wrapper._global_llm_semaphore]

    def test_unbucketed_scope_only_global(self):
        retain_sem = asyncio.Semaphore(2)
        reflect_sem = asyncio.Semaphore(2)
        consolidation_sem = asyncio.Semaphore(2)
        with patch.object(
            llm_wrapper,
            "_per_op_llm_semaphores",
            {
                "retain": retain_sem,
                "reflect": reflect_sem,
                "consolidation": consolidation_sem,
            },
        ):
            assert _semaphores_for_scope("mental_model_delta_ops") == [
                llm_wrapper._global_llm_semaphore
            ]
            assert _semaphores_for_scope("memory_think") == [llm_wrapper._global_llm_semaphore]
            assert _semaphores_for_scope("verification") == [llm_wrapper._global_llm_semaphore]


class TestBuildPerOpSemaphores:
    """`_build_per_op_semaphores()` reads env vars and validates them."""

    def test_empty_when_no_env_vars(self, monkeypatch):
        monkeypatch.delenv("HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT", raising=False)
        monkeypatch.delenv("HINDSIGHT_API_REFLECT_LLM_MAX_CONCURRENT", raising=False)
        monkeypatch.delenv("HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT", raising=False)
        assert llm_wrapper._build_per_op_semaphores() == {}

    def test_populated_when_env_vars_set(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT", "2")
        monkeypatch.setenv("HINDSIGHT_API_REFLECT_LLM_MAX_CONCURRENT", "3")
        monkeypatch.delenv("HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT", raising=False)
        result = llm_wrapper._build_per_op_semaphores()
        assert set(result.keys()) == {"retain", "reflect"}
        # asyncio.Semaphore's internal counter is _value; assert it matches.
        assert result["retain"]._value == 2
        assert result["reflect"]._value == 3

    def test_empty_string_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT", "")
        monkeypatch.delenv("HINDSIGHT_API_REFLECT_LLM_MAX_CONCURRENT", raising=False)
        monkeypatch.delenv("HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT", raising=False)
        assert llm_wrapper._build_per_op_semaphores() == {}

    @pytest.mark.parametrize("bad_value", ["0", "-1"])
    def test_rejects_non_positive(self, monkeypatch, bad_value):
        monkeypatch.setenv("HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT", bad_value)
        with pytest.raises(ValueError, match="must be a positive integer"):
            llm_wrapper._build_per_op_semaphores()


class TestSemaphoreEnforcement:
    """End-to-end: a fake provider call gated through `_semaphores_for_scope`
    actually respects the per-op and global caps under concurrent load."""

    @pytest.mark.asyncio
    async def test_per_op_cap_limits_concurrency(self):
        retain_sem = asyncio.Semaphore(2)
        # Global is wide-open so we isolate the per-op cap.
        global_sem = asyncio.Semaphore(100)

        in_flight = 0
        peak = 0

        async def fake_call():
            nonlocal in_flight, peak
            sems = [retain_sem, global_sem]
            async with AsyncExitStack() as stack:
                for s in sems:
                    await stack.enter_async_context(s)
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.01)
                in_flight -= 1

        await asyncio.gather(*[fake_call() for _ in range(10)])

        assert peak == 2, f"per-op cap should limit to 2, observed peak={peak}"

    @pytest.mark.asyncio
    async def test_global_cap_still_applies_when_per_op_unset(self):
        global_sem = asyncio.Semaphore(2)

        in_flight = 0
        peak = 0

        async def fake_call():
            nonlocal in_flight, peak
            async with global_sem:
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.01)
                in_flight -= 1

        await asyncio.gather(*[fake_call() for _ in range(10)])

        assert peak == 2, f"global cap should limit to 2, observed peak={peak}"

    @pytest.mark.asyncio
    async def test_llm_provider_call_respects_per_op_cap(self):
        """End-to-end: `LLMProvider.call()` actually goes through
        `_semaphores_for_scope`, so patching the per-op registry caps real
        provider calls."""
        provider = LLMProvider(provider="mock", api_key="", base_url="", model="test-model")

        in_flight = 0
        peak = 0

        # Replace the mock provider's call with one that holds the semaphore long
        # enough to observe concurrency. We can't sleep inside the real MockLLM
        # path without rewriting its internals.
        async def slow_mock_call(**kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return "ok"

        provider._provider_impl.call = slow_mock_call  # type: ignore[assignment]

        retain_sem = asyncio.Semaphore(2)
        with patch.object(llm_wrapper, "_per_op_llm_semaphores", {"retain": retain_sem}):
            await asyncio.gather(
                *[
                    provider.call(
                        messages=[{"role": "user", "content": "x"}],
                        scope="retain_extract_facts",
                    )
                    for _ in range(8)
                ]
            )

        assert peak == 2, f"retain cap should hold even for end-to-end calls, peak={peak}"

    @pytest.mark.asyncio
    async def test_per_op_composes_with_global(self):
        """When both caps are set, the tighter one wins on its operation but
        the global cap still constrains the sum across operations."""
        retain_sem = asyncio.Semaphore(2)
        reflect_sem = asyncio.Semaphore(10)
        global_sem = asyncio.Semaphore(3)

        retain_in_flight = 0
        retain_peak = 0
        total_in_flight = 0
        total_peak = 0

        async def retain_call():
            nonlocal retain_in_flight, retain_peak, total_in_flight, total_peak
            async with AsyncExitStack() as stack:
                await stack.enter_async_context(retain_sem)
                await stack.enter_async_context(global_sem)
                retain_in_flight += 1
                total_in_flight += 1
                retain_peak = max(retain_peak, retain_in_flight)
                total_peak = max(total_peak, total_in_flight)
                await asyncio.sleep(0.01)
                retain_in_flight -= 1
                total_in_flight -= 1

        async def reflect_call():
            nonlocal total_in_flight, total_peak
            async with AsyncExitStack() as stack:
                await stack.enter_async_context(reflect_sem)
                await stack.enter_async_context(global_sem)
                total_in_flight += 1
                total_peak = max(total_peak, total_in_flight)
                await asyncio.sleep(0.01)
                total_in_flight -= 1

        tasks = [retain_call() for _ in range(6)] + [reflect_call() for _ in range(6)]
        await asyncio.gather(*tasks)

        assert retain_peak <= 2, f"retain cap exceeded: peak={retain_peak}"
        assert total_peak <= 3, f"global cap exceeded: peak={total_peak}"
