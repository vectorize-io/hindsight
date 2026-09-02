"""Tests for per-operation LLM concurrency caps.

These tests exercise the dispatch logic in `llm_wrapper` that gates calls on
per-operation semaphores when `HINDSIGHT_API_{RETAIN,REFLECT,CONSOLIDATION}_LLM_MAX_CONCURRENT`
is set. They patch the module-level semaphore registry so they can run without
needing to re-import the module with custom env vars.
"""

import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from openai import APIConnectionError

from hindsight_api.engine import llm_wrapper
from hindsight_api.engine.llm_wrapper import (
    LLMProvider,
    _scope_to_operation,
    _semaphores_for_scope,
)
from hindsight_api.worker.stage import StageHolder, bind_holder


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
            assert _semaphores_for_scope("mental_model_delta_ops") == [llm_wrapper._global_llm_semaphore]
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
    @pytest.mark.parametrize("with_tools", [False, True])
    async def test_unrelated_call_can_run_while_retrying_call_backs_off(self, with_tools):
        """A logical call's retry sleep must not occupy the global permit."""
        provider = LLMProvider(provider="openai", api_key="test", base_url="", model="test-model")
        entered_backoff = asyncio.Event()
        release_backoff = asyncio.Event()
        unrelated_started = asyncio.Event()

        async def retrying_mock_call(**kwargs):
            content = kwargs["messages"][0]["content"]
            attempt_context = kwargs["attempt_context"]
            async with attempt_context():
                if content != "retrying":
                    unrelated_started.set()
            if content == "retrying":
                entered_backoff.set()
                await release_backoff.wait()
            return "ok"

        method = "call_with_tools" if with_tools else "call"
        setattr(provider._provider_impl, method, retrying_mock_call)

        async def invoke(content):
            kwargs = {"messages": [{"role": "user", "content": content}], "scope": "retain"}
            if with_tools:
                kwargs["tools"] = []
                return await provider.call_with_tools(**kwargs)
            return await provider.call(**kwargs)

        with patch.object(llm_wrapper, "_global_llm_semaphore", asyncio.Semaphore(1)):
            retrying = asyncio.create_task(invoke("retrying"))
            await asyncio.wait_for(entered_backoff.wait(), timeout=1)
            unrelated = asyncio.create_task(invoke("unrelated"))
            try:
                await asyncio.wait_for(unrelated_started.wait(), timeout=0.1)
            finally:
                release_backoff.set()
                await asyncio.gather(retrying, unrelated)

    @pytest.mark.asyncio
    async def test_real_retry_loop_releases_permit_during_backoff_and_reacquires(self):
        """End-to-end through the real OpenAI-compatible retry loop: attempt 1
        fails with a retryable error and, while the provider sleeps out its
        backoff, an unrelated call takes the global permit; attempt 2 then
        reacquires it before hitting the upstream."""
        provider = LLMProvider(provider="openai", api_key="test", base_url="", model="test-model")
        global_sem = asyncio.Semaphore(1)
        events: list[str] = []
        first_attempt_failed = asyncio.Event()

        def _ok_response():
            choice = SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="ok", tool_calls=None, refusal=None),
            )
            return SimpleNamespace(error=None, usage=None, choices=[choice])

        async def create(**kwargs):
            content = kwargs["messages"][0]["content"]
            assert global_sem.locked(), "an upstream attempt must hold the global permit"
            if content == "retrying" and not first_attempt_failed.is_set():
                events.append("retrying-attempt-1")
                first_attempt_failed.set()
                raise APIConnectionError(request=httpx.Request("POST", "http://test"))
            events.append(content)
            return _ok_response()

        provider._provider_impl._client.chat.completions.create = create

        with (
            patch.object(llm_wrapper, "_per_op_llm_semaphores", {}),
            patch.object(llm_wrapper, "_global_llm_semaphore", global_sem),
        ):
            retrying = asyncio.create_task(
                provider.call(
                    messages=[{"role": "user", "content": "retrying"}],
                    scope="retain",
                    max_retries=1,
                    initial_backoff=1.0,
                    max_backoff=1.0,
                )
            )
            await asyncio.wait_for(first_attempt_failed.wait(), timeout=1)
            # While the retrying call sleeps out its ~1s backoff, an unrelated
            # call must be able to take the global permit and complete.
            await asyncio.wait_for(
                provider.call(messages=[{"role": "user", "content": "unrelated"}], scope="retain"),
                timeout=0.5,
            )
            assert await asyncio.wait_for(retrying, timeout=3) == "ok"

        assert events == ["retrying-attempt-1", "unrelated", "retrying"]

    @pytest.mark.asyncio
    async def test_cancel_while_waiting_for_global_releases_per_op_permit(self):
        """Cancelling an attempt that holds the per-op permit but is still
        queued on the global one must release the per-op permit."""
        retain_sem = asyncio.Semaphore(1)
        global_sem = asyncio.Semaphore(1)
        with (
            patch.object(llm_wrapper, "_per_op_llm_semaphores", {"retain": retain_sem}),
            patch.object(llm_wrapper, "_global_llm_semaphore", global_sem),
        ):
            await global_sem.acquire()  # an unrelated holder saturates the global cap

            async def waiter():
                async with llm_wrapper._attempt_permits("retain"):
                    pass

            task = asyncio.create_task(waiter())
            for _ in range(50):
                if retain_sem.locked():
                    break
                await asyncio.sleep(0)
            assert retain_sem.locked(), "waiter should hold the per-op permit while queued on global"

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not retain_sem.locked(), "cancellation must release the per-op permit"
            global_sem.release()

    @pytest.mark.asyncio
    async def test_stage_stays_queued_until_permit_then_marks_backoff(self):
        """Through the real wrapper flow: the stage keeps its `.queued` suffix
        while the call waits for a permit (#3002), shows `attempt=N` only while
        permits are held, and gains a `.backoff` suffix while the provider
        sleeps between attempts without permits."""
        provider = LLMProvider(provider="openai", api_key="test", base_url="", model="test-model")
        holder = StageHolder()
        global_sem = asyncio.Semaphore(1)
        first_attempt_failed = asyncio.Event()

        async def create(**kwargs):
            assert holder.stage == f"llm.openai.retain.attempt={1 if not first_attempt_failed.is_set() else 2}/2"
            if not first_attempt_failed.is_set():
                first_attempt_failed.set()
                raise APIConnectionError(request=httpx.Request("POST", "http://test"))
            choice = SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="ok", tool_calls=None, refusal=None),
            )
            return SimpleNamespace(error=None, usage=None, choices=[choice])

        provider._provider_impl._client.chat.completions.create = create

        async def invoke():
            bind_holder(holder)
            return await provider.call(
                messages=[{"role": "user", "content": "hello"}],
                scope="retain",
                max_retries=1,
                initial_backoff=0.2,
                max_backoff=0.2,
            )

        with (
            patch.object(llm_wrapper, "_per_op_llm_semaphores", {}),
            patch.object(llm_wrapper, "_global_llm_semaphore", global_sem),
        ):
            await global_sem.acquire()  # saturate so the call queues
            task = asyncio.create_task(invoke())
            for _ in range(20):
                await asyncio.sleep(0)
            assert holder.stage == "llm.openai.retain.queued", "must stay `.queued` while waiting for the permit"

            global_sem.release()
            await asyncio.wait_for(first_attempt_failed.wait(), timeout=1)
            for _ in range(20):
                await asyncio.sleep(0)
            assert holder.stage == "llm.openai.retain.attempt=1/2.backoff", (
                "backoff sleep must be visible in the stage while no permit is held"
            )
            assert await asyncio.wait_for(task, timeout=3) == "ok"

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
