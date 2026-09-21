"""Reflect's per-call LLM deadline follows the prompt (issue #4568).

``DEFAULT_REFLECT_LLM_TIMEOUT`` (30 s) was chosen against 1-4 s tool-calling turns.
Reflect's final synthesis is not one of those: its prompt is the accumulated tool
results -- 30-90k tokens on a real bank -- and a healthy provider took 5-48 s over
it, so the flat deadline killed healthy calls and retried the same prompt into the
same wall. The fix keeps the 30 s base for small prompts and lengthens the deadline
per call, as a *floor* the providers apply on top of their configured timeout:

* the agent computes the floor from the prompt size (``scaled_reflect_deadline``),
  only when the operator left the deadline at its default;
* the agent arms it around each call with ``request_timeout_floor(...)``, a
  ContextVar the providers read, so no provider signature changes;
* each provider arms ``effective_request_timeout(configured)`` -- never less than
  what the operator configured;
* a deadline that still expires surfaces as ``ReflectLLMDeadlineError`` -> HTTP 504
  naming the knob, instead of a bare 500.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import httpx
import pytest

from hindsight_api.config import (
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_REFLECT_LLM_TIMEOUT,
    DEFAULT_REFLECT_LLM_TIMEOUT_PER_1K_PROMPT_TOKENS,
    ENV_LLM_TIMEOUT,
    ENV_REFLECT_LLM_TIMEOUT,
    _reflect_llm_timeout_is_adaptive,
    clear_config_cache,
    get_config,
)
from hindsight_api.engine.llm_transport import (
    RequestDeadlineExceeded,
    _request_timeout_floor_ctx,
    effective_request_timeout,
    is_request_deadline_error,
    request_timeout_floor,
)
from hindsight_api.engine.llm_wrapper import LLMConfig
from hindsight_api.engine.reflect import ReflectLLMDeadlineError
from hindsight_api.engine.reflect.agent import _reflect_timeout_floor, scaled_reflect_deadline

# ---------------------------------------------------------------------------
# The scaling rule
# ---------------------------------------------------------------------------


class TestScaledReflectDeadline:
    def test_a_small_tool_turn_keeps_the_base(self):
        assert scaled_reflect_deadline(0, base=30.0, per_1k_tokens=1.0, ceiling=120.0) == 30.0
        assert scaled_reflect_deadline(2_000, base=30.0, per_1k_tokens=1.0, ceiling=120.0) == 32.0

    def test_a_large_synthesis_prompt_gets_proportionally_more(self):
        """The two real synthesis calls from #4568: 63k tokens took 19 s, 88k took 38 s."""
        assert scaled_reflect_deadline(60_000, base=30.0, per_1k_tokens=1.0, ceiling=120.0) == 90.0
        assert scaled_reflect_deadline(88_000, base=30.0, per_1k_tokens=1.0, ceiling=120.0) == 118.0

    def test_the_global_deadline_is_the_ceiling(self):
        assert scaled_reflect_deadline(500_000, base=30.0, per_1k_tokens=1.0, ceiling=120.0) == 120.0

    def test_a_ceiling_below_the_base_does_not_invert_them(self):
        """A global deadline set under 30 s must not hand a reflect call less than its base."""
        assert scaled_reflect_deadline(80_000, base=30.0, per_1k_tokens=1.0, ceiling=10.0) == 30.0

    def test_negative_token_estimates_are_treated_as_zero(self):
        assert scaled_reflect_deadline(-5, base=30.0, per_1k_tokens=1.0, ceiling=120.0) == 30.0

    def test_shipped_defaults_cover_the_observed_healthy_calls(self):
        """Guard the constants against the evidence: 2-3x headroom over the measured prefill."""
        for tokens, observed_seconds in ((62_711, 18.5), (88_373, 37.7)):
            deadline = scaled_reflect_deadline(
                tokens,
                base=DEFAULT_REFLECT_LLM_TIMEOUT,
                per_1k_tokens=DEFAULT_REFLECT_LLM_TIMEOUT_PER_1K_PROMPT_TOKENS,
                ceiling=DEFAULT_LLM_TIMEOUT,
            )
            assert deadline >= 2 * observed_seconds, (tokens, deadline)


class TestReflectTimeoutFloor:
    """``_reflect_timeout_floor`` applies the rule only while the deadline is the default."""

    @staticmethod
    def _config(adaptive: bool, llm_timeout: float = 120.0) -> MagicMock:
        return MagicMock(reflect_llm_timeout_adaptive=adaptive, llm_timeout=llm_timeout)

    def test_none_when_the_operator_fixed_the_deadline(self):
        with patch("hindsight_api.engine.reflect.agent.get_config", return_value=self._config(adaptive=False)):
            assert _reflect_timeout_floor(90_000) is None

    def test_scaled_when_the_deadline_is_the_default(self):
        with patch("hindsight_api.engine.reflect.agent.get_config", return_value=self._config(adaptive=True)):
            assert _reflect_timeout_floor(0) == DEFAULT_REFLECT_LLM_TIMEOUT
            assert _reflect_timeout_floor(60_000) == DEFAULT_REFLECT_LLM_TIMEOUT + 60.0

    def test_capped_at_the_configured_global_deadline(self):
        with patch("hindsight_api.engine.reflect.agent.get_config", return_value=self._config(True, llm_timeout=75.0)):
            assert _reflect_timeout_floor(90_000) == 75.0


# ---------------------------------------------------------------------------
# Config: when is the deadline adaptive?
# ---------------------------------------------------------------------------


class TestAdaptiveFlagResolution:
    @pytest.fixture(autouse=True)
    def _fresh_config(self):
        clear_config_cache()
        yield
        clear_config_cache()

    def test_adaptive_only_when_neither_deadline_is_set(self, monkeypatch):
        monkeypatch.delenv(ENV_REFLECT_LLM_TIMEOUT, raising=False)
        monkeypatch.delenv(ENV_LLM_TIMEOUT, raising=False)
        assert _reflect_llm_timeout_is_adaptive() is True
        assert get_config().reflect_llm_timeout_adaptive is True
        assert get_config().reflect_llm_timeout == DEFAULT_REFLECT_LLM_TIMEOUT

    def test_an_explicit_reflect_deadline_is_fixed(self, monkeypatch):
        monkeypatch.delenv(ENV_LLM_TIMEOUT, raising=False)
        monkeypatch.setenv(ENV_REFLECT_LLM_TIMEOUT, "180")
        assert _reflect_llm_timeout_is_adaptive() is False
        assert get_config().reflect_llm_timeout_adaptive is False
        assert get_config().reflect_llm_timeout == 180.0

    def test_an_explicit_global_deadline_is_fixed_too(self, monkeypatch):
        """The operator chose a global deadline deliberately; reflect inherits it as is (#3982)."""
        monkeypatch.delenv(ENV_REFLECT_LLM_TIMEOUT, raising=False)
        monkeypatch.setenv(ENV_LLM_TIMEOUT, "300")
        assert _reflect_llm_timeout_is_adaptive() is False
        assert get_config().reflect_llm_timeout_adaptive is False
        assert get_config().reflect_llm_timeout is None  # inherit llm_timeout


# ---------------------------------------------------------------------------
# Transport helpers
# ---------------------------------------------------------------------------


class TestEffectiveRequestTimeout:
    def test_no_floor_returns_the_configured_value(self):
        assert _request_timeout_floor_ctx.get() is None
        assert effective_request_timeout(30.0) == 30.0

    def test_a_floor_above_the_configured_value_lengthens_it_inside_the_block_only(self):
        with request_timeout_floor(90.0):
            assert effective_request_timeout(30.0) == 90.0
        assert effective_request_timeout(30.0) == 30.0

    def test_a_floor_below_the_configured_value_never_shortens_it(self):
        """An operator's explicit 2700 s refresh deadline must survive a 90 s floor."""
        with request_timeout_floor(90.0):
            assert effective_request_timeout(2700.0) == 2700.0

    def test_none_arms_nothing_and_shadows_an_outer_floor(self):
        with request_timeout_floor(90.0):
            with request_timeout_floor(None):
                assert effective_request_timeout(30.0) == 30.0
            assert effective_request_timeout(30.0) == 90.0

    def test_the_floor_is_cleared_when_the_block_raises(self):
        with pytest.raises(RuntimeError):
            with request_timeout_floor(90.0):
                raise RuntimeError("boom")
        assert _request_timeout_floor_ctx.get() is None


class TestIsRequestDeadlineError:
    def test_bare_timeouts(self):
        assert is_request_deadline_error(TimeoutError())
        assert is_request_deadline_error(asyncio.TimeoutError())

    def test_sdk_error_raised_from_an_httpx_timeout(self):
        try:
            try:
                raise httpx.ReadTimeout("read")
            except httpx.ReadTimeout as inner:
                raise RuntimeError("Request timed out.") from inner
        except RuntimeError as outer:
            assert is_request_deadline_error(outer)

    def test_marker_mixin(self):
        class Marked(RequestDeadlineExceeded, ValueError):
            pass

        assert is_request_deadline_error(Marked("x"))

    def test_codex_runaway_stream_error_is_recognised(self):
        from hindsight_api.engine.providers.codex_llm import CodexRunawayStreamError

        assert is_request_deadline_error(CodexRunawayStreamError("runaway"))

    def test_unrelated_errors_are_not(self):
        assert not is_request_deadline_error(RuntimeError("provider is down"))
        assert not is_request_deadline_error(ValueError("bad json"))

    def test_only_the_cause_chain_counts_not_the_context(self):
        """A timeout that merely happened to be in flight must not relabel a different error."""
        try:
            try:
                raise TimeoutError()
            except TimeoutError:
                raise ValueError("unrelated")  # implicit __context__, no ``from``
        except ValueError as e:
            assert not is_request_deadline_error(e)

    def test_a_cyclic_cause_chain_terminates(self):
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__cause__ = b
        b.__cause__ = a
        assert not is_request_deadline_error(a)


# ---------------------------------------------------------------------------
# Providers arm the effective deadline
# ---------------------------------------------------------------------------


@pytest.fixture
def floor():
    """Arm a 90 s floor for the test body."""
    with request_timeout_floor(90.0):
        yield 90.0


class TestOpenAICompatibleProvider:
    @staticmethod
    def _impl(timeout: float | None = 30.0):
        impl = LLMConfig(provider="openai", api_key="k", base_url="", model="m", timeout=timeout)._provider_impl
        impl._client = MagicMock()
        return impl

    def test_without_a_floor_the_configured_client_is_used(self):
        impl = self._impl()
        assert impl._client_for_call() is impl._client
        impl._client.with_options.assert_not_called()

    def test_a_floor_above_the_configured_timeout_lengthens_this_call_only(self, floor):
        impl = self._impl()
        impl._client_for_call()
        armed = impl._client.with_options.call_args.kwargs["timeout"]
        assert isinstance(armed, httpx.Timeout)
        assert armed.read == floor

    def test_a_floor_below_the_configured_timeout_changes_nothing(self, floor):
        impl = self._impl(timeout=300.0)
        assert impl._client_for_call() is impl._client


class TestAnthropicProvider:
    @staticmethod
    def _impl(timeout: float | None = 30.0):
        pytest.importorskip("anthropic")
        impl = LLMConfig(provider="anthropic", api_key="k", base_url="", model="m", timeout=timeout)._provider_impl
        impl._client = MagicMock()
        return impl

    def test_without_a_floor_the_configured_client_is_used(self):
        impl = self._impl()
        assert impl._client_for_call() is impl._client

    def test_a_floor_above_the_configured_timeout_lengthens_this_call_only(self, floor):
        impl = self._impl()
        impl._client_for_call()
        assert impl._client.with_options.call_args.kwargs["timeout"].read == floor

    def test_a_floor_below_the_configured_timeout_changes_nothing(self, floor):
        impl = self._impl(timeout=300.0)
        assert impl._client_for_call() is impl._client


class TestLiteLLMProvider:
    @staticmethod
    def _impl():
        pytest.importorskip("litellm")
        return LLMConfig(provider="litellm", api_key="k", base_url="", model="m", timeout=30.0)._provider_impl

    def test_request_kwargs_carry_the_configured_timeout(self):
        assert self._impl()._build_common_kwargs(messages=[])["timeout"] == 30.0

    def test_request_kwargs_carry_the_floor_when_longer(self, floor):
        assert self._impl()._build_common_kwargs(messages=[])["timeout"] == floor


class TestGeminiProvider:
    """The floor reaches Gemini's ``asyncio.wait_for`` deadline: a slow answer survives it."""

    @pytest.mark.asyncio
    async def test_a_slow_healthy_answer_is_cut_without_the_floor_and_kept_with_it(self):
        pytest.importorskip("google.genai")
        from tests.test_gemini_deadline_retry import _generate_content, _make_gemini_provider, _text_response

        async def _slow_answer():
            await asyncio.sleep(0.3)
            return _text_response("done")

        provider = _make_gemini_provider(timeout=0.05)
        provider._client.aio.models.generate_content = _generate_content(_slow_answer)
        with pytest.raises(TimeoutError):
            await provider.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

        provider._client.aio.models.generate_content = _generate_content(_slow_answer)
        with request_timeout_floor(5.0):
            result = await provider.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)
        assert result.content == "done"


class TestCodexProvider:
    """The floor lengthens Codex's wall deadline and its per-phase socket timeouts together."""

    @staticmethod
    async def _delayed_sse_server(delay: float, body: bytes):
        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            while await reader.readline() not in (b"\r\n", b"\n", b""):
                pass
            # Silent for ``delay`` seconds -- a backend prefilling a large prompt -- then
            # the whole (short) answer at once.
            await asyncio.sleep(delay)
            try:
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                    + b"%x\r\n" % len(body)
                    + body
                    + b"\r\n0\r\n\r\n"
                )
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError):
                return
            finally:
                writer.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        return server, f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"

    @pytest.mark.asyncio
    async def test_a_silent_prefill_longer_than_the_configured_deadline_survives_under_the_floor(self):
        from tests.test_codex_stream_deadline import _build_llm, _delta_chunk

        body = _delta_chunk("hello ") + _delta_chunk("world") + b"data: [DONE]\n\n"
        server, base_url = await self._delayed_sse_server(delay=1.2, body=body)
        llm = _build_llm(base_url, timeout=0.4)
        try:
            with pytest.raises(Exception) as excinfo:
                await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)
            assert is_request_deadline_error(excinfo.value), repr(excinfo.value)

            with request_timeout_floor(10.0):
                result = await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)
            assert result.content == "hello world"
        finally:
            await llm.cleanup()
            server.close()


# ---------------------------------------------------------------------------
# HTTP: the failure names the knob and is a 504, not a bare 500
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reflect_deadline_error_is_a_504_naming_the_setting(api_client, memory, monkeypatch):
    message = (
        "An LLM call inside reflect exceeded its per-request deadline, retries included: "
        "TimeoutError. HINDSIGHT_API_REFLECT_LLM_TIMEOUT gives every reflect call one fixed, larger deadline instead."
    )

    async def _deadline(*args, **kwargs):
        raise ReflectLLMDeadlineError(message) from TimeoutError()

    monkeypatch.setattr(memory, "reflect_async", _deadline)
    response = await api_client.post("/v1/default/banks/deadline-bank/reflect", json={"query": "where do we stand?"})
    assert response.status_code == 504, response.text
    assert response.json()["detail"] == message
