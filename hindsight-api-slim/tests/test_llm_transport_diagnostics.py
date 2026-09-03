"""Transport-level diagnostics for stalled LLM requests (issue #3881).

A reflect call that stalls for exactly ``llm_timeout`` and raises
``APIConnectionError: Request timed out`` says nothing about *which* httpx phase
stalled -- connect, pool, write and read are four different faults with four
different owners, and the SDK stringifies all of them identically. These tests
cover the three pieces that make that distinction visible:

* the connect phase is capped independently of the total request budget,
* the underlying httpx/httpcore exception chain reaches the log line,
* time spent queued behind a concurrency permit is reported apart from the time
  the request was actually in flight.
"""

import asyncio
import logging
from unittest.mock import patch

import httpx
import pytest

from hindsight_api.config import DEFAULT_LLM_CONNECT_TIMEOUT, ENV_LLM_CONNECT_TIMEOUT, ENV_LLM_HTTP_LOG_LEVEL
from hindsight_api.engine.llm_trace import (
    LLMQueueWait,
    record_queue_wait,
    reset_queue_wait_sink,
    set_queue_wait_sink,
)
from hindsight_api.engine.llm_transport import (
    build_sdk_timeout,
    configure_http_logging,
    describe_llm_error,
    describe_transport_error,
)


# How long the test holds the only permit shut. Long enough that scheduling jitter is
# a rounding error against the assertion floor below, short enough to stay a unit test.
_PERMIT_HOLD_SECONDS = 0.2


def _api_timeout_wrapping(cause: Exception):
    """An ``openai.APITimeoutError`` raised from ``cause``, as the SDK builds it."""
    from openai import APITimeoutError

    request = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    err = APITimeoutError(request=request)
    err.__cause__ = cause
    return err


# -- connect-phase cap --------------------------------------------------------


def test_connect_phase_capped_below_request_timeout(monkeypatch):
    """A 150s request budget must not become a 150s connect timeout."""
    monkeypatch.delenv(ENV_LLM_CONNECT_TIMEOUT, raising=False)
    timeout = build_sdk_timeout(150.0)

    assert timeout.connect == DEFAULT_LLM_CONNECT_TIMEOUT
    # Everything else keeps the full budget: a slow model is not a broken endpoint.
    assert timeout.read == 150.0
    assert timeout.write == 150.0
    assert timeout.pool == 150.0


def test_connect_cap_never_exceeds_the_request_timeout(monkeypatch):
    """A request budget below the cap wins -- no phase may outlive the request."""
    monkeypatch.delenv(ENV_LLM_CONNECT_TIMEOUT, raising=False)
    assert build_sdk_timeout(3.0).connect == 3.0


def test_connect_cap_configurable(monkeypatch):
    monkeypatch.setenv(ENV_LLM_CONNECT_TIMEOUT, "45")
    assert build_sdk_timeout(150.0).connect == 45.0


def test_connect_cap_disabled_with_zero(monkeypatch):
    """0 restores the pre-#3881 behaviour: one value across all four phases."""
    monkeypatch.setenv(ENV_LLM_CONNECT_TIMEOUT, "0")
    timeout = build_sdk_timeout(150.0)
    assert timeout.connect == 150.0
    assert timeout.read == 150.0


def test_openai_compatible_client_gets_a_capped_connect_phase(monkeypatch):
    """The cap reaches the real client, not just the helper."""
    monkeypatch.delenv(ENV_LLM_CONNECT_TIMEOUT, raising=False)
    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    provider = OpenAICompatibleLLM(provider="openai", api_key="k", base_url="", model="m", timeout=150.0)

    timeout = provider._client.timeout
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == DEFAULT_LLM_CONNECT_TIMEOUT
    assert timeout.read == 150.0


def test_every_provider_that_owns_an_httpx_client_caps_its_connect_phase(monkeypatch):
    """Parity guard: providers that build their own client, not just the SDK-backed ones.

    codex and xai-oauth construct ``httpx.AsyncClient`` directly instead of going through
    an SDK, so they were the two that carried the bare-float defect after the first pass
    at #3881. Enumerated here rather than tested one by one, so a provider added later
    with its own client fails this instead of silently inheriting the whole budget.
    """
    monkeypatch.delenv(ENV_LLM_CONNECT_TIMEOUT, raising=False)

    from hindsight_api.engine.providers.codex_llm import CodexLLM
    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM
    from hindsight_api.engine.providers.openai_responses_llm import OpenAIResponsesLLM

    # Codex reads OAuth credentials from disk at construction; stub them the way the
    # rest of the codex suite does so this runs on a machine that has never logged in.
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        codex = CodexLLM(provider="openai-codex", api_key="k", base_url="", model="m", timeout=150.0)

    providers = [
        OpenAICompatibleLLM(provider="openai", api_key="k", base_url="", model="m", timeout=150.0),
        OpenAIResponsesLLM(provider="openai-responses", api_key="k", base_url="", model="m", timeout=150.0),
        codex,
    ]

    for provider in providers:
        timeout = provider._client.timeout
        label = type(provider).__name__
        assert isinstance(timeout, httpx.Timeout), f"{label} passed a bare float"
        assert timeout.connect == DEFAULT_LLM_CONNECT_TIMEOUT, f"{label} left connect uncapped"
        assert timeout.read == 150.0, f"{label} lost the request budget"


# -- transport cause ----------------------------------------------------------


def test_describe_transport_error_names_the_stalled_phase():
    """The whole point: ReadTimeout and ConnectTimeout must not read alike."""
    read_timed_out = _api_timeout_wrapping(httpx.ReadTimeout("timed out"))
    connect_timed_out = _api_timeout_wrapping(httpx.ConnectTimeout("timed out"))

    assert "httpx.ReadTimeout" in describe_transport_error(read_timed_out)
    assert "httpx.ConnectTimeout" in describe_transport_error(connect_timed_out)
    # ...even though the SDK renders both identically.
    assert str(read_timed_out) == str(connect_timed_out)


def test_describe_transport_error_renders_the_wrapping_chain():
    """httpx wraps httpcore; both links are worth seeing."""
    inner = RuntimeError("connection closed")
    outer = httpx.ReadTimeout("timed out")
    outer.__cause__ = inner
    described = describe_transport_error(_api_timeout_wrapping(outer))

    assert described.startswith("httpx.ReadTimeout <- RuntimeError")
    assert "connection closed" in described


def test_describe_transport_error_without_a_cause():
    assert describe_transport_error(RuntimeError("bare")) == "<no cause>"


def test_describe_transport_error_survives_a_cycle():
    """A self-referencing chain must terminate, not hang the error path."""
    first = RuntimeError("first")
    second = RuntimeError("second")
    first.__cause__ = second
    second.__cause__ = first

    # Terminates at the first already-seen link rather than walking forever.
    assert describe_transport_error(first) == "RuntimeError: second"


def test_describe_transport_error_survives_an_exploding_str():
    class Exploding(Exception):
        def __str__(self):
            raise ValueError("nope")

    err = RuntimeError("outer")
    err.__cause__ = Exploding()

    assert describe_transport_error(err).endswith("Exploding")


def test_describe_llm_error_names_an_exception_that_stringifies_to_nothing():
    """The reason this exists: a bare TimeoutError logs as an empty message.

    ``[REFLECT ...] LLM error on iteration 2:  (120002ms)`` names neither the failure
    nor the provider, and it is what every non-httpx provider produced on a stall.
    """
    empty = asyncio.TimeoutError()
    assert str(empty) == ""
    # asyncio.TimeoutError is builtins.TimeoutError on 3.11+, and builtins is stripped.
    assert describe_llm_error(empty) == "TimeoutError"


def test_describe_llm_error_includes_message_and_cause():
    err = _api_timeout_wrapping(httpx.ReadTimeout("timed out"))
    described = describe_llm_error(err)

    assert described.startswith("openai.APITimeoutError: Request timed out.")
    assert "[httpx.ReadTimeout" in described


def test_describe_llm_error_omits_the_bracket_without_a_cause():
    assert describe_llm_error(RuntimeError("boom")) == "RuntimeError: boom"


# -- http logging -------------------------------------------------------------


def test_http_logging_level_is_configurable(monkeypatch):
    """httpcore at DEBUG is the instrument that names the stalled phase."""
    monkeypatch.setenv(ENV_LLM_HTTP_LOG_LEVEL, "DEBUG")
    try:
        configure_http_logging()
        assert logging.getLogger("httpx").level == logging.DEBUG
        assert logging.getLogger("httpcore").level == logging.DEBUG
    finally:
        monkeypatch.delenv(ENV_LLM_HTTP_LOG_LEVEL, raising=False)
        configure_http_logging()

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


def test_unparseable_http_log_level_falls_back(monkeypatch):
    monkeypatch.setenv(ENV_LLM_HTTP_LOG_LEVEL, "chatty")
    try:
        configure_http_logging()
        assert logging.getLogger("httpx").level == logging.WARNING
    finally:
        monkeypatch.delenv(ENV_LLM_HTTP_LOG_LEVEL, raising=False)
        configure_http_logging()


# -- permit wait accounting ---------------------------------------------------


def test_queue_wait_is_a_noop_without_a_sink():
    """Callers that don't ask must not pay, and must not blow up."""
    record_queue_wait(1.5)  # no sink bound


def test_queue_wait_accumulates_across_attempts():
    sink = LLMQueueWait()
    token = set_queue_wait_sink(sink)
    try:
        record_queue_wait(0.25)
        record_queue_wait(0.75)
    finally:
        reset_queue_wait_sink(token)

    assert sink.seconds == 1.0
    # Unbound again: a later call must not leak into the finished sink.
    record_queue_wait(9.0)
    assert sink.seconds == 1.0


@pytest.mark.asyncio
async def test_permit_wait_is_reported_separately_from_request_time(monkeypatch):
    """A call queued behind a saturated semaphore must not look like a slow provider.

    This is the #3881 ambiguity: with the wait folded into the measured duration,
    ``agent_N=Xms`` could mean "the provider took X" or "we waited X for a permit".
    """
    from hindsight_api.engine import llm_wrapper

    saturated = asyncio.Semaphore(1)
    monkeypatch.setattr(llm_wrapper, "_global_llm_semaphore", saturated)
    monkeypatch.setattr(llm_wrapper, "_per_op_llm_semaphores", {})

    await saturated.acquire()

    sink = LLMQueueWait()
    token = set_queue_wait_sink(sink)
    try:
        llm = llm_wrapper.LLMConfig(provider="mock", api_key="", base_url="", model="m")

        async def fake_call(**kwargs):
            return "ok"

        monkeypatch.setattr(llm._provider_impl, "call", fake_call)

        call = asyncio.create_task(llm.call(messages=[{"role": "user", "content": "hi"}], scope="reflect"))
        await asyncio.sleep(_PERMIT_HOLD_SECONDS)
        assert not call.done(), "call should be queued behind the permit"
        saturated.release()
        assert await call == "ok"
    finally:
        reset_queue_wait_sink(token)

    # Floor sits below the hold: the wait is only timed once the task reaches the
    # acquire, which is a scheduling hop after create_task. Asserting the full hold
    # loses that hop and fails by a fraction of a millisecond under load.
    assert sink.seconds >= _PERMIT_HOLD_SECONDS * 0.75
