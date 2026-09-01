"""Failure classification through the public provider interface, without live auth."""

import httpx
import pytest

from hindsight_api.engine.llm_interface import LLMCooldownFailure, LLMFailureCategory, LLMTerminalFailure
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.providers.codex_auth import CodexReauthenticationRequiredError, CodexRefreshExpiredError
from hindsight_api.engine.providers.codex_llm import CodexLLM


def test_other_providers_leave_failures_unclassified() -> None:
    provider = LLMProvider(provider="mock", api_key="", base_url="", model="mock")
    assert provider.classify_failure(RuntimeError("failure")) is None


@pytest.mark.parametrize("retry_after", ["12", "0", "1.5"])
def test_codex_classifies_explicit_quota(retry_after: str) -> None:
    # Classification needs no credentials or network; construct only the interface.
    provider = CodexLLM.__new__(CodexLLM)
    request = httpx.Request("POST", "https://example.invalid/responses")
    response = httpx.Response(429, headers={"Retry-After": retry_after}, request=request)
    error = httpx.HTTPStatusError("quota", request=request, response=response)
    assert provider.classify_failure(error) == LLMCooldownFailure(
        category=LLMFailureCategory.RATE_LIMIT, retry_after_seconds=float(retry_after)
    )


def test_codex_retry_after_http_date_and_wrapped_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hindsight_api.engine.providers.codex_llm.time.time", lambda: 1788260400.0)
    request = httpx.Request("POST", "https://example.invalid/responses")
    response = httpx.Response(429, headers={"Retry-After": "Tue, 01 Sep 2026 11:01:00 GMT"}, request=request)
    error = RuntimeError("wrapped")
    error.__cause__ = httpx.HTTPStatusError("quota", request=request, response=response)
    assert CodexLLM.__new__(CodexLLM).classify_failure(error) == LLMCooldownFailure(retry_after_seconds=60)


@pytest.mark.parametrize("retry_after", ["", "garbage", "NaN", "inf", "-1"])
def test_codex_invalid_retry_after_uses_default(retry_after: str) -> None:
    request = httpx.Request("POST", "https://example.invalid/responses")
    error = httpx.HTTPStatusError(
        "quota", request=request, response=httpx.Response(429, headers={"Retry-After": retry_after}, request=request)
    )
    assert CodexLLM.__new__(CodexLLM).classify_failure(error) == LLMCooldownFailure()


@pytest.mark.parametrize("status", [401, 403, 500, 503])
def test_codex_other_http_errors_are_unclassified(status: int) -> None:
    request = httpx.Request("POST", "https://example.invalid/responses")
    error = httpx.HTTPStatusError("failure", request=request, response=httpx.Response(status, request=request))
    assert CodexLLM.__new__(CodexLLM).classify_failure(error) is None


def test_past_retry_after_date_allows_immediate_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hindsight_api.engine.providers.codex_llm.time.time", lambda: 1788260400.0)
    request = httpx.Request("POST", "https://example.invalid/responses")
    error = httpx.HTTPStatusError(
        "quota",
        request=request,
        response=httpx.Response(429, headers={"Retry-After": "Tue, 01 Sep 2026 10:59:00 GMT"}, request=request),
    )
    assert CodexLLM.__new__(CodexLLM).classify_failure(error) == LLMCooldownFailure(retry_after_seconds=0)


def test_only_positive_terminal_type_is_classified_and_cycles_are_bounded() -> None:
    provider = CodexLLM.__new__(CodexLLM)
    assert provider.classify_failure(CodexReauthenticationRequiredError("confirmed")) == LLMTerminalFailure()
    assert provider.classify_failure(CodexRefreshExpiredError("unknown refresh 401")) is None
    error = RuntimeError("cyclic")
    error.__cause__ = error
    assert provider.classify_failure(error) is None
    assert provider.classify_failure(httpx.ConnectError("network unavailable")) is None
