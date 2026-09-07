"""Provider failure classification without credentials or network access."""

import httpx
import pytest

from hindsight_api.engine.llm_interface import LLMCooldownFailure, LLMFailureCategory, LLMTerminalFailure
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.providers.codex_auth import CodexReauthenticationRequiredError, CodexRefreshExpiredError
from hindsight_api.engine.providers.codex_llm import CodexLLM


def _http_error(status: int, retry_after: str | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.invalid/responses")
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    response = httpx.Response(status, headers=headers, request=request)
    return httpx.HTTPStatusError("provider failure", request=request, response=response)


def test_other_providers_leave_failures_unclassified() -> None:
    provider = LLMProvider(provider="mock", api_key="", base_url="", model="mock")
    assert provider.classify_failure(RuntimeError("failure")) is None


@pytest.mark.parametrize(("retry_after", "seconds"), [("12", 12.0), ("0", 0.0), ("1.5", 1.5), ("1e300", 1e300)])
def test_codex_classifies_explicit_quota(retry_after: str, seconds: float) -> None:
    provider = CodexLLM.__new__(CodexLLM)
    assert provider.classify_failure(_http_error(429, retry_after)) == LLMCooldownFailure(
        category=LLMFailureCategory.RATE_LIMIT,
        retry_after_seconds=seconds,
    )


def test_codex_parses_retry_after_http_date_through_explicit_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hindsight_api.engine.providers.codex_llm.time.time", lambda: 1788260400.0)
    error = RuntimeError("wrapped")
    error.__cause__ = _http_error(429, "Tue, 01 Sep 2026 11:01:00 GMT")
    assert CodexLLM.__new__(CodexLLM).classify_failure(error) == LLMCooldownFailure(retry_after_seconds=60.0)


@pytest.mark.parametrize("retry_after", [None, "", "garbage", "NaN", "inf", "-1"])
def test_codex_invalid_retry_after_uses_sixty_second_fallback(retry_after: str | None) -> None:
    assert CodexLLM.__new__(CodexLLM).classify_failure(_http_error(429, retry_after)) == LLMCooldownFailure(
        retry_after_seconds=60.0
    )


@pytest.mark.parametrize("status", [401, 403, 500, 503])
def test_codex_other_http_errors_are_unclassified(status: int) -> None:
    assert CodexLLM.__new__(CodexLLM).classify_failure(_http_error(status)) is None


def test_past_retry_after_date_allows_immediate_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hindsight_api.engine.providers.codex_llm.time.time", lambda: 1788260400.0)
    error = _http_error(429, "Tue, 01 Sep 2026 10:59:00 GMT")
    assert CodexLLM.__new__(CodexLLM).classify_failure(error) == LLMCooldownFailure(retry_after_seconds=0.0)


def test_only_positive_terminal_subtype_is_classified_and_cause_walk_is_bounded() -> None:
    provider = CodexLLM.__new__(CodexLLM)
    confirmed = RuntimeError("outer")
    confirmed.__cause__ = CodexReauthenticationRequiredError("confirmed")
    assert provider.classify_failure(confirmed) == LLMTerminalFailure()
    assert provider.classify_failure(CodexRefreshExpiredError("unrecognized refresh response")) is None

    cyclic = RuntimeError("cyclic")
    cyclic.__cause__ = cyclic
    assert provider.classify_failure(cyclic) is None
    assert provider.classify_failure(httpx.ConnectError("network unavailable")) is None


def test_incidental_context_does_not_terminal_classify() -> None:
    error = RuntimeError("active failure")
    error.__context__ = CodexReauthenticationRequiredError("incidental context")
    assert CodexLLM.__new__(CodexLLM).classify_failure(error) is None
