"""
Regression test: LiteLLM errors are retried by HTTP status, not only by message text.

Bedrock's transient ``ServiceUnavailableError`` carries 503 on ``status_code``, but its
message ("The system encountered an unexpected error during processing. Try your
request again.") contains none of the retry keywords, so ``call`` and
``call_with_tools`` used to raise it on the first attempt.
"""

import litellm
import pytest

from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM

_BEDROCK_TRANSIENT = (
    'BedrockException - {"message":"The system encountered an unexpected error during processing. '
    'Try your request again."}'
)


def _make_provider() -> LiteLLMLLM:
    return LiteLLMLLM(
        provider="litellm",
        api_key="unused",
        base_url="http://localhost:0/v1",
        model="litellm_proxy/test-model",
        timeout=5,
    )


def _raising(monkeypatch, provider: LiteLLMLLM, error: Exception) -> list[int]:
    """Make every completion attempt raise ``error``; return the attempt counter."""
    attempts: list[int] = []

    async def _fail(**kwargs):
        attempts.append(1)
        raise error

    monkeypatch.setattr(provider, "_acompletion", _fail)
    return attempts


def _service_unavailable() -> litellm.ServiceUnavailableError:
    return litellm.ServiceUnavailableError(message=_BEDROCK_TRANSIENT, llm_provider="bedrock", model="test-model")


async def test_call_retries_service_unavailable_without_status_in_message(monkeypatch):
    provider = _make_provider()
    attempts = _raising(monkeypatch, provider, _service_unavailable())

    with pytest.raises(litellm.ServiceUnavailableError):
        await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=2,
            initial_backoff=0.001,
            max_backoff=0.001,
        )

    assert len(attempts) == 3


async def test_call_with_tools_retries_service_unavailable_without_status_in_message(monkeypatch):
    provider = _make_provider()
    attempts = _raising(monkeypatch, provider, _service_unavailable())

    with pytest.raises(litellm.ServiceUnavailableError):
        await provider.call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            max_retries=2,
            initial_backoff=0.001,
            max_backoff=0.001,
        )

    assert len(attempts) == 3


async def test_call_does_not_retry_client_error(monkeypatch):
    provider = _make_provider()
    attempts = _raising(
        monkeypatch,
        provider,
        litellm.BadRequestError(message="invalid request", model="test-model", llm_provider="bedrock"),
    )

    with pytest.raises(litellm.BadRequestError):
        await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=2,
            initial_backoff=0.001,
            max_backoff=0.001,
        )

    assert len(attempts) == 1
