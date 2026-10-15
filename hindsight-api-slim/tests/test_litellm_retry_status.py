"""
LiteLLM provider retries on the HTTP status LiteLLM maps onto its exceptions (#5009).

Bedrock's transient 503 surfaces as ``litellm.ServiceUnavailableError`` whose message
names no status or retry keyword, so the old keyword match failed it on the first
attempt. The status code decides now: 408/409/429/5xx retry, other 4xx fail fast.
"""

import litellm
import pytest

from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM

_TOOLS = [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object", "properties": {}}}}]


def _bedrock_503() -> Exception:
    return litellm.ServiceUnavailableError(
        message='BedrockException - {"message":"The system encountered an unexpected error during processing. '
        'Try your request again."}',
        llm_provider="bedrock",
        model="m",
    )


def _auth_401() -> Exception:
    return litellm.AuthenticationError(message="BedrockException - bad key", llm_provider="bedrock", model="m")


def _bad_request_400() -> Exception:
    # "limit" used to match the retry keywords, so this deterministic 400 was retried.
    return litellm.BadRequestError(message="max_tokens exceeds the model limit", llm_provider="bedrock", model="m")


async def _attempts(monkeypatch: pytest.MonkeyPatch, error: Exception, *, tools: bool) -> int:
    provider = LiteLLMLLM(provider="litellm", api_key="unused", base_url="http://localhost:0/v1", model="m")
    calls = 0

    async def _fail(**kwargs):
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(provider, "_acompletion", _fail)
    messages = [{"role": "user", "content": "hi"}]
    with pytest.raises(type(error)):
        if tools:
            await provider.call_with_tools(messages=messages, tools=_TOOLS, max_retries=2, initial_backoff=0.0)
        else:
            await provider.call(messages=messages, max_retries=2, initial_backoff=0.0)
    return calls


@pytest.mark.parametrize("tools", [False, True])
@pytest.mark.parametrize(
    ("make_error", "expected_attempts"),
    [(_bedrock_503, 3), (_auth_401, 1), (_bad_request_400, 1)],
)
async def test_retry_follows_status_code(monkeypatch, make_error, expected_attempts, tools):
    assert await _attempts(monkeypatch, make_error(), tools=tools) == expected_attempts
