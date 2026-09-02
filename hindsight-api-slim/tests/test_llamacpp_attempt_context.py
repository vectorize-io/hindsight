"""Regression coverage for llama.cpp's OpenAI-compatible delegation boundary."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.providers.llamacpp_llm import LlamaCppLLM


@pytest.mark.asyncio
@pytest.mark.parametrize("with_tools", [False, True])
async def test_llamacpp_forwards_attempt_context_to_delegate(with_tools):
    provider = LlamaCppLLM(
        provider="llamacpp",
        api_key="",
        base_url="",
        model="test-model",
    )
    provider._initialized = True
    provider._delegate = AsyncMock()

    @asynccontextmanager
    async def attempt_context():
        yield

    if with_tools:
        await provider.call_with_tools(
            messages=[{"role": "user", "content": "test"}],
            tools=[],
            attempt_context=attempt_context,
        )
        forwarded = provider._delegate.call_with_tools.await_args.kwargs
    else:
        await provider.call(
            messages=[{"role": "user", "content": "test"}],
            attempt_context=attempt_context,
        )
        forwarded = provider._delegate.call.await_args.kwargs

    assert provider.supports_attempt_scoped_concurrency()
    assert forwarded["attempt_context"] is attempt_context
