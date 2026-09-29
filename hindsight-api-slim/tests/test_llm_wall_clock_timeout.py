"""An upstream that trickles bytes must not hang an LLM call (#4763).

The SDK/aiohttp timeouts are per phase, and the read timeout restarts on every
byte, so a server sending keep-alive whitespace before the body resets it
forever. The call held its worker slot indefinitely with no error, stalling the
retain queue. Every provider must bound the call by the configured timeout on
the wall clock. (xai-oauth has its own case in test_xai_oauth_llm.py: it needs
a token store to reach the request.)
"""

import asyncio
import time
from collections.abc import Awaitable

import pytest
from aiohttp import web
from pydantic import BaseModel

from hindsight_api.engine.llm_interface import LLMInterface
from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM
from hindsight_api.engine.providers.openai_responses_llm import OpenAIResponsesLLM
from tests.aiohttp_stub import stub_server

TIMEOUT = 0.5
MESSAGES = [{"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object", "properties": {}}}}]


class _Ok(BaseModel):
    ok: bool


async def trickle(request: web.Request) -> web.StreamResponse:
    """Send headers, then one space every 50 ms, forever."""
    response = web.StreamResponse(headers={"Content-Type": "application/json"})
    await response.prepare(request)
    while True:
        await response.write(b" ")
        await asyncio.sleep(0.05)


def _openai(url: str) -> LLMInterface:
    return OpenAICompatibleLLM(
        provider="openai", api_key="k", base_url=f"{url}/v1", model="gpt-4o-mini", timeout=TIMEOUT
    )


def _ollama(url: str) -> LLMInterface:
    return OpenAICompatibleLLM(provider="ollama", api_key="local", base_url=f"{url}/v1", model="qwen", timeout=TIMEOUT)


def _responses(url: str) -> LLMInterface:
    return OpenAIResponsesLLM(
        provider="openai-responses", api_key="k", base_url=f"{url}/v1", model="gpt-5", timeout=TIMEOUT
    )


def _anthropic(url: str) -> LLMInterface:
    return AnthropicLLM(provider="anthropic", api_key="k", base_url=url, model="claude-sonnet-5", timeout=TIMEOUT)


def _call(llm: LLMInterface) -> Awaitable[object]:
    return llm.call(messages=MESSAGES, max_retries=0)


def _structured(llm: LLMInterface) -> Awaitable[object]:
    return llm.call(messages=MESSAGES, response_format=_Ok, max_retries=0)


def _tools(llm: LLMInterface) -> Awaitable[object]:
    return llm.call_with_tools(messages=MESSAGES, tools=TOOLS, max_retries=0)


CASES = {
    "openai-call": (_openai, _call),
    "openai-call-structured": (_openai, _structured),
    "openai-tools": (_openai, _tools),
    "ollama-native": (_ollama, _structured),
    "openai-responses": (_responses, _call),
    "anthropic-call": (_anthropic, _call),
    "anthropic-tools": (_anthropic, _tools),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES)
async def test_trickling_upstream_times_out_on_the_wall_clock(case: str):
    make_llm, invoke = CASES[case]
    async with stub_server(trickle) as base_url:
        llm = make_llm(base_url)
        start = time.monotonic()
        try:
            # The outer wait_for is only the test's own safety net: before the fix
            # the call never returned, and it is what fails the test then.
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(invoke(llm), timeout=10)
            assert time.monotonic() - start < 5, "the outer safety net fired, not the provider's own deadline"
        finally:
            await llm.cleanup()
