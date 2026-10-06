"""OpenRouter structured-output calls pin routing to upstreams that honour response_format (#3494)."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


class _Out(BaseModel):
    answer: str


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        error=None,
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content, tool_calls=None, refusal=None),
            )
        ],
    )


async def _extra_body(
    provider: str,
    base_url: str,
    response_format: type[BaseModel] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    llm = OpenAICompatibleLLM(
        provider=provider, api_key="k", base_url=base_url, model="z-ai/glm-5.2", extra_body=extra_body
    )
    content = '{"answer": "x"}' if response_format else "plain"
    llm._client.chat.completions.create = AsyncMock(return_value=_response(content))
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], response_format=response_format, max_retries=0)
    return llm._client.chat.completions.create.call_args.kwargs.get("extra_body")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "base_url"),
    [("openrouter", ""), ("openai", "https://openrouter.ai/api/v1")],
)
async def test_structured_call_requires_parameters(provider: str, base_url: str) -> None:
    body = await _extra_body(provider, base_url, response_format=_Out)
    assert body == {"provider": {"require_parameters": True}}


@pytest.mark.asyncio
async def test_operator_provider_routing_wins_and_config_is_not_mutated() -> None:
    configured = {"provider": {"only": ["deepinfra"], "require_parameters": False}}
    body = await _extra_body("openrouter", "", response_format=_Out, extra_body=configured)
    assert body == {"provider": {"only": ["deepinfra"], "require_parameters": False}}
    assert configured == {"provider": {"only": ["deepinfra"], "require_parameters": False}}

    body = await _extra_body("openrouter", "", response_format=_Out, extra_body={"provider": {"only": ["x"]}})
    assert body == {"provider": {"require_parameters": True, "only": ["x"]}}


@pytest.mark.asyncio
async def test_untouched_without_structured_output_or_off_openrouter() -> None:
    assert await _extra_body("openrouter", "") is None
    assert await _extra_body("openai", "https://api.openai.com/v1", response_format=_Out) is None
