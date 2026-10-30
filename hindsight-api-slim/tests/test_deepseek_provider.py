"""DeepSeek as a first-class OpenAI-compatible provider: its defaults and request shape."""

import pytest

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


def _make_deepseek_llm(model: str = "deepseek-v4-flash") -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openai",
        api_key="sk-test",
        base_url="https://api.deepseek.com",
        model=model,
    )


def test_deepseek_first_class_provider_sets_default_base_url():
    """provider="deepseek" is a configuration shortcut for the api.deepseek.com endpoint."""
    llm = OpenAICompatibleLLM(
        provider="deepseek",
        api_key="sk-test",
        base_url="",
        model="deepseek-v4-flash",
    )

    assert llm.base_url == "https://api.deepseek.com"


def test_deepseek_first_class_provider_requires_api_key():
    with pytest.raises(ValueError, match="API key is required for deepseek"):
        OpenAICompatibleLLM(
            provider="deepseek",
            api_key="",
            base_url="",
            model="deepseek-v4-flash",
        )


def test_deepseek_flash_is_not_treated_as_reasoning_model():
    llm = _make_deepseek_llm("deepseek-v4-flash")

    assert llm._supports_reasoning_model() is False


def test_deepseek_reasoning_models_still_use_reasoning_parameters():
    llm = _make_deepseek_llm("deepseek-v4-pro")

    assert llm._supports_reasoning_model() is True


@pytest.mark.asyncio
async def test_a_strict_schema_goes_in_the_prompt_on_deepseek():
    """DeepSeek rejects a json_schema response_format with HTTP 400, so it gets json_object."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from hindsight_api.engine.providers.tools_via_schema import ResponseSchema, schema_format

    llm = OpenAICompatibleLLM(provider="deepseek", api_key="k", base_url="", model="deepseek-v4-flash")
    reply = MagicMock()
    reply.choices[0].message.content = '{"query": "x"}'
    reply.choices[0].finish_reason = "stop"
    reply.usage.prompt_tokens, reply.usage.completion_tokens, reply.usage.total_tokens = 1, 1, 2
    reply.usage.completion_tokens_details = None
    reply.error = None
    reply.model_dump.return_value = {}
    fmt = schema_format(
        ResponseSchema(name="recall", schema={"type": "object", "properties": {"query": {"type": "string"}}})
    )
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock, return_value=reply) as create:
        result = await llm.call(
            [{"role": "user", "content": "q"}],
            response_format=fmt,
            strict_schema=True,
            skip_validation=True,
            max_retries=0,
        )
    sent = create.await_args.kwargs
    assert sent["response_format"] == {"type": "json_object"}
    assert '"query"' in sent["messages"][0]["content"]
    assert result.content == {"query": "x"}
