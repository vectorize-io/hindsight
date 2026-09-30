"""Provider usage adapters preserve visible output and reasoning separately."""

from types import SimpleNamespace

import pytest

from hindsight_api.engine.providers.gemini_llm import _usage_from_gemini_response
from hindsight_api.engine.providers.litellm_llm import _usage_from_litellm_response
from hindsight_api.engine.providers.openai_compatible_llm import _usage_from_openai_response


@pytest.mark.parametrize("adapter", [_usage_from_openai_response, _usage_from_litellm_response])
@pytest.mark.parametrize(
    ("completion_tokens", "total_tokens"),
    [(100, 110), (40, 110)],
    ids=["reasoning-folded-into-completion", "reasoning-outside-completion"],
)
def test_openai_shaped_usage_splits_reasoning(adapter, completion_tokens, total_tokens):
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=5),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=60),
        )
    )

    usage = adapter(response)

    assert usage.input_tokens == 10
    assert usage.output_tokens == 40
    assert usage.cached_tokens == 5
    assert usage.thoughts_tokens == 60


def test_gemini_usage_stashes_thoughts_separately():
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=40,
            cached_content_token_count=5,
            thoughts_token_count=60,
        )
    )

    usage = _usage_from_gemini_response(response)

    assert usage.input_tokens == 10
    assert usage.output_tokens == 40
    assert usage.cached_tokens == 5
    assert usage.thoughts_tokens == 60
