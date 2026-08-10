"""Gemini deterministic-400 handling (#3256).

An HTTP 400 ``INVALID_ARGUMENT`` is a deterministic client-side rejection: the
schema/prompt/generation-config the bank compiled is malformed, so every retry
repeats an identical rejected call. These tests pin the two fixes:

1. Retry classification — a 400 fails fast; it must NOT consume the LLM retry
   budget (nor, above it, the batch retry ladder). A retryable 503 still burns
   the full budget, so the distinction is real.
2. Diagnosability — a 400 always emits the content-free structural profile
   (``[LLM_4XX_DUMP]``) even with the opt-in flag off, so the otherwise-opaque
   failure is diagnosable on first occurrence. A recoverable cache-400 must not
   be mistaken for a deterministic rejection.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("google.genai")

from google.genai import errors as genai_errors  # noqa: E402


def _make_gemini_provider():
    """Return a GeminiLLM instance with a mocked genai.Client."""
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        from hindsight_api.engine.providers.gemini_llm import GeminiLLM

        provider = GeminiLLM(
            provider="gemini",
            api_key="fake-api-key",
            base_url="",
            model="gemini-2.5-flash",
        )
        provider._client = MagicMock()
        return provider


def _api_error(code: int, status: str = "INVALID_ARGUMENT") -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"message": f"{status}: rejected", "status": status}})


@pytest.mark.asyncio
async def test_call_400_fails_fast_without_consuming_retries():
    """A 400 raises after a single attempt — no retry budget is burned."""
    provider = _make_gemini_provider()
    generate = AsyncMock(side_effect=_api_error(400))
    provider._client.aio.models.generate_content = generate

    with pytest.raises(genai_errors.APIError) as excinfo:
        await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            scope="consolidation",
            max_retries=4,
            initial_backoff=0.0,
        )

    assert excinfo.value.code == 400
    assert generate.call_count == 1  # NOT 5 (1 + 4 retries)


@pytest.mark.asyncio
async def test_call_503_still_consumes_full_retry_budget():
    """A retryable error still burns the budget — the fail-fast is 400-specific."""
    provider = _make_gemini_provider()
    generate = AsyncMock(side_effect=_api_error(503, status="UNAVAILABLE"))
    provider._client.aio.models.generate_content = generate

    with pytest.raises(genai_errors.APIError):
        await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            scope="consolidation",
            max_retries=3,
            initial_backoff=0.0,
        )

    assert generate.call_count == 4  # 1 + 3 retries


@pytest.mark.asyncio
async def test_call_with_tools_400_fails_fast():
    """The tool path fails fast on 400 too."""
    provider = _make_gemini_provider()
    generate = AsyncMock(side_effect=_api_error(400))
    provider._client.aio.models.generate_content = generate

    with pytest.raises(genai_errors.APIError) as excinfo:
        await provider.call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "noop", "description": "n", "parameters": {"type": "object"}},
                }
            ],
            scope="consolidation",
            max_retries=4,
            initial_backoff=0.0,
        )

    assert excinfo.value.code == 400
    assert generate.call_count == 1


@pytest.mark.asyncio
async def test_call_400_always_dumps_structural_profile(monkeypatch, caplog):
    """The structural profile is logged on a 400 even with the opt-in flag off,
    and carries no user content (only per-part sizes)."""
    from hindsight_api.config import ENV_LLM_DEBUG_DUMP_4XX, clear_config_cache

    monkeypatch.delenv(ENV_LLM_DEBUG_DUMP_4XX, raising=False)
    clear_config_cache()

    provider = _make_gemini_provider()
    provider._client.aio.models.generate_content = AsyncMock(side_effect=_api_error(400))

    with caplog.at_level(logging.ERROR):
        with pytest.raises(genai_errors.APIError):
            await provider.call(
                messages=[{"role": "user", "content": "sensitive memory text"}],
                scope="consolidation",
                max_retries=4,
                initial_backoff=0.0,
            )

    assert "[LLM_4XX_DUMP]" in caplog.text
    assert "code=400" in caplog.text
    assert "sensitive memory text" not in caplog.text  # forced dump omits previews
    clear_config_cache()


@pytest.mark.asyncio
async def test_call_503_does_not_force_dump(monkeypatch, caplog):
    """A retryable non-4xx never triggers the forced structural dump."""
    from hindsight_api.config import ENV_LLM_DEBUG_DUMP_4XX, clear_config_cache

    monkeypatch.delenv(ENV_LLM_DEBUG_DUMP_4XX, raising=False)
    clear_config_cache()

    provider = _make_gemini_provider()
    provider._client.aio.models.generate_content = AsyncMock(side_effect=_api_error(503, status="UNAVAILABLE"))

    with caplog.at_level(logging.ERROR):
        with pytest.raises(genai_errors.APIError):
            await provider.call(
                messages=[{"role": "user", "content": "hi"}],
                scope="consolidation",
                max_retries=1,
                initial_backoff=0.0,
            )

    assert "[LLM_4XX_DUMP]" not in caplog.text
    clear_config_cache()
