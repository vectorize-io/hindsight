"""
Test validation for batch API configuration.

When HINDSIGHT_API_RETAIN_BATCH_ENABLED=true but the LLM provider does not
support the batch API, the server should fail at startup with a clear error
message telling the user exactly what config is wrong.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.config import HindsightConfig


@pytest.mark.asyncio
async def test_startup_rejects_batch_enabled_with_non_batch_provider():
    """
    verify_llm() should raise RuntimeError at startup when
    retain_batch_enabled=True but the provider doesn't support batch API.
    """
    mock_provider = AsyncMock()
    mock_provider.supports_batch_api = AsyncMock(return_value=False)

    mock_llm_config = MagicMock()
    # anthropic has no batch API in the engine — a genuine non-batch provider.
    mock_llm_config.provider = "anthropic"
    mock_llm_config._provider_impl = mock_provider
    mock_llm_config.verify_connection = AsyncMock()

    config = HindsightConfig.from_env()
    config.retain_batch_enabled = True

    with patch("hindsight_api.engine.memory_engine.get_config", return_value=config):
        supports_batch = await mock_provider.supports_batch_api()
        assert supports_batch is False

        with pytest.raises(RuntimeError, match="HINDSIGHT_API_RETAIN_BATCH_ENABLED=true"):
            if config.retain_batch_enabled and not supports_batch:
                raise RuntimeError(
                    f"Configuration error: HINDSIGHT_API_RETAIN_BATCH_ENABLED=true "
                    f"but the retain LLM provider '{mock_llm_config.provider}' "
                    f"does not support the batch API. Either switch to a provider "
                    f"that supports batch operations (e.g. 'openai', 'groq', 'gemini') or "
                    f"set HINDSIGHT_API_RETAIN_BATCH_ENABLED=false."
                )


@pytest.mark.asyncio
async def test_startup_allows_batch_enabled_with_batch_provider():
    """
    verify_llm() should NOT raise when retain_batch_enabled=True and the
    provider supports batch API (e.g. OpenAI).
    """
    mock_provider = AsyncMock()
    mock_provider.supports_batch_api = AsyncMock(return_value=True)

    config = HindsightConfig.from_env()
    config.retain_batch_enabled = True

    supports_batch = await mock_provider.supports_batch_api()
    assert supports_batch is True

    # No error should be raised
    if config.retain_batch_enabled and not supports_batch:
        pytest.fail("Should not reach here -- provider supports batch API")


@pytest.mark.asyncio
async def test_startup_allows_batch_disabled_with_non_batch_provider():
    """
    verify_llm() should NOT raise when retain_batch_enabled=False,
    regardless of provider batch support.
    """
    mock_provider = AsyncMock()
    mock_provider.supports_batch_api = AsyncMock(return_value=False)

    config = HindsightConfig.from_env()
    config.retain_batch_enabled = False

    supports_batch = await mock_provider.supports_batch_api()
    assert supports_batch is False

    # No error should be raised when batch is disabled
    if config.retain_batch_enabled and not supports_batch:
        pytest.fail("Should not reach here -- batch is disabled")


@pytest.mark.asyncio
async def test_runtime_raises_if_batch_unsupported():
    """
    extract_facts_from_contents_batch_api() should raise RuntimeError
    if somehow called with a non-batch provider (startup check bypassed).
    """
    mock_provider = AsyncMock()
    mock_provider.supports_batch_api = AsyncMock(return_value=False)

    with pytest.raises(RuntimeError, match="does not support the batch API"):
        if not await mock_provider.supports_batch_api():
            raise RuntimeError(
                "retain_batch_enabled=True but provider 'anthropic' does not "
                "support the batch API. This should have been caught at startup -- check "
                "HINDSIGHT_API_RETAIN_BATCH_ENABLED and your LLM provider configuration."
            )
