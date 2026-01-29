"""
Test Vertex AI provider integration including token refresh and API calls.
"""

import asyncio
import os
from unittest.mock import MagicMock, Mock, patch

import pytest

# Skip all tests if google-auth not available
pytest.importorskip("google.auth")


@pytest.mark.asyncio
async def test_token_refresher_initialization():
    """Test token refresher initialization with mocked credentials."""
    from hindsight_api.engine.vertexai_token_refresher import VertexAITokenRefresher

    # Mock credentials
    mock_credentials = MagicMock()
    mock_credentials.token = "test-token-123"
    mock_credentials.expiry = None

    with patch("google.auth.transport.requests.Request"):
        refresher = VertexAITokenRefresher(mock_credentials, "test-project", "us-central1")

        # Verify token was fetched
        assert refresher.get_token() == "test-token-123"

        # Verify base URL is correctly formatted
        expected_url = (
            "https://us-central1-aiplatform.googleapis.com/v1beta1/"
            "projects/test-project/locations/us-central1/endpoints/openapi"
        )
        assert refresher.get_base_url() == expected_url


@pytest.mark.asyncio
async def test_token_refresher_background_refresh():
    """Test that background refresh task starts and stops correctly."""
    from hindsight_api.engine.vertexai_token_refresher import VertexAITokenRefresher

    # Mock credentials
    mock_credentials = MagicMock()
    mock_credentials.token = "test-token-123"
    mock_credentials.expiry = None

    with patch("google.auth.transport.requests.Request"):
        refresher = VertexAITokenRefresher(mock_credentials, "test-project", "us-central1")

        # Start refresh task
        refresher.start_refresh_task()
        assert refresher._refresh_task is not None
        assert not refresher._refresh_task.done()

        # Stop refresh task
        await refresher.stop()
        assert refresher._refresh_task.done()


@pytest.mark.asyncio
async def test_token_refresher_thread_safety():
    """Test that token access is thread-safe."""
    from hindsight_api.engine.vertexai_token_refresher import VertexAITokenRefresher

    # Mock credentials
    mock_credentials = MagicMock()
    mock_credentials.token = "test-token-123"
    mock_credentials.expiry = None

    with patch("google.auth.transport.requests.Request"):
        refresher = VertexAITokenRefresher(mock_credentials, "test-project", "us-central1")

        # Access token from multiple tasks concurrently
        async def get_token_task():
            return refresher.get_token()

        results = await asyncio.gather(*[get_token_task() for _ in range(10)])

        # All should return the same token
        assert all(token == "test-token-123" for token in results)


@pytest.mark.asyncio
async def test_token_refresher_no_token_error():
    """Test that getting token without refresh raises error."""
    from hindsight_api.engine.vertexai_token_refresher import VertexAITokenRefresher

    # Mock credentials that fail to refresh
    mock_credentials = MagicMock()
    mock_credentials.token = None

    with patch("google.auth.transport.requests.Request") as mock_request:
        mock_request.side_effect = Exception("Refresh failed")

        with pytest.raises(Exception, match="Refresh failed"):
            VertexAITokenRefresher(mock_credentials, "test-project", "us-central1")


def test_llm_wrapper_vertexai_missing_dependency():
    """Test error when google-auth is not available."""
    from hindsight_api.engine import llm_wrapper

    # Temporarily disable Vertex AI availability
    original_available = llm_wrapper.VERTEXAI_AVAILABLE
    try:
        llm_wrapper.VERTEXAI_AVAILABLE = False

        with pytest.raises(ValueError, match="google-auth"):
            from hindsight_api.engine.llm_wrapper import LLMProvider

            LLMProvider(
                provider="vertexai",
                api_key="",
                base_url="",
                model="google/gemini-2.0-flash-001",
            )
    finally:
        llm_wrapper.VERTEXAI_AVAILABLE = original_available


def test_llm_wrapper_vertexai_missing_project_id():
    """Test error when project ID is not configured."""
    with patch.dict(os.environ, {"HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID": ""}, clear=False):
        # Clear config cache to reload from env
        from hindsight_api.config import clear_config_cache

        clear_config_cache()

        with pytest.raises(ValueError, match="HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID"):
            from hindsight_api.engine.llm_wrapper import LLMProvider

            LLMProvider(
                provider="vertexai",
                api_key="",
                base_url="",
                model="google/gemini-2.0-flash-001",
            )

        # Restore config cache
        clear_config_cache()


@pytest.mark.asyncio
async def test_llm_wrapper_vertexai_adc_auth():
    """Test Vertex AI with ADC authentication (mocked)."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    mock_credentials = MagicMock()
    mock_credentials.token = "test-token-adc"
    mock_credentials.expiry = None

    with patch.dict(
        os.environ,
        {"HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID": "test-project"},
        clear=False,
    ):
        # Clear config cache to reload from env
        from hindsight_api.config import clear_config_cache

        clear_config_cache()

        with patch("google.auth.default", return_value=(mock_credentials, "test-project")):
            with patch("google.auth.transport.requests.Request"):
                provider = LLMProvider(
                    provider="vertexai",
                    api_key="",
                    base_url="",
                    model="google/gemini-2.0-flash-001",
                )

                assert provider.provider == "vertexai"
                assert provider._vertexai_refresher is not None
                assert "aiplatform.googleapis.com" in provider.base_url

                # Cleanup
                await provider.cleanup()

        # Restore config cache
        clear_config_cache()


@pytest.mark.asyncio
async def test_llm_wrapper_vertexai_sa_auth():
    """Test Vertex AI with service account authentication (mocked)."""
    from hindsight_api.engine.llm_wrapper import LLMProvider
    import google.auth.exceptions

    mock_credentials = MagicMock()
    mock_credentials.token = "test-token-sa"
    mock_credentials.expiry = None

    with patch.dict(
        os.environ,
        {
            "HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID": "test-project",
            "HINDSIGHT_API_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY": "/path/to/key.json",
        },
        clear=False,
    ):
        # Clear config cache to reload from env
        from hindsight_api.config import clear_config_cache

        clear_config_cache()

        # Mock ADC failure, SA success
        with patch(
            "google.auth.default",
            side_effect=google.auth.exceptions.DefaultCredentialsError("ADC not available"),
        ):
            with patch(
                "google.oauth2.service_account.Credentials.from_service_account_file",
                return_value=mock_credentials,
            ):
                with patch("google.auth.transport.requests.Request"):
                    provider = LLMProvider(
                        provider="vertexai",
                        api_key="",
                        base_url="",
                        model="google/gemini-2.0-flash-001",
                    )

                    assert provider.provider == "vertexai"
                    assert provider._vertexai_refresher is not None

                    # Cleanup
                    await provider.cleanup()

        # Restore config cache
        clear_config_cache()


@pytest.mark.asyncio
async def test_llm_wrapper_vertexai_auth_failure():
    """Test Vertex AI with both ADC and SA auth failing."""
    import google.auth.exceptions

    with patch.dict(
        os.environ,
        {"HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID": "test-project"},
        clear=False,
    ):
        # Clear config cache to reload from env
        from hindsight_api.config import clear_config_cache

        clear_config_cache()

        # Mock both ADC and SA failures
        with patch(
            "google.auth.default",
            side_effect=google.auth.exceptions.DefaultCredentialsError("ADC failed"),
        ):
            with pytest.raises(ValueError, match="authentication failed"):
                from hindsight_api.engine.llm_wrapper import LLMProvider

                LLMProvider(
                    provider="vertexai",
                    api_key="",
                    base_url="",
                    model="google/gemini-2.0-flash-001",
                )

        # Restore config cache
        clear_config_cache()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID"),
    reason="Vertex AI integration tests require HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID",
)
async def test_vertexai_integration_actual_api():
    """
    Integration test with actual Vertex AI API.

    Requires:
    - HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID
    - ADC or HINDSIGHT_API_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY
    """
    from hindsight_api.engine.llm_wrapper import LLMProvider

    provider = LLMProvider(
        provider="vertexai",
        api_key="",
        base_url="",
        model="google/gemini-2.0-flash-001",
    )

    try:
        # Simple test call
        response = await provider.call(
            messages=[{"role": "user", "content": "Say 'ok' and nothing else"}],
            max_completion_tokens=10,
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    finally:
        # Cleanup
        await provider.cleanup()
