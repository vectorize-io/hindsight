"""
Test Vertex AI provider integration using native genai SDK.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if google-auth not available
pytest.importorskip("google.auth")


def test_llm_wrapper_vertexai_missing_dependency():
    """Test error when google-auth is not available and service account key is set."""
    from hindsight_api.engine import llm_wrapper

    # VERTEXAI_AVAILABLE only matters when a service account key is provided
    original_available = llm_wrapper.VERTEXAI_AVAILABLE
    try:
        llm_wrapper.VERTEXAI_AVAILABLE = False

        with patch.dict(
            os.environ,
            {
                "HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID": "test-project",
                "HINDSIGHT_API_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY": "/path/to/key.json",
            },
            clear=False,
        ):
            from hindsight_api.config import clear_config_cache

            clear_config_cache()

            with pytest.raises(ValueError, match="google-auth"):
                from hindsight_api.engine.llm_wrapper import LLMProvider

                LLMProvider(
                    provider="vertexai",
                    api_key="",
                    base_url="",
                    model="google/gemini-2.0-flash-001",
                )

            clear_config_cache()
    finally:
        llm_wrapper.VERTEXAI_AVAILABLE = original_available


def test_llm_wrapper_vertexai_missing_project_id():
    """Test error when project ID is not configured."""
    with patch.dict(os.environ, {"HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID": ""}, clear=False):
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

        clear_config_cache()


def test_llm_wrapper_vertexai_adc_auth():
    """Test Vertex AI with ADC authentication creates native genai client."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    with patch.dict(
        os.environ,
        {"HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID": "test-project"},
        clear=False,
    ):
        from hindsight_api.config import clear_config_cache

        clear_config_cache()

        # genai.Client handles ADC internally — just verify it creates the client
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()

            provider = LLMProvider(
                provider="vertexai",
                api_key="",
                base_url="",
                model="google/gemini-2.0-flash-001",
            )

            assert provider.provider == "vertexai"
            assert provider.model == "gemini-2.0-flash-001"  # google/ prefix stripped
            assert provider._gemini_client is not None

            # Verify genai.Client was called with vertexai=True
            mock_client_cls.assert_called_once_with(
                vertexai=True,
                project="test-project",
                location="us-central1",
            )

        clear_config_cache()


def test_llm_wrapper_vertexai_sa_auth():
    """Test Vertex AI with service account authentication passes credentials to genai client."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    mock_credentials = MagicMock()

    with patch.dict(
        os.environ,
        {
            "HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID": "test-project",
            "HINDSIGHT_API_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY": "/path/to/key.json",
        },
        clear=False,
    ):
        from hindsight_api.config import clear_config_cache

        clear_config_cache()

        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_credentials,
        ):
            with patch("google.genai.Client") as mock_client_cls:
                mock_client_cls.return_value = MagicMock()

                provider = LLMProvider(
                    provider="vertexai",
                    api_key="",
                    base_url="",
                    model="google/gemini-2.0-flash-001",
                )

                assert provider.provider == "vertexai"
                assert provider._gemini_client is not None

                # Verify credentials were passed to genai.Client
                mock_client_cls.assert_called_once_with(
                    vertexai=True,
                    project="test-project",
                    location="us-central1",
                    credentials=mock_credentials,
                )

        clear_config_cache()


def test_llm_wrapper_vertexai_strips_google_prefix():
    """Test that google/ prefix is stripped from model name for native SDK."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    with patch.dict(
        os.environ,
        {"HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID": "test-project"},
        clear=False,
    ):
        from hindsight_api.config import clear_config_cache

        clear_config_cache()

        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()

            provider = LLMProvider(
                provider="vertexai",
                api_key="",
                base_url="",
                model="google/gemini-2.0-flash-lite-001",
            )

            assert provider.model == "gemini-2.0-flash-lite-001"

        clear_config_cache()


def test_llm_wrapper_vertexai_no_prefix_model():
    """Test that model without google/ prefix is unchanged."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    with patch.dict(
        os.environ,
        {"HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID": "test-project"},
        clear=False,
    ):
        from hindsight_api.config import clear_config_cache

        clear_config_cache()

        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()

            provider = LLMProvider(
                provider="vertexai",
                api_key="",
                base_url="",
                model="gemini-2.0-flash-001",
            )

            assert provider.model == "gemini-2.0-flash-001"

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
        await provider.cleanup()
