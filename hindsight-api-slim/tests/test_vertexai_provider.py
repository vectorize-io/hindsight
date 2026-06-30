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

        with pytest.raises(ValueError, match="google-auth"):
            from hindsight_api.engine.llm_wrapper import LLMProvider

            LLMProvider(
                provider="vertexai",
                api_key="",
                base_url="",
                model="google/gemini-2.0-flash-001",
                vertexai_project_id="test-project",
                vertexai_service_account_key="/path/to/key.json",
            )
    finally:
        llm_wrapper.VERTEXAI_AVAILABLE = original_available


def test_llm_wrapper_vertexai_missing_project_id():
    """Test error when no project ID is provided."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    with pytest.raises(ValueError, match="HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID"):
        LLMProvider(
            provider="vertexai",
            api_key="",
            base_url="",
            model="google/gemini-2.0-flash-001",
        )


def test_llm_wrapper_vertexai_adc_auth():
    """Test Vertex AI with ADC authentication creates native genai client."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    # No service account key → ADC path. genai.Client handles ADC internally —
    # just verify it creates the client with the project/region passed in.
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()

        provider = LLMProvider(
            provider="vertexai",
            api_key="",
            base_url="",
            model="google/gemini-2.0-flash-001",
            vertexai_project_id="test-project",
        )

        assert provider.provider == "vertexai"
        assert provider.model == "gemini-2.0-flash-001"  # google/ prefix stripped
        assert provider._gemini_client is not None

        # Verify genai.Client was called with vertexai=True
        call_kwargs = mock_client_cls.call_args.kwargs
        assert call_kwargs["vertexai"] is True
        assert call_kwargs["project"] == "test-project"
        assert call_kwargs["location"] == "us-central1"


def test_llm_wrapper_vertexai_sa_auth():
    """Test Vertex AI with service account authentication passes credentials to genai client."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    mock_credentials = MagicMock()

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
                vertexai_project_id="test-project",
                vertexai_service_account_key="/path/to/key.json",
            )

            assert provider.provider == "vertexai"
            assert provider._gemini_client is not None

            # Verify credentials were passed to genai.Client
            call_kwargs = mock_client_cls.call_args.kwargs
            assert call_kwargs["vertexai"] is True
            assert call_kwargs["project"] == "test-project"
            assert call_kwargs["location"] == "us-central1"
            assert call_kwargs["credentials"] is mock_credentials


def test_llm_wrapper_vertexai_strips_google_prefix():
    """Test that google/ prefix is stripped from model name for native SDK."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()

        provider = LLMProvider(
            provider="vertexai",
            api_key="",
            base_url="",
            model="google/gemini-2.0-flash-lite-001",
            vertexai_project_id="test-project",
        )

        assert provider.model == "gemini-2.0-flash-lite-001"


def test_llm_wrapper_vertexai_no_prefix_model():
    """Test that model without google/ prefix is unchanged."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()

        provider = LLMProvider(
            provider="vertexai",
            api_key="",
            base_url="",
            model="gemini-2.0-flash-001",
            vertexai_project_id="test-project",
        )

        assert provider.model == "gemini-2.0-flash-001"


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
        model="google/gemini-2.5-flash-lite",
        vertexai_project_id=os.getenv("HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID"),
        vertexai_service_account_key=os.getenv("HINDSIGHT_API_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY") or None,
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
