"""
Tests for CohereCrossEncoder.

Tests the Cohere cross-encoder implementation, including Azure AI Foundry endpoint support.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hindsight_api.engine.cross_encoder import CohereCrossEncoder, create_cross_encoder_from_env


class TestCohereCrossEncoder:
    """Test suite for CohereCrossEncoder class."""

    @pytest.mark.asyncio
    async def test_initialization_native_cohere(self):
        """Test successful initialization with native Cohere API (no base_url)."""
        encoder = CohereCrossEncoder(
            api_key="test_key",
            model="rerank-english-v3.0",
        )

        assert encoder.provider_name == "cohere"
        assert encoder.api_key == "test_key"
        assert encoder.model == "rerank-english-v3.0"
        assert encoder._client is None
        assert encoder._http_client is None

        # Mock the cohere import
        mock_cohere = MagicMock()
        mock_cohere.Client = MagicMock()
        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            await encoder.initialize()
            assert encoder._client is not None
            assert encoder._http_client is None
            mock_cohere.Client.assert_called_once_with(api_key="test_key", timeout=60.0)

    @pytest.mark.asyncio
    async def test_initialization_azure_endpoint(self):
        """Test initialization with Azure AI Foundry endpoint (uses httpx)."""
        encoder = CohereCrossEncoder(
            api_key="test_key",
            model="cohere-rerank-v3-english",
            base_url="https://my-endpoint.inference.ai.azure.com/models/cohere-rerank-v3-english/invoke",
        )

        assert encoder.base_url == "https://my-endpoint.inference.ai.azure.com/models/cohere-rerank-v3-english/invoke"

        await encoder.initialize()

        assert encoder._http_client is not None
        assert encoder._client is None
        assert isinstance(encoder._http_client._async_client, httpx.AsyncClient)
        assert encoder._http_client.include_top_n is False
        assert (
            encoder._http_client.rerank_url
            == "https://my-endpoint.inference.ai.azure.com/models/cohere-rerank-v3-english/invoke"
        )

    @pytest.mark.asyncio
    async def test_initialization_missing_package(self):
        """Test initialization fails when cohere package is missing (native API)."""
        encoder = CohereCrossEncoder(
            api_key="test_key",
            model="rerank-english-v3.0",
        )

        with patch.dict("sys.modules", {"cohere": None}):
            with pytest.raises(ImportError, match="cohere is required"):
                await encoder.initialize()

    @pytest.mark.asyncio
    async def test_initialization_idempotent(self):
        """Test that calling initialize() multiple times is safe."""
        encoder = CohereCrossEncoder(
            api_key="test_key",
            model="rerank-english-v3.0",
        )

        mock_cohere = MagicMock()
        mock_cohere.Client = MagicMock()
        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            await encoder.initialize()
            assert encoder._client is not None

            # Second call should be no-op
            await encoder.initialize()
            # Should only create client once
            mock_cohere.Client.assert_called_once()

    @pytest.mark.asyncio
    async def test_predict_native_cohere_single_query(self):
        """Test prediction with native Cohere SDK."""
        encoder = CohereCrossEncoder(
            api_key="test_key",
            model="rerank-english-v3.0",
        )

        # Create mock Cohere response
        mock_result_1 = MagicMock()
        mock_result_1.index = 0
        mock_result_1.relevance_score = 0.9

        mock_result_2 = MagicMock()
        mock_result_2.index = 1
        mock_result_2.relevance_score = 0.7

        mock_result_3 = MagicMock()
        mock_result_3.index = 2
        mock_result_3.relevance_score = 0.5

        mock_response = MagicMock()
        mock_response.results = [mock_result_1, mock_result_2, mock_result_3]

        mock_cohere_client = MagicMock()
        mock_cohere_client.rerank = MagicMock(return_value=mock_response)

        mock_cohere = MagicMock()
        mock_cohere.Client = MagicMock(return_value=mock_cohere_client)

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            await encoder.initialize()

            pairs = [
                ("What is Python?", "Python is a programming language"),
                ("What is Python?", "Python is a snake"),
                ("What is Python?", "Python is a British comedy group"),
            ]

            scores = await encoder.predict(pairs)

            assert len(scores) == 3
            assert scores == [0.9, 0.7, 0.5]

            # Verify rerank was called correctly
            mock_cohere_client.rerank.assert_called_once()
            call_args = mock_cohere_client.rerank.call_args
            assert call_args.kwargs["model"] == "rerank-english-v3.0"
            assert call_args.kwargs["query"] == "What is Python?"
            assert len(call_args.kwargs["documents"]) == 3
            assert call_args.kwargs["return_documents"] is False

    @pytest.mark.asyncio
    async def test_predict_azure_endpoint_single_query(self):
        """Test prediction with Azure AI Foundry endpoint (httpx direct call)."""
        encoder = CohereCrossEncoder(
            api_key="test_key",
            model="cohere-rerank-v3-english",
            base_url="https://my-endpoint.inference.ai.azure.com/models/cohere-rerank-v3-english/invoke",
        )

        await encoder.initialize()

        # Mock async httpx response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.7},
                {"index": 2, "relevance_score": 0.5},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        encoder._http_client._async_client.post = AsyncMock(return_value=mock_response)

        pairs = [
            ("What is Python?", "Python is a programming language"),
            ("What is Python?", "Python is a snake"),
            ("What is Python?", "Python is a British comedy group"),
        ]

        scores = await encoder.predict(pairs)

        assert len(scores) == 3
        assert scores == [0.9, 0.7, 0.5]

        # Verify httpx.post was called with correct URL and payload
        encoder._http_client._async_client.post.assert_called_once()
        call_args = encoder._http_client._async_client.post.call_args
        assert call_args[0][0] == "https://my-endpoint.inference.ai.azure.com/models/cohere-rerank-v3-english/invoke"
        assert call_args.kwargs["json"]["model"] == "cohere-rerank-v3-english"
        assert call_args.kwargs["json"]["query"] == "What is Python?"
        assert len(call_args.kwargs["json"]["documents"]) == 3
        assert call_args.kwargs["json"]["return_documents"] is False
        # Azure endpoints expect no top_n in the body
        assert "top_n" not in call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_predict_multiple_queries(self):
        """Test prediction with multiple different queries (grouped efficiently)."""
        encoder = CohereCrossEncoder(
            api_key="test_key",
            model="rerank-english-v3.0",
        )

        # First query response
        mock_result_1_1 = MagicMock()
        mock_result_1_1.index = 0
        mock_result_1_1.relevance_score = 0.9

        mock_result_1_2 = MagicMock()
        mock_result_1_2.index = 1
        mock_result_1_2.relevance_score = 0.7

        mock_response1 = MagicMock()
        mock_response1.results = [mock_result_1_1, mock_result_1_2]

        # Second query response
        mock_result_2_1 = MagicMock()
        mock_result_2_1.index = 0
        mock_result_2_1.relevance_score = 0.8

        mock_response2 = MagicMock()
        mock_response2.results = [mock_result_2_1]

        mock_cohere_client = MagicMock()
        mock_cohere_client.rerank = MagicMock(side_effect=[mock_response1, mock_response2])

        mock_cohere = MagicMock()
        mock_cohere.Client = MagicMock(return_value=mock_cohere_client)

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            await encoder.initialize()

            pairs = [
                ("What is Python?", "Python is a programming language"),
                ("What is Python?", "Python is a snake"),
                ("What is Java?", "Java is a programming language"),
            ]

            scores = await encoder.predict(pairs)

            assert len(scores) == 3
            assert scores[0] == 0.9  # First query, first doc
            assert scores[1] == 0.7  # First query, second doc
            assert scores[2] == 0.8  # Second query, first doc

            # Verify rerank was called twice (once per unique query)
            assert mock_cohere_client.rerank.call_count == 2

    @pytest.mark.asyncio
    async def test_predict_empty_pairs(self):
        """Test prediction with empty input."""
        encoder = CohereCrossEncoder(
            api_key="test_key",
            model="rerank-english-v3.0",
        )

        mock_cohere = MagicMock()
        mock_cohere.Client = MagicMock()
        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            await encoder.initialize()
            scores = await encoder.predict([])
            assert scores == []

    @pytest.mark.asyncio
    async def test_predict_not_initialized(self):
        """Test that predict fails if encoder not initialized."""
        encoder = CohereCrossEncoder(
            api_key="test_key",
            model="rerank-english-v3.0",
        )

        pairs = [("query", "document")]

        with pytest.raises(RuntimeError, match="not initialized"):
            await encoder.predict(pairs)

    @pytest.mark.asyncio
    async def test_azure_endpoint_http_error(self):
        """Test that HTTP errors from Azure endpoint are raised."""
        encoder = CohereCrossEncoder(
            api_key="test_key",
            model="cohere-rerank-v3-english",
            base_url="https://my-endpoint.inference.ai.azure.com/models/cohere-rerank-v3-english/invoke",
        )

        await encoder.initialize()

        # Mock httpx to raise HTTP error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )
        encoder._http_client._async_client.post = AsyncMock(return_value=mock_response)

        pairs = [("What is Python?", "Python is a programming language")]

        # Should raise the HTTP error
        with pytest.raises(httpx.HTTPStatusError):
            await encoder.predict(pairs)


class TestFactoryFunction:
    """Test suite for create_cross_encoder_from_env factory function."""

    @pytest.mark.asyncio
    async def test_create_cohere_from_env(self):
        """Test creating Cohere cross-encoder from environment variables."""
        env_vars = {
            "HINDSIGHT_API_RERANKER_PROVIDER": "cohere",
            "HINDSIGHT_API_RERANKER_COHERE_API_KEY": "test_key",
            "HINDSIGHT_API_RERANKER_COHERE_MODEL": "rerank-english-v3.0",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            from hindsight_api.config import HindsightConfig

            config = HindsightConfig.from_env()

            with patch("hindsight_api.config.get_config", return_value=config):
                encoder = create_cross_encoder_from_env()

                assert isinstance(encoder, CohereCrossEncoder)
                assert encoder.api_key == "test_key"
                assert encoder.model == "rerank-english-v3.0"
                assert encoder.base_url is None

    @pytest.mark.asyncio
    async def test_create_cohere_with_azure_base_url_from_env(self):
        """Test creating Cohere cross-encoder with Azure base URL from environment."""
        env_vars = {
            "HINDSIGHT_API_RERANKER_PROVIDER": "cohere",
            "HINDSIGHT_API_RERANKER_COHERE_API_KEY": "test_key",
            "HINDSIGHT_API_RERANKER_COHERE_MODEL": "cohere-rerank-v3-english",
            "HINDSIGHT_API_RERANKER_COHERE_BASE_URL": "https://my-endpoint.inference.ai.azure.com/models/cohere-rerank-v3-english/invoke",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            from hindsight_api.config import HindsightConfig

            config = HindsightConfig.from_env()

            with patch("hindsight_api.config.get_config", return_value=config):
                encoder = create_cross_encoder_from_env()

                assert isinstance(encoder, CohereCrossEncoder)
                assert encoder.api_key == "test_key"
                assert encoder.model == "cohere-rerank-v3-english"
                assert encoder.base_url == "https://my-endpoint.inference.ai.azure.com/models/cohere-rerank-v3-english/invoke"
