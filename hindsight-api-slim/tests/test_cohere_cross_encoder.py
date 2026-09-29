"""
Tests for CohereCrossEncoder.

Tests the Cohere cross-encoder implementation, including Azure AI Foundry endpoint support.
The native API goes through the Cohere SDK's async client (mocked here); a custom
base_url goes through our aiohttp client, stubbed with a real in-process server.
"""

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from hindsight_api.engine.aiohttp_session import UpstreamHTTPError
from hindsight_api.engine.cross_encoder import CohereCrossEncoder, create_cross_encoder_from_env
from tests.aiohttp_stub import stub_server

AZURE_INVOKE_PATH = "/models/cohere-rerank-v3-english/invoke"


def _mock_cohere_module(async_client: Any = None) -> MagicMock:
    """A stand-in for the ``cohere`` package whose AsyncClient returns ``async_client``."""
    mock_cohere = MagicMock()
    mock_cohere.AsyncClient = MagicMock(return_value=async_client if async_client is not None else MagicMock())
    return mock_cohere


def _rerank_result(index: int, score: float) -> MagicMock:
    result = MagicMock()
    result.index = index
    result.relevance_score = score
    return result


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

        mock_cohere = _mock_cohere_module()
        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            await encoder.initialize()
            assert encoder._client is not None
            assert encoder._http_client is None
            # Built lazily per loop: its pooled connections belong to the loop that opened them.
            mock_cohere.AsyncClient.assert_not_called()
            encoder._client.get()
            # The async client: a sync one would need a thread per in-flight rerank.
            mock_cohere.AsyncClient.assert_called_once_with(api_key="test_key", timeout=60.0)
            mock_cohere.Client.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialization_azure_endpoint(self):
        """Test initialization with Azure AI Foundry endpoint (uses our aiohttp client)."""
        encoder = CohereCrossEncoder(
            api_key="test_key",
            model="cohere-rerank-v3-english",
            base_url="https://my-endpoint.inference.ai.azure.com/models/cohere-rerank-v3-english/invoke",
        )

        assert encoder.base_url == "https://my-endpoint.inference.ai.azure.com/models/cohere-rerank-v3-english/invoke"

        await encoder.initialize()

        assert encoder._http_client is not None
        assert encoder._client is None
        assert encoder._http_client.initialized
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

        mock_cohere = _mock_cohere_module()
        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            await encoder.initialize()
            assert encoder._client is not None

            # Second call should be no-op
            await encoder.initialize()
            # Should only create one client per loop
            assert encoder._client.get() is encoder._client.get()
            mock_cohere.AsyncClient.assert_called_once()

    @pytest.mark.asyncio
    async def test_predict_native_cohere_single_query(self):
        """Test prediction with native Cohere SDK (awaited on the loop)."""
        encoder = CohereCrossEncoder(
            api_key="test_key",
            model="rerank-english-v3.0",
        )

        mock_response = MagicMock()
        mock_response.results = [_rerank_result(0, 0.9), _rerank_result(1, 0.7), _rerank_result(2, 0.5)]

        mock_cohere_client = MagicMock()
        mock_cohere_client.rerank = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"cohere": _mock_cohere_module(mock_cohere_client)}):
            await encoder.initialize()

            pairs = [
                ("What is Python?", "Python is a programming language"),
                ("What is Python?", "Python is a snake"),
                ("What is Python?", "Python is a British comedy group"),
            ]

            scores = await encoder.predict(pairs)

            assert len(scores) == 3
            assert scores == [0.9, 0.7, 0.5]

            # Verify rerank was awaited correctly
            mock_cohere_client.rerank.assert_awaited_once()
            call_args = mock_cohere_client.rerank.call_args
            assert call_args.kwargs["model"] == "rerank-english-v3.0"
            assert call_args.kwargs["query"] == "What is Python?"
            assert len(call_args.kwargs["documents"]) == 3
            assert call_args.kwargs["return_documents"] is False

    @pytest.mark.asyncio
    async def test_predict_azure_endpoint_single_query(self):
        """Test prediction with Azure AI Foundry endpoint (direct HTTP call)."""
        requests: list[dict[str, Any]] = []

        async def handler(request: web.Request) -> web.StreamResponse:
            requests.append({"path": request.path, "headers": dict(request.headers), "json": await request.json()})
            return web.json_response(
                {
                    "results": [
                        {"index": 0, "relevance_score": 0.9},
                        {"index": 1, "relevance_score": 0.7},
                        {"index": 2, "relevance_score": 0.5},
                    ]
                }
            )

        pairs = [
            ("What is Python?", "Python is a programming language"),
            ("What is Python?", "Python is a snake"),
            ("What is Python?", "Python is a British comedy group"),
        ]

        async with stub_server(handler) as base_url:
            encoder = CohereCrossEncoder(
                api_key="test_key",
                model="cohere-rerank-v3-english",
                base_url=f"{base_url}{AZURE_INVOKE_PATH}",
            )
            await encoder.initialize()

            with patch(
                "hindsight_api.engine.cross_encoder.reranker_bank_attribution_headers",
                return_value={"X-Hindsight-Bank-Id": "bank-cohere-http"},
            ):
                scores = await encoder.predict(pairs)

        assert len(scores) == 3
        assert scores == [0.9, 0.7, 0.5]

        # Verify the POST went to the full invoke URL with the right payload
        assert len(requests) == 1
        request = requests[0]
        assert request["path"] == AZURE_INVOKE_PATH
        assert request["headers"]["Authorization"] == "Bearer test_key"
        assert request["headers"]["X-Hindsight-Bank-Id"] == "bank-cohere-http"
        assert request["json"]["model"] == "cohere-rerank-v3-english"
        assert request["json"]["query"] == "What is Python?"
        assert len(request["json"]["documents"]) == 3
        assert request["json"]["return_documents"] is False
        # Azure endpoints expect no top_n in the body
        assert "top_n" not in request["json"]

    @pytest.mark.asyncio
    async def test_predict_multiple_queries(self):
        """Test prediction with multiple different queries (grouped efficiently)."""
        encoder = CohereCrossEncoder(
            api_key="test_key",
            model="rerank-english-v3.0",
        )

        mock_response1 = MagicMock()
        mock_response1.results = [_rerank_result(0, 0.9), _rerank_result(1, 0.7)]
        mock_response2 = MagicMock()
        mock_response2.results = [_rerank_result(0, 0.8)]

        mock_cohere_client = MagicMock()
        mock_cohere_client.rerank = AsyncMock(side_effect=[mock_response1, mock_response2])

        with patch.dict("sys.modules", {"cohere": _mock_cohere_module(mock_cohere_client)}):
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
            assert mock_cohere_client.rerank.await_count == 2

    @pytest.mark.asyncio
    async def test_predict_empty_pairs(self):
        """Test prediction with empty input."""
        encoder = CohereCrossEncoder(
            api_key="test_key",
            model="rerank-english-v3.0",
        )

        with patch.dict("sys.modules", {"cohere": _mock_cohere_module()}):
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
        """Test that HTTP errors from Azure endpoint are raised, carrying status and body."""

        async def handler(request: web.Request) -> web.StreamResponse:
            return web.Response(status=404, text="deployment not found")

        async with stub_server(handler) as base_url:
            encoder = CohereCrossEncoder(
                api_key="test_key",
                model="cohere-rerank-v3-english",
                base_url=f"{base_url}{AZURE_INVOKE_PATH}",
            )
            await encoder.initialize()

            # Should raise the HTTP error
            with pytest.raises(UpstreamHTTPError) as exc_info:
                await encoder.predict([("What is Python?", "Python is a programming language")])

        assert exc_info.value.status_code == 404
        assert "deployment not found" in exc_info.value.body


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
                assert (
                    encoder.base_url
                    == "https://my-endpoint.inference.ai.azure.com/models/cohere-rerank-v3-english/invoke"
                )
