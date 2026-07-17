"""
Tests for the llamacpp reranker provider (issue #2665).

llama.cpp's llama-server exposes a Cohere/Jina-compatible rerank endpoint
(aliases /rerank, /reranking, /v1/rerank, /v1/reranking) when started with
--rerank. These tests cover the factory wiring from env vars and the HTTP
behavior against a mocked server response.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.engine.cross_encoder import LlamaCppCrossEncoder, create_cross_encoder_from_env


class TestLlamaCppCrossEncoder:
    """Test suite for LlamaCppCrossEncoder against a mocked llama.cpp server."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "relevance_scores",
        [
            # already-normalized-looking scores
            [0.9, 0.7, 0.5],
            # raw logits as llama.cpp actually returns them (res->score = embd[0]):
            # negative and >1 values must be passed through untouched
            [-2.3, 4.1, 0.0],
        ],
    )
    async def test_predict_single_query(self, relevance_scores):
        """Scores map back to original pair order by results[].index, raw values preserved."""
        encoder = LlamaCppCrossEncoder(base_url="http://localhost:8080", model="bge-reranker-v2-m3")
        await encoder.initialize()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"index": i, "relevance_score": s} for i, s in enumerate(relevance_scores)]
        }
        mock_response.raise_for_status = MagicMock()
        encoder._client._async_client.post = AsyncMock(return_value=mock_response)

        pairs = [
            ("What is a panda?", "The giant panda is a bear species endemic to China."),
            ("What is a panda?", "It is a bear."),
            ("What is a panda?", "hi"),
        ]
        scores = await encoder.predict(pairs)

        assert scores == relevance_scores

        encoder._client._async_client.post.assert_called_once()
        call_args = encoder._client._async_client.post.call_args
        assert call_args[0][0] == "http://localhost:8080/rerank"
        body = call_args.kwargs["json"]
        assert body["model"] == "bge-reranker-v2-m3"
        assert body["query"] == "What is a panda?"
        assert len(body["documents"]) == 3
        assert body["top_n"] == 3
        assert body["return_documents"] is False

    @pytest.mark.asyncio
    async def test_predict_multiple_queries(self):
        """Pairs are grouped per unique query: one POST per query, indices remapped."""
        encoder = LlamaCppCrossEncoder(base_url="http://localhost:8080")
        await encoder.initialize()

        response_1 = MagicMock()
        response_1.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.7}]
        }
        response_1.raise_for_status = MagicMock()
        response_2 = MagicMock()
        response_2.json.return_value = {"results": [{"index": 0, "relevance_score": 0.8}]}
        response_2.raise_for_status = MagicMock()
        encoder._client._async_client.post = AsyncMock(side_effect=[response_1, response_2])

        pairs = [
            ("What is Python?", "Python is a programming language"),
            ("What is Python?", "Python is a snake"),
            ("What is Java?", "Java is a programming language"),
        ]
        scores = await encoder.predict(pairs)

        assert scores == [0.9, 0.7, 0.8]
        assert encoder._client._async_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_predict_remaps_permuted_result_indices(self):
        """results[].index is authoritative even when the server returns scores out of order."""
        encoder = LlamaCppCrossEncoder(base_url="http://localhost:8080")
        await encoder.initialize()

        mock_response = MagicMock()
        # Server returns documents scored in reverse/permuted order relative to request.
        mock_response.json.return_value = {
            "results": [
                {"index": 2, "relevance_score": 0.1},
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.5},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        encoder._client._async_client.post = AsyncMock(return_value=mock_response)

        pairs = [
            ("What is a panda?", "The giant panda is a bear species endemic to China."),
            ("What is a panda?", "It is a bear."),
            ("What is a panda?", "hi"),
        ]
        scores = await encoder.predict(pairs)

        assert scores == [0.9, 0.5, 0.1]

    @pytest.mark.asyncio
    async def test_predict_empty_pairs(self):
        encoder = LlamaCppCrossEncoder(base_url="http://localhost:8080")
        await encoder.initialize()
        assert await encoder.predict([]) == []

    @pytest.mark.asyncio
    async def test_predict_not_initialized(self):
        encoder = LlamaCppCrossEncoder(base_url="http://localhost:8080")
        with pytest.raises(RuntimeError, match="not initialized"):
            await encoder.predict([("query", "document")])

    @pytest.mark.asyncio
    async def test_http_error_propagates(self):
        """HTTP errors from the llama.cpp server surface to the caller."""
        encoder = LlamaCppCrossEncoder(base_url="http://localhost:8080")
        await encoder.initialize()

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503 Service Unavailable",
            request=MagicMock(),
            response=MagicMock(status_code=503),
        )
        encoder._client._async_client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(httpx.HTTPStatusError):
            await encoder.predict([("query", "document")])

    @pytest.mark.asyncio
    async def test_no_api_key_sends_dummy_bearer(self):
        """Without an API key the Authorization header stays well-formed (Bearer no-key)."""
        encoder = LlamaCppCrossEncoder(base_url="http://localhost:8080")
        await encoder.initialize()
        assert encoder._client._async_client.headers["Authorization"] == "Bearer no-key"

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        encoder = LlamaCppCrossEncoder(base_url="http://localhost:8080")
        await encoder.initialize()
        client = encoder._client._async_client
        await encoder.initialize()
        assert encoder._client._async_client is client


class TestFactoryFunction:
    """Test suite for create_cross_encoder_from_env with provider=llamacpp."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "base_url, expected_rerank_url",
        [
            # bare host → /rerank
            ("http://localhost:8080", "http://localhost:8080/rerank"),
            # trailing slash is stripped
            ("http://localhost:8080/", "http://localhost:8080/rerank"),
            # /v1-suffixed base also works: llama.cpp aliases /v1/rerank to the same handler
            ("http://reranker.internal:8012/v1", "http://reranker.internal:8012/v1/rerank"),
        ],
    )
    async def test_create_llamacpp_from_env(self, base_url, expected_rerank_url):
        """Factory builds a llamacpp encoder from env vars; base URL shape is forgiving."""
        env_vars = {
            "HINDSIGHT_API_RERANKER_PROVIDER": "llamacpp",
            "HINDSIGHT_API_RERANKER_LLAMACPP_BASE_URL": base_url,
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = HindsightConfig.from_env()

            with patch("hindsight_api.config.get_config", return_value=config):
                encoder = create_cross_encoder_from_env()

                assert encoder.provider_name == "llamacpp"
                assert encoder._client.rerank_url == expected_rerank_url
                # No API key configured: dummy value keeps the Bearer header well-formed
                # (llama.cpp ignores it unless started with --api-key).
                assert encoder._client.api_key == "no-key"
                # Model defaults to empty: single-model llama.cpp servers ignore it.
                assert encoder._client.model == ""

    @pytest.mark.asyncio
    async def test_create_llamacpp_with_model_and_api_key(self):
        """Optional model and API key env vars reach the HTTP client."""
        env_vars = {
            "HINDSIGHT_API_RERANKER_PROVIDER": "llamacpp",
            "HINDSIGHT_API_RERANKER_LLAMACPP_BASE_URL": "http://localhost:8080",
            "HINDSIGHT_API_RERANKER_LLAMACPP_MODEL": "bge-reranker-v2-m3",
            "HINDSIGHT_API_RERANKER_LLAMACPP_API_KEY": "secret",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = HindsightConfig.from_env()

            with patch("hindsight_api.config.get_config", return_value=config):
                encoder = create_cross_encoder_from_env()

                assert encoder.provider_name == "llamacpp"
                assert encoder._client.model == "bge-reranker-v2-m3"
                assert encoder._client.api_key == "secret"

    @pytest.mark.asyncio
    async def test_missing_base_url_raises(self):
        """provider=llamacpp without a base URL fails fast, naming the env var."""
        env_vars = {"HINDSIGHT_API_RERANKER_PROVIDER": "llamacpp"}

        with patch.dict(os.environ, env_vars, clear=False):
            os.environ.pop("HINDSIGHT_API_RERANKER_LLAMACPP_BASE_URL", None)
            config = HindsightConfig.from_env()

            with patch("hindsight_api.config.get_config", return_value=config):
                with pytest.raises(ValueError, match="HINDSIGHT_API_RERANKER_LLAMACPP_BASE_URL"):
                    create_cross_encoder_from_env()
