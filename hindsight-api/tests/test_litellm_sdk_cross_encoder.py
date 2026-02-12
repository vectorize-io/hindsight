"""
Tests for LiteLLMSDKCrossEncoder.

Tests the LiteLLM SDK-based cross-encoder implementation for reranking.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.cross_encoder import LiteLLMSDKCrossEncoder, create_cross_encoder_from_env


class TestLiteLLMSDKCrossEncoder:
    """Test suite for LiteLLMSDKCrossEncoder class."""

    @pytest.mark.asyncio
    async def test_initialization_success(self):
        """Test successful initialization with valid config."""
        encoder = LiteLLMSDKCrossEncoder(
            api_key="test_key",
            model="deepinfra/Qwen3-reranker-8B",
        )

        assert encoder.provider_name == "litellm-sdk"
        assert encoder.api_key == "test_key"
        assert encoder.model == "deepinfra/Qwen3-reranker-8B"
        assert encoder._initialized is False

        # Mock the litellm import
        mock_litellm = MagicMock()
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            await encoder.initialize()
            assert encoder._initialized is True
            # Check that DeepInfra API key was set
            assert os.environ.get("DEEPINFRA_API_KEY") == "test_key"

    @pytest.mark.asyncio
    async def test_initialization_missing_package(self):
        """Test initialization fails when litellm package is missing."""
        encoder = LiteLLMSDKCrossEncoder(
            api_key="test_key",
            model="cohere/rerank-english-v3.0",
        )

        with patch.dict("sys.modules", {"litellm": None}):
            with pytest.raises(ImportError, match="litellm is required"):
                await encoder.initialize()

    @pytest.mark.asyncio
    async def test_initialization_idempotent(self):
        """Test that calling initialize() multiple times is safe."""
        encoder = LiteLLMSDKCrossEncoder(
            api_key="test_key",
            model="cohere/rerank-english-v3.0",
        )

        mock_litellm = MagicMock()
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            await encoder.initialize()
            assert encoder._initialized is True

            # Second call should be no-op
            await encoder.initialize()
            assert encoder._initialized is True

    @pytest.mark.asyncio
    async def test_predict_single_query(self):
        """Test prediction with a single query and multiple documents."""
        encoder = LiteLLMSDKCrossEncoder(
            api_key="test_key",
            model="deepinfra/Qwen3-reranker-8B",
        )

        # Create mock response with results as TypedDicts
        mock_response = MagicMock()
        mock_response.results = [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.7},
            {"index": 2, "relevance_score": 0.5},
        ]

        mock_litellm = MagicMock()
        mock_litellm.arerank = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            await encoder.initialize()

            pairs = [
                ("What is Python?", "Python is a programming language"),
                ("What is Python?", "Python is a snake"),
                ("What is Python?", "Python is a British comedy group"),
            ]

            scores = await encoder.predict(pairs)

            assert len(scores) == 3
            assert scores == [0.9, 0.7, 0.5]

            # Verify arerank was called correctly
            mock_litellm.arerank.assert_called_once()
            call_args = mock_litellm.arerank.call_args
            assert call_args.kwargs["model"] == "deepinfra/Qwen3-reranker-8B"
            assert call_args.kwargs["query"] == "What is Python?"
            assert len(call_args.kwargs["documents"]) == 3

    @pytest.mark.asyncio
    async def test_predict_multiple_queries(self):
        """Test prediction with multiple different queries (grouped efficiently)."""
        encoder = LiteLLMSDKCrossEncoder(
            api_key="test_key",
            model="cohere/rerank-english-v3.0",
        )

        # First query response
        mock_response1 = MagicMock()
        mock_response1.results = [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.7},
        ]

        # Second query response
        mock_response2 = MagicMock()
        mock_response2.results = [
            {"index": 0, "relevance_score": 0.8},
        ]

        mock_litellm = MagicMock()
        mock_litellm.arerank = AsyncMock(side_effect=[mock_response1, mock_response2])

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
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

            # Verify arerank was called twice (once per unique query)
            assert mock_litellm.arerank.call_count == 2

    @pytest.mark.asyncio
    async def test_predict_empty_pairs(self):
        """Test prediction with empty input."""
        encoder = LiteLLMSDKCrossEncoder(
            api_key="test_key",
            model="cohere/rerank-english-v3.0",
        )

        mock_litellm = MagicMock()
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            await encoder.initialize()
            scores = await encoder.predict([])
            assert scores == []

    @pytest.mark.asyncio
    async def test_predict_not_initialized(self):
        """Test that predict fails if encoder not initialized."""
        encoder = LiteLLMSDKCrossEncoder(
            api_key="test_key",
            model="cohere/rerank-english-v3.0",
        )

        pairs = [("query", "document")]

        with pytest.raises(RuntimeError, match="not initialized"):
            await encoder.predict(pairs)

    @pytest.mark.asyncio
    async def test_predict_error_handling(self):
        """Test that errors during prediction are handled gracefully."""
        encoder = LiteLLMSDKCrossEncoder(
            api_key="test_key",
            model="cohere/rerank-english-v3.0",
        )

        # Mock litellm to raise an error
        mock_litellm = MagicMock()
        mock_litellm.arerank = AsyncMock(side_effect=Exception("API Error"))

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            await encoder.initialize()

            pairs = [
                ("What is Python?", "Python is a programming language"),
            ]

            scores = await encoder.predict(pairs)

            # Should return zeros for failed queries
            assert scores == [0.0]

    @pytest.mark.asyncio
    async def test_provider_specific_api_keys(self):
        """Test that provider-specific API keys are set correctly."""
        test_cases = [
            ("cohere/rerank-english-v3.0", "COHERE_API_KEY"),
            ("deepinfra/Qwen3-reranker-8B", "DEEPINFRA_API_KEY"),
            ("together_ai/Salesforce/Llama-Rank-V1", "TOGETHERAI_API_KEY"),
            ("huggingface/BAAI/bge-reranker-v2-m3", "HUGGINGFACE_API_KEY"),
            ("jina_ai/jina-reranker-v2", "JINA_API_KEY"),
            ("voyage/rerank-2", "VOYAGE_API_KEY"),
        ]

        for model, expected_env_var in test_cases:
            # Clear env var before test
            if expected_env_var in os.environ:
                del os.environ[expected_env_var]

            encoder = LiteLLMSDKCrossEncoder(
                api_key="test_key_123",
                model=model,
            )

            mock_litellm = MagicMock()
            with patch.dict("sys.modules", {"litellm": mock_litellm}):
                await encoder.initialize()
                assert os.environ.get(expected_env_var) == "test_key_123"

    @pytest.mark.asyncio
    async def test_custom_api_base(self):
        """Test that custom API base URL is set correctly."""
        encoder = LiteLLMSDKCrossEncoder(
            api_key="test_key",
            model="cohere/rerank-english-v3.0",
            api_base="https://custom.api.example.com",
        )

        mock_litellm = MagicMock()
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            await encoder.initialize()
            assert os.environ.get("LITELLM_API_BASE") == "https://custom.api.example.com"

    @pytest.mark.asyncio
    async def test_fallback_to_sync_rerank(self):
        """Test fallback to sync rerank when arerank is not available."""
        encoder = LiteLLMSDKCrossEncoder(
            api_key="test_key",
            model="cohere/rerank-english-v3.0",
        )

        # Mock litellm without arerank (only sync rerank)
        mock_litellm = MagicMock()
        # Don't set arerank attribute
        mock_litellm.arerank = None
        delattr(mock_litellm, "arerank")

        mock_response = MagicMock()
        mock_response.results = [
            {"index": 0, "relevance_score": 0.9},
        ]
        mock_litellm.rerank = MagicMock(return_value=mock_response)

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            await encoder.initialize()

            pairs = [("query", "document")]
            scores = await encoder.predict(pairs)

            assert scores == [0.9]
            # Verify sync rerank was called
            mock_litellm.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_response_with_direct_score_list(self):
        """Test handling of response format with direct score list."""
        encoder = LiteLLMSDKCrossEncoder(
            api_key="test_key",
            model="some-provider/model",
        )

        # Mock litellm to return direct list of scores
        mock_litellm = MagicMock()
        mock_litellm.arerank = AsyncMock(return_value=[0.9, 0.7, 0.5])

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            await encoder.initialize()

            pairs = [
                ("query", "doc1"),
                ("query", "doc2"),
                ("query", "doc3"),
            ]

            scores = await encoder.predict(pairs)

            assert scores == [0.9, 0.7, 0.5]


class TestFactoryFunction:
    """Test suite for create_cross_encoder_from_env factory function."""

    @pytest.mark.asyncio
    async def test_create_litellm_sdk_from_env(self):
        """Test creating LiteLLM SDK cross-encoder from environment variables."""
        env_vars = {
            "HINDSIGHT_API_RERANKER_PROVIDER": "litellm-sdk",
            "HINDSIGHT_API_RERANKER_LITELLM_SDK_API_KEY": "test_key",
            "HINDSIGHT_API_RERANKER_LITELLM_SDK_MODEL": "deepinfra/Qwen3-reranker-8B",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            # Need to reload config to pick up env vars
            from hindsight_api.config import HindsightConfig

            config = HindsightConfig.from_env()

            with patch("hindsight_api.config.get_config", return_value=config):
                encoder = create_cross_encoder_from_env()

                assert isinstance(encoder, LiteLLMSDKCrossEncoder)
                assert encoder.api_key == "test_key"
                assert encoder.model == "deepinfra/Qwen3-reranker-8B"

    @pytest.mark.asyncio
    async def test_create_litellm_sdk_missing_api_key(self):
        """Test that factory raises error when API key is missing."""
        env_vars = {
            "HINDSIGHT_API_RERANKER_PROVIDER": "litellm-sdk",
            "HINDSIGHT_API_RERANKER_LITELLM_SDK_MODEL": "deepinfra/Qwen3-reranker-8B",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            # Remove API key if set
            if "HINDSIGHT_API_RERANKER_LITELLM_SDK_API_KEY" in os.environ:
                del os.environ["HINDSIGHT_API_RERANKER_LITELLM_SDK_API_KEY"]

            from hindsight_api.config import HindsightConfig

            config = HindsightConfig.from_env()

            with patch("hindsight_api.config.get_config", return_value=config):
                with pytest.raises(ValueError, match="HINDSIGHT_API_RERANKER_LITELLM_SDK_API_KEY is required"):
                    create_cross_encoder_from_env()

    @pytest.mark.asyncio
    async def test_create_litellm_sdk_with_custom_api_base(self):
        """Test creating LiteLLM SDK cross-encoder with custom API base."""
        env_vars = {
            "HINDSIGHT_API_RERANKER_PROVIDER": "litellm-sdk",
            "HINDSIGHT_API_RERANKER_LITELLM_SDK_API_KEY": "test_key",
            "HINDSIGHT_API_RERANKER_LITELLM_SDK_MODEL": "cohere/rerank-english-v3.0",
            "HINDSIGHT_API_RERANKER_LITELLM_SDK_API_BASE": "https://custom.api.example.com",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            from hindsight_api.config import HindsightConfig

            config = HindsightConfig.from_env()

            with patch("hindsight_api.config.get_config", return_value=config):
                encoder = create_cross_encoder_from_env()

                assert isinstance(encoder, LiteLLMSDKCrossEncoder)
                assert encoder.api_base == "https://custom.api.example.com"


class TestLiteLLMSDKCohereCrossEncoder:
    """Tests for LiteLLM SDK calling Cohere (runs in CI with COHERE_API_KEY)."""

    @pytest.fixture(scope="class")
    def litellm_cohere_cross_encoder(self):
        """Create LiteLLM SDK cross-encoder instance for Cohere."""
        if not os.environ.get("COHERE_API_KEY"):
            pytest.skip("Cohere API key not available (set COHERE_API_KEY)")

        encoder = LiteLLMSDKCrossEncoder(
            api_key=os.environ["COHERE_API_KEY"],
            model="cohere/rerank-english-v3.0",
        )
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(encoder.initialize())
        finally:
            loop.close()
        return encoder

    def test_litellm_sdk_cohere_initialization(self, litellm_cohere_cross_encoder):
        """Test that LiteLLM SDK Cohere cross-encoder initializes correctly."""
        assert litellm_cohere_cross_encoder.provider_name == "litellm-sdk"
        assert litellm_cohere_cross_encoder.model == "cohere/rerank-english-v3.0"

    @pytest.mark.asyncio
    async def test_litellm_sdk_cohere_predict(self, litellm_cohere_cross_encoder):
        """Test that LiteLLM SDK can call Cohere rerank API."""
        pairs = [
            ("What is the capital of France?", "Paris is the capital of France."),
            ("What is the capital of France?", "The Eiffel Tower is in Paris."),
            ("What is the capital of France?", "Python is a programming language."),
        ]
        scores = await litellm_cohere_cross_encoder.predict(pairs)

        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        # The first result should be most relevant
        assert scores[0] > scores[2], "Direct answer should score higher than unrelated text"
        # All scores should be in valid range
        assert all(0.0 <= score <= 1.0 for score in scores)


class TestIntegration:
    """Integration tests with real API (optional - requires API keys)."""

    @pytest.mark.skipif(
        not os.environ.get("DEEPINFRA_API_KEY"),
        reason="DEEPINFRA_API_KEY not set - skipping integration test",
    )
    @pytest.mark.asyncio
    async def test_real_deepinfra_api(self):
        """Test with real DeepInfra API (requires DEEPINFRA_API_KEY env var)."""
        encoder = LiteLLMSDKCrossEncoder(
            api_key=os.environ["DEEPINFRA_API_KEY"],
            model="deepinfra/Qwen3-reranker-8B",
        )

        await encoder.initialize()

        pairs = [
            ("What is Python?", "Python is a high-level programming language"),
            ("What is Python?", "Python is a species of snake"),
            ("What is Python?", "Python is unrelated text about cars"),
        ]

        scores = await encoder.predict(pairs)

        # First doc should have highest score (most relevant)
        assert len(scores) == 3
        assert scores[0] > scores[1]
        assert scores[1] > scores[2]
        assert all(0.0 <= score <= 1.0 for score in scores)
