"""
Tests for LiteLLM SDK embeddings implementation.

These tests cover:
1. Initialization (success, missing package, missing API key, idempotent)
2. Encode (single text, multiple texts, batching, error handling)
3. Provider-specific configuration (Cohere, OpenAI, etc.)
4. Factory function (create from env, validation errors)
5. Dimension detection
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.config import (
    ENV_EMBEDDINGS_LITELLM_SDK_API_KEY,
    ENV_EMBEDDINGS_LITELLM_SDK_MODEL,
    ENV_EMBEDDINGS_PROVIDER,
    HindsightConfig,
)
from hindsight_api.engine.embeddings import LiteLLMSDKEmbeddings, create_embeddings_from_env


class TestLiteLLMSDKEmbeddings:
    """Unit tests for LiteLLMSDKEmbeddings with mocked litellm responses."""

    @pytest.fixture
    def mock_litellm(self):
        """Mock litellm module."""
        mock = MagicMock()

        # Mock aembedding (async) for initialization
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1] * 768, "index": 0}]
        mock.aembedding = AsyncMock(return_value=mock_response)

        # Mock embedding (sync) for encode
        mock_sync_response = MagicMock()
        mock_sync_response.data = [
            {"embedding": [0.1] * 768, "index": 0},
            {"embedding": [0.2] * 768, "index": 1},
        ]
        mock.embedding = MagicMock(return_value=mock_sync_response)

        return mock

    @pytest.fixture
    async def embeddings(self, mock_litellm):
        """Create initialized LiteLLMSDKEmbeddings instance."""
        emb = LiteLLMSDKEmbeddings(
            api_key="test_key",
            model="cohere/embed-english-v3.0",
            api_base=None,
            batch_size=100,
            timeout=60.0,
        )
        # Manually set the mock (simulating successful initialization)
        emb._litellm = mock_litellm
        emb._dimension = 768
        return emb

    async def test_initialization_success(self, mock_litellm):
        """Test successful initialization."""
        with patch("builtins.__import__", side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args)):
            emb = LiteLLMSDKEmbeddings(
                api_key="test_key",
                model="cohere/embed-english-v3.0",
                api_base=None,
                batch_size=100,
                timeout=60.0,
            )

            assert emb._litellm is None
            assert emb._dimension is None

            await emb.initialize()

            assert emb._litellm is not None
            assert emb._dimension == 768

            # Verify test embedding was called
            mock_litellm.aembedding.assert_called_once_with(
                model="cohere/embed-english-v3.0",
                input=["test"],
            )

    async def test_initialization_sets_provider_env_vars(self, mock_litellm):
        """Test that initialization sets provider-specific environment variables."""
        with patch("builtins.__import__", side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args)):
            emb = LiteLLMSDKEmbeddings(
                api_key="my_cohere_key",
                model="cohere/embed-english-v3.0",
                api_base=None,
                batch_size=100,
                timeout=60.0,
            )

            await emb.initialize()

            # Check that COHERE_API_KEY was set
            assert os.environ.get("COHERE_API_KEY") == "my_cohere_key"

    async def test_initialization_missing_package(self):
        """Test initialization fails gracefully when litellm is not installed."""
        def mock_import(name, *args):
            if name == "litellm":
                raise ImportError("No module named 'litellm'")
            return __import__(name, *args)

        with patch("builtins.__import__", side_effect=mock_import):
            emb = LiteLLMSDKEmbeddings(
                api_key="test_key",
                model="cohere/embed-english-v3.0",
                api_base=None,
                batch_size=100,
                timeout=60.0,
            )

            with pytest.raises(ImportError, match="litellm is required"):
                await emb.initialize()

    async def test_initialization_idempotent(self, embeddings, mock_litellm):
        """Test that calling initialize() multiple times is safe."""
        # embeddings._litellm is already set in fixture
        assert embeddings._litellm is not None

        # Call again
        await embeddings.initialize()

        # Should still have same litellm instance
        assert embeddings._litellm is not None

    async def test_encode_single_text(self, embeddings, mock_litellm):
        """Test encoding a single text."""
        # Set up mock response
        mock_litellm.embedding.return_value.data = [
            {"embedding": [0.5] * 768, "index": 0},
        ]

        result = embeddings.encode(["Hello world"])

        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) == 768
        assert all(isinstance(x, float) for x in result[0])
        assert all(abs(x - 0.5) < 0.001 for x in result[0])

        # Verify call
        mock_litellm.embedding.assert_called_once_with(
            model="cohere/embed-english-v3.0",
            input=["Hello world"],
        )

    async def test_encode_multiple_texts(self, embeddings, mock_litellm):
        """Test encoding multiple texts."""
        # Set up mock response
        mock_litellm.embedding.return_value.data = [
            {"embedding": [0.1] * 768, "index": 0},
            {"embedding": [0.2] * 768, "index": 1},
            {"embedding": [0.3] * 768, "index": 2},
        ]

        texts = ["First text", "Second text", "Third text"]
        result = embeddings.encode(texts)

        assert isinstance(result, list)
        assert len(result) == 3
        assert len(result[0]) == 768
        assert len(result[1]) == 768
        assert len(result[2]) == 768
        assert all(abs(x - 0.1) < 0.001 for x in result[0])
        assert all(abs(x - 0.2) < 0.001 for x in result[1])
        assert all(abs(x - 0.3) < 0.001 for x in result[2])

    async def test_encode_batching(self, embeddings, mock_litellm):
        """Test that large inputs are batched correctly."""
        # Create embeddings with small batch size
        emb = LiteLLMSDKEmbeddings(
            api_key="test_key",
            model="cohere/embed-english-v3.0",
            api_base=None,
            batch_size=2,  # Small batch for testing
            timeout=60.0,
        )
        emb._litellm = mock_litellm
        emb._initialized = True
        emb._dimension = 768

        # Mock responses for each batch
        def mock_embedding_side_effect(model, input, **kwargs):
            mock_response = MagicMock()
            mock_response.data = [
                {"embedding": [float(i)] * 768, "index": i} for i in range(len(input))
            ]
            return mock_response

        mock_litellm.embedding.side_effect = mock_embedding_side_effect

        # Encode 5 texts (should create 3 batches: 2, 2, 1)
        texts = [f"Text {i}" for i in range(5)]
        result = emb.encode(texts)

        assert isinstance(result, list)
        assert len(result) == 5
        assert all(len(embedding) == 768 for embedding in result)

        # Verify batching: should be called 3 times
        assert mock_litellm.embedding.call_count == 3

        # Verify batch sizes
        calls = mock_litellm.embedding.call_args_list
        assert len(calls[0][1]["input"]) == 2  # First batch
        assert len(calls[1][1]["input"]) == 2  # Second batch
        assert len(calls[2][1]["input"]) == 1  # Third batch

    async def test_encode_empty_list(self, embeddings):
        """Test encoding empty list returns empty list."""
        result = embeddings.encode([])

        assert isinstance(result, list)
        assert len(result) == 0

    async def test_encode_before_initialization(self, mock_litellm):
        """Test that encode raises error if not initialized."""
        emb = LiteLLMSDKEmbeddings(
            api_key="test_key",
            model="cohere/embed-english-v3.0",
            api_base=None,
            batch_size=100,
            timeout=60.0,
        )

        with pytest.raises(RuntimeError, match="not initialized"):
            emb.encode(["test"])

    async def test_encode_error_handling(self, embeddings, mock_litellm):
        """Test error handling during encoding."""
        # Make embedding raise an error
        mock_litellm.embedding.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            embeddings.encode(["test"])

    async def test_dimension_property(self, embeddings):
        """Test dimension property."""
        assert embeddings.dimension == 768

    async def test_dimension_before_initialization(self, mock_litellm):
        """Test dimension raises error if not initialized."""
        emb = LiteLLMSDKEmbeddings(
            api_key="test_key",
            model="cohere/embed-english-v3.0",
            api_base=None,
            batch_size=100,
            timeout=60.0,
        )

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = emb.dimension

    async def test_different_providers(self, mock_litellm):
        """Test different provider configurations."""
        providers = [
            ("cohere/embed-english-v3.0", "COHERE_API_KEY"),
            ("openai/text-embedding-3-small", "OPENAI_API_KEY"),
            ("together_ai/togethercomputer/m2-bert-80M-8k-retrieval", "TOGETHERAI_API_KEY"),
            ("huggingface/sentence-transformers/all-MiniLM-L6-v2", "HUGGINGFACE_API_KEY"),
        ]

        with patch("builtins.__import__", side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args)):
            for model, env_var in providers:
                emb = LiteLLMSDKEmbeddings(
                    api_key=f"test_key_{env_var}",
                    model=model,
                    api_base=None,
                    batch_size=100,
                    timeout=60.0,
                )

                await emb.initialize()

                # Check that provider-specific env var was set
                prefix = model.split("/")[0]
                if prefix in ["cohere", "openai", "together_ai", "huggingface"]:
                    assert env_var in os.environ

    async def test_custom_api_base(self, mock_litellm):
        """Test custom API base URL configuration."""
        with patch("builtins.__import__", side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args)):
            emb = LiteLLMSDKEmbeddings(
                api_key="test_key",
                model="cohere/embed-english-v3.0",
                api_base="https://custom.api.com",
                batch_size=100,
                timeout=60.0,
            )

            await emb.initialize()

            # Verify api_base is set
            assert emb.api_base == "https://custom.api.com"


class TestLiteLLMSDKEmbeddingsFactory:
    """Test the factory function for creating LiteLLM SDK embeddings."""

    def test_create_from_env_success(self, monkeypatch):
        """Test creating embeddings from environment variables."""
        # Mock get_config() to return configured HindsightConfig
        mock_config = MagicMock()
        mock_config.embeddings_provider = "litellm-sdk"
        mock_config.embeddings_litellm_sdk_api_key = "test_key"
        mock_config.embeddings_litellm_sdk_model = "cohere/embed-english-v3.0"
        mock_config.embeddings_litellm_sdk_api_base = None

        with patch("hindsight_api.config.get_config", return_value=mock_config):
            embeddings = create_embeddings_from_env()

            assert isinstance(embeddings, LiteLLMSDKEmbeddings)
            assert embeddings.api_key == "test_key"
            assert embeddings.model == "cohere/embed-english-v3.0"

    def test_create_from_env_missing_api_key(self, monkeypatch):
        """Test that missing API key raises error."""
        # Mock get_config() with missing API key
        mock_config = MagicMock()
        mock_config.embeddings_provider = "litellm-sdk"
        mock_config.embeddings_litellm_sdk_api_key = None  # Missing key
        mock_config.embeddings_litellm_sdk_model = "cohere/embed-english-v3.0"

        with patch("hindsight_api.config.get_config", return_value=mock_config):
            with pytest.raises(ValueError, match=ENV_EMBEDDINGS_LITELLM_SDK_API_KEY):
                create_embeddings_from_env()

    def test_create_from_env_with_api_base(self, monkeypatch):
        """Test creating embeddings with custom API base."""
        # Mock get_config() with custom API base
        mock_config = MagicMock()
        mock_config.embeddings_provider = "litellm-sdk"
        mock_config.embeddings_litellm_sdk_api_key = "test_key"
        mock_config.embeddings_litellm_sdk_model = "cohere/embed-english-v3.0"
        mock_config.embeddings_litellm_sdk_api_base = "https://custom.api.com"

        with patch("hindsight_api.config.get_config", return_value=mock_config):
            embeddings = create_embeddings_from_env()

            assert isinstance(embeddings, LiteLLMSDKEmbeddings)
            assert embeddings.api_base == "https://custom.api.com"


class TestLiteLLMSDKCohereEmbeddings:
    """Integration tests calling real Cohere API (matches CI pattern)."""

    @pytest.fixture
    async def litellm_cohere_embeddings(self):
        """Create embeddings instance with real Cohere API key."""
        if not os.environ.get("COHERE_API_KEY"):
            pytest.skip("Cohere API key not available")

        emb = LiteLLMSDKEmbeddings(
            api_key=os.environ["COHERE_API_KEY"],
            model="cohere/embed-english-v3.0",
            api_base=None,
            batch_size=100,
            timeout=60.0,
        )

        try:
            await emb.initialize()
        except Exception as e:
            # Skip if API key is invalid or initialization fails
            pytest.skip(f"Cohere API initialization failed: {e}")

        return emb

    @pytest.mark.asyncio
    async def test_litellm_sdk_cohere_encode(self, litellm_cohere_embeddings):
        """Test real Cohere API call for embeddings."""
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Python is a popular programming language",
        ]

        result = litellm_cohere_embeddings.encode(texts)

        # Verify result type and shape
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(len(embedding) > 0 for embedding in result)
        assert all(isinstance(x, float) for x in result[0])

        # Verify embeddings are not zeros (common API failure mode)
        for i, embedding in enumerate(result):
            assert not all(abs(x) < 0.0001 for x in embedding), f"Embedding {i} is all zeros"

        # Verify embeddings are normalized (Cohere returns normalized vectors)
        for i, embedding in enumerate(result):
            norm = sum(x * x for x in embedding) ** 0.5
            assert 0.9 < norm < 1.1, f"Embedding {i} norm {norm} is not close to 1.0"

    @pytest.mark.asyncio
    async def test_litellm_sdk_cohere_dimension(self, litellm_cohere_embeddings):
        """Test dimension detection with real Cohere API."""
        dimension = litellm_cohere_embeddings.dimension

        # Cohere embed-english-v3.0 has 1024 dimensions
        assert dimension == 1024

    @pytest.mark.asyncio
    async def test_litellm_sdk_cohere_single_text(self, litellm_cohere_embeddings):
        """Test encoding single text with real Cohere API."""
        result = litellm_cohere_embeddings.encode(["Hello world"])

        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) == 1024
        assert not all(abs(x) < 0.0001 for x in result[0])
