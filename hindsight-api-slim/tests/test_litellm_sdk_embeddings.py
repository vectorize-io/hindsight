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
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args),
        ):
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
                api_key="test_key",
                encoding_format="float",
            )

    async def test_initialization_without_api_key(self, mock_litellm):
        """Test initialization without api_key (e.g. AWS Bedrock with IAM auth)."""
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args),
        ):
            emb = LiteLLMSDKEmbeddings(
                model="bedrock/amazon.titan-embed-text-v2:0",
                batch_size=100,
                timeout=60.0,
            )

            await emb.initialize()

            assert emb._litellm is not None
            assert emb._dimension == 768

            call_kwargs = mock_litellm.aembedding.call_args.kwargs
            assert "api_key" not in call_kwargs

    async def test_encode_without_api_key(self, mock_litellm):
        """Test encode omits api_key when not set (IAM/ambient credentials)."""
        emb = LiteLLMSDKEmbeddings(
            model="bedrock/amazon.titan-embed-text-v2:0",
        )
        emb._litellm = mock_litellm
        emb._dimension = 768

        mock_litellm.embedding.return_value.data = [
            {"embedding": [0.5] * 768, "index": 0},
        ]

        result = emb.encode(["Hello world"])

        assert len(result) == 1
        call_kwargs = mock_litellm.embedding.call_args.kwargs
        assert "api_key" not in call_kwargs

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
            api_key="test_key",
            encoding_format="float",
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
            mock_response.data = [{"embedding": [float(i)] * 768, "index": i} for i in range(len(input))]
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

    async def test_custom_api_base(self, mock_litellm):
        """Test custom API base URL is passed to embedding calls."""
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args),
        ):
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

            # Verify api_base is passed to aembedding
            mock_litellm.aembedding.assert_called_once()
            call_args = mock_litellm.aembedding.call_args
            assert call_args.kwargs["api_base"] == "https://custom.api.com"

            # Test encode also passes api_base
            mock_litellm.embedding.return_value.data = [{"embedding": [0.1] * 768, "index": 0}]
            emb.encode(["test"])

            mock_litellm.embedding.assert_called_once()
            call_args = mock_litellm.embedding.call_args
            assert call_args.kwargs["api_base"] == "https://custom.api.com"

    async def test_output_dimensions_passed_when_set(self, mock_litellm):
        """Test output dimensions are passed to LiteLLM when configured."""
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args),
        ):
            emb = LiteLLMSDKEmbeddings(
                api_key="test_key",
                model="cohere/embed-english-v3.0",
                output_dimensions=768,
            )
            await emb.initialize()

            init_call_args = mock_litellm.aembedding.call_args
            assert init_call_args.kwargs["dimensions"] == 768
            assert "allowed_openai_params" not in init_call_args.kwargs

            mock_litellm.embedding.return_value.data = [{"embedding": [0.1] * 768, "index": 0}]
            emb.encode(["test"])

            encode_call_args = mock_litellm.embedding.call_args
            assert encode_call_args.kwargs["dimensions"] == 768
            assert "allowed_openai_params" not in encode_call_args.kwargs

    async def test_openai_output_dimensions_allows_litellm_dimensions_param(self, mock_litellm):
        """OpenAI-compatible custom models need LiteLLM's explicit dimensions allow-list."""
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args),
        ):
            emb = LiteLLMSDKEmbeddings(
                api_key="test_key",
                model="openai/Qwen3-Embedding-4B-4bit-DWQ",
                api_base="https://custom.api.com",
                output_dimensions=2000,
            )
            await emb.initialize()

            init_call_args = mock_litellm.aembedding.call_args
            assert init_call_args.kwargs["dimensions"] == 2000
            assert init_call_args.kwargs["allowed_openai_params"] == ["dimensions"]

            mock_litellm.embedding.return_value.data = [{"embedding": [0.1] * 2000, "index": 0}]
            emb.encode(["test"])

            encode_call_args = mock_litellm.embedding.call_args
            assert encode_call_args.kwargs["dimensions"] == 2000
            assert encode_call_args.kwargs["allowed_openai_params"] == ["dimensions"]

    async def test_output_dimensions_omitted_when_unset(self, mock_litellm):
        """Test output dimensions are omitted when not configured."""
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args),
        ):
            emb = LiteLLMSDKEmbeddings(
                api_key="test_key",
                model="cohere/embed-english-v3.0",
            )
            await emb.initialize()

            init_call_args = mock_litellm.aembedding.call_args
            assert "dimensions" not in init_call_args.kwargs

            mock_litellm.embedding.return_value.data = [{"embedding": [0.1] * 768, "index": 0}]
            emb.encode(["test"])

            encode_call_args = mock_litellm.embedding.call_args
            assert "dimensions" not in encode_call_args.kwargs

    async def test_output_dimensions_and_api_base_passed_when_both_set(self, mock_litellm):
        """Test both dimensions and api_base are forwarded when configured together."""
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args),
        ):
            emb = LiteLLMSDKEmbeddings(
                api_key="test_key",
                model="cohere/embed-english-v3.0",
                api_base="https://custom.api.com",
                output_dimensions=768,
            )
            await emb.initialize()

            init_call_args = mock_litellm.aembedding.call_args
            assert init_call_args.kwargs["api_base"] == "https://custom.api.com"
            assert init_call_args.kwargs["dimensions"] == 768
            assert "allowed_openai_params" not in init_call_args.kwargs

            mock_litellm.embedding.return_value.data = [{"embedding": [0.1] * 768, "index": 0}]
            emb.encode(["test"])

            encode_call_args = mock_litellm.embedding.call_args
            assert encode_call_args.kwargs["api_base"] == "https://custom.api.com"
            assert encode_call_args.kwargs["dimensions"] == 768
            assert "allowed_openai_params" not in encode_call_args.kwargs

    async def test_encoding_format_default_is_float(self, mock_litellm):
        """Test that encoding_format defaults to 'float' for backwards compatibility."""
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args),
        ):
            emb = LiteLLMSDKEmbeddings(
                api_key="test_key",
                model="cohere/embed-english-v3.0",
            )
            await emb.initialize()

            init_call_args = mock_litellm.aembedding.call_args
            assert init_call_args.kwargs["encoding_format"] == "float"

            mock_litellm.embedding.return_value.data = [{"embedding": [0.1] * 768, "index": 0}]
            emb.encode(["test"])

            encode_call_args = mock_litellm.embedding.call_args
            assert encode_call_args.kwargs["encoding_format"] == "float"

    async def test_encoding_format_omitted_when_none(self, mock_litellm):
        """Test that encoding_format is omitted when set to None (for Voyage AI, Gemini)."""
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args),
        ):
            emb = LiteLLMSDKEmbeddings(
                api_key="test_key",
                model="voyage/voyage-4-large",
                encoding_format=None,
            )
            await emb.initialize()

            init_call_args = mock_litellm.aembedding.call_args
            assert "encoding_format" not in init_call_args.kwargs

            mock_litellm.embedding.return_value.data = [{"embedding": [0.1] * 768, "index": 0}]
            emb.encode(["test"])

            encode_call_args = mock_litellm.embedding.call_args
            assert "encoding_format" not in encode_call_args.kwargs

    async def test_encoding_format_omitted_when_empty_string(self, mock_litellm):
        """Test that encoding_format is omitted when set to empty string."""
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args),
        ):
            emb = LiteLLMSDKEmbeddings(
                api_key="test_key",
                model="gemini/gemini-embedding-2",
                encoding_format="",
            )
            await emb.initialize()

            init_call_args = mock_litellm.aembedding.call_args
            assert "encoding_format" not in init_call_args.kwargs

    async def test_openai_invalid_output_dimensions_raises(self, mock_litellm):
        """Invalid dimensions fail during initialize() (probe call), not per HTTP request.

        MemoryEngine runs this at app lifespan startup; the process typically fails to become
        ready rather than returning a JSON error for a single API call. The RuntimeError
        message should still chain the underlying provider/LiteLLM detail for logs.
        """
        mock_litellm.aembedding.side_effect = Exception("invalid dimensions for model")

        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args),
        ):
            emb = LiteLLMSDKEmbeddings(
                api_key="test_key",
                model="openai/text-embedding-3-small",
                output_dimensions=9999,
            )

            with pytest.raises(
                RuntimeError,
                match="Failed to initialize LiteLLM SDK embeddings:.*invalid dimensions for model",
            ):
                await emb.initialize()

    async def test_resolve_max_input_tokens_from_known_table(self, mock_litellm):
        """Known embedding models resolve their input-token limit from the table."""
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args),
        ):
            emb = LiteLLMSDKEmbeddings(model="bedrock/amazon.titan-embed-text-v2:0")
            await emb.initialize()

            assert emb._max_input_tokens == 8192
            mock_litellm.get_model_info.assert_not_called()

    async def test_resolve_max_input_tokens_falls_back_to_litellm_metadata(self, mock_litellm):
        """Unknown models fall back to litellm.get_model_info()['max_input_tokens']."""
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 2048}
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args),
        ):
            emb = LiteLLMSDKEmbeddings(model="some-provider/mystery-embed")
            await emb.initialize()

            assert emb._max_input_tokens == 2048
            mock_litellm.get_model_info.assert_called_once_with("some-provider/mystery-embed")

    async def test_resolve_max_input_tokens_unknown_is_none(self, mock_litellm):
        """When neither the table nor litellm metadata yields a limit, it stays None."""
        mock_litellm.get_model_info.side_effect = Exception("no metadata")
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: mock_litellm if name == "litellm" else __import__(name, *args),
        ):
            emb = LiteLLMSDKEmbeddings(model="some-provider/mystery-embed")
            await emb.initialize()

            assert emb._max_input_tokens is None

    async def test_encode_truncates_oversized_input(self, mock_litellm):
        """Oversized single input is shortened to the model's token budget before embedding."""
        oversized_tokens = list(range(9000))
        mock_litellm.encode.return_value = oversized_tokens
        mock_litellm.decode.return_value = "truncated-content"
        mock_litellm.embedding.return_value.data = [{"embedding": [0.1] * 768, "index": 0}]

        emb = LiteLLMSDKEmbeddings(model="bedrock/amazon.titan-embed-text-v2:0")
        emb._litellm = mock_litellm
        emb._dimension = 768
        emb._max_input_tokens = 8192

        result = emb.encode(["x" * 100000])

        assert len(result) == 1
        decode_kwargs = mock_litellm.decode.call_args.kwargs
        assert len(decode_kwargs["tokens"]) == int(8192 * 0.95)
        embed_kwargs = mock_litellm.embedding.call_args.kwargs
        assert embed_kwargs["input"] == ["truncated-content"]

    async def test_encode_oversized_falls_back_to_char_cut_when_decode_empty(self, mock_litellm):
        """If decode() yields an empty string, an oversized input is still cut (never sent whole).

        Regression guard: a decode that returns "" must not let the original
        oversized text through — that is the exact #2501 failure mode.
        """
        mock_litellm.encode.return_value = list(range(9000))
        mock_litellm.decode.return_value = ""  # decode unavailable / empty
        mock_litellm.embedding.return_value.data = [{"embedding": [0.1] * 768, "index": 0}]

        emb = LiteLLMSDKEmbeddings(model="bedrock/amazon.titan-embed-text-v2:0")
        emb._litellm = mock_litellm
        emb._dimension = 768
        emb._max_input_tokens = 8192

        budget = int(8192 * 0.95)
        # Dense text where char count ~ token count (each "char" is one token
        # in this mock: encode returns 9000 tokens for a 9000-char string), so a
        # fixed budget*4 char cut would wrongly pass it through. The proportional
        # fallback must cut to ~budget chars here.
        dense = "d" * 9000
        result = emb.encode([dense])

        assert len(result) == 1
        sent = mock_litellm.embedding.call_args.kwargs["input"][0]
        # proportional cut: len(text) * budget / token_count = 9000 * budget / 9000 = budget
        expected = int(9000 * budget / 9000)
        assert sent == "d" * expected
        assert len(sent) < len(dense)

    async def test_encode_preserves_input_order_and_count_in_mixed_batch(self, mock_litellm):
        """A mixed-size batch keeps 1:1 input->vector alignment; only oversized inputs change."""

        # index 0 oversized, index 1 small, index 2 oversized.
        def fake_encode(model, text):
            return list(range(9000)) if text.startswith("BIG") else list(range(5))

        mock_litellm.encode.side_effect = fake_encode
        mock_litellm.decode.return_value = "CUT"

        def fake_embedding(model, input, **kwargs):
            resp = MagicMock()
            resp.data = [{"embedding": [float(i)] * 768, "index": i} for i in range(len(input))]
            return resp

        mock_litellm.embedding.side_effect = fake_embedding

        emb = LiteLLMSDKEmbeddings(model="bedrock/amazon.titan-embed-text-v2:0", batch_size=100)
        emb._litellm = mock_litellm
        emb._dimension = 768
        emb._max_input_tokens = 8192

        result = emb.encode(["BIG-a", "small-b", "BIG-c"])

        assert len(result) == 3  # 1:1 alignment preserved
        sent = mock_litellm.embedding.call_args.kwargs["input"]
        assert sent[0] == "CUT"  # oversized truncated
        assert sent[1] == "small-b"  # small untouched, position preserved
        assert sent[2] == "CUT"  # oversized truncated

    async def test_encode_does_not_truncate_within_budget(self, mock_litellm):
        """Inputs already under the token budget are sent unchanged (no decode)."""
        mock_litellm.encode.return_value = list(range(10))
        mock_litellm.embedding.return_value.data = [{"embedding": [0.1] * 768, "index": 0}]

        emb = LiteLLMSDKEmbeddings(model="bedrock/amazon.titan-embed-text-v2:0")
        emb._litellm = mock_litellm
        emb._dimension = 768
        emb._max_input_tokens = 8192

        result = emb.encode(["short text"])

        assert len(result) == 1
        mock_litellm.decode.assert_not_called()
        embed_kwargs = mock_litellm.embedding.call_args.kwargs
        assert embed_kwargs["input"] == ["short text"]

    async def test_encode_skips_truncation_when_limit_unknown(self, mock_litellm):
        """When no input-token limit is known, inputs are sent unchanged (prior behavior)."""
        mock_litellm.embedding.return_value.data = [{"embedding": [0.1] * 768, "index": 0}]

        emb = LiteLLMSDKEmbeddings(model="some-provider/mystery-embed")
        emb._litellm = mock_litellm
        emb._dimension = 768
        emb._max_input_tokens = None

        big_text = "x" * 100000
        result = emb.encode([big_text])

        assert len(result) == 1
        mock_litellm.encode.assert_not_called()
        mock_litellm.decode.assert_not_called()
        embed_kwargs = mock_litellm.embedding.call_args.kwargs
        assert embed_kwargs["input"] == [big_text]


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

    def test_create_from_env_without_api_key(self, monkeypatch):
        """Test that litellm-sdk works without an API key (e.g. AWS Bedrock with IAM)."""
        mock_config = MagicMock()
        mock_config.embeddings_provider = "litellm-sdk"
        mock_config.embeddings_litellm_sdk_api_key = None
        mock_config.embeddings_litellm_sdk_model = "bedrock/amazon.titan-embed-text-v2:0"
        mock_config.embeddings_litellm_sdk_api_base = None

        with patch("hindsight_api.config.get_config", return_value=mock_config):
            embeddings = create_embeddings_from_env()

            assert isinstance(embeddings, LiteLLMSDKEmbeddings)
            assert embeddings.api_key is None
            assert embeddings.model == "bedrock/amazon.titan-embed-text-v2:0"

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

    def test_create_from_env_with_output_dimensions(self, monkeypatch):
        """Test creating embeddings with configured output dimensions."""
        mock_config = MagicMock()
        mock_config.embeddings_provider = "litellm-sdk"
        mock_config.embeddings_litellm_sdk_api_key = "test_key"
        mock_config.embeddings_litellm_sdk_model = "gemini/gemini-embedding-2"
        mock_config.embeddings_litellm_sdk_api_base = None
        mock_config.embeddings_litellm_sdk_output_dimensions = 768

        with patch("hindsight_api.config.get_config", return_value=mock_config):
            embeddings = create_embeddings_from_env()

            assert isinstance(embeddings, LiteLLMSDKEmbeddings)
            assert embeddings.output_dimensions == 768


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
        await emb.initialize()
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
