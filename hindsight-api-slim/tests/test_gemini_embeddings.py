"""
Tests for Google embeddings implementation (Gemini API + Vertex AI).

These tests cover:
1. Initialization (Gemini API key, Vertex AI with ADC/service account)
2. Dimension detection via test embedding
3. Output dimensionality configuration
4. Encode (single text, multiple texts, batching, empty list, uninitialized)
5. Provider name and model name normalization
6. Factory function (create from env, validation errors)
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.config import (
    ENV_EMBEDDINGS_GEMINI_API_KEY,
    ENV_EMBEDDINGS_PROVIDER,
    HindsightConfig,
)
from hindsight_api.engine.embeddings import GeminiEmbeddings, create_embeddings_from_env


def _make_mock_embedding(values: list[float]) -> MagicMock:
    emb = MagicMock()
    emb.values = values
    return emb


def _make_mock_embed_result(embeddings_data: list[list[float]]) -> MagicMock:
    result = MagicMock()
    result.embeddings = [_make_mock_embedding(v) for v in embeddings_data]
    return result


def _make_mock_genai(embed_result: Any = None) -> MagicMock:
    if embed_result is None:
        embed_result = _make_mock_embed_result([[0.1] * 768])
    mock_genai = MagicMock()
    mock_client = MagicMock()
    mock_client.models.embed_content = MagicMock(return_value=embed_result)
    mock_genai.Client = MagicMock(return_value=mock_client)
    return mock_genai


def _make_mock_google_module(mock_genai: MagicMock) -> MagicMock:
    mod = MagicMock()
    mod.genai = mock_genai
    mod.genai.types.EmbedContentConfig = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    mod.genai.types.HttpOptions = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    return mod


def _patch_google_import(mock_genai: MagicMock):
    original_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "google":
            return _make_mock_google_module(mock_genai)
        if name == "google.genai":
            return mock_genai
        return original_import(name, *args, **kwargs)

    return patch("builtins.__import__", side_effect=mock_import)


class TestGeminiEmbeddings:
    """Unit tests for GeminiEmbeddings with mocked google.genai."""

    async def test_initialization_api_key_success(self):
        """Test successful Gemini API key initialization."""
        mock_genai = _make_mock_genai()
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key")

        with _patch_google_import(mock_genai):
            await emb.initialize()

        assert emb._client is not None
        assert emb.dimension == 768
        assert emb.provider_name == "google"
        assert emb._is_vertexai is False
        mock_genai.Client.return_value.models.embed_content.assert_called_once()

    async def test_initialization_vertexai_success(self):
        """Test successful Vertex AI initialization."""
        mock_genai = _make_mock_genai()
        emb = GeminiEmbeddings(
            model="gemini-embedding-001",
            vertexai_project_id="test-project",
            vertexai_region="us-central1",
        )

        with _patch_google_import(mock_genai):
            await emb.initialize()

        assert emb._client is not None
        assert emb.dimension == 768
        assert emb.provider_name == "google"
        assert emb._is_vertexai is True
        mock_genai.Client.assert_called_once_with(
            vertexai=True,
            project="test-project",
            location="us-central1",
        )

    async def test_initialization_missing_api_key(self):
        """Test that missing API key raises ValueError when no vertexai_project_id."""
        mock_genai = _make_mock_genai()
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key=None)

        with _patch_google_import(mock_genai):
            with pytest.raises(ValueError, match="requires an API key"):
                await emb.initialize()

    async def test_initialization_vertexai_missing_project_id(self):
        """Test that Vertex AI mode requires project_id."""
        mock_genai = _make_mock_genai()
        emb = GeminiEmbeddings(model="gemini-embedding-001", vertexai_project_id="temp")
        emb.vertexai_project_id = None  # Simulate misconfiguration

        with _patch_google_import(mock_genai):
            with pytest.raises(ValueError, match="is required for Vertex AI"):
                await emb.initialize()

    async def test_initialization_idempotent(self):
        """Test that calling initialize() twice is a no-op."""
        mock_genai = _make_mock_genai()
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key")

        with _patch_google_import(mock_genai):
            await emb.initialize()
            first_client = emb._client
            await emb.initialize()
            assert emb._client is first_client

    async def test_dimension_detection_via_test_embedding(self):
        """Test that dimension is detected via a test embedding call."""
        test_embed = _make_mock_embed_result([[0.5] * 256])
        mock_genai = _make_mock_genai(embed_result=test_embed)
        emb = GeminiEmbeddings(model="some-new-model", api_key="test-key")

        with _patch_google_import(mock_genai):
            await emb.initialize()

        assert emb.dimension == 256

    async def test_output_dimensionality(self):
        """Test that output_dimensionality is passed via EmbedContentConfig."""
        test_embed = _make_mock_embed_result([[0.1] * 256])
        mock_genai = _make_mock_genai(embed_result=test_embed)
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key", output_dimensionality=256)

        with _patch_google_import(mock_genai):
            await emb.initialize()

        assert emb.dimension == 256
        assert emb._embed_config is not None
        call_kwargs = mock_genai.Client.return_value.models.embed_content.call_args
        assert "config" in call_kwargs.kwargs

    async def test_no_output_dimensionality(self):
        """Test that no EmbedContentConfig is built when output_dimensionality is None."""
        mock_genai = _make_mock_genai()
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key", output_dimensionality=None)

        with _patch_google_import(mock_genai):
            await emb.initialize()

        assert emb._embed_config is None
        call_kwargs = mock_genai.Client.return_value.models.embed_content.call_args
        assert "config" not in call_kwargs.kwargs

    async def test_force_ipv4_passes_http_options(self):
        """Test that force_ipv4 configures the Gemini client with custom HTTP options."""
        mock_genai = _make_mock_genai()
        mock_transport = MagicMock()
        mock_httpx_client = MagicMock()
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key", force_ipv4=True)

        with _patch_google_import(mock_genai):
            with patch("httpx.HTTPTransport", return_value=mock_transport) as mock_http_transport:
                with patch("httpx.Client", return_value=mock_httpx_client) as mock_http_client:
                    await emb.initialize()

        mock_http_transport.assert_called_once_with(local_address="0.0.0.0")
        mock_http_client.assert_called_once_with(timeout=10, transport=mock_transport)
        assert emb._httpx_client is mock_httpx_client
        assert "http_options" in mock_genai.Client.call_args.kwargs

    def test_auto_detect_vertexai(self):
        """Test that _is_vertexai is auto-detected from vertexai_project_id."""
        assert GeminiEmbeddings(model="m", api_key="k")._is_vertexai is False
        assert GeminiEmbeddings(model="m", vertexai_project_id="p")._is_vertexai is True

    def test_encode_single_text(self):
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key")
        mock_client = MagicMock()
        mock_client.models.embed_content = MagicMock(return_value=_make_mock_embed_result([[0.1, 0.2, 0.3]]))
        emb._client = mock_client
        emb._dimension = 3

        assert emb.encode(["hello"]) == [[0.1, 0.2, 0.3]]

    def test_encode_multiple_texts(self):
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key")
        mock_client = MagicMock()
        mock_client.models.embed_content = MagicMock(
            return_value=_make_mock_embed_result([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        )
        emb._client = mock_client
        emb._dimension = 2

        result = emb.encode(["a", "b", "c"])
        assert len(result) == 3
        assert result[1] == [0.3, 0.4]

    def test_encode_batching(self):
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key", batch_size=2)
        mock_client = MagicMock()
        mock_client.models.embed_content = MagicMock(
            side_effect=[_make_mock_embed_result([[0.1], [0.2]]), _make_mock_embed_result([[0.3]])]
        )
        emb._client = mock_client
        emb._dimension = 1

        assert emb.encode(["a", "b", "c"]) == [[0.1], [0.2], [0.3]]
        assert mock_client.models.embed_content.call_count == 2

    def test_encode_passes_config(self):
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key")
        mock_client = MagicMock()
        mock_client.models.embed_content = MagicMock(return_value=_make_mock_embed_result([[0.1, 0.2]]))
        emb._client = mock_client
        emb._dimension = 2
        emb._embed_config = MagicMock()

        emb.encode(["hello"])
        assert mock_client.models.embed_content.call_args.kwargs["config"] is emb._embed_config

    def test_encode_empty_list(self):
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key")
        emb._client = MagicMock()
        emb._dimension = 768
        assert emb.encode([]) == []

    def test_encode_before_initialization(self):
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key")
        with pytest.raises(RuntimeError, match="not initialized"):
            emb.encode(["test"])

    def test_dimension_before_initialization(self):
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key")
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = emb.dimension

    def test_provider_name_always_google(self):
        assert GeminiEmbeddings(model="m", api_key="k").provider_name == "google"
        assert GeminiEmbeddings(model="m", vertexai_project_id="p").provider_name == "google"

    def test_vertexai_strips_google_prefix(self):
        mock_genai = _make_mock_genai()
        emb = GeminiEmbeddings(model="google/gemini-embedding-001", vertexai_project_id="test-project")
        emb._init_vertexai(mock_genai)
        assert emb.model == "gemini-embedding-001"

    def test_default_region(self):
        emb = GeminiEmbeddings(model="m", vertexai_project_id="proj")
        assert emb.vertexai_region == "us-central1"

    def test_custom_region(self):
        emb = GeminiEmbeddings(model="m", vertexai_project_id="proj", vertexai_region="europe-west1")
        assert emb.vertexai_region == "europe-west1"


class TestGeminiEmbeddingsFactory:
    """Tests for create_embeddings_from_env() with 'google' provider."""

    def _make_config(self, **overrides) -> HindsightConfig:
        from dataclasses import fields

        defaults = {}
        for f in fields(HindsightConfig):
            if f.type == "str":
                defaults[f.name] = ""
            elif f.type == "str | None":
                defaults[f.name] = None
            elif f.type == "int":
                defaults[f.name] = 0
            elif f.type == "int | None":
                defaults[f.name] = None
            elif f.type == "float":
                defaults[f.name] = 0.0
            elif f.type == "float | None":
                defaults[f.name] = None
            elif f.type == "bool":
                defaults[f.name] = False
            elif f.type == "list | None":
                defaults[f.name] = None
            else:
                defaults[f.name] = None

        defaults["embeddings_provider"] = "google"
        defaults["embeddings_gemini_api_key"] = "test-key"
        defaults["embeddings_gemini_model"] = "gemini-embedding-001"
        defaults["embeddings_gemini_output_dimensionality"] = 768
        defaults["embeddings_gemini_force_ipv4"] = False
        defaults["embeddings_vertexai_project_id"] = None
        defaults["embeddings_vertexai_region"] = None
        defaults["embeddings_vertexai_service_account_key"] = None

        defaults.update(overrides)
        return HindsightConfig(**defaults)

    def test_create_with_api_key(self):
        config = self._make_config()
        with patch("hindsight_api.config.get_config", return_value=config):
            emb = create_embeddings_from_env()
        assert isinstance(emb, GeminiEmbeddings)
        assert emb.provider_name == "google"
        assert emb.api_key == "test-key"
        assert emb._is_vertexai is False
        assert emb.force_ipv4 is False

    def test_create_with_force_ipv4(self):
        config = self._make_config(embeddings_gemini_force_ipv4=True)
        with patch("hindsight_api.config.get_config", return_value=config):
            emb = create_embeddings_from_env()
        assert isinstance(emb, GeminiEmbeddings)
        assert emb.force_ipv4 is True

    def test_create_with_vertexai(self):
        config = self._make_config(
            embeddings_gemini_api_key=None,
            embeddings_vertexai_project_id="my-project",
            embeddings_vertexai_region="us-east1",
        )
        with patch("hindsight_api.config.get_config", return_value=config):
            emb = create_embeddings_from_env()
        assert isinstance(emb, GeminiEmbeddings)
        assert emb._is_vertexai is True
        assert emb.api_key is None
        assert emb.vertexai_project_id == "my-project"

    def test_create_missing_all_credentials(self):
        config = self._make_config(embeddings_gemini_api_key=None, embeddings_vertexai_project_id=None)
        with patch("hindsight_api.config.get_config", return_value=config):
            with pytest.raises(ValueError, match="is required"):
                create_embeddings_from_env()

    def test_vertexai_takes_priority(self):
        config = self._make_config(embeddings_gemini_api_key="key", embeddings_vertexai_project_id="proj")
        with patch("hindsight_api.config.get_config", return_value=config):
            emb = create_embeddings_from_env()
        assert emb._is_vertexai is True
        assert emb.api_key is None

    def test_create_with_custom_dimensionality(self):
        config = self._make_config(embeddings_gemini_output_dimensionality=256)
        with patch("hindsight_api.config.get_config", return_value=config):
            emb = create_embeddings_from_env()
        assert emb.output_dimensionality == 256
