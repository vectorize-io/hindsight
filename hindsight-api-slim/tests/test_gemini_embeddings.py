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

import asyncio
import os
import socket
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.engine.embeddings import (
    RetryPolicy,
    GeminiEmbeddings,
    create_embeddings_from_env,
)


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
    mock_client.aio.models.embed_content = AsyncMock(return_value=embed_result)
    mock_genai.Client = MagicMock(return_value=mock_client)
    return mock_genai


def _make_mock_google_module(mock_genai: MagicMock) -> MagicMock:
    mod = MagicMock()
    mod.genai = mock_genai
    mod.genai.types.EmbedContentConfig = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    mod.genai.types.HttpOptions = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    mod.genai.types.Content = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    mod.genai.types.Part.from_text = MagicMock(side_effect=lambda text="": MagicMock(text=text))
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
        mock_genai.Client.return_value.aio.models.embed_content.assert_called_once()

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

    @pytest.mark.parametrize(
        "model,vertexai,expected_requests",
        [
            # Vertex routes these to the single-content embedContent endpoint, which
            # rejects a second Content client-side — one request per text is the only
            # shape that comes back 1:1.
            ("gemini-embedding-2-preview", True, 3),
            ("google/gemini-embedding-2-preview", True, 3),
            ("text-multilingual-maas-002", True, 3),
            # The one Vertex gemini model the SDK still batches, and everything on the
            # Gemini API, keep the configured batch size: one request for all three.
            ("gemini-embedding-001", True, 1),
            ("text-embedding-005", True, 1),
            ("gemini-embedding-2-preview", False, 1),
        ],
    )
    async def test_vertex_single_content_models_get_one_request_per_text(self, model, vertexai, expected_requests):
        """Batching is capped at one text where the API takes one Content (#4001 follow-up).

        Each text is already sent as its own Content, which is what keeps a
        multimodal model from fusing a batch into a single vector. On Vertex the
        SDK then refuses more than one Content for these models outright —
        ``ValueError: The embedContent API for this model only supports one content
        at a time.`` — so every encode() against gemini-embedding-2 raised before
        reaching the network. Counting requests rather than asserting on the
        exception is deliberate: the fix is the request shape, and a regression
        would show up here as three texts back in one call.
        """
        mock_genai = _make_mock_genai()
        embed_content = mock_genai.Client.return_value.aio.models.embed_content
        emb = GeminiEmbeddings(
            model=model,
            api_key=None if vertexai else "test-key",
            vertexai_project_id="test-project" if vertexai else None,
        )
        with _patch_google_import(mock_genai):
            await emb.initialize()

        texts = ["a", "b", "c"]
        # One vector per text in the batch the call actually carries, so the 1:1
        # check inside _embed_batch passes for either shape.
        embed_content.side_effect = lambda **kw: _make_mock_embed_result([[0.1] * 768] * len(kw["contents"]))
        embed_content.reset_mock()
        vectors = await emb.encode(texts)

        assert len(vectors) == len(texts)
        assert embed_content.call_count == expected_requests
        assert [len(c.kwargs["contents"]) for c in embed_content.call_args_list] == (
            [1, 1, 1] if expected_requests == 3 else [3]
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
        call_kwargs = mock_genai.Client.return_value.aio.models.embed_content.call_args
        assert "config" in call_kwargs.kwargs

    async def test_no_output_dimensionality(self):
        """Test that no EmbedContentConfig is built when output_dimensionality is None."""
        mock_genai = _make_mock_genai()
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key", output_dimensionality=None)

        with _patch_google_import(mock_genai):
            await emb.initialize()

        assert emb._embed_config is None
        call_kwargs = mock_genai.Client.return_value.aio.models.embed_content.call_args
        assert "config" not in call_kwargs.kwargs

    async def test_force_ipv4_passes_an_ipv4_only_aiohttp_session(self):
        """force_ipv4 hands the SDK an aiohttp session whose connector dials IPv4 only."""
        mock_genai = _make_mock_genai()
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key", force_ipv4=True)

        with _patch_google_import(mock_genai):
            await emb.initialize()

        http_options = mock_genai.Client.call_args.kwargs["http_options"]
        assert http_options.timeout == 10000
        session = http_options.aiohttp_client
        assert isinstance(session, aiohttp.ClientSession)
        assert session.connector is not None
        assert session.connector.family == socket.AF_INET
        await session.close()

    async def test_force_ipv4_builds_one_genai_client_per_event_loop(self):
        """A session binds to its loop, so another loop must get its own client and session."""
        mock_genai = _make_mock_genai()
        mock_genai.Client.side_effect = lambda **kwargs: MagicMock(http_options=kwargs["http_options"])
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key", force_ipv4=True)

        with _patch_google_import(mock_genai):
            emb._init_gemini(mock_genai, _make_mock_google_module(mock_genai).genai.types)

        assert emb._ipv4_clients is not None
        here = emb._ipv4_clients.get()
        assert emb._ipv4_clients.get() is here

        async def other_loop_client() -> Any:
            client = emb._ipv4_clients.get()
            await client.http_options.aiohttp_client.close()
            return client

        elsewhere = await asyncio.to_thread(asyncio.run, other_loop_client())
        assert elsewhere is not here
        assert elsewhere.http_options.aiohttp_client is not here.http_options.aiohttp_client
        await here.http_options.aiohttp_client.close()

    async def test_auto_detect_vertexai(self):
        """Test that _is_vertexai is auto-detected from vertexai_project_id."""
        assert GeminiEmbeddings(model="m", api_key="k")._is_vertexai is False
        assert GeminiEmbeddings(model="m", vertexai_project_id="p")._is_vertexai is True

    async def test_encode_single_text(self):
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key")
        mock_client = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=_make_mock_embed_result([[0.1, 0.2, 0.3]]))
        emb._client = mock_client
        emb._dimension = 3

        assert await emb.encode(["hello"]) == [[0.1, 0.2, 0.3]]

    async def test_encode_multiple_texts(self):
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key")
        mock_client = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(
            return_value=_make_mock_embed_result([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        )
        emb._client = mock_client
        emb._dimension = 2

        result = await emb.encode(["a", "b", "c"])
        assert len(result) == 3
        assert result[1] == [0.3, 0.4]

    async def test_encode_batching(self):
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key", batch_size=2)
        mock_client = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(
            side_effect=[_make_mock_embed_result([[0.1], [0.2]]), _make_mock_embed_result([[0.3]])]
        )
        emb._client = mock_client
        emb._dimension = 1

        assert await emb.encode(["a", "b", "c"]) == [[0.1], [0.2], [0.3]]
        assert mock_client.aio.models.embed_content.call_count == 2

    async def test_encode_passes_config(self):
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key")
        mock_client = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=_make_mock_embed_result([[0.1, 0.2]]))
        emb._client = mock_client
        emb._dimension = 2
        emb._embed_config = MagicMock()

        await emb.encode(["hello"])
        assert mock_client.aio.models.embed_content.call_args.kwargs["config"] is emb._embed_config

    async def test_encode_gemini_embedding_2_batches_multiple_inputs(self):
        """Gemini Embedding 2+ wraps inputs in Content objects so batch_size=100
        correctly batches texts into a single request with 1:1 vector alignment."""
        emb = GeminiEmbeddings(model="gemini-embedding-2-preview", api_key="test-key", batch_size=100)
        mock_client = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=_make_mock_embed_result([[0.1], [0.2], [0.3]]))
        emb._client = mock_client
        emb._dimension = 1

        assert await emb.encode(["a", "b", "c"]) == [[0.1], [0.2], [0.3]]
        # 1 call for 3 inputs when batch_size=100
        assert mock_client.aio.models.embed_content.call_count == 1
        call_contents = mock_client.aio.models.embed_content.call_args.kwargs["contents"]
        assert len(call_contents) == 3

    async def test_encode_raises_on_misaligned_vector_count(self):
        """A backend that aggregates inputs (returns fewer vectors than texts)
        must raise rather than silently misalign vectors with inputs."""
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key", batch_size=100)
        mock_client = MagicMock()
        # 3 inputs in one batch but only 1 vector returned (aggregation).
        mock_client.aio.models.embed_content = AsyncMock(return_value=_make_mock_embed_result([[0.1]]))
        emb._client = mock_client
        emb._dimension = 1

        with pytest.raises(RuntimeError, match="expected exact 1:1 alignment"):
            await emb.encode(["a", "b", "c"])

    async def test_encode_empty_list(self):
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key")
        emb._client = MagicMock()
        emb._dimension = 768
        assert await emb.encode([]) == []

    async def test_encode_before_initialization(self):
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key")
        with pytest.raises(RuntimeError, match="not initialized"):
            await emb.encode(["test"])

    async def test_dimension_before_initialization(self):
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key")
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = emb.dimension

    async def test_provider_name_always_google(self):
        assert GeminiEmbeddings(model="m", api_key="k").provider_name == "google"
        assert GeminiEmbeddings(model="m", vertexai_project_id="p").provider_name == "google"

    async def test_vertexai_strips_google_prefix(self):
        mock_genai = _make_mock_genai()
        emb = GeminiEmbeddings(model="google/gemini-embedding-001", vertexai_project_id="test-project")
        emb._init_vertexai(mock_genai)
        assert emb.model == "gemini-embedding-001"

    async def test_default_region(self):
        emb = GeminiEmbeddings(model="m", vertexai_project_id="proj")
        assert emb.vertexai_region == "us-central1"

    async def test_custom_region(self):
        emb = GeminiEmbeddings(model="m", vertexai_project_id="proj", vertexai_region="europe-west1")
        assert emb.vertexai_region == "europe-west1"


class _GenAIError(Exception):
    """Stand-in for google.genai.errors.APIError, which carries the status on `code`."""

    def __init__(self, code: int, message: str = "RESOURCE_EXHAUSTED"):
        super().__init__(f"{code} {message}")
        self.code = code
        self.response = None


# Fast policy so these tests exercise the retry logic, not the sleeps.
_FAST_POLICY = RetryPolicy(max_retries=3, initial_backoff=0.01, max_backoff=0.02, budget_seconds=5.0)


class TestGeminiEmbeddingsRetry:
    """Bounded retry for quota and transient service responses (#4103).

    A shared Gemini project hands out 429s well before anything is actually wrong.
    Without a request-level retry the worker's coarse whole-task retry dead-letters
    retain and consolidation operations during an ordinary quota window.
    """

    def _make_embeddings(self, side_effect, policy: RetryPolicy = _FAST_POLICY) -> GeminiEmbeddings:
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key", retry_policy=policy)
        client = MagicMock()
        client.aio.models.embed_content = AsyncMock(side_effect=side_effect)
        emb._client = client
        emb._dimension = 768
        return emb

    async def test_transient_status_then_success(self):
        """A 429 is retried and the later success is returned."""
        calls = {"n": 0}

        def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _GenAIError(429)
            return _make_mock_embed_result([[0.1] * 768])

        emb = self._make_embeddings(flaky)
        assert await emb.encode(["hello"]) == [[0.1] * 768]
        assert calls["n"] == 2

    async def test_transient_5xx_is_retried(self):
        calls = {"n": 0}

        def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _GenAIError(503, "UNAVAILABLE")
            return _make_mock_embed_result([[0.2] * 768])

        emb = self._make_embeddings(flaky)
        assert await emb.encode(["hello"]) == [[0.2] * 768]
        assert calls["n"] == 3

    @pytest.mark.parametrize("code", [400, 401, 403, 404])
    async def test_permanent_client_errors_fail_fast(self, code):
        """Auth and validation failures must not be retried — retrying cannot fix them."""
        calls = {"n": 0}

        def always_fail(**kwargs):
            calls["n"] += 1
            raise _GenAIError(code, "INVALID_ARGUMENT")

        emb = self._make_embeddings(always_fail)
        with pytest.raises(_GenAIError):
            await emb.encode(["hello"])
        assert calls["n"] == 1

    async def test_exhausted_retries_propagate(self):
        """A sustained outage still surfaces to the worker, after a bounded attempt count."""
        calls = {"n": 0}

        def always_429(**kwargs):
            calls["n"] += 1
            raise _GenAIError(429)

        emb = self._make_embeddings(always_429)
        with pytest.raises(_GenAIError):
            await emb.encode(["hello"])
        assert calls["n"] == _FAST_POLICY.max_retries + 1

    async def test_retry_budget_is_shared_across_batches(self):
        """Batching must not multiply the worst-case added latency of one encode().

        The batches of one encode() go out concurrently, so a per-batch budget would let
        a degraded provider be hammered in proportion to the text volume. One budget for
        the whole call bounds it: here four concurrent batches with five retries each
        would be 24 upstream calls unshared, and the shared budget cuts it to a handful.
        """
        policy = RetryPolicy(max_retries=5, initial_backoff=0.05, max_backoff=0.05, budget_seconds=0.06)
        calls = {"n": 0}
        lock = threading.Lock()

        def always_429(**kwargs):
            with lock:
                calls["n"] += 1
            raise _GenAIError(429)

        emb = self._make_embeddings(always_429, policy=policy)
        emb.batch_size = 1
        emb.max_concurrent_requests = 4

        started = time.monotonic()
        with pytest.raises(_GenAIError):
            await emb.encode(["a", "b", "c", "d"])

        assert calls["n"] < 4 * (policy.max_retries + 1) / 2
        # The budget caps wall-clock too, not just the attempt count.
        assert time.monotonic() - started < 1.0

    async def test_initialize_probe_retries_transient_failures(self):
        """A quota blip during the startup dimension probe must not crash-loop the daemon."""
        mock_genai = _make_mock_genai()
        calls = {"n": 0}

        def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _GenAIError(429)
            return _make_mock_embed_result([[0.3] * 768])

        mock_genai.Client.return_value.aio.models.embed_content = AsyncMock(side_effect=flaky)
        emb = GeminiEmbeddings(model="gemini-embedding-001", api_key="test-key", retry_policy=_FAST_POLICY)

        with _patch_google_import(mock_genai):
            await emb.initialize()

        assert emb.dimension == 768
        assert calls["n"] == 2


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

    def test_create_wires_the_configured_retry_policy(self):
        """The provider honours HINDSIGHT_API_EMBEDDINGS_* like the LiteLLM backends do (#4103)."""
        config = self._make_config(
            embeddings_max_retries=7,
            embeddings_initial_backoff=0.25,
            embeddings_max_backoff=2.0,
            embeddings_retry_budget=42.5,
        )
        with patch("hindsight_api.config.get_config", return_value=config):
            emb = create_embeddings_from_env()

        assert isinstance(emb, GeminiEmbeddings)
        assert emb.retry_policy.max_retries == 7
        assert emb.retry_policy.initial_backoff == 0.25
        assert emb.retry_policy.max_backoff == 2.0
        assert emb.retry_policy.budget_seconds == 42.5

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

    def test_create_with_batch_size(self):
        config = self._make_config(embeddings_gemini_batch_size=50)
        with patch("hindsight_api.config.get_config", return_value=config):
            emb = create_embeddings_from_env()
        assert emb.batch_size == 50


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID"),
    reason="Vertex AI integration tests require HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID",
)
async def test_gemini_embedding_2_vertexai_one_vector_per_input():
    """Real Vertex AI check that the gemini-embedding-2 family stays 1:1.

    These multimodal models fuse the several Parts of one Content into a single
    embedding, so encode() sends each text as its own Content (the bug in #1139
    returned one aggregated vector for the whole batch). This is the only test
    that can confirm it: the behaviour lives in the API, not in our code, so a
    mocked client would pass either way. All three texts go out in ONE request
    at the default batch size, which is exactly the case that used to aggregate.
    Runs in the CI jobs that provide GCP credentials; skips locally otherwise.
    """
    project_id = os.getenv("HINDSIGHT_API_EMBEDDINGS_VERTEXAI_PROJECT_ID") or os.getenv(
        "HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID"
    )
    region = (
        os.getenv("HINDSIGHT_API_EMBEDDINGS_VERTEXAI_REGION")
        or os.getenv("HINDSIGHT_API_LLM_VERTEXAI_REGION")
        or "us-central1"
    )
    service_account_key = os.getenv("HINDSIGHT_API_EMBEDDINGS_VERTEXAI_SERVICE_ACCOUNT_KEY") or os.getenv(
        "HINDSIGHT_API_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY"
    )
    # Overridable so the model can be bumped (e.g. to GA) without a code change.
    model = os.getenv("HINDSIGHT_API_EMBEDDINGS_GEMINI_MODEL", "gemini-embedding-2-preview")

    emb = GeminiEmbeddings(
        model=model,
        vertexai_project_id=project_id,
        vertexai_region=region,
        vertexai_service_account_key=service_account_key,
        output_dimensionality=768,
    )

    texts = [
        "The sky is blue.",
        "I visited Paris in 2023.",
        "Python is a programming language.",
    ]

    from google.genai.errors import APIError

    try:
        await emb.initialize()
        vectors = await emb.encode(texts)
    except APIError as e:
        # gemini-embedding-2 is a preview/allowlisted model not enabled in every
        # Vertex project (e.g. CI returns 400 FAILED_PRECONDITION). Skip rather
        # than fail when the project lacks access — a real aggregation regression
        # surfaces below as a wrong vector count (RuntimeError/AssertionError),
        # never as an APIError, so this skip cannot mask the behavior under test.
        pytest.skip(f"gemini-embedding-2 not available in this Vertex project: {e}")

    # The fix: one vector per input, not a single aggregated vector.
    assert len(vectors) == len(texts)
    assert all(len(v) == emb.dimension for v in vectors)
    # Distinct inputs must produce distinct vectors (proves no aggregation).
    assert vectors[0] != vectors[1]
    assert vectors[1] != vectors[2]
    assert vectors[0] != vectors[2]
