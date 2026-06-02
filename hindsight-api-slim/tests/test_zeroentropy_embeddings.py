"""Tests for the native ZeroEntropy zembed embeddings provider."""

import base64
import struct

import httpx
import pytest
from pydantic import BaseModel

ZEROENTROPY_ENV_VARS = [
    "HINDSIGHT_API_EMBEDDINGS_PROVIDER",
    "HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_API_KEY",
    "HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_MODEL",
    "HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_BASE_URL",
    "HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_DIMENSIONS",
    "HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_ENCODING_FORMAT",
    "HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_BATCH_SIZE",
    "HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_LATENCY",
    "HINDSIGHT_API_RERANKER_PROVIDER",
    "HINDSIGHT_API_RERANKER_ZEROENTROPY_API_KEY",
    "HINDSIGHT_API_RERANKER_ZEROENTROPY_MODEL",
    "HINDSIGHT_API_RERANKER_ZEROENTROPY_BASE_URL",
    "ZEROENTROPY_API_KEY",
]


class CapturedZeroEntropyEmbedRequest(BaseModel):
    model: str
    input: list[str]
    input_type: str
    dimensions: int
    encoding_format: str
    latency: str | None = None


@pytest.fixture(autouse=True)
def clean_zeroentropy_env(monkeypatch):
    from hindsight_api.config import clear_config_cache

    for env_var in ZEROENTROPY_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PROVIDER", "rrf")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "zeroentropy")
    clear_config_cache()

    yield

    clear_config_cache()


def test_zeroentropy_config_defaults():
    from hindsight_api.config import HindsightConfig

    config = HindsightConfig.from_env()

    assert config.embeddings_zeroentropy_model == "zembed-1"
    assert config.embeddings_zeroentropy_base_url == "https://api.zeroentropy.dev"
    assert config.embeddings_zeroentropy_dimensions == 1280
    assert config.embeddings_zeroentropy_encoding_format == "float"
    assert config.embeddings_zeroentropy_batch_size == 100
    assert config.embeddings_zeroentropy_latency is None


def test_zeroentropy_create_from_env(monkeypatch):
    from hindsight_api.engine.embeddings import ZeroEntropyEmbeddings, create_embeddings_from_env

    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "zeroentropy")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_API_KEY", "ze-test")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_MODEL", "zembed-1")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_BASE_URL", "https://ze.example")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_DIMENSIONS", "640")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_ENCODING_FORMAT", "base64")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_BATCH_SIZE", "2")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_LATENCY", "fast")

    embeddings = create_embeddings_from_env()

    assert isinstance(embeddings, ZeroEntropyEmbeddings)
    assert embeddings.api_key == "ze-test"
    assert embeddings.model == "zembed-1"
    assert embeddings.base_url == "https://ze.example"
    assert embeddings.dimensions == 640
    assert embeddings.encoding_format == "base64"
    assert embeddings.batch_size == 2
    assert embeddings.latency == "fast"


def test_zeroentropy_create_from_env_requires_api_key(monkeypatch):
    from hindsight_api.engine.embeddings import create_embeddings_from_env

    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "zeroentropy")

    with pytest.raises(ValueError, match="HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_API_KEY"):
        create_embeddings_from_env()


def test_zeroentropy_create_from_env_uses_standard_api_key_env(monkeypatch):
    from hindsight_api.engine.embeddings import ZeroEntropyEmbeddings, create_embeddings_from_env

    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "zeroentropy")
    monkeypatch.setenv("ZEROENTROPY_API_KEY", "ze-standard")

    embeddings = create_embeddings_from_env()

    assert isinstance(embeddings, ZeroEntropyEmbeddings)
    assert embeddings.api_key == "ze-standard"


def test_zeroentropy_rejects_invalid_dimension(monkeypatch):
    from hindsight_api.engine.embeddings import create_embeddings_from_env

    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "zeroentropy")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_API_KEY", "ze-test")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_DIMENSIONS", "1024")

    with pytest.raises(ValueError, match="must be one of"):
        create_embeddings_from_env()


def test_zeroentropy_rejects_invalid_encoding_format(monkeypatch):
    from hindsight_api.engine.embeddings import create_embeddings_from_env

    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "zeroentropy")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_API_KEY", "ze-test")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_ENCODING_FORMAT", "binary")

    with pytest.raises(ValueError, match="HINDSIGHT_API_EMBEDDINGS_ZEROENTROPY_ENCODING_FORMAT"):
        create_embeddings_from_env()


def test_zeroentropy_encode_documents_batches_and_sends_expected_payload():
    from hindsight_api.engine.embeddings import ZeroEntropyEmbeddings

    requests: list[CapturedZeroEntropyEmbedRequest] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = CapturedZeroEntropyEmbedRequest.model_validate_json(request.content)
        requests.append(body)
        assert str(request.url) == "https://api.zeroentropy.dev/v1/models/embed"
        assert request.headers["authorization"] == "Bearer ze-test"
        return httpx.Response(
            200,
            json={"results": [{"embedding": [float(i), 0.0]} for i, _ in enumerate(body.input)]},
        )

    embeddings = ZeroEntropyEmbeddings(api_key="ze-test", dimensions=1280, batch_size=2, latency="fast")
    embeddings._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer ze-test", "Content-Type": "application/json"},
    )
    embeddings._dimension = 1280

    vectors = embeddings.encode_documents(["alpha", "beta", "gamma"])

    assert len(vectors) == 3
    assert [request.input for request in requests] == [["alpha", "beta"], ["gamma"]]
    assert all(request.model == "zembed-1" for request in requests)
    assert all(request.input_type == "document" for request in requests)
    assert all(request.dimensions == 1280 for request in requests)
    assert all(request.encoding_format == "float" for request in requests)
    assert all(request.latency == "fast" for request in requests)


def test_zeroentropy_omits_latency_when_unset():
    import json

    from hindsight_api.engine.embeddings import ZeroEntropyEmbeddings

    seen_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"results": [{"embedding": [1.0, 2.0]}]})

    embeddings = ZeroEntropyEmbeddings(api_key="ze-test", dimensions=1280, latency=None)
    embeddings._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer ze-test", "Content-Type": "application/json"},
    )
    embeddings._dimension = 1280

    embeddings.encode_documents(["alpha"])

    assert "latency" not in seen_bodies[0]


def test_zeroentropy_encode_query_sends_query_input_type():
    from hindsight_api.engine.embeddings import ZeroEntropyEmbeddings

    seen_input_types: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = CapturedZeroEntropyEmbedRequest.model_validate_json(request.content)
        seen_input_types.append(body.input_type)
        return httpx.Response(200, json={"results": [{"embedding": [1.0, 2.0]}]})

    embeddings = ZeroEntropyEmbeddings(api_key="ze-test", dimensions=1280)
    embeddings._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer ze-test", "Content-Type": "application/json"},
    )
    embeddings._dimension = 1280

    assert embeddings.encode_query(["where is this?"]) == [[1.0, 2.0]]
    assert seen_input_types == ["query"]


def test_zeroentropy_base64_response_is_decoded():
    from hindsight_api.engine.embeddings import ZeroEntropyEmbeddings

    encoded = base64.b64encode(struct.pack("<2f", 0.25, 0.5)).decode("ascii")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"embedding": encoded}]})

    embeddings = ZeroEntropyEmbeddings(api_key="ze-test", dimensions=1280, encoding_format="base64")
    embeddings._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer ze-test", "Content-Type": "application/json"},
    )
    embeddings._dimension = 1280

    assert embeddings.encode_documents(["alpha"]) == [[0.25, 0.5]]


def test_zeroentropy_reranker_create_from_env_uses_base_url(monkeypatch):
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.cross_encoder import ZeroEntropyCrossEncoder, create_cross_encoder_from_env

    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PROVIDER", "zeroentropy")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_ZEROENTROPY_API_KEY", "ze-test")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_ZEROENTROPY_BASE_URL", "https://rerank.example")
    clear_config_cache()

    encoder = create_cross_encoder_from_env()

    assert isinstance(encoder, ZeroEntropyCrossEncoder)
    assert encoder.base_url == "https://rerank.example"
    assert encoder._client.rerank_url == "https://rerank.example/v1/models/rerank"


@pytest.mark.asyncio
async def test_embedding_utils_routes_query_embeddings_to_provider_hook():
    from hindsight_api.engine.retain.embedding_utils import generate_embeddings_batch

    class QueryAwareEmbeddings:
        def encode(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("query embeddings should not use the generic encode method")

        def encode_query(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text))] for text in texts]

    vectors = await generate_embeddings_batch(QueryAwareEmbeddings(), ["query text"], input_type="query")

    assert vectors == [[10.0]]


@pytest.mark.asyncio
async def test_embedding_utils_routes_document_embeddings_to_provider_hook():
    from hindsight_api.engine.retain.embedding_utils import generate_embeddings_batch

    class DocumentAwareEmbeddings:
        def encode(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("document embeddings should not use the generic encode method")

        def encode_documents(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text))] for text in texts]

    vectors = await generate_embeddings_batch(DocumentAwareEmbeddings(), ["document text"], input_type="document")

    assert vectors == [[13.0]]
