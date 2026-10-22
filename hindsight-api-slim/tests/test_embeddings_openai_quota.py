"""OpenAI embeddings: no credits defers the operation (#5588); partial usage never fails it (#5941)."""

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from openai import RateLimitError
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage

from hindsight_api import tracing
from hindsight_api.engine.aiohttp_session import LoopLocal
from hindsight_api.engine.embeddings import EMBEDDINGS_QUOTA_DEFER_SECONDS, OpenAIEmbeddings
from hindsight_api.engine.llm_interface import ProviderRateLimitResetError
from hindsight_api.engine.retain.embedding_utils import generate_embeddings_batch
from tests.test_llm_trace import _CapturingRecorder


def _rate_limit_error(code: str) -> RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"))
    return RateLimitError("Error code: 429", response=response, body={"code": code, "message": "no credits"})


class _FailingEmbeddingsAPI:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def create(self, **_: object) -> None:
        raise self.error


class _FailingClient:
    def __init__(self, error: Exception) -> None:
        self.api_key = "sk-test"
        self.embeddings = _FailingEmbeddingsAPI(error)


def _backend(error: Exception) -> OpenAIEmbeddings:
    backend = OpenAIEmbeddings(api_key="sk-test", model="text-embedding-3-small")
    backend._dimension = 1536
    backend._clients = LoopLocal(lambda: _FailingClient(error))
    return backend


@pytest.mark.asyncio
async def test_insufficient_quota_defers_through_generate_embeddings_batch():
    before = datetime.now(UTC)
    with pytest.raises(ProviderRateLimitResetError) as exc_info:
        await generate_embeddings_batch(_backend(_rate_limit_error("insufficient_quota")), ["hello"])

    expected = before + timedelta(seconds=EMBEDDINGS_QUOTA_DEFER_SECONDS)
    assert abs((exc_info.value.retry_at - expected).total_seconds()) < 5


@pytest.mark.asyncio
async def test_ordinary_rate_limit_still_fails_with_the_cause_kept():
    error = _rate_limit_error("rate_limit_exceeded")
    with pytest.raises(Exception, match="Failed to generate batch embeddings") as exc_info:
        await generate_embeddings_batch(_backend(error), ["hello"])

    assert not isinstance(exc_info.value, ProviderRateLimitResetError)
    assert exc_info.value.__cause__ is error


@pytest.mark.asyncio
async def test_client_carries_timeout_and_configured_retries(monkeypatch):
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.embeddings import create_embeddings_from_env

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "openai")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY", "sk-test")
    # A known model, so initialize() reads its dimension instead of probing the API.
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL", "text-embedding-3-small")
    monkeypatch.delenv("HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_MAX_RETRIES", "1")
    clear_config_cache()
    try:
        backend = create_embeddings_from_env()
        assert isinstance(backend, OpenAIEmbeddings)
        await backend.initialize()
        client = backend._loop_client()
        assert client.max_retries == 1
        assert client.timeout == 60.0
    finally:
        clear_config_cache()


class _UsageEmbeddingsAPI:
    """Answers like Voyage's OpenAI-compatible endpoint: usage without prompt_tokens (#5941)."""

    def __init__(self, total_tokens: int | None) -> None:
        self.total_tokens = total_tokens

    async def create(self, **_: object) -> CreateEmbeddingResponse:
        return CreateEmbeddingResponse.model_construct(
            data=[Embedding(embedding=[0.1] * 1536, index=0, object="embedding")],
            model="voyage-4",
            object="list",
            usage=Usage.model_construct(prompt_tokens=None, total_tokens=self.total_tokens),
        )


class _UsageClient:
    def __init__(self, total_tokens: int | None) -> None:
        self.api_key = "sk-test"
        self.embeddings = _UsageEmbeddingsAPI(total_tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize("total_tokens,traced", [(7, 7), (None, None)])
async def test_usage_without_prompt_tokens_does_not_fail_the_call(total_tokens: int | None, traced: int | None):
    backend = OpenAIEmbeddings(api_key="sk-test", model="voyage-4")
    backend._dimension = 1536
    backend._clients = LoopLocal(lambda: _UsageClient(total_tokens))
    recorder = _CapturingRecorder()
    tracing.register_span_recorder(recorder)
    try:
        vectors = await generate_embeddings_batch(backend, ["hello"])
    finally:
        tracing.unregister_span_recorder(recorder)

    assert len(vectors) == 1
    assert [r.input_tokens for r in recorder.records] == [traced]
