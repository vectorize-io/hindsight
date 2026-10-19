"""OpenAI embeddings out of credits defers the operation instead of failing it (#5588)."""

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from openai import RateLimitError

from hindsight_api.engine.aiohttp_session import LoopLocal
from hindsight_api.engine.embeddings import EMBEDDINGS_QUOTA_DEFER_SECONDS, OpenAIEmbeddings
from hindsight_api.engine.llm_interface import ProviderRateLimitResetError
from hindsight_api.engine.retain.embedding_utils import generate_embeddings_batch


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
