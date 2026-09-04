"""Bounded retry/backoff on the native Cohere embeddings provider.

The Cohere SDK does not retry on its own — its request-level ``max_retries``
defaults to 0 — so before this every quota or transient service response failed the
whole retain/consolidation operation, exactly as the native Gemini provider did
before #4103. These tests pin the same contract the LiteLLM and Gemini backends
already hold: transient upstream failures are retried within a bounded attempt count
and wall-clock budget, permanent 4xx fail fast, and the knobs come from
``HINDSIGHT_API_EMBEDDINGS_*``.
"""

import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.engine.embeddings import (
    CohereEmbeddings,
    EmbeddingRetryPolicy,
    create_embeddings_from_env,
)

COHERE_ENV_VARS = [
    "HINDSIGHT_API_EMBEDDINGS_PROVIDER",
    "HINDSIGHT_API_EMBEDDINGS_COHERE_API_KEY",
    "HINDSIGHT_API_EMBEDDINGS_COHERE_MODEL",
    "HINDSIGHT_API_EMBEDDINGS_COHERE_BASE_URL",
    "HINDSIGHT_API_EMBEDDINGS_COHERE_OUTPUT_DIMENSIONS",
    "HINDSIGHT_API_EMBEDDINGS_MAX_RETRIES",
    "HINDSIGHT_API_EMBEDDINGS_INITIAL_BACKOFF",
    "HINDSIGHT_API_EMBEDDINGS_MAX_BACKOFF",
    "HINDSIGHT_API_EMBEDDINGS_RETRY_BUDGET",
    "COHERE_API_KEY",
]


@pytest.fixture(autouse=True)
def clean_cohere_env(monkeypatch):
    from hindsight_api.config import clear_config_cache

    for env_var in COHERE_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PROVIDER", "rrf")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "cohere")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_COHERE_API_KEY", "co-test")
    clear_config_cache()

    yield

    clear_config_cache()


class _CohereApiError(Exception):
    """Stand-in for cohere.core.api_error.ApiError, which exposes `status_code`."""

    def __init__(self, status_code: int, message: str = "upstream error"):
        super().__init__(f"{message} (status {status_code})")
        self.status_code = status_code


# Fast policy so these tests exercise the retry logic, not the sleeps.
_FAST_POLICY = EmbeddingRetryPolicy(max_retries=3, initial_backoff=0.01, max_backoff=0.02, budget_seconds=5.0)


def _make_embeddings(side_effect, policy: EmbeddingRetryPolicy = _FAST_POLICY) -> CohereEmbeddings:
    emb = CohereEmbeddings(api_key="co-test", model="embed-english-v3.0", retry_policy=policy)
    client = MagicMock()
    client.embed = MagicMock(side_effect=side_effect)
    emb._client = client
    emb._dimension = 1024
    return emb


def _response(batch: list[str]) -> MagicMock:
    response = MagicMock()
    response.embeddings = [[0.1] * 1024 for _ in batch]
    return response


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 408])
def test_transient_status_then_success(status):
    """Quota and transient service responses are retried, and the later success returned."""
    calls = {"n": 0}

    def flaky(texts, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _CohereApiError(status)
        return _response(texts)

    emb = _make_embeddings(flaky)
    assert len(emb.encode(["hello"])) == 1
    assert calls["n"] == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_client_errors_fail_fast(status):
    """Auth and validation failures must not be retried — retrying cannot fix them."""
    calls = {"n": 0}

    def always_fail(texts, **kwargs):
        calls["n"] += 1
        raise _CohereApiError(status)

    emb = _make_embeddings(always_fail)
    with pytest.raises(_CohereApiError):
        emb.encode(["hello"])
    assert calls["n"] == 1


def test_exhausted_retries_propagate():
    """A sustained outage still surfaces to the worker, after a bounded attempt count."""
    calls = {"n": 0}

    def always_429(texts, **kwargs):
        calls["n"] += 1
        raise _CohereApiError(429)

    emb = _make_embeddings(always_429)
    with pytest.raises(_CohereApiError):
        emb.encode(["hello"])
    assert calls["n"] == _FAST_POLICY.max_retries + 1


def test_v2_output_dimension_path_is_retried():
    """The Matryoshka path goes through client.v2.embed — it needs the same retry."""
    calls = {"n": 0}

    def flaky(texts, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _CohereApiError(429)
        response = MagicMock()
        response.embeddings.float_ = [[0.2] * 256 for _ in texts]
        return response

    emb = CohereEmbeddings(
        api_key="co-test",
        model="embed-english-v3.0",
        output_dimensions=256,
        retry_policy=_FAST_POLICY,
    )
    client = MagicMock()
    client.v2.embed = MagicMock(side_effect=flaky)
    emb._client = client
    emb._dimension = 256

    assert len(emb.encode(["hello"])) == 1
    assert calls["n"] == 2


def test_retry_budget_is_shared_across_batches():
    """Batching must not multiply the worst-case added latency of one encode().

    Four concurrent batches at five retries each would be 24 upstream calls if every
    batch got its own budget; one budget for the whole call cuts it to a handful.
    """
    policy = EmbeddingRetryPolicy(max_retries=5, initial_backoff=0.05, max_backoff=0.05, budget_seconds=0.06)
    calls = {"n": 0}
    lock = threading.Lock()

    def always_429(texts, **kwargs):
        with lock:
            calls["n"] += 1
        raise _CohereApiError(429)

    emb = _make_embeddings(always_429, policy=policy)
    emb.batch_size = 1
    emb.max_concurrent_requests = 4

    started = time.monotonic()
    with pytest.raises(_CohereApiError):
        emb.encode(["a", "b", "c", "d"])

    assert calls["n"] < 4 * (policy.max_retries + 1) / 2
    # The budget caps wall-clock too, not just the attempt count.
    assert time.monotonic() - started < 1.0


async def test_initialize_probe_retries_transient_failures():
    """A quota blip during the startup dimension probe must not crash-loop the daemon."""
    calls = {"n": 0}

    def flaky(texts, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _CohereApiError(429)
        return _response(texts)

    # An unknown model forces the probe — a model in MODEL_DIMENSIONS skips it.
    emb = CohereEmbeddings(api_key="co-test", model="embed-future-v9.0", retry_policy=_FAST_POLICY)
    client = MagicMock()
    client.embed = MagicMock(side_effect=flaky)

    fake_cohere = types.ModuleType("cohere")
    fake_cohere.Client = MagicMock(return_value=client)
    with patch.dict(sys.modules, {"cohere": fake_cohere}):
        await emb.initialize()

    assert emb.dimension == 1024
    assert calls["n"] == 2


def test_factory_wires_the_configured_policy(monkeypatch):
    """The provider honours HINDSIGHT_API_EMBEDDINGS_* like the LiteLLM backends do."""
    from hindsight_api.config import clear_config_cache

    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_MAX_RETRIES", "7")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_INITIAL_BACKOFF", "0.25")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_MAX_BACKOFF", "2.0")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_RETRY_BUDGET", "42.5")
    clear_config_cache()

    emb = create_embeddings_from_env()

    assert isinstance(emb, CohereEmbeddings)
    assert emb.retry_policy.max_retries == 7
    assert emb.retry_policy.initial_backoff == 0.25
    assert emb.retry_policy.max_backoff == 2.0
    assert emb.retry_policy.budget_seconds == 42.5
