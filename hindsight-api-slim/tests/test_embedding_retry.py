"""
Tests for bounded retry/backoff on remote embedding calls.

Recall generates its query embedding synchronously on the request path, so a
single upstream 5xx used to surface as a failed recall. These tests pin the
contract of the retry wrapper:

1. Transient failures (5xx, timeouts, connection errors) are retried and a later
   success is returned.
2. Persistent transient failures still raise, after a bounded number of attempts.
3. Non-transient failures (4xx auth/validation) raise immediately, with no retry.
4. The wall-clock retry budget caps how long a single encode() call can spend
   retrying, and batching does not multiply it.
"""

import time
from unittest.mock import MagicMock

import httpx
import pytest

from hindsight_api.engine.embeddings import (
    EmbeddingRetryPolicy,
    LiteLLMEmbeddings,
    LiteLLMSDKEmbeddings,
    _is_transient_embedding_error,
)


class _StatusError(Exception):
    """Stand-in for litellm/openai errors, which expose `status_code`."""

    def __init__(self, status_code: int, message: str = "upstream error"):
        super().__init__(f"{message} (status {status_code})")
        self.status_code = status_code


class _InternalServerError(_StatusError):
    """Matches litellm.InternalServerError closely enough for classification."""

    def __init__(self, message: str = "Venice AI returned 500"):
        super().__init__(500, message)


class _AuthenticationError(_StatusError):
    def __init__(self, message: str = "invalid api key"):
        super().__init__(401, message)


def _response_for(texts):
    response = MagicMock()
    response.data = [{"embedding": [0.1] * 768, "index": i} for i in range(len(texts))]
    return response


def _make_embeddings(policy: EmbeddingRetryPolicy, batch_size: int = 100) -> LiteLLMSDKEmbeddings:
    emb = LiteLLMSDKEmbeddings(
        api_key="test_key",
        model="openai/text-embedding-qwen3-8b",
        api_base="https://example.invalid/api/v1",
        batch_size=batch_size,
        timeout=60.0,
        retry_policy=policy,
    )
    emb._litellm = MagicMock()
    emb._dimension = 768
    return emb


# Fast policy so the tests exercise the logic, not the sleeps.
FAST_POLICY = EmbeddingRetryPolicy(max_retries=4, initial_backoff=0.01, max_backoff=0.04, budget_seconds=5.0)


class TestTransientClassification:
    """Only genuinely transient failures should be retried."""

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 408, 429])
    def test_transient_status_codes(self, status):
        assert _is_transient_embedding_error(_StatusError(status)) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_client_error_status_codes_are_not_transient(self, status):
        assert _is_transient_embedding_error(_StatusError(status)) is False

    def test_transport_errors_are_transient(self):
        import httpx

        assert _is_transient_embedding_error(httpx.ConnectError("connection refused")) is True
        assert _is_transient_embedding_error(httpx.ReadTimeout("read timed out")) is True
        assert _is_transient_embedding_error(TimeoutError("timed out")) is True
        assert _is_transient_embedding_error(ConnectionError("reset by peer")) is True

    def test_named_sdk_errors_are_transient_without_status(self):
        class APITimeoutError(Exception):
            pass

        assert _is_transient_embedding_error(APITimeoutError("no status attribute")) is True

    def test_unknown_errors_are_not_transient(self):
        assert _is_transient_embedding_error(ValueError("bad input")) is False
        assert _is_transient_embedding_error(Exception("API Error")) is False


class TestEncodeRetries:
    """encode() is the synchronous path recall runs inline."""

    def test_transient_then_success(self):
        emb = _make_embeddings(FAST_POLICY)
        calls = {"n": 0}

        def side_effect(**kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _InternalServerError()
            return _response_for(kwargs["input"])

        emb._litellm.embedding.side_effect = side_effect

        result = emb.encode(["what did we decide about embeddings?"])

        assert len(result) == 1
        assert len(result[0]) == 768
        assert emb._litellm.embedding.call_count == 3

    def test_persistent_transient_failure_raises_after_attempt_cap(self):
        emb = _make_embeddings(FAST_POLICY)
        emb._litellm.embedding.side_effect = _InternalServerError()

        with pytest.raises(_InternalServerError):
            emb.encode(["query"])

        # max_retries=4 -> 5 attempts total, then the original error propagates.
        assert emb._litellm.embedding.call_count == FAST_POLICY.max_retries + 1

    def test_auth_error_raises_immediately_with_no_retries(self):
        emb = _make_embeddings(FAST_POLICY)
        emb._litellm.embedding.side_effect = _AuthenticationError()

        with pytest.raises(_AuthenticationError):
            emb.encode(["query"])

        assert emb._litellm.embedding.call_count == 1

    @pytest.mark.parametrize("status", [400, 403, 404, 422])
    def test_other_client_errors_raise_immediately(self, status):
        emb = _make_embeddings(FAST_POLICY)
        emb._litellm.embedding.side_effect = _StatusError(status)

        with pytest.raises(_StatusError):
            emb.encode(["query"])

        assert emb._litellm.embedding.call_count == 1

    def test_retries_disabled_by_zero_max_retries(self):
        emb = _make_embeddings(EmbeddingRetryPolicy(max_retries=0, budget_seconds=5.0))
        emb._litellm.embedding.side_effect = _InternalServerError()

        with pytest.raises(_InternalServerError):
            emb.encode(["query"])

        assert emb._litellm.embedding.call_count == 1


class TestRetryBudget:
    """The budget is what keeps a degraded upstream from stalling a recall."""

    def test_budget_cuts_retries_short_and_bounds_wall_clock(self):
        # Each failed attempt burns ~0.2s, so a 0.5s budget cannot fund the full
        # 5 attempts even though max_retries would allow them.
        policy = EmbeddingRetryPolicy(max_retries=4, initial_backoff=0.05, max_backoff=0.2, budget_seconds=0.5)
        emb = _make_embeddings(policy)

        def slow_failure(**kwargs):
            time.sleep(0.2)
            raise _InternalServerError()

        emb._litellm.embedding.side_effect = slow_failure

        started = time.monotonic()
        with pytest.raises(_InternalServerError):
            emb.encode(["query"])
        elapsed = time.monotonic() - started

        assert emb._litellm.embedding.call_count < policy.max_retries + 1
        # Budget + one in-flight attempt is the ceiling; the point is that it is
        # bounded and small, not the exact figure.
        assert elapsed < policy.budget_seconds + 0.5

    def test_budget_is_shared_across_batches(self):
        # 6 texts at batch_size=2 -> 3 batches. A per-batch budget would let the
        # call spend 3x the configured ceiling.
        policy = EmbeddingRetryPolicy(max_retries=4, initial_backoff=0.05, max_backoff=0.2, budget_seconds=0.4)
        emb = _make_embeddings(policy, batch_size=2)

        def slow_failure(**kwargs):
            time.sleep(0.15)
            raise _InternalServerError()

        emb._litellm.embedding.side_effect = slow_failure

        started = time.monotonic()
        with pytest.raises(_InternalServerError):
            emb.encode([f"text {i}" for i in range(6)])
        elapsed = time.monotonic() - started

        # The first batch never succeeds, so the call aborts there.
        assert elapsed < policy.budget_seconds + 0.5

    def test_budget_not_consumed_by_successful_batches(self):
        # A long run of successful batches must not starve a later batch of its
        # retries — only failures and backoff sleeps count against the budget.
        policy = EmbeddingRetryPolicy(max_retries=4, initial_backoff=0.01, max_backoff=0.04, budget_seconds=0.5)
        emb = _make_embeddings(policy, batch_size=1)
        calls = {"n": 0}

        def side_effect(**kwargs):
            calls["n"] += 1
            time.sleep(0.1)  # successful but slow
            if calls["n"] == 4:
                raise _InternalServerError()
            return _response_for(kwargs["input"])

        emb._litellm.embedding.side_effect = side_effect

        result = emb.encode([f"text {i}" for i in range(5)])

        assert len(result) == 5
        # 5 batches + 1 retry of the batch that failed.
        assert emb._litellm.embedding.call_count == 6


class TestInitializeRetries:
    """Dimension detection gates startup; a flaky provider must not blow it up."""

    async def test_initialize_retries_transient_failure(self, monkeypatch):
        import litellm

        calls = {"n": 0}

        async def flaky_aembedding(**kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _InternalServerError()
            return _response_for(["test"])

        monkeypatch.setattr(litellm, "aembedding", flaky_aembedding, raising=False)

        emb = LiteLLMSDKEmbeddings(
            api_key="test_key",
            model="openai/text-embedding-qwen3-8b",
            api_base="https://example.invalid/api/v1",
            retry_policy=FAST_POLICY,
        )
        await emb.initialize()

        assert emb.dimension == 768
        assert calls["n"] == 3

    async def test_initialize_does_not_retry_auth_failure(self, monkeypatch):
        import litellm

        calls = {"n": 0}

        async def failing_aembedding(**kwargs):
            calls["n"] += 1
            raise _AuthenticationError()

        monkeypatch.setattr(litellm, "aembedding", failing_aembedding, raising=False)

        emb = LiteLLMSDKEmbeddings(
            api_key="bad_key",
            model="openai/text-embedding-qwen3-8b",
            api_base="https://example.invalid/api/v1",
            retry_policy=FAST_POLICY,
        )
        with pytest.raises(RuntimeError, match="Failed to initialize LiteLLM SDK embeddings"):
            await emb.initialize()

        assert calls["n"] == 1


class TestPolicyFromConfig:
    """Attempts and budget must be tunable without a code change."""

    def test_env_overrides_are_picked_up(self, monkeypatch):
        from hindsight_api.config import (
            ENV_EMBEDDINGS_INITIAL_BACKOFF,
            ENV_EMBEDDINGS_MAX_BACKOFF,
            ENV_EMBEDDINGS_MAX_RETRIES,
            ENV_EMBEDDINGS_RETRY_BUDGET,
            HindsightConfig,
        )
        from hindsight_api.engine.embeddings import _retry_policy_from_config

        monkeypatch.setenv(ENV_EMBEDDINGS_MAX_RETRIES, "2")
        monkeypatch.setenv(ENV_EMBEDDINGS_INITIAL_BACKOFF, "0.25")
        monkeypatch.setenv(ENV_EMBEDDINGS_MAX_BACKOFF, "2.5")
        monkeypatch.setenv(ENV_EMBEDDINGS_RETRY_BUDGET, "7.5")

        config = HindsightConfig.from_env()
        policy = _retry_policy_from_config(config)

        assert policy.max_retries == 2
        assert policy.initial_backoff == 0.25
        assert policy.max_backoff == 2.5
        assert policy.budget_seconds == 7.5

    def test_defaults_are_bounded(self):
        policy = EmbeddingRetryPolicy()

        assert 3 <= policy.max_retries + 1 <= 5
        assert policy.budget_seconds <= 15.0


class TestLiteLLMProxyEncodeRetries:
    """
    The `litellm` proxy provider is the sibling of `litellm-sdk` and shares the
    retry wrapper. It needs its own coverage because it fails differently: the
    proxy returns an ordinary HTTP response, so the 5xx only becomes an exception
    when `raise_for_status()` runs — and that call had to move *inside* the
    retried closure for a proxy 5xx to be retried rather than raised straight
    through to the caller.
    """

    def _make_proxy(self, policy: EmbeddingRetryPolicy, batch_size: int = 100) -> LiteLLMEmbeddings:
        emb = LiteLLMEmbeddings(
            api_base="https://proxy.invalid",
            api_key="test_key",
            model="text-embedding-3-small",
            batch_size=batch_size,
            retry_policy=policy,
        )
        emb._client = MagicMock()
        emb._dimension = 768
        return emb

    def _http_response(self, status_code: int, texts: list[str] | None = None) -> MagicMock:
        response = MagicMock()
        response.status_code = status_code
        if status_code >= 400:
            request = httpx.Request("POST", "https://proxy.invalid/embeddings")
            error = httpx.HTTPStatusError(
                f"{status_code} error", request=request, response=httpx.Response(status_code, request=request)
            )
            response.raise_for_status.side_effect = error
        else:
            response.json.return_value = {
                "data": [{"embedding": [0.1] * 768, "index": i} for i in range(len(texts or []))]
            }
        return response

    def test_proxy_5xx_is_retried_then_succeeds(self):
        emb = self._make_proxy(FAST_POLICY)
        texts = ["what did we decide about embeddings?"]
        emb._client.post.side_effect = [
            self._http_response(503),
            self._http_response(200, texts),
        ]

        result = emb.encode(texts)

        assert len(result) == 1
        assert len(result[0]) == 768
        assert emb._client.post.call_count == 2

    def test_proxy_4xx_raises_immediately_without_retrying(self):
        emb = self._make_proxy(FAST_POLICY)
        emb._client.post.return_value = self._http_response(401)

        with pytest.raises(httpx.HTTPStatusError):
            emb.encode(["query"])

        assert emb._client.post.call_count == 1
