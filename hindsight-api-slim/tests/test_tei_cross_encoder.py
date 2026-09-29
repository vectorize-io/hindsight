"""
Tests for RemoteTEICrossEncoder (TEI reranker client).

Tests cover:
- Initialization and server connectivity
- Basic predict functionality
- Batch splitting
- Parallel request handling
- Backpressure/semaphore behavior
- Retry logic on transient errors
- Multiple queries handling

The upstream is a real in-process HTTP server (``tests/aiohttp_stub.py``), so these
exercise the actual aiohttp transport rather than a mocked client.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import patch

import pytest
from aiohttp import web

from hindsight_api.engine.aiohttp_session import UpstreamHTTPError
from hindsight_api.engine.cross_encoder import RemoteTEICrossEncoder
from tests.aiohttp_stub import stub_server

# Nothing listens on port 1, so connecting is refused immediately.
UNREACHABLE_URL = "http://127.0.0.1:1"


def rerank_handler(
    score_for: Callable[[dict[str, Any]], list[dict[str, Any]]],
    *,
    calls: list[dict[str, Any]] | None = None,
    before: Callable[[], Awaitable[None]] | None = None,
) -> Callable[[web.Request], Awaitable[web.StreamResponse]]:
    """A TEI stub: answers /info, and /rerank with ``score_for(body)``."""

    async def handler(request: web.Request) -> web.StreamResponse:
        if request.path == "/info":
            return web.json_response({"model_id": "test-model"})
        if request.path == "/rerank":
            body = await request.json()
            if calls is not None:
                calls.append({"body": body, "headers": dict(request.headers)})
            if before is not None:
                await before()
            return web.json_response(score_for(body))
        return web.Response(status=404)

    return handler


def constant_scores(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"index": i, "score": 0.5} for i in range(len(body["texts"]))]


def ready_encoder(base_url: str, **kwargs: Any) -> RemoteTEICrossEncoder:
    """An encoder marked initialized without the /info round trip."""
    encoder = RemoteTEICrossEncoder(base_url=base_url, **kwargs)
    encoder._model_id = "test-model"
    return encoder


class TestRemoteTEICrossEncoderInitialization:
    """Tests for TEI cross-encoder initialization."""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization with valid TEI server."""

        async def handler(request: web.Request) -> web.StreamResponse:
            if request.path == "/info":
                return web.json_response({"model_id": "BAAI/bge-reranker-base", "version": "1.0"})
            return web.Response(status=404)

        async with stub_server(handler) as base_url:
            encoder = RemoteTEICrossEncoder(base_url=base_url)
            await encoder.initialize()

            assert encoder._model_id == "BAAI/bge-reranker-base"

    @pytest.mark.asyncio
    async def test_initialize_server_unreachable(self):
        """Test initialization fails when server is unreachable."""
        encoder = RemoteTEICrossEncoder(
            base_url=UNREACHABLE_URL,
            max_retries=1,
            retry_delay=0.01,
        )

        with pytest.raises(RuntimeError, match="Failed to connect to TEI server"):
            await encoder.initialize()
        assert encoder._model_id is None

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test that initialize() is idempotent."""
        call_count = 0

        async def handler(request: web.Request) -> web.StreamResponse:
            nonlocal call_count
            if request.path == "/info":
                call_count += 1
                return web.json_response({"model_id": "test-model"})
            return web.Response(status=404)

        async with stub_server(handler) as base_url:
            encoder = RemoteTEICrossEncoder(base_url=base_url)
            await encoder.initialize()
            await encoder.initialize()
            await encoder.initialize()

        assert call_count == 1


class TestRemoteTEICrossEncoderPredict:
    """Tests for TEI cross-encoder predict functionality."""

    @pytest.mark.asyncio
    async def test_predict_not_initialized(self):
        """Test predict raises error when not initialized."""
        encoder = RemoteTEICrossEncoder(base_url="http://localhost:8080")

        with pytest.raises(RuntimeError, match="Reranker not initialized"):
            await encoder.predict([("query", "doc")])

    @pytest.mark.asyncio
    async def test_predict_empty_pairs(self):
        """Test predict returns empty list for empty input (no request made)."""
        encoder = ready_encoder(UNREACHABLE_URL)

        result = await encoder.predict([])
        assert result == []

    @pytest.mark.asyncio
    async def test_predict_single_query(self):
        """Test predict with single query and multiple documents."""
        calls: list[dict[str, Any]] = []

        def descending(body: dict[str, Any]) -> list[dict[str, Any]]:
            # Return scores in descending order with original indices
            return [{"index": i, "score": 1.0 - (i * 0.1)} for i in range(len(body["texts"]))]

        pairs = [
            ("What is Python?", "Python is a programming language."),
            ("What is Python?", "Python is a snake."),
            ("What is Python?", "Java is also a language."),
        ]

        async with stub_server(rerank_handler(descending, calls=calls)) as base_url:
            encoder = ready_encoder(base_url)
            with patch(
                "hindsight_api.engine.cross_encoder.reranker_bank_attribution_headers",
                return_value={"X-Hindsight-Bank-Id": "bank-tei"},
            ):
                scores = await encoder.predict(pairs)

        assert len(scores) == 3
        assert len(calls) == 1
        assert calls[0]["body"]["query"] == "What is Python?"
        assert len(calls[0]["body"]["texts"]) == 3
        assert calls[0]["body"]["return_text"] is False
        assert calls[0]["headers"]["X-Hindsight-Bank-Id"] == "bank-tei"
        # Scores should be mapped back correctly
        assert scores[0] == 1.0
        assert scores[1] == 0.9
        assert scores[2] == pytest.approx(0.8, rel=0.01)

    @pytest.mark.asyncio
    async def test_predict_multiple_queries(self):
        """Test predict with multiple different queries."""
        calls: list[dict[str, Any]] = []

        def ascending(body: dict[str, Any]) -> list[dict[str, Any]]:
            return [{"index": i, "score": 0.5 + (i * 0.1)} for i in range(len(body["texts"]))]

        pairs = [
            ("Query A", "Doc A1"),
            ("Query B", "Doc B1"),
            ("Query A", "Doc A2"),
            ("Query B", "Doc B2"),
        ]

        async with stub_server(rerank_handler(ascending, calls=calls)) as base_url:
            scores = await ready_encoder(base_url).predict(pairs)

        assert len(scores) == 4
        # Two queries = two rerank calls (run in parallel)
        assert len(calls) == 2


class TestRemoteTEICrossEncoderBatching:
    """Tests for batch splitting behavior."""

    @pytest.mark.asyncio
    async def test_batch_splitting(self):
        """Test that large inputs are split into batches."""
        calls: list[dict[str, Any]] = []

        # 7 documents with same query, batch_size=3 -> 3 batches (3+3+1)
        pairs = [("Query", f"Doc {i}") for i in range(7)]

        async with stub_server(rerank_handler(constant_scores, calls=calls)) as base_url:
            scores = await ready_encoder(base_url, batch_size=3).predict(pairs)

        assert len(scores) == 7
        assert len(calls) == 3
        # Check batch sizes
        batch_sizes = sorted([len(call["body"]["texts"]) for call in calls])
        assert batch_sizes == [1, 3, 3]

    @pytest.mark.asyncio
    async def test_score_mapping_across_batches(self):
        """Test that scores are correctly mapped back across batches."""

        def score_by_text(body: dict[str, Any]) -> list[dict[str, Any]]:
            # Score each doc by the number in its text, so the mapping is checkable
            # regardless of the order batches arrive in.
            return [{"index": i, "score": float(text.split()[-1])} for i, text in enumerate(body["texts"])]

        pairs = [("Query", f"Doc {i}") for i in range(7)]

        async with stub_server(rerank_handler(score_by_text)) as base_url:
            scores = await ready_encoder(base_url, batch_size=3).predict(pairs)

        assert scores == [float(i) for i in range(7)]


class TestRemoteTEICrossEncoderParallelism:
    """Tests for parallel request handling and backpressure."""

    @pytest.mark.asyncio
    async def test_parallel_requests(self):
        """Test that requests are made in parallel."""
        concurrent_count = [0]
        max_concurrent_observed = [0]

        async def simulate_latency() -> None:
            concurrent_count[0] += 1
            max_concurrent_observed[0] = max(max_concurrent_observed[0], concurrent_count[0])
            await asyncio.sleep(0.1)
            concurrent_count[0] -= 1

        # 6 docs = 3 batches, should run in parallel
        pairs = [("Query", f"Doc {i}") for i in range(6)]

        async with stub_server(rerank_handler(constant_scores, before=simulate_latency)) as base_url:
            encoder = ready_encoder(
                base_url,
                batch_size=2,
                max_concurrent=10,  # High limit to allow parallelism
            )
            start = time.time()
            scores = await encoder.predict(pairs)
            elapsed = time.time() - start

        assert len(scores) == 6
        # 6 docs = 3 batches at ~0.1s each: ~0.1s if parallel vs ~0.3s if serial.
        # Assert comfortably below the serial time so CI scheduling jitter can't flake
        # it; max_concurrent_observed below is the deterministic proof that the batches
        # actually overlapped.
        assert elapsed < 0.25, f"Requests should run in parallel, took {elapsed}s"
        assert max_concurrent_observed[0] > 1, "Multiple requests should run concurrently"

    @pytest.mark.asyncio
    async def test_backpressure_semaphore(self):
        """Test that semaphore limits concurrent requests."""
        concurrent_count = [0]
        max_concurrent_observed = [0]

        async def simulate_latency() -> None:
            concurrent_count[0] += 1
            max_concurrent_observed[0] = max(max_concurrent_observed[0], concurrent_count[0])
            await asyncio.sleep(0.01)
            concurrent_count[0] -= 1

        max_concurrent_limit = 2
        # 10 docs = 10 batches, but only 2 should run at a time
        pairs = [("Query", f"Doc {i}") for i in range(10)]

        async with stub_server(rerank_handler(constant_scores, before=simulate_latency)) as base_url:
            encoder = ready_encoder(
                base_url,
                batch_size=1,  # 1 doc per batch to maximize requests
                max_concurrent=max_concurrent_limit,
            )
            scores = await encoder.predict(pairs)

        assert len(scores) == 10
        assert max_concurrent_observed[0] <= max_concurrent_limit, (
            f"Semaphore should limit to {max_concurrent_limit}, observed {max_concurrent_observed[0]}"
        )


class _RefuseFirstConnects:
    """Session wrapper whose first ``failures`` requests go to a port nothing listens on.

    That produces a genuine aiohttp connect error from the real transport, then lets
    later attempts through to the stub.
    """

    def __init__(self, session: Any, failures: int) -> None:
        self._session = session
        self._failures = failures
        self.attempts = 0

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        self.attempts += 1
        if self.attempts <= self._failures:
            url = f"{UNREACHABLE_URL}/rerank"
        return self._session.request(method, url, **kwargs)


class TestRemoteTEICrossEncoderRetry:
    """Tests for retry logic on transient errors."""

    @pytest.mark.asyncio
    async def test_retry_on_connect_error(self):
        """Test that connect errors trigger retries."""
        async with stub_server(rerank_handler(constant_scores)) as base_url:
            encoder = ready_encoder(base_url, max_retries=3, retry_delay=0.01)
            session = encoder._session.get()
            wrapper = _RefuseFirstConnects(session, failures=2)
            with patch.object(encoder._session, "get", return_value=wrapper):
                scores = await encoder.predict([("Query", "Doc 1")])

        assert len(scores) == 1
        assert wrapper.attempts == 3  # 2 failures + 1 success

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status_code", "error_message"),
        [(429, "Too many requests"), (503, "Service unavailable")],
    )
    async def test_retry_on_transient_status(self, status_code, error_message):
        """TEI overload and 5xx responses should trigger retries."""
        attempt_count = [0]

        async def handler(request: web.Request) -> web.StreamResponse:
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                return web.json_response({"error": error_message}, status=status_code)
            body = await request.json()
            return web.json_response(constant_scores(body))

        async with stub_server(handler) as base_url:
            encoder = ready_encoder(base_url, max_retries=3, retry_delay=0.01)
            scores = await encoder.predict([("Query", "Doc 1")])

        assert len(scores) == 1
        assert attempt_count[0] == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_client_error(self):
        """Test that 4xx errors do not trigger retries."""
        attempt_count = [0]

        async def handler(request: web.Request) -> web.StreamResponse:
            attempt_count[0] += 1
            return web.json_response({"error": "Bad request"}, status=400)

        async with stub_server(handler) as base_url:
            encoder = ready_encoder(base_url, max_retries=3, retry_delay=0.01)
            with pytest.raises(RuntimeError, match="TEI rerank request failed"):
                await encoder.predict([("Query", "Doc 1")])

        assert attempt_count[0] == 1  # No retries for 4xx

    @pytest.mark.asyncio
    async def test_persistent_too_many_requests_exhausts_retry_budget(self):
        """A persistent overload should make exactly max_retries + 1 attempts."""
        attempt_count = 0

        async def handler(request: web.Request) -> web.StreamResponse:
            nonlocal attempt_count
            attempt_count += 1
            return web.json_response(
                {"error": "Model is overloaded"},
                status=429,
                headers={"Retry-After": "Infinity"},
            )

        async with stub_server(handler) as base_url:
            encoder = ready_encoder(base_url, max_retries=2, retry_delay=0)
            with pytest.raises(RuntimeError, match="TEI rerank request failed") as exc_info:
                await encoder.predict([("Query", "Doc 1")])

        assert attempt_count == 3
        assert isinstance(exc_info.value.__context__, UpstreamHTTPError)
        assert exc_info.value.__context__.status_code == 429


class TestRemoteTEICrossEncoderConfig:
    """Tests for configuration from environment variables."""

    def test_default_values(self):
        """Test default configuration values."""
        encoder = RemoteTEICrossEncoder(base_url="http://localhost:8080")

        assert encoder.batch_size == 128
        assert encoder.max_concurrent == 8
        assert encoder.timeout == 30.0
        assert encoder.max_retries == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        encoder = RemoteTEICrossEncoder(
            base_url="http://localhost:8080",
            batch_size=64,
            max_concurrent=4,
            timeout=60.0,
            max_retries=5,
            retry_delay=1.0,
        )

        assert encoder.batch_size == 64
        assert encoder.max_concurrent == 4
        assert encoder.timeout == 60.0
        assert encoder.max_retries == 5
        assert encoder.retry_delay == 1.0

    def test_create_from_env(self):
        """Test creating encoder from environment variables."""
        import os

        from hindsight_api.config import clear_config_cache
        from hindsight_api.engine.cross_encoder import create_cross_encoder_from_env

        with patch.dict(
            os.environ,
            {
                "HINDSIGHT_API_RERANKER_PROVIDER": "tei",
                "HINDSIGHT_API_RERANKER_TEI_URL": "http://test:9000",
                "HINDSIGHT_API_RERANKER_TEI_BATCH_SIZE": "256",
                "HINDSIGHT_API_RERANKER_TEI_MAX_CONCURRENT": "16",
            },
        ):
            clear_config_cache()  # Clear cache to pick up patched env vars
            encoder = create_cross_encoder_from_env()

            assert isinstance(encoder, RemoteTEICrossEncoder)
            assert encoder.base_url == "http://test:9000"
            assert encoder.batch_size == 256
            assert encoder.max_concurrent == 16

        clear_config_cache()  # Clear cache after test

    def test_create_from_env_with_custom_timeout(self):
        """Test that HINDSIGHT_API_RERANKER_TEI_HTTP_TIMEOUT is respected."""
        import os

        from hindsight_api.config import clear_config_cache
        from hindsight_api.engine.cross_encoder import create_cross_encoder_from_env

        with patch.dict(
            os.environ,
            {
                "HINDSIGHT_API_RERANKER_PROVIDER": "tei",
                "HINDSIGHT_API_RERANKER_TEI_URL": "http://test:9000",
                "HINDSIGHT_API_RERANKER_TEI_HTTP_TIMEOUT": "120.0",
            },
        ):
            clear_config_cache()
            encoder = create_cross_encoder_from_env()

            assert isinstance(encoder, RemoteTEICrossEncoder)
            assert encoder.timeout == 120.0

        clear_config_cache()


# ============================================================================
# TEI Reranker Performance Benchmark Tests
# ============================================================================
# These tests require a running TEI server to measure actual performance.
# Set TEI_RERANKER_URL environment variable to run.
# Example:
#   TEI_RERANKER_URL=http://localhost:8000 \
#   pytest tests/test_tei_cross_encoder.py::test_tei_reranker_performance -v -s -n0

import os

TEI_RERANKER_URL = os.environ.get("TEI_RERANKER_URL")

requires_tei_server = pytest.mark.skipif(
    TEI_RERANKER_URL is None,
    reason="TEI_RERANKER_URL not set - skipping TEI performance benchmark",
)


@requires_tei_server
@pytest.mark.asyncio
async def test_tei_reranker_performance():
    """
    Benchmark TEI reranker performance with different configurations.

    This test measures latency for different batch sizes and concurrency levels
    to find the optimal configuration for your TEI server.

    Example usage:
        TEI_RERANKER_URL=http://localhost:8000 \
        pytest tests/test_tei_cross_encoder.py::test_tei_reranker_performance -v -s -n0
    """
    import aiohttp

    # Get server info
    async with aiohttp.ClientSession() as client, client.get(f"{TEI_RERANKER_URL}/info") as response:
        info = await response.json()
        print("\n📊 TEI Server Info:")
        print(f"   URL: {TEI_RERANKER_URL}")
        print(f"   Model: {info.get('model_id', 'unknown')}")
        if "reranker_model" in info:
            print(f"   Reranker Model: {info['reranker_model']}")

    # Generate test data (800 pairs to simulate real workload)
    num_pairs = 800
    query = "What did I say about training machine learning models and artificial intelligence?"
    test_pairs = [
        (query, f"Document {i} about machine learning, neural networks, and AI training techniques.")
        for i in range(num_pairs)
    ]

    # Test configurations: (batch_size, max_concurrent)
    configs = [
        (128, 8),  # Default
        (256, 4),  # Larger batches, fewer concurrent
        (256, 8),  # Larger batches, same concurrent
        (512, 2),  # Very large batches, few concurrent
        (512, 4),  # Very large batches, moderate concurrent
        (64, 16),  # Smaller batches, more concurrent
        (800, 1),  # Single batch (all at once)
    ]

    results = []
    print(f"\n⏱️  Benchmarking {num_pairs} pairs with different configurations:\n")

    for batch_size, max_concurrent in configs:
        encoder = RemoteTEICrossEncoder(
            base_url=TEI_RERANKER_URL,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            timeout=60.0,
        )
        await encoder.initialize()

        # Warm-up run
        await encoder.predict(test_pairs[:100])

        # Timed runs (3 iterations)
        times = []
        for _ in range(3):
            start = time.time()
            scores = await encoder.predict(test_pairs)
            elapsed = time.time() - start
            times.append(elapsed)
            assert len(scores) == num_pairs

        avg_time = sum(times) / len(times)
        min_time = min(times)
        results.append(
            {
                "batch_size": batch_size,
                "max_concurrent": max_concurrent,
                "avg_ms": avg_time * 1000,
                "min_ms": min_time * 1000,
                "num_batches": (num_pairs + batch_size - 1) // batch_size,
            }
        )

        print(
            f"   batch_size={batch_size:4d}, max_concurrent={max_concurrent:2d}: "
            f"avg={avg_time * 1000:6.1f}ms, min={min_time * 1000:6.1f}ms "
            f"({results[-1]['num_batches']} batches)"
        )

    # Find best configuration
    best = min(results, key=lambda x: x["avg_ms"])
    print("\n🏆 Best Configuration:")
    print(f"   batch_size={best['batch_size']}, max_concurrent={best['max_concurrent']}")
    print(f"   Average: {best['avg_ms']:.1f}ms, Min: {best['min_ms']:.1f}ms")

    # Performance target check
    target_ms = 100
    if best["avg_ms"] <= target_ms:
        print(f"\n✅ Target met! Average {best['avg_ms']:.1f}ms <= {target_ms}ms")
    else:
        print(f"\n⚠️ Target NOT met. Average {best['avg_ms']:.1f}ms > {target_ms}ms")
        print("   Consider: larger batch size, GPU optimization, or faster network")


@requires_tei_server
@pytest.mark.asyncio
async def test_tei_reranker_concurrent_requests():
    """
    Test TEI reranker performance under concurrent request load.

    This simulates multiple parallel recall requests hitting the reranker
    at the same time.
    """
    # Smaller batches to simulate typical recall workload
    num_pairs_per_request = 200
    num_concurrent_requests = 4

    query = "Tell me about machine learning and AI training"
    test_pairs = [(query, f"Document {i} about ML and training.") for i in range(num_pairs_per_request)]

    # Test configurations
    configs = [
        (128, 8),  # Default
        (256, 4),  # Larger batches
        (512, 2),  # Very large batches
        (200, 1),  # Single batch per request
    ]

    print(
        f"\n⏱️  Concurrent Load Test: {num_concurrent_requests} parallel requests, {num_pairs_per_request} pairs each:\n"
    )

    for batch_size, max_concurrent in configs:
        encoder = RemoteTEICrossEncoder(
            base_url=TEI_RERANKER_URL,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            timeout=60.0,
        )
        await encoder.initialize()

        # Warm-up
        await encoder.predict(test_pairs[:50])

        async def run_single_request():
            start = time.time()
            scores = await encoder.predict(test_pairs)
            return time.time() - start, len(scores)

        # Run concurrent requests
        times = []
        for _ in range(3):  # 3 iterations
            start = time.time()
            results = await asyncio.gather(*[run_single_request() for _ in range(num_concurrent_requests)])
            total_time = time.time() - start

            individual_times = [r[0] for r in results]
            times.append(
                {
                    "total": total_time,
                    "max_individual": max(individual_times),
                    "avg_individual": sum(individual_times) / len(individual_times),
                }
            )

        avg_total = sum(t["total"] for t in times) / len(times)
        avg_max_individual = sum(t["max_individual"] for t in times) / len(times)

        print(
            f"   batch_size={batch_size:4d}, max_concurrent={max_concurrent:2d}: "
            f"total={avg_total * 1000:6.1f}ms, slowest_req={avg_max_individual * 1000:6.1f}ms"
        )


@requires_tei_server
@pytest.mark.asyncio
async def test_tei_reranker_latency_breakdown():
    """
    Measure latency breakdown for TEI reranker requests.

    This helps identify where time is spent: network vs processing.
    """
    import aiohttp

    print("\n⏱️  Latency Breakdown Test:\n")

    timeout = aiohttp.ClientTimeout(total=30.0)

    # Test single document latency (network overhead)
    async with aiohttp.ClientSession(timeout=timeout) as client:
        times = []
        for _ in range(10):
            start = time.time()
            async with client.post(
                f"{TEI_RERANKER_URL}/rerank",
                json={
                    "query": "test query",
                    "texts": ["test document"],
                    "return_text": False,
                },
            ) as response:
                await response.read()
            times.append((time.time() - start) * 1000)

        avg_single = sum(times) / len(times)
        print(f"   Single doc latency (raw HTTP): {avg_single:.2f}ms")

    # Test batch latencies
    batch_sizes = [10, 50, 100, 200, 500]
    for batch_size in batch_sizes:
        texts = [f"Document {i} about machine learning" for i in range(batch_size)]
        async with aiohttp.ClientSession(timeout=timeout) as client:
            times = []
            for _ in range(5):
                start = time.time()
                async with client.post(
                    f"{TEI_RERANKER_URL}/rerank",
                    json={
                        "query": "What about machine learning?",
                        "texts": texts,
                        "return_text": False,
                    },
                ) as response:
                    await response.read()
                times.append((time.time() - start) * 1000)

            avg = sum(times) / len(times)
            per_doc = avg / batch_size
            print(f"   Batch size {batch_size:4d}: {avg:6.1f}ms total, {per_doc:.2f}ms/doc")

    print("\n   💡 Insight: Higher per-doc time at small batches = network overhead dominant")
    print("   💡 Insight: Lower per-doc time at large batches = GPU efficiently utilized")
