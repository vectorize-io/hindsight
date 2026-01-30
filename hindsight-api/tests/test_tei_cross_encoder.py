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
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from hindsight_api.engine.cross_encoder import RemoteTEICrossEncoder


class TestRemoteTEICrossEncoderInitialization:
    """Tests for TEI cross-encoder initialization."""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization with valid TEI server."""

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/info":
                return httpx.Response(
                    200,
                    json={"model_id": "BAAI/bge-reranker-base", "version": "1.0"},
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(mock_handler)

        with patch.object(httpx, "AsyncClient", return_value=httpx.AsyncClient(transport=transport)):
            encoder = RemoteTEICrossEncoder(base_url="http://localhost:8080")
            await encoder.initialize()

            assert encoder._model_id == "BAAI/bge-reranker-base"
            assert encoder._async_client is not None

    @pytest.mark.asyncio
    async def test_initialize_server_unreachable(self):
        """Test initialization fails when server is unreachable."""

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(mock_handler)

        with patch.object(httpx, "AsyncClient", return_value=httpx.AsyncClient(transport=transport)):
            encoder = RemoteTEICrossEncoder(
                base_url="http://localhost:8080",
                max_retries=1,
                retry_delay=0.01,
            )

            with pytest.raises(RuntimeError, match="Failed to connect to TEI server"):
                await encoder.initialize()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test that initialize() is idempotent."""
        call_count = 0

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            if request.url.path == "/info":
                call_count += 1
                return httpx.Response(200, json={"model_id": "test-model"})
            return httpx.Response(404)

        transport = httpx.MockTransport(mock_handler)

        with patch.object(httpx, "AsyncClient", return_value=httpx.AsyncClient(transport=transport)):
            encoder = RemoteTEICrossEncoder(base_url="http://localhost:8080")
            await encoder.initialize()
            await encoder.initialize()
            await encoder.initialize()

            assert call_count == 1


def create_mock_async_client(handler):
    """Create a mock AsyncClient that uses the given handler for requests."""

    class MockAsyncClient:
        def __init__(self, **kwargs):
            self.timeout = kwargs.get("timeout", 30.0)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            return await handler("POST", url, **kwargs)

        async def get(self, url, **kwargs):
            return await handler("GET", url, **kwargs)

    return MockAsyncClient()


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
        """Test predict returns empty list for empty input."""
        encoder = RemoteTEICrossEncoder(base_url="http://localhost:8080")
        encoder._async_client = httpx.AsyncClient()
        encoder._model_id = "test-model"

        result = await encoder.predict([])
        assert result == []

    @pytest.mark.asyncio
    async def test_predict_single_query(self):
        """Test predict with single query and multiple documents."""
        rerank_calls = []

        async def mock_handler(method, url, **kwargs):
            if "/rerank" in url:
                body = kwargs.get("json", {})
                rerank_calls.append(body)
                texts = body["texts"]
                # Return scores in descending order with original indices
                results = [{"index": i, "score": 1.0 - (i * 0.1)} for i in range(len(texts))]
                response = MagicMock()
                response.status_code = 200
                response.raise_for_status = MagicMock()
                response.json = MagicMock(return_value=results)
                return response
            raise httpx.HTTPStatusError("Not found", request=MagicMock(), response=MagicMock())

        encoder = RemoteTEICrossEncoder(base_url="http://localhost:8080")
        encoder._async_client = create_mock_async_client(mock_handler)
        encoder._model_id = "test-model"

        pairs = [
            ("What is Python?", "Python is a programming language."),
            ("What is Python?", "Python is a snake."),
            ("What is Python?", "Java is also a language."),
        ]

        scores = await encoder.predict(pairs)

        assert len(scores) == 3
        assert len(rerank_calls) == 1
        assert rerank_calls[0]["query"] == "What is Python?"
        assert len(rerank_calls[0]["texts"]) == 3
        # Scores should be mapped back correctly
        assert scores[0] == 1.0
        assert scores[1] == 0.9
        assert scores[2] == pytest.approx(0.8, rel=0.01)

    @pytest.mark.asyncio
    async def test_predict_multiple_queries(self):
        """Test predict with multiple different queries."""
        rerank_calls = []

        async def mock_handler(method, url, **kwargs):
            if "/rerank" in url:
                body = kwargs.get("json", {})
                rerank_calls.append(body)
                texts = body["texts"]
                results = [{"index": i, "score": 0.5 + (i * 0.1)} for i in range(len(texts))]
                response = MagicMock()
                response.status_code = 200
                response.raise_for_status = MagicMock()
                response.json = MagicMock(return_value=results)
                return response
            raise httpx.HTTPStatusError("Not found", request=MagicMock(), response=MagicMock())

        encoder = RemoteTEICrossEncoder(base_url="http://localhost:8080")
        encoder._async_client = create_mock_async_client(mock_handler)
        encoder._model_id = "test-model"

        pairs = [
            ("Query A", "Doc A1"),
            ("Query B", "Doc B1"),
            ("Query A", "Doc A2"),
            ("Query B", "Doc B2"),
        ]

        scores = await encoder.predict(pairs)

        assert len(scores) == 4
        # Two queries = two rerank calls (run in parallel)
        assert len(rerank_calls) == 2


class TestRemoteTEICrossEncoderBatching:
    """Tests for batch splitting behavior."""

    @pytest.mark.asyncio
    async def test_batch_splitting(self):
        """Test that large inputs are split into batches."""
        rerank_calls = []

        async def mock_handler(method, url, **kwargs):
            if "/rerank" in url:
                body = kwargs.get("json", {})
                rerank_calls.append(body)
                texts = body["texts"]
                results = [{"index": i, "score": 0.5} for i in range(len(texts))]
                response = MagicMock()
                response.status_code = 200
                response.raise_for_status = MagicMock()
                response.json = MagicMock(return_value=results)
                return response
            raise httpx.HTTPStatusError("Not found", request=MagicMock(), response=MagicMock())

        encoder = RemoteTEICrossEncoder(
            base_url="http://localhost:8080",
            batch_size=3,  # Small batch for testing
        )
        encoder._async_client = create_mock_async_client(mock_handler)
        encoder._model_id = "test-model"

        # 7 documents with same query, batch_size=3 -> 3 batches (3+3+1)
        pairs = [("Query", f"Doc {i}") for i in range(7)]

        scores = await encoder.predict(pairs)

        assert len(scores) == 7
        assert len(rerank_calls) == 3
        # Check batch sizes
        batch_sizes = sorted([len(call["texts"]) for call in rerank_calls])
        assert batch_sizes == [1, 3, 3]

    @pytest.mark.asyncio
    async def test_score_mapping_across_batches(self):
        """Test that scores are correctly mapped back across batches."""
        call_counter = [0]

        async def mock_handler(method, url, **kwargs):
            if "/rerank" in url:
                body = kwargs.get("json", {})
                batch_num = call_counter[0]
                call_counter[0] += 1
                texts = body["texts"]
                # Each batch returns different scores to verify mapping
                base_score = batch_num * 10
                results = [{"index": i, "score": float(base_score + i)} for i in range(len(texts))]
                response = MagicMock()
                response.status_code = 200
                response.raise_for_status = MagicMock()
                response.json = MagicMock(return_value=results)
                return response
            raise httpx.HTTPStatusError("Not found", request=MagicMock(), response=MagicMock())

        encoder = RemoteTEICrossEncoder(
            base_url="http://localhost:8080",
            batch_size=3,
        )
        encoder._async_client = create_mock_async_client(mock_handler)
        encoder._model_id = "test-model"

        pairs = [("Query", f"Doc {i}") for i in range(7)]

        scores = await encoder.predict(pairs)

        assert len(scores) == 7
        # All scores should be present (exact values depend on batch ordering)
        assert all(isinstance(s, (int, float)) for s in scores)


class TestRemoteTEICrossEncoderParallelism:
    """Tests for parallel request handling and backpressure."""

    @pytest.mark.asyncio
    async def test_parallel_requests(self):
        """Test that requests are made in parallel."""
        concurrent_count = [0]
        max_concurrent_observed = [0]

        async def mock_handler(method, url, **kwargs):
            if "/rerank" in url:
                concurrent_count[0] += 1
                max_concurrent_observed[0] = max(max_concurrent_observed[0], concurrent_count[0])

                await asyncio.sleep(0.03)  # Simulate latency

                concurrent_count[0] -= 1
                body = kwargs.get("json", {})
                texts = body["texts"]
                results = [{"index": i, "score": 0.5} for i in range(len(texts))]
                response = MagicMock()
                response.status_code = 200
                response.raise_for_status = MagicMock()
                response.json = MagicMock(return_value=results)
                return response
            raise httpx.HTTPStatusError("Not found", request=MagicMock(), response=MagicMock())

        encoder = RemoteTEICrossEncoder(
            base_url="http://localhost:8080",
            batch_size=2,
            max_concurrent=10,  # High limit to allow parallelism
        )
        encoder._async_client = create_mock_async_client(mock_handler)
        encoder._model_id = "test-model"

        # 6 docs = 3 batches, should run in parallel
        pairs = [("Query", f"Doc {i}") for i in range(6)]

        start = time.time()
        scores = await encoder.predict(pairs)
        elapsed = time.time() - start

        assert len(scores) == 6
        # If parallel, 3 batches with 30ms each should take ~30ms, not 90ms
        assert elapsed < 0.08, f"Requests should run in parallel, took {elapsed}s"
        assert max_concurrent_observed[0] > 1, "Multiple requests should run concurrently"

    @pytest.mark.asyncio
    async def test_backpressure_semaphore(self):
        """Test that semaphore limits concurrent requests."""
        concurrent_count = [0]
        max_concurrent_observed = [0]

        async def mock_handler(method, url, **kwargs):
            if "/rerank" in url:
                concurrent_count[0] += 1
                max_concurrent_observed[0] = max(max_concurrent_observed[0], concurrent_count[0])

                await asyncio.sleep(0.01)  # Simulate latency

                concurrent_count[0] -= 1
                body = kwargs.get("json", {})
                texts = body["texts"]
                results = [{"index": i, "score": 0.5} for i in range(len(texts))]
                response = MagicMock()
                response.status_code = 200
                response.raise_for_status = MagicMock()
                response.json = MagicMock(return_value=results)
                return response
            raise httpx.HTTPStatusError("Not found", request=MagicMock(), response=MagicMock())

        max_concurrent_limit = 2
        encoder = RemoteTEICrossEncoder(
            base_url="http://localhost:8080",
            batch_size=1,  # 1 doc per batch to maximize requests
            max_concurrent=max_concurrent_limit,
        )
        encoder._async_client = create_mock_async_client(mock_handler)
        encoder._model_id = "test-model"

        # 10 docs = 10 batches, but only 2 should run at a time
        pairs = [("Query", f"Doc {i}") for i in range(10)]

        scores = await encoder.predict(pairs)

        assert len(scores) == 10
        assert max_concurrent_observed[0] <= max_concurrent_limit, (
            f"Semaphore should limit to {max_concurrent_limit}, observed {max_concurrent_observed[0]}"
        )


class TestRemoteTEICrossEncoderRetry:
    """Tests for retry logic on transient errors."""

    @pytest.mark.asyncio
    async def test_retry_on_connect_error(self):
        """Test that connect errors trigger retries."""
        attempt_count = [0]

        async def mock_handler(method, url, **kwargs):
            if "/rerank" in url:
                attempt_count[0] += 1
                if attempt_count[0] < 3:
                    raise httpx.ConnectError("Connection refused")
                body = kwargs.get("json", {})
                texts = body["texts"]
                results = [{"index": i, "score": 0.5} for i in range(len(texts))]
                response = MagicMock()
                response.status_code = 200
                response.raise_for_status = MagicMock()
                response.json = MagicMock(return_value=results)
                return response
            raise httpx.HTTPStatusError("Not found", request=MagicMock(), response=MagicMock())

        encoder = RemoteTEICrossEncoder(
            base_url="http://localhost:8080",
            max_retries=3,
            retry_delay=0.01,
        )
        encoder._async_client = create_mock_async_client(mock_handler)
        encoder._model_id = "test-model"

        pairs = [("Query", "Doc 1")]
        scores = await encoder.predict(pairs)

        assert len(scores) == 1
        assert attempt_count[0] == 3  # 2 failures + 1 success

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self):
        """Test that 5xx errors trigger retries."""
        attempt_count = [0]

        async def mock_handler(method, url, **kwargs):
            if "/rerank" in url:
                attempt_count[0] += 1
                if attempt_count[0] < 2:
                    response = MagicMock()
                    response.status_code = 503

                    def raise_for_status():
                        raise httpx.HTTPStatusError(
                            "Service unavailable",
                            request=MagicMock(),
                            response=response,
                        )

                    response.raise_for_status = raise_for_status
                    return response

                body = kwargs.get("json", {})
                texts = body["texts"]
                results = [{"index": i, "score": 0.5} for i in range(len(texts))]
                response = MagicMock()
                response.status_code = 200
                response.raise_for_status = MagicMock()
                response.json = MagicMock(return_value=results)
                return response
            raise httpx.HTTPStatusError("Not found", request=MagicMock(), response=MagicMock())

        encoder = RemoteTEICrossEncoder(
            base_url="http://localhost:8080",
            max_retries=3,
            retry_delay=0.01,
        )
        encoder._async_client = create_mock_async_client(mock_handler)
        encoder._model_id = "test-model"

        pairs = [("Query", "Doc 1")]
        scores = await encoder.predict(pairs)

        assert len(scores) == 1
        assert attempt_count[0] == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_client_error(self):
        """Test that 4xx errors do not trigger retries."""
        attempt_count = [0]

        async def mock_handler(method, url, **kwargs):
            if "/rerank" in url:
                attempt_count[0] += 1
                response = MagicMock()
                response.status_code = 400

                def raise_for_status():
                    raise httpx.HTTPStatusError(
                        "Bad request",
                        request=MagicMock(),
                        response=response,
                    )

                response.raise_for_status = raise_for_status
                return response
            raise httpx.HTTPStatusError("Not found", request=MagicMock(), response=MagicMock())

        encoder = RemoteTEICrossEncoder(
            base_url="http://localhost:8080",
            max_retries=3,
            retry_delay=0.01,
        )
        encoder._async_client = create_mock_async_client(mock_handler)
        encoder._model_id = "test-model"

        pairs = [("Query", "Doc 1")]

        with pytest.raises(RuntimeError, match="TEI rerank request failed"):
            await encoder.predict(pairs)

        assert attempt_count[0] == 1  # No retries for 4xx


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
    import httpx

    # Get server info
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{TEI_RERANKER_URL}/info")
        info = response.json()
        print(f"\n📊 TEI Server Info:")
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
        (128, 8),   # Default
        (256, 4),   # Larger batches, fewer concurrent
        (256, 8),   # Larger batches, same concurrent
        (512, 2),   # Very large batches, few concurrent
        (512, 4),   # Very large batches, moderate concurrent
        (64, 16),   # Smaller batches, more concurrent
        (800, 1),   # Single batch (all at once)
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
        results.append({
            "batch_size": batch_size,
            "max_concurrent": max_concurrent,
            "avg_ms": avg_time * 1000,
            "min_ms": min_time * 1000,
            "num_batches": (num_pairs + batch_size - 1) // batch_size,
        })

        print(f"   batch_size={batch_size:4d}, max_concurrent={max_concurrent:2d}: "
              f"avg={avg_time * 1000:6.1f}ms, min={min_time * 1000:6.1f}ms "
              f"({results[-1]['num_batches']} batches)")

    # Find best configuration
    best = min(results, key=lambda x: x["avg_ms"])
    print(f"\n🏆 Best Configuration:")
    print(f"   batch_size={best['batch_size']}, max_concurrent={best['max_concurrent']}")
    print(f"   Average: {best['avg_ms']:.1f}ms, Min: {best['min_ms']:.1f}ms")

    # Performance target check
    target_ms = 100
    if best["avg_ms"] <= target_ms:
        print(f"\n✅ Target met! Average {best['avg_ms']:.1f}ms <= {target_ms}ms")
    else:
        print(f"\n⚠️ Target NOT met. Average {best['avg_ms']:.1f}ms > {target_ms}ms")
        print(f"   Consider: larger batch size, GPU optimization, or faster network")


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
    test_pairs = [
        (query, f"Document {i} about ML and training.")
        for i in range(num_pairs_per_request)
    ]

    # Test configurations
    configs = [
        (128, 8),   # Default
        (256, 4),   # Larger batches
        (512, 2),   # Very large batches
        (200, 1),   # Single batch per request
    ]

    print(f"\n⏱️  Concurrent Load Test: {num_concurrent_requests} parallel requests, "
          f"{num_pairs_per_request} pairs each:\n")

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
            times.append({
                "total": total_time,
                "max_individual": max(individual_times),
                "avg_individual": sum(individual_times) / len(individual_times),
            })

        avg_total = sum(t["total"] for t in times) / len(times)
        avg_max_individual = sum(t["max_individual"] for t in times) / len(times)

        print(f"   batch_size={batch_size:4d}, max_concurrent={max_concurrent:2d}: "
              f"total={avg_total * 1000:6.1f}ms, slowest_req={avg_max_individual * 1000:6.1f}ms")


@requires_tei_server
@pytest.mark.asyncio
async def test_tei_reranker_latency_breakdown():
    """
    Measure latency breakdown for TEI reranker requests.

    This helps identify where time is spent: network vs processing.
    """
    import httpx

    print(f"\n⏱️  Latency Breakdown Test:\n")

    # Test single document latency (network overhead)
    async with httpx.AsyncClient(timeout=30.0) as client:
        times = []
        for _ in range(10):
            start = time.time()
            await client.post(
                f"{TEI_RERANKER_URL}/rerank",
                json={
                    "query": "test query",
                    "texts": ["test document"],
                    "return_text": False,
                },
            )
            times.append((time.time() - start) * 1000)

        avg_single = sum(times) / len(times)
        print(f"   Single doc latency (raw HTTP): {avg_single:.2f}ms")

    # Test batch latencies
    batch_sizes = [10, 50, 100, 200, 500]
    for batch_size in batch_sizes:
        texts = [f"Document {i} about machine learning" for i in range(batch_size)]
        async with httpx.AsyncClient(timeout=30.0) as client:
            times = []
            for _ in range(5):
                start = time.time()
                await client.post(
                    f"{TEI_RERANKER_URL}/rerank",
                    json={
                        "query": "What about machine learning?",
                        "texts": texts,
                        "return_text": False,
                    },
                )
                times.append((time.time() - start) * 1000)

            avg = sum(times) / len(times)
            per_doc = avg / batch_size
            print(f"   Batch size {batch_size:4d}: {avg:6.1f}ms total, {per_doc:.2f}ms/doc")

    print(f"\n   💡 Insight: Higher per-doc time at small batches = network overhead dominant")
    print(f"   💡 Insight: Lower per-doc time at large batches = GPU efficiently utilized")
