"""
Tests for the Groq Batch API module.

Tests cover:
- Batch request/response model serialization
- JSONL building
- Batch queue enqueue/flush/poll lifecycle
- Batch result processing
- Config validation for batch mode
- Graceful fallback on batch failure
"""

import asyncio
import json
import os

import pytest

from hindsight_api.engine.batch.client import GroqBatchClient
from hindsight_api.engine.batch.models import (
    BatchJob,
    BatchRequest,
    BatchRequestBody,
    BatchRequestCounts,
    BatchResponse,
    BatchResponseBody,
    BatchResponseChoice,
    BatchResponseHTTP,
    BatchResponseUsage,
    BatchStatus,
    FileUploadResponse,
)
from hindsight_api.engine.batch.processor import extract_token_usage, parse_batch_response_content
from hindsight_api.engine.batch.queue import BatchQueue, PendingRequest


# ===== Model Tests =====


class TestBatchStatus:
    def test_terminal_statuses(self):
        assert BatchStatus.COMPLETED.is_terminal()
        assert BatchStatus.FAILED.is_terminal()
        assert BatchStatus.EXPIRED.is_terminal()
        assert BatchStatus.CANCELLED.is_terminal()

    def test_non_terminal_statuses(self):
        assert not BatchStatus.VALIDATING.is_terminal()
        assert not BatchStatus.IN_PROGRESS.is_terminal()
        assert not BatchStatus.FINALIZING.is_terminal()
        assert not BatchStatus.CANCELLING.is_terminal()

    def test_success_status(self):
        assert BatchStatus.COMPLETED.is_success()
        assert not BatchStatus.FAILED.is_success()
        assert not BatchStatus.IN_PROGRESS.is_success()


class TestBatchRequest:
    def test_serialization(self):
        req = BatchRequest(
            custom_id="test-123",
            body=BatchRequestBody(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ],
                temperature=0.1,
                max_completion_tokens=1000,
                seed=4242,
            ),
        )

        data = json.loads(req.model_dump_json())
        assert data["custom_id"] == "test-123"
        assert data["method"] == "POST"
        assert data["url"] == "/v1/chat/completions"
        assert data["body"]["model"] == "openai/gpt-oss-120b"
        assert len(data["body"]["messages"]) == 2
        assert data["body"]["temperature"] == 0.1
        assert data["body"]["seed"] == 4242

    def test_defaults(self):
        req = BatchRequest(
            custom_id="test",
            body=BatchRequestBody(
                model="test-model",
                messages=[{"role": "user", "content": "Hi"}],
            ),
        )
        assert req.method == "POST"
        assert req.url == "/v1/chat/completions"
        assert req.body.temperature is None
        assert req.body.seed is None


class TestBatchResponse:
    def test_success_response(self):
        resp = BatchResponse(
            custom_id="test-123",
            id="batch_req_abc",
            response=BatchResponseHTTP(
                status_code=200,
                body=BatchResponseBody(
                    id="chatcmpl-xyz",
                    choices=[
                        BatchResponseChoice(
                            message={"role": "assistant", "content": '{"facts": []}'},
                            finish_reason="stop",
                        )
                    ],
                    usage=BatchResponseUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                ),
            ),
        )

        assert resp.is_success()
        assert resp.get_content() == '{"facts": []}'

    def test_error_response(self):
        resp = BatchResponse(
            custom_id="test-123",
            id="batch_req_abc",
            error={"code": "rate_limit", "message": "Too many requests"},
        )

        assert not resp.is_success()
        assert resp.get_content() is None

    def test_http_error_response(self):
        resp = BatchResponse(
            custom_id="test-123",
            id="batch_req_abc",
            response=BatchResponseHTTP(status_code=500),
        )

        assert not resp.is_success()

    def test_empty_choices(self):
        resp = BatchResponse(
            custom_id="test-123",
            response=BatchResponseHTTP(
                status_code=200,
                body=BatchResponseBody(choices=[]),
            ),
        )
        assert resp.is_success()
        assert resp.get_content() is None


# ===== Client Tests =====


class TestGroqBatchClient:
    def test_build_jsonl(self):
        client = GroqBatchClient(api_key="test-key")

        requests = [
            BatchRequest(
                custom_id="req-1",
                body=BatchRequestBody(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": "Hello"}],
                ),
            ),
            BatchRequest(
                custom_id="req-2",
                body=BatchRequestBody(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": "World"}],
                ),
            ),
        ]

        jsonl = client.build_jsonl(requests)
        lines = jsonl.decode().strip().split("\n")

        assert len(lines) == 2

        line1 = json.loads(lines[0])
        assert line1["custom_id"] == "req-1"
        assert line1["body"]["messages"][0]["content"] == "Hello"

        line2 = json.loads(lines[1])
        assert line2["custom_id"] == "req-2"
        assert line2["body"]["messages"][0]["content"] == "World"

    def test_build_jsonl_empty(self):
        client = GroqBatchClient(api_key="test-key")
        jsonl = client.build_jsonl([])
        assert jsonl == b""

    def test_headers(self):
        client = GroqBatchClient(api_key="gsk_test123")
        headers = client._headers()
        assert headers["Authorization"] == "Bearer gsk_test123"


# ===== Queue Tests =====


class TestPendingRequest:
    def test_to_batch_request(self):
        req = PendingRequest(
            custom_id="test-1",
            messages=[{"role": "user", "content": "Hello"}],
            model="openai/gpt-oss-120b",
            temperature=0.1,
            max_completion_tokens=1000,
            seed=4242,
            scope="retain_extract_facts",
        )

        batch_req = req.to_batch_request()
        assert batch_req.custom_id == "test-1"
        assert batch_req.body.model == "openai/gpt-oss-120b"
        assert batch_req.body.temperature == 0.1
        assert batch_req.body.seed == 4242


class TestBatchQueue:
    @pytest.fixture
    def mock_client(self):
        """Create a mock GroqBatchClient that records calls."""

        class MockBatchClient:
            def __init__(self):
                self.submitted_requests = []
                self.batch_status = BatchStatus.IN_PROGRESS
                self.results: list[BatchResponse] = []
                self._batch_counter = 0

            async def submit_requests(self, requests, completion_window="24h"):
                self._batch_counter += 1
                self.submitted_requests.extend(requests)
                return BatchJob(
                    id=f"batch_{self._batch_counter}",
                    input_file_id="file_123",
                    status=BatchStatus.VALIDATING,
                    request_counts=BatchRequestCounts(total=len(requests)),
                )

            async def get_batch_status(self, batch_id):
                return BatchJob(
                    id=batch_id,
                    input_file_id="file_123",
                    status=self.batch_status,
                    output_file_id="output_456" if self.batch_status == BatchStatus.COMPLETED else None,
                    request_counts=BatchRequestCounts(total=len(self.results), completed=len(self.results)),
                )

            async def download_results(self, output_file_id):
                return self.results

            async def cancel_batch(self, batch_id):
                return BatchJob(
                    id=batch_id,
                    input_file_id="file_123",
                    status=BatchStatus.CANCELLED,
                )

        return MockBatchClient()

    @pytest.mark.asyncio
    async def test_enqueue_returns_future(self, mock_client):
        queue = BatchQueue(client=mock_client, model="openai/gpt-oss-120b")

        future = await queue.enqueue(
            messages=[{"role": "user", "content": "Hello"}],
            scope="retain_extract_facts",
        )

        assert isinstance(future, asyncio.Future)
        assert not future.done()
        assert queue.pending_count == 1

    @pytest.mark.asyncio
    async def test_flush_submits_batch(self, mock_client):
        queue = BatchQueue(client=mock_client, model="openai/gpt-oss-120b")

        await queue.enqueue(messages=[{"role": "user", "content": "Hello"}])
        await queue.enqueue(messages=[{"role": "user", "content": "World"}])

        assert queue.pending_count == 2

        batch = await queue.flush()

        assert batch is not None
        assert batch.id == "batch_1"
        assert queue.pending_count == 0
        assert queue.in_flight_count == 1
        assert len(mock_client.submitted_requests) == 2

    @pytest.mark.asyncio
    async def test_flush_empty_queue(self, mock_client):
        queue = BatchQueue(client=mock_client, model="openai/gpt-oss-120b")
        batch = await queue.flush()
        assert batch is None

    @pytest.mark.asyncio
    async def test_poll_completions_resolves_futures(self, mock_client):
        queue = BatchQueue(client=mock_client, model="openai/gpt-oss-120b")

        future1 = await queue.enqueue(messages=[{"role": "user", "content": "Hello"}])
        future2 = await queue.enqueue(messages=[{"role": "user", "content": "World"}])

        await queue.flush()

        # Get custom_ids from submitted requests
        custom_ids = [req.custom_id for req in mock_client.submitted_requests]

        # Set up mock results
        mock_client.batch_status = BatchStatus.COMPLETED
        mock_client.results = [
            BatchResponse(
                custom_id=custom_ids[0],
                response=BatchResponseHTTP(
                    status_code=200,
                    body=BatchResponseBody(
                        choices=[BatchResponseChoice(message={"content": "Hello response"})],
                    ),
                ),
            ),
            BatchResponse(
                custom_id=custom_ids[1],
                response=BatchResponseHTTP(
                    status_code=200,
                    body=BatchResponseBody(
                        choices=[BatchResponseChoice(message={"content": "World response"})],
                    ),
                ),
            ),
        ]

        # Poll for completions
        completed = await queue.poll_completions()

        assert completed == 1
        assert queue.in_flight_count == 0
        assert future1.done()
        assert future2.done()
        assert await future1 == "Hello response"
        assert await future2 == "World response"

    @pytest.mark.asyncio
    async def test_poll_still_in_progress(self, mock_client):
        queue = BatchQueue(client=mock_client, model="openai/gpt-oss-120b")

        await queue.enqueue(messages=[{"role": "user", "content": "Hello"}])
        await queue.flush()

        # Status is still IN_PROGRESS (default)
        completed = await queue.poll_completions()
        assert completed == 0
        assert queue.in_flight_count == 1

    @pytest.mark.asyncio
    async def test_poll_failed_batch_rejects_futures(self, mock_client):
        queue = BatchQueue(client=mock_client, model="openai/gpt-oss-120b")

        future = await queue.enqueue(messages=[{"role": "user", "content": "Hello"}])
        await queue.flush()

        mock_client.batch_status = BatchStatus.FAILED

        completed = await queue.poll_completions()
        assert completed == 1
        assert future.done()

        with pytest.raises(RuntimeError, match="ended with status"):
            await future

    @pytest.mark.asyncio
    async def test_auto_flush_on_threshold(self, mock_client):
        queue = BatchQueue(client=mock_client, model="openai/gpt-oss-120b", max_requests_per_batch=2)

        await queue.enqueue(messages=[{"role": "user", "content": "Hello"}])
        # This should trigger auto-flush
        await queue.enqueue(messages=[{"role": "user", "content": "World"}])

        assert queue.pending_count == 0
        assert queue.in_flight_count == 1
        assert len(mock_client.submitted_requests) == 2

    @pytest.mark.asyncio
    async def test_cancel_all(self, mock_client):
        queue = BatchQueue(client=mock_client, model="openai/gpt-oss-120b")

        future1 = await queue.enqueue(messages=[{"role": "user", "content": "Hello"}])
        await queue.flush()

        future2 = await queue.enqueue(messages=[{"role": "user", "content": "Pending"}])

        await queue.cancel_all()

        assert queue.pending_count == 0
        assert queue.in_flight_count == 0

        with pytest.raises(RuntimeError):
            await future1
        with pytest.raises(RuntimeError):
            await future2

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_client):
        queue = BatchQueue(client=mock_client, model="openai/gpt-oss-120b")

        await queue.enqueue(messages=[{"role": "user", "content": "Hello"}])
        await queue.enqueue(messages=[{"role": "user", "content": "World"}])
        await queue.flush()
        await queue.enqueue(messages=[{"role": "user", "content": "Pending"}])

        stats = queue.get_stats()
        assert stats["pending_requests"] == 1
        assert stats["in_flight_batches"] == 1
        assert stats["in_flight_requests"] == 2
        assert len(stats["in_flight_batch_ids"]) == 1


# ===== Processor Tests =====


class TestBatchProcessor:
    def test_parse_text_response(self):
        resp = BatchResponse(
            custom_id="test-1",
            response=BatchResponseHTTP(
                status_code=200,
                body=BatchResponseBody(
                    choices=[BatchResponseChoice(message={"content": "Hello world"})],
                ),
            ),
        )

        result = parse_batch_response_content(resp, response_format=None)
        assert result == "Hello world"

    def test_parse_json_response_skip_validation(self):
        resp = BatchResponse(
            custom_id="test-1",
            response=BatchResponseHTTP(
                status_code=200,
                body=BatchResponseBody(
                    choices=[BatchResponseChoice(message={"content": '{"facts": ["sky is blue"]}'})],
                ),
            ),
        )

        result = parse_batch_response_content(resp, response_format=None, skip_validation=True)
        # Without response_format, returns raw text
        assert result == '{"facts": ["sky is blue"]}'

    def test_parse_error_response_raises(self):
        resp = BatchResponse(
            custom_id="test-1",
            error={"code": "rate_limit", "message": "Too many requests"},
        )

        with pytest.raises(RuntimeError, match="failed"):
            parse_batch_response_content(resp)

    def test_parse_no_content_raises(self):
        resp = BatchResponse(
            custom_id="test-1",
            response=BatchResponseHTTP(
                status_code=200,
                body=BatchResponseBody(choices=[]),
            ),
        )

        with pytest.raises(RuntimeError, match="no content"):
            parse_batch_response_content(resp)

    def test_extract_token_usage(self):
        resp = BatchResponse(
            custom_id="test-1",
            response=BatchResponseHTTP(
                status_code=200,
                body=BatchResponseBody(
                    choices=[BatchResponseChoice(message={"content": "test"})],
                    usage=BatchResponseUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                ),
            ),
        )

        input_t, output_t, total_t = extract_token_usage(resp)
        assert input_t == 100
        assert output_t == 50
        assert total_t == 150

    def test_extract_token_usage_error(self):
        resp = BatchResponse(
            custom_id="test-1",
            error={"code": "error"},
        )

        input_t, output_t, total_t = extract_token_usage(resp)
        assert input_t == 0
        assert output_t == 0
        assert total_t == 0


# ===== Config Tests =====


@pytest.fixture(autouse=True)
def clean_config_cache():
    """Clear config cache before and after each test."""
    from hindsight_api.config import clear_config_cache

    # Save and clear batch-related env vars
    batch_vars = [
        "HINDSIGHT_API_BATCH_MODE",
        "HINDSIGHT_API_BATCH_GROQ_API_KEY",
        "HINDSIGHT_API_BATCH_COMPLETION_WINDOW",
        "HINDSIGHT_API_BATCH_MAX_REQUESTS",
        "HINDSIGHT_API_BATCH_FLUSH_INTERVAL_S",
        "HINDSIGHT_API_BATCH_POLL_INTERVAL_S",
        "HINDSIGHT_API_LLM_PROVIDER",
        "HINDSIGHT_API_LLM_API_KEY",
    ]
    saved = {k: os.environ.get(k) for k in batch_vars}

    clear_config_cache()

    yield

    # Restore env
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    clear_config_cache()


class TestBatchConfig:
    def test_batch_mode_defaults_to_off(self):
        from hindsight_api.config import HindsightConfig

        os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
        os.environ.pop("HINDSIGHT_API_BATCH_MODE", None)

        config = HindsightConfig.from_env()
        assert config.batch_mode == "off"

    def test_batch_mode_groq(self):
        from hindsight_api.config import HindsightConfig

        os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
        os.environ["HINDSIGHT_API_LLM_API_KEY"] = "test-key"
        os.environ["HINDSIGHT_API_BATCH_MODE"] = "groq"

        config = HindsightConfig.from_env()
        assert config.batch_mode == "groq"

    def test_invalid_batch_mode_raises(self):
        from hindsight_api.config import HindsightConfig

        os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
        os.environ["HINDSIGHT_API_BATCH_MODE"] = "invalid"

        with pytest.raises(ValueError, match="Invalid batch mode"):
            HindsightConfig.from_env()

    def test_batch_groq_requires_api_key(self):
        from hindsight_api.config import HindsightConfig

        os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
        os.environ["HINDSIGHT_API_BATCH_MODE"] = "groq"
        os.environ.pop("HINDSIGHT_API_LLM_API_KEY", None)
        os.environ.pop("HINDSIGHT_API_BATCH_GROQ_API_KEY", None)

        with pytest.raises(ValueError, match="requires an API key"):
            HindsightConfig.from_env()

    def test_batch_groq_falls_back_to_llm_api_key(self):
        from hindsight_api.config import HindsightConfig

        os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
        os.environ["HINDSIGHT_API_LLM_API_KEY"] = "global-key"
        os.environ["HINDSIGHT_API_BATCH_MODE"] = "groq"

        config = HindsightConfig.from_env()
        assert config.batch_mode == "groq"
        assert config.batch_groq_api_key is None  # Falls back to llm_api_key

    def test_batch_config_values(self):
        from hindsight_api.config import HindsightConfig

        os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
        os.environ["HINDSIGHT_API_LLM_API_KEY"] = "test-key"
        os.environ["HINDSIGHT_API_BATCH_MODE"] = "groq"
        os.environ["HINDSIGHT_API_BATCH_GROQ_API_KEY"] = "batch-key"
        os.environ["HINDSIGHT_API_BATCH_COMPLETION_WINDOW"] = "7d"
        os.environ["HINDSIGHT_API_BATCH_MAX_REQUESTS"] = "10000"
        os.environ["HINDSIGHT_API_BATCH_FLUSH_INTERVAL_S"] = "120"
        os.environ["HINDSIGHT_API_BATCH_POLL_INTERVAL_S"] = "60"

        config = HindsightConfig.from_env()
        assert config.batch_groq_api_key == "batch-key"
        assert config.batch_completion_window == "7d"
        assert config.batch_max_requests == 10000
        assert config.batch_flush_interval_s == 120
        assert config.batch_poll_interval_s == 60
