"""
Async batch request queue.

Collects LLM requests and provides Future-based result delivery.
When flushed, all queued requests are submitted as a single Groq batch job.
Callers await their individual Futures to get results when the batch completes.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from .client import GroqBatchClient
from .models import BatchJob, BatchRequest, BatchRequestBody, BatchResponse, BatchStatus

logger = logging.getLogger(__name__)


class PendingRequest:
    """A queued LLM request awaiting batch processing."""

    def __init__(
        self,
        custom_id: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
        scope: str = "memory",
    ):
        self.custom_id = custom_id
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.response_format = response_format
        self.seed = seed
        self.scope = scope
        self.future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        self.created_at = time.time()

    def to_batch_request(self) -> BatchRequest:
        """Convert to a BatchRequest for JSONL submission."""
        body = BatchRequestBody(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            response_format=self.response_format,
            seed=self.seed,
        )
        return BatchRequest(custom_id=self.custom_id, body=body)


class InFlightBatch:
    """Tracks a submitted batch job and its pending requests."""

    def __init__(self, batch_job: BatchJob, requests: dict[str, PendingRequest]):
        self.batch_job = batch_job
        self.requests = requests  # custom_id -> PendingRequest
        self.submitted_at = time.time()

    @property
    def batch_id(self) -> str:
        return self.batch_job.id


class BatchQueue:
    """Async queue that collects LLM requests for batch submission.

    Usage:
        queue = BatchQueue(client, model="openai/gpt-oss-120b")

        # Queue requests (returns Future)
        future = await queue.enqueue(messages=[...], ...)

        # Flush queued requests as a batch (or auto-flush on threshold)
        await queue.flush()

        # Result is available when batch completes
        result = await future  # Blocks until batch is done

        # Poll in-flight batches for completion
        await queue.poll_completions()
    """

    def __init__(
        self,
        client: GroqBatchClient,
        model: str,
        completion_window: str = "24h",
        max_requests_per_batch: int = 50000,
        seed: int | None = None,
    ):
        """Initialize the batch queue.

        Args:
            client: Groq batch API client
            model: Model name for all requests in this queue
            completion_window: Batch processing window
            max_requests_per_batch: Max requests per batch file (Groq limit: 50000)
            seed: Optional seed for deterministic behavior
        """
        self._client = client
        self._model = model
        self._completion_window = completion_window
        self._max_requests = max_requests_per_batch
        self._seed = seed

        self._pending: list[PendingRequest] = []
        self._in_flight: list[InFlightBatch] = []
        self._lock = asyncio.Lock()

    @property
    def pending_count(self) -> int:
        """Number of requests waiting to be submitted."""
        return len(self._pending)

    @property
    def in_flight_count(self) -> int:
        """Number of batch jobs currently in flight."""
        return len(self._in_flight)

    async def enqueue(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        scope: str = "memory",
    ) -> asyncio.Future[str]:
        """Queue an LLM request for batch processing.

        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_completion_tokens: Max output tokens
            response_format: JSON response format spec
            scope: Scope identifier for tracking

        Returns:
            Future that resolves to the LLM response content string
        """
        custom_id = str(uuid.uuid4())

        request = PendingRequest(
            custom_id=custom_id,
            messages=messages,
            model=self._model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            response_format=response_format,
            seed=self._seed,
            scope=scope,
        )

        async with self._lock:
            self._pending.append(request)

            # Auto-flush if we hit the max requests threshold
            if len(self._pending) >= self._max_requests:
                logger.info(f"Batch queue hit max threshold ({self._max_requests}), auto-flushing")
                await self._flush_locked()

        return request.future

    async def flush(self) -> BatchJob | None:
        """Submit all pending requests as a batch job.

        Returns:
            BatchJob if requests were submitted, None if queue was empty
        """
        async with self._lock:
            return await self._flush_locked()

    async def _flush_locked(self) -> BatchJob | None:
        """Submit pending requests (must be called with lock held)."""
        if not self._pending:
            return None

        requests_to_submit = self._pending[:]
        self._pending.clear()

        # Build batch requests
        batch_requests = [req.to_batch_request() for req in requests_to_submit]

        try:
            # Submit batch
            batch_job = await self._client.submit_requests(
                requests=batch_requests,
                completion_window=self._completion_window,
            )

            # Track in-flight batch
            requests_map = {req.custom_id: req for req in requests_to_submit}
            in_flight = InFlightBatch(batch_job=batch_job, requests=requests_map)
            self._in_flight.append(in_flight)

            logger.info(
                f"Flushed batch queue: {len(requests_to_submit)} requests -> "
                f"batch {batch_job.id} (status={batch_job.status})"
            )

            return batch_job

        except Exception as e:
            # On failure, reject all futures so callers know to fall back
            logger.error(f"Batch submission failed: {e}")
            for req in requests_to_submit:
                if not req.future.done():
                    req.future.set_exception(e)
            raise

    async def poll_completions(self) -> int:
        """Poll all in-flight batches for completion and resolve futures.

        Returns:
            Number of batches that completed (success or failure)
        """
        completed_count = 0

        async with self._lock:
            still_in_flight = []

            for batch in self._in_flight:
                try:
                    status = await self._client.get_batch_status(batch.batch_id)
                    batch.batch_job = status

                    if status.status.is_terminal():
                        await self._resolve_batch(batch, status)
                        completed_count += 1
                    else:
                        still_in_flight.append(batch)
                        logger.debug(
                            f"Batch {batch.batch_id} still processing: "
                            f"status={status.status}, "
                            f"completed={status.request_counts.completed}/"
                            f"{status.request_counts.total}"
                        )

                except Exception as e:
                    logger.warning(f"Error polling batch {batch.batch_id}: {e}")
                    still_in_flight.append(batch)

            self._in_flight = still_in_flight

        return completed_count

    async def _resolve_batch(self, batch: InFlightBatch, status: BatchJob) -> None:
        """Resolve futures for a completed batch (must be called with lock held)."""
        if status.status.is_success() and status.output_file_id:
            # Download results
            try:
                results = await self._client.download_results(status.output_file_id)
                results_map = {r.custom_id: r for r in results}

                # Resolve each future
                for custom_id, pending_req in batch.requests.items():
                    if pending_req.future.done():
                        continue

                    result = results_map.get(custom_id)
                    if result and result.is_success():
                        content = result.get_content()
                        if content is not None:
                            pending_req.future.set_result(content)
                        else:
                            pending_req.future.set_exception(
                                RuntimeError(f"Batch response for {custom_id} had no content")
                            )
                    elif result and result.error:
                        pending_req.future.set_exception(
                            RuntimeError(f"Batch request {custom_id} failed: {result.error}")
                        )
                    else:
                        pending_req.future.set_exception(RuntimeError(f"No result found for batch request {custom_id}"))

                logger.info(
                    f"Batch {batch.batch_id} completed: {len(results)} results resolved "
                    f"(completed={status.request_counts.completed}, "
                    f"failed={status.request_counts.failed})"
                )

            except Exception as e:
                logger.error(f"Error downloading batch results for {batch.batch_id}: {e}")
                for pending_req in batch.requests.values():
                    if not pending_req.future.done():
                        pending_req.future.set_exception(e)

        else:
            # Batch failed, expired, or cancelled
            error_msg = f"Batch {batch.batch_id} ended with status {status.status}"
            logger.warning(error_msg)
            for pending_req in batch.requests.values():
                if not pending_req.future.done():
                    pending_req.future.set_exception(RuntimeError(error_msg))

    async def cancel_all(self) -> None:
        """Cancel all in-flight batches and reject pending requests."""
        async with self._lock:
            # Reject pending requests
            for req in self._pending:
                if not req.future.done():
                    req.future.set_exception(RuntimeError("Batch queue cancelled"))
            self._pending.clear()

            # Cancel in-flight batches
            for batch in self._in_flight:
                try:
                    await self._client.cancel_batch(batch.batch_id)
                except Exception as e:
                    logger.warning(f"Error cancelling batch {batch.batch_id}: {e}")

                for req in batch.requests.values():
                    if not req.future.done():
                        req.future.set_exception(RuntimeError("Batch cancelled"))

            self._in_flight.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        return {
            "pending_requests": len(self._pending),
            "in_flight_batches": len(self._in_flight),
            "in_flight_requests": sum(len(b.requests) for b in self._in_flight),
            "in_flight_batch_ids": [b.batch_id for b in self._in_flight],
        }
