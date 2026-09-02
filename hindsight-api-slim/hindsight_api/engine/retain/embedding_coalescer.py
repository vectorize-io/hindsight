"""Merge the streaming retain producer's per-chunk embedding calls into batched requests.

The streaming producer fans out one task per chunk and each task embeds only its own
chunk's facts. That is the right shape for the *extraction* path — every chunk needs
its own LLM call and the fan-out lets those overlap — but it also makes every embedding
call exactly one batch of one document wide. In ``chunks`` extraction mode extraction is
a no-op, so that single round trip is the entire per-chunk cost and ingest throughput
ends up bounded by it (issue #3784).

``CoalescingEmbedder`` keeps the fan-out and moves the batching one layer down:
concurrent callers park on a shared pending list, and each backend request carries
whatever has accumulated. Nothing waits on a timer — a caller that arrives while a
backend slot is free is dispatched on the very next event-loop tick — so a lone caller
pays no added latency while a fan-out of N naturally forms batches of up to the
backend's own per-request limit.
"""

import asyncio
from collections import deque
from dataclasses import dataclass

from . import embedding_utils
from .embedding_utils import EmbeddingsBackend

# How many backend requests may be in flight at once. The dispatcher hands the first
# request everything that is already runnable, so extra slots only come into play once
# more than one batch worth of text is pending — there they let the overflow batches go
# out concurrently instead of one after another. Kept small because each in-flight
# request also occupies a thread of the default executor.
MAX_CONCURRENT_REQUESTS = 4

# Fallback per-request batch size for a backend that does not publish one (the local
# SentenceTransformers and ONNX backends do their own internal batching).
DEFAULT_MAX_BATCH_SIZE = 64


@dataclass
class _Waiter:
    """One caller's texts and the future its embeddings are delivered to."""

    texts: list[str]
    future: asyncio.Future[list[list[float]]]


@dataclass
class CoalescerStats:
    """What the coalescer actually sent, for the retain log.

    Running totals rather than a per-request list: a large document produces
    thousands of requests, and retain is the path that has to stay flat in
    memory (#3756).
    """

    texts: int = 0
    requests: int = 0
    largest_request: int = 0

    def record(self, batch_texts: int) -> None:
        self.texts += batch_texts
        self.requests += 1
        self.largest_request = max(self.largest_request, batch_texts)

    def describe(self) -> str:
        if not self.requests:
            return "Embeddings: no texts embedded"
        return (
            f"Embeddings: {self.texts} texts in {self.requests} backend request(s) "
            f"(avg {self.texts / self.requests:.1f}, max {self.largest_request} texts per request)"
        )


def resolve_max_batch_size(embeddings_backend: EmbeddingsBackend) -> int:
    """Batch at the backend's own per-request limit, not above it.

    Every remote backend splits an oversized list into ``batch_size``-sized requests
    *inside one executor thread*, so handing it more than that trades N concurrent
    requests for N sequential ones. Sizing the coalescer's batches from the backend
    means TEI's ``HINDSIGHT_API_EMBEDDINGS_TEI_BATCH_SIZE`` (and the OpenAI/Cohere/
    ZeroEntropy equivalents) tunes the coalescer too, with no second knob.
    """
    backend_batch_size = getattr(embeddings_backend, "batch_size", None)
    if isinstance(backend_batch_size, int) and backend_batch_size > 0:
        return backend_batch_size
    return DEFAULT_MAX_BATCH_SIZE


class CoalescingEmbedder:
    """Batches concurrent document-embedding requests into shared backend calls.

    One instance serves ONE retain call. That is deliberate: the backends read the
    ambient bank id for per-bank cost attribution (see ``bank_attribution``), so texts
    from different banks must never share a request.

    Duck-typed against the retain embedding path: ``embedding_processing.
    generate_embeddings_batch`` dispatches to ``embed_documents_async`` when the object
    it is handed exposes one, and falls back to the plain backend call otherwise.
    """

    def __init__(
        self,
        embeddings_backend: EmbeddingsBackend,
        *,
        max_batch_size: int | None = None,
        max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS,
    ) -> None:
        self._backend = embeddings_backend
        self._max_batch_size = max_batch_size or resolve_max_batch_size(embeddings_backend)
        self._slots = asyncio.Semaphore(max_concurrent_requests)
        self._pending: deque[_Waiter] = deque()
        self._dispatcher: asyncio.Task | None = None
        self._in_flight: set[asyncio.Task] = set()
        self._closed = False
        self.stats = CoalescerStats()

    @property
    def dimension(self) -> int:
        return self._backend.dimension

    async def embed_documents_async(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` as retained document text, sharing a request where possible."""
        if not texts:
            return []
        if self._closed:
            raise RuntimeError("CoalescingEmbedder is closed")
        waiter = _Waiter(texts=texts, future=asyncio.get_running_loop().create_future())
        self._pending.append(waiter)
        self._ensure_dispatcher()
        return await waiter.future

    def close(self) -> None:
        """Stop dispatching and cancel anything still parked.

        Synchronous so it is safe in a ``finally`` that runs while the retain is being
        cancelled — there is nothing to wait for. Requests already in flight are left
        to finish; they hold their own reference to this object, deliver into futures
        that are checked before use, and their ``finally`` finds the coalescer closed
        so no further request is dispatched.
        """
        self._closed = True
        dispatcher, self._dispatcher = self._dispatcher, None
        if dispatcher is not None and not dispatcher.done():
            dispatcher.cancel()
        for waiter in self._pending:
            if not waiter.future.done():
                waiter.future.cancel()
        self._pending.clear()

    def _ensure_dispatcher(self) -> None:
        if self._closed:
            return
        if self._dispatcher is None or self._dispatcher.done():
            self._dispatcher = asyncio.create_task(self._dispatch_loop())

    async def _dispatch_loop(self) -> None:
        """Turn the pending list into backend requests until it runs dry.

        Waiting for a free slot *before* taking the batch is what makes this coalesce:
        while every slot is busy the pending list keeps growing, and the next request
        carries all of it. When a slot is free the single ``sleep(0)`` is the only
        delay — just enough for the callers that are already runnable (the producer
        schedules its whole chunk fan-out in one go) to join this batch.
        """
        while self._pending:
            await self._slots.acquire()
            try:
                await asyncio.sleep(0)
                batch = self._take_batch()
            except BaseException:
                self._slots.release()
                raise
            if not batch:
                self._slots.release()
                return
            request = asyncio.create_task(self._run_request(batch))
            self._in_flight.add(request)
            request.add_done_callback(self._in_flight.discard)

    def _take_batch(self) -> list[_Waiter]:
        """Pop whole waiters off the pending list up to the per-request text budget.

        A waiter is never split across requests: keeping its texts together is what
        lets ``_run_request`` slice the results back out by length alone. A single
        waiter larger than the budget goes out on its own, where the backend applies
        its own splitting.
        """
        batch: list[_Waiter] = []
        budget = 0
        while self._pending:
            waiter = self._pending[0]
            if batch and budget + len(waiter.texts) > self._max_batch_size:
                break
            batch.append(self._pending.popleft())
            budget += len(waiter.texts)
        return batch

    async def _run_request(self, batch: list[_Waiter]) -> None:
        try:
            texts = [text for waiter in batch for text in waiter.texts]
            try:
                embeddings = await embedding_utils.generate_embeddings_batch(self._backend, texts)
            except BaseException as exc:
                # Every rider on a failed request has to see the failure — silently
                # dropping them would commit the document with those facts missing.
                for waiter in batch:
                    if not waiter.future.done():
                        waiter.future.set_exception(exc)
                        # The caller may already be gone (a cancelled producer task).
                        # Retrieve the exception so it is not reported as unhandled.
                        waiter.future.exception()
                if isinstance(exc, asyncio.CancelledError):
                    raise
                return
            self.stats.record(len(texts))
            offset = 0
            for waiter in batch:
                count = len(waiter.texts)
                if not waiter.future.done():
                    waiter.future.set_result(embeddings[offset : offset + count])
                offset += count
        finally:
            self._slots.release()
            # A waiter that arrived after the dispatch loop drained the list has no
            # dispatcher to pick it up; restart one now that a slot is free again.
            if self._pending:
                self._ensure_dispatcher()
