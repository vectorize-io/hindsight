"""Tests for the streaming retain producer's embedding coalescer (issue #3784).

The producer fans out one task per chunk and each task embeds only its own chunk's
facts, so every embedding call used to be one text wide — in ``chunks`` extraction
mode that single round trip is the whole per-chunk cost. ``CoalescingEmbedder`` keeps
the fan-out and batches the concurrent calls underneath it.
"""

import asyncio

import pytest

from hindsight_api.engine.retain.embedding_coalescer import (
    DEFAULT_MAX_BATCH_SIZE,
    CoalescingEmbedder,
    resolve_max_batch_size,
)

pytestmark = pytest.mark.asyncio


class FakeBackend:
    """Records every encode() call so tests can assert on request shape.

    The vector encodes its own text so a mis-sliced result is caught, not just a
    mis-counted one.
    """

    def __init__(self, *, batch_size: int | None = None, delay: float = 0.0) -> None:
        self.calls: list[list[str]] = []
        self.delay = delay
        if batch_size is not None:
            self.batch_size = batch_size

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return 2

    async def initialize(self) -> None:
        return None

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.delay:
            import time

            time.sleep(self.delay)
        return [[float(len(text)), float(hash(text) % 1000)] for text in texts]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self.encode(texts)

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        return self.encode(texts)


def _expected(text: str) -> list[float]:
    return [float(len(text)), float(hash(text) % 1000)]


async def test_concurrent_callers_share_one_backend_request() -> None:
    """The fan-out the producer creates collapses into a single encode() call."""
    backend = FakeBackend(batch_size=100)
    embedder = CoalescingEmbedder(backend)

    texts = [f"chunk-{i}" for i in range(30)]
    results = await asyncio.gather(*[embedder.embed_documents_async([text]) for text in texts])

    assert len(backend.calls) == 1
    assert backend.calls[0] == texts
    assert results == [[_expected(text)] for text in texts]
    assert embedder.stats.requests == 1
    assert embedder.stats.texts == 30


async def test_each_caller_gets_its_own_slice_in_order() -> None:
    """Multi-text callers are sliced back out of the shared response by length."""
    backend = FakeBackend(batch_size=100)
    embedder = CoalescingEmbedder(backend)

    groups = [["a"], ["bb", "ccc"], ["dddd", "eeeee", "ffffff"]]
    results = await asyncio.gather(*[embedder.embed_documents_async(group) for group in groups])

    assert len(backend.calls) == 1
    assert results == [[_expected(text) for text in group] for group in groups]


async def test_lone_caller_is_dispatched_without_waiting() -> None:
    """A single caller pays no batching delay — nothing waits on a timer."""
    backend = FakeBackend(batch_size=100)
    embedder = CoalescingEmbedder(backend)

    assert await embedder.embed_documents_async(["only"]) == [_expected("only")]
    assert backend.calls == [["only"]]


async def test_batches_never_exceed_the_backend_batch_size() -> None:
    """Requests are sized from the backend's own per-request limit."""
    backend = FakeBackend(batch_size=8)
    embedder = CoalescingEmbedder(backend)

    texts = [f"chunk-{i}" for i in range(25)]
    results = await asyncio.gather(*[embedder.embed_documents_async([text]) for text in texts])

    assert all(len(call) <= 8 for call in backend.calls)
    assert [text for call in backend.calls for text in call] == texts
    assert results == [[_expected(text)] for text in texts]


async def test_caller_larger_than_the_budget_goes_out_alone() -> None:
    """A waiter is never split: the backend applies its own splitting instead."""
    backend = FakeBackend(batch_size=4)
    embedder = CoalescingEmbedder(backend)

    big = [f"big-{i}" for i in range(10)]
    results = await asyncio.gather(
        embedder.embed_documents_async(big),
        embedder.embed_documents_async(["small"]),
    )

    assert big in backend.calls
    assert results[0] == [_expected(text) for text in big]
    assert results[1] == [_expected("small")]


async def test_backend_failure_reaches_every_caller_in_the_batch() -> None:
    """One bad request must fail all of its riders, not silently drop them."""

    class ExplodingBackend(FakeBackend):
        def encode(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embeddings backend down")

    embedder = CoalescingEmbedder(ExplodingBackend(batch_size=100))

    results = await asyncio.gather(
        *[embedder.embed_documents_async([f"chunk-{i}"]) for i in range(5)],
        return_exceptions=True,
    )

    assert len(results) == 5
    for result in results:
        assert isinstance(result, Exception)
        assert "embeddings backend down" in str(result)


async def test_empty_input_never_reaches_the_backend() -> None:
    backend = FakeBackend(batch_size=100)
    embedder = CoalescingEmbedder(backend)

    assert await embedder.embed_documents_async([]) == []
    assert backend.calls == []


async def test_close_cancels_parked_callers_and_stops_dispatching() -> None:
    """A cancelled retain must not leave the dispatcher or a caller's future alive."""
    backend = FakeBackend(batch_size=100, delay=0.05)
    embedder = CoalescingEmbedder(backend, max_concurrent_requests=1)

    first = asyncio.create_task(embedder.embed_documents_async(["first"]))
    await asyncio.sleep(0)
    # Parked behind the in-flight request, so it is still on the pending list.
    parked = asyncio.create_task(embedder.embed_documents_async(["parked"]))
    await asyncio.sleep(0)

    embedder.close()

    with pytest.raises(asyncio.CancelledError):
        await parked
    await asyncio.gather(first, return_exceptions=True)
    assert embedder._dispatcher is None

    with pytest.raises(RuntimeError, match="closed"):
        await embedder.embed_documents_async(["after-close"])


async def test_serves_more_batches_than_the_concurrency_limit() -> None:
    """Every caller is served even when the pending list outruns the slot count."""
    backend = FakeBackend(batch_size=2)
    embedder = CoalescingEmbedder(backend, max_concurrent_requests=2)

    texts = [f"chunk-{i}" for i in range(20)]
    results = await asyncio.gather(*[embedder.embed_documents_async([text]) for text in texts])

    assert results == [[_expected(text)] for text in texts]
    assert sum(len(call) for call in backend.calls) == 20


async def test_callers_arriving_after_a_drain_restart_the_dispatcher() -> None:
    """A second wave with no dispatcher running still gets served."""
    backend = FakeBackend(batch_size=100)
    embedder = CoalescingEmbedder(backend)

    await asyncio.gather(*[embedder.embed_documents_async([f"first-{i}"]) for i in range(3)])
    await asyncio.gather(*[embedder.embed_documents_async([f"second-{i}"]) for i in range(3)])

    assert len(backend.calls) == 2
    assert backend.calls[1] == ["second-0", "second-1", "second-2"]


async def test_resolve_max_batch_size_prefers_the_backend_limit() -> None:
    assert resolve_max_batch_size(FakeBackend(batch_size=17)) == 17


async def test_resolve_max_batch_size_falls_back_for_backends_without_one() -> None:
    """Local backends do their own internal batching and publish no per-request limit."""
    assert resolve_max_batch_size(FakeBackend()) == DEFAULT_MAX_BATCH_SIZE
