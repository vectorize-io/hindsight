"""The retain pipeline is bounded in bytes, not just in chunks (issue #3756).

``retain_chunk_batch_size`` bounds how many chunks the streaming pipeline holds, which is
only a memory bound if chunks cost a predictable amount — and they do not, because a chunk
carries however many facts the extractor found in it. ``RetainMemoryBudget`` puts a ceiling
on what those chunks weigh, so a worker can be sized against a number that means something.

The properties that have to hold: the budget must actually block a producer that has run
ahead, it must never block one that would otherwise make no progress, and it must give back
exactly what was taken.
"""

import asyncio
from array import array
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.response_models import TokenUsage
from hindsight_api.engine.retain import orchestrator
from hindsight_api.engine.retain.memory_budget import (
    PRODUCER_WAITING,
    RetainMemoryBudget,
    estimate_chunk_bytes,
)
from hindsight_api.engine.retain.types import ChunkMetadata, ExtractedFact, ProcessedFact

_MB = 1024 * 1024


def _fact(text: str = "Ada shipped the parser", context: str = "standup notes", dims: int = 384) -> ProcessedFact:
    return ProcessedFact(
        fact_text=text,
        fact_type="world",
        embedding=array("f", [0.01] * dims),
        occurred_start=None,
        occurred_end=None,
        mentioned_at=None,
        context=context,
        metadata={},
    )


def _extracted(text: str = "Ada shipped the parser") -> ExtractedFact:
    return ExtractedFact(fact_text=text, fact_type="world", context="standup notes")


def _meta(chunk_text: str = "x" * 1500) -> ChunkMetadata:
    return ChunkMetadata(chunk_text=chunk_text, fact_count=1, content_index=0, chunk_index=0)


# ---------------------------------------------------------------------------
# estimate_chunk_bytes
# ---------------------------------------------------------------------------


def test_estimate_counts_the_things_that_actually_scale():
    """Text, context, embedding and chunk text — the parts that grow with extraction."""
    small = estimate_chunk_bytes([_fact()], [_extracted()], [_meta()])
    more_facts = estimate_chunk_bytes([_fact()] * 10, [_extracted()] * 10, [_meta()])
    bigger_vectors = estimate_chunk_bytes([_fact(dims=1536)], [_extracted()], [_meta()])
    longer_text = estimate_chunk_bytes([_fact(text="y" * 5000)], [_extracted()], [_meta()])

    assert more_facts > small * 5
    assert bigger_vectors > small
    assert longer_text > small


def test_estimate_of_nothing_is_nothing():
    """A chunk that yielded no facts reserves nothing."""
    assert estimate_chunk_bytes([], [], []) == 0


def test_estimate_tolerates_a_null_context():
    """A retain with no context must not be failed by its own memory estimate.

    ``ProcessedFact.context`` is annotated ``str``, but a converted file upload retains
    without one and puts ``None`` there. The first version of this estimator called
    ``len()`` on it and turned every file retain into an HTTP 500 — a heuristic breaking
    the operation it exists to protect.
    """
    fact = _fact()
    fact.context = None  # type: ignore[assignment]
    raw = _extracted()
    raw.context = None  # type: ignore[assignment]

    assert estimate_chunk_bytes([fact], [raw], [_meta()]) > 0


def test_estimate_tracks_embedding_width():
    """A 1536-dim model costs 4x a 384-dim one per fact, and the estimate says so."""
    narrow = estimate_chunk_bytes([_fact(dims=384)], [], [])
    wide = estimate_chunk_bytes([_fact(dims=1536)], [], [])

    assert wide - narrow == (1536 - 384) * 4


# ---------------------------------------------------------------------------
# RetainMemoryBudget
# ---------------------------------------------------------------------------


async def test_reservations_under_the_budget_never_wait():
    budget = RetainMemoryBudget(limit_bytes=_MB)

    for _ in range(4):
        await asyncio.wait_for(budget.reserve(100_000), timeout=1)

    assert budget.held_bytes == 400_000


async def test_a_producer_over_the_budget_waits_for_the_consumer():
    """The point of the whole thing: extraction throttles instead of the worker dying."""
    budget = RetainMemoryBudget(limit_bytes=1000)
    await budget.reserve(900)

    blocked = asyncio.create_task(budget.reserve(500))
    await asyncio.sleep(0)
    assert not blocked.done(), "a reservation past the budget should not have been admitted"

    budget.release(900)
    await asyncio.wait_for(blocked, timeout=1)
    assert budget.held_bytes == 500


async def test_an_oversized_chunk_is_admitted_rather_than_deadlocking():
    """Nothing else is holding memory, so nobody can free any — admit it and move on.

    A single chunk whose facts exceed the entire budget is possible (a dense chunk, a wide
    embedding model, a small budget). Blocking it would wait forever for room that only it
    could release.
    """
    budget = RetainMemoryBudget(limit_bytes=1000)

    await asyncio.wait_for(budget.reserve(50_000), timeout=1)

    assert budget.held_bytes == 50_000


async def test_an_oversized_chunk_still_waits_its_turn():
    """It is admitted when the pipeline is empty — not while someone else holds memory."""
    budget = RetainMemoryBudget(limit_bytes=1000)
    await budget.reserve(600)

    blocked = asyncio.create_task(budget.reserve(50_000))
    await asyncio.sleep(0)
    assert not blocked.done()

    budget.release(600)
    await asyncio.wait_for(blocked, timeout=1)


async def test_release_gives_back_exactly_what_was_taken():
    budget = RetainMemoryBudget(limit_bytes=_MB)
    await budget.reserve(300)
    await budget.reserve(700)

    budget.release(300)
    assert budget.held_bytes == 700
    budget.release(700)
    assert budget.held_bytes == 0


async def test_over_release_cannot_drive_the_budget_negative():
    """A double release must not manufacture headroom that does not exist."""
    budget = RetainMemoryBudget(limit_bytes=_MB)
    await budget.reserve(100)

    budget.release(100)
    budget.release(100)

    assert budget.held_bytes == 0


async def test_disabling_the_budget_restores_the_count_only_bound():
    """``0`` means "I have tuned the chunk count myself" — nothing waits, nothing flushes."""
    budget = RetainMemoryBudget(limit_bytes=0)

    await asyncio.wait_for(budget.reserve(10 * _MB), timeout=1)

    assert not budget.enabled
    assert budget.held_bytes == 0
    assert budget.should_flush(10 * _MB) is False


def test_should_flush_at_half_the_budget():
    """Half for the open batch, half for the producer to keep extracting into."""
    budget = RetainMemoryBudget(limit_bytes=1000)

    assert budget.should_flush(499) is False
    assert budget.should_flush(500) is True


async def test_many_producers_against_one_consumer_stay_within_the_budget():
    """Whatever the interleaving, the pipeline never holds more than it reserved for.

    Models the real shape: many extraction tasks handing work to a single writer. The
    invariant is that held bytes never exceed the budget while more than one chunk is in
    flight — the oversized-chunk escape hatch only applies to an empty pipeline.
    """
    budget = RetainMemoryBudget(limit_bytes=10_000)
    peak = 0
    committed = 0

    async def produce(cost: int) -> None:
        nonlocal peak
        await budget.reserve(cost)
        peak = max(peak, budget.held_bytes)
        await asyncio.sleep(0)

    async def consume(costs: list[int]) -> None:
        nonlocal committed
        for cost in costs:
            await asyncio.sleep(0)
            budget.release(cost)
            committed += cost

    costs = [1500] * 40
    producers = asyncio.gather(*(produce(cost) for cost in costs))
    consumer = asyncio.create_task(consume(costs))
    await asyncio.wait_for(asyncio.gather(producers, consumer), timeout=5)

    assert committed == sum(costs)
    assert budget.held_bytes == 0
    # 10,000 budget with 1,500-byte chunks admits 6 before blocking the 7th.
    assert peak <= 10_000


@pytest.mark.parametrize("limit_mb", [1, 16, 128])
def test_configured_budget_converts_to_bytes(limit_mb: int):
    """The config is in MB because that is the unit a worker limit is written in."""
    budget = RetainMemoryBudget(limit_bytes=limit_mb * _MB)

    assert budget.enabled
    assert budget.limit_bytes == limit_mb * 1024 * 1024


# ---------------------------------------------------------------------------
# A small open batch + a large next chunk must not deadlock (#5296)
# ---------------------------------------------------------------------------


async def _pipeline(budget: RetainMemoryBudget, costs: list[int], batch_size: int) -> list[list[int]]:
    """The orchestrator's producer/consumer protocol over the real budget, returning batches."""
    queue: asyncio.Queue = asyncio.Queue()
    written: list[list[int]] = []

    async def produce() -> None:
        for cost in costs:
            await budget.reserve(cost)
            await queue.put(cost)
        await queue.put(None)

    async def consume() -> None:
        batch: list[int] = []
        while True:
            item = await budget.next_item_or_producer_waiting(queue) if batch else await queue.get()
            if item is None:
                if batch:
                    written.append(batch)
                    budget.release(sum(batch))
                return
            if item is not PRODUCER_WAITING:
                batch.append(item)
            if item is PRODUCER_WAITING or len(batch) >= batch_size or budget.should_flush(sum(batch)):
                written.append(batch)
                budget.release(sum(batch))
                batch = []

    await asyncio.wait_for(asyncio.gather(produce(), consume()), timeout=2)
    return written


async def test_a_held_batch_is_written_when_the_producer_needs_its_room():
    """The issue's case: 40 held (under the 50 flush mark), then 70 that only fits once it goes."""
    budget = RetainMemoryBudget(limit_bytes=100)

    written = await _pipeline(budget, [40, 70], batch_size=100)

    assert written == [[40], [70]]
    assert budget.held_bytes == 0


async def test_batching_is_unchanged_while_the_producer_fits():
    """No early writes unless the producer is actually stuck: small chunks still batch up."""
    budget = RetainMemoryBudget(limit_bytes=1000)

    written = await _pipeline(budget, [10] * 9, batch_size=3)

    assert written == [[10, 10, 10]] * 3


async def test_queued_items_are_taken_before_the_waiting_signal():
    """With work queued the consumer keeps batching; the signal only wins over an empty queue."""
    budget = RetainMemoryBudget(limit_bytes=100)
    await budget.reserve(90)
    blocked = asyncio.create_task(budget.reserve(50))
    await asyncio.sleep(0)
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait("queued")

    assert await budget.next_item_or_producer_waiting(queue) == "queued"
    assert await budget.next_item_or_producer_waiting(queue) is PRODUCER_WAITING

    budget.release(90)
    await asyncio.wait_for(blocked, timeout=1)


async def test_the_waiting_signal_does_not_swallow_a_queue_item():
    """The abandoned get must leave the queue intact for the next call."""
    budget = RetainMemoryBudget(limit_bytes=100)
    await budget.reserve(90)
    blocked = asyncio.create_task(budget.reserve(50))
    await asyncio.sleep(0)
    queue: asyncio.Queue = asyncio.Queue()

    assert await budget.next_item_or_producer_waiting(queue) is PRODUCER_WAITING
    queue.put_nowait("next")
    assert queue.get_nowait() == "next"

    budget.release(90)
    await asyncio.wait_for(blocked, timeout=1)


async def test_the_waiting_signal_clears_once_no_reservation_waits():
    """Admitted or cancelled, a reservation that stops waiting stops signalling."""
    budget = RetainMemoryBudget(limit_bytes=100)
    await budget.reserve(90)
    admitted = asyncio.create_task(budget.reserve(50))
    cancelled = asyncio.create_task(budget.reserve(50))
    await asyncio.sleep(0)

    cancelled.cancel()
    await asyncio.gather(cancelled, return_exceptions=True)
    budget.release(90)
    await asyncio.wait_for(admitted, timeout=1)

    queue: asyncio.Queue = asyncio.Queue()
    get = asyncio.create_task(budget.next_item_or_producer_waiting(queue))
    await asyncio.sleep(0.05)
    assert not get.done(), "no reservation is waiting, so only a queue item should wake the consumer"
    queue.put_nowait("item")
    assert await asyncio.wait_for(get, timeout=1) == "item"


class _ReachedWrite(Exception):
    pass


class _WriteProbePool:
    """Backend-shaped pool whose first acquire — the consumer's write — proves it got there."""

    _wraps_backend = True
    ops = None

    def acquire(self):
        raise _ReachedWrite


async def test_streaming_retain_writes_a_held_batch_instead_of_hanging(monkeypatch):
    """The real pipeline: a 1 MB budget, a 0.4 MB chunk then a 0.7 MB one, batches of 10.

    Before #5296 the consumer held the first chunk waiting for a ninth more, while the
    producer waited for room only that held chunk could free — hanging until the retain's
    wall-clock ceiling. Reaching the write is the proof it moved; the write itself fails on
    purpose so the test needs no database.
    """
    costs = iter([int(0.4 * _MB), int(0.7 * _MB)])
    calls = 0

    async def fake_extract_and_embed(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01 * calls)  # queue order: the small chunk, then the large one
        return orchestrator._EmbeddedExtraction([], [], [], TokenUsage())

    # A SQL-shaped store, so the consumer's write goes through the pool below whatever
    # backend the suite runs against.
    store = MagicMock()
    store.store_owned_for.return_value = False
    store.recovery_chunk_hashes = AsyncMock(return_value=set())

    monkeypatch.setattr(orchestrator, "_extract_and_embed", fake_extract_and_embed)
    monkeypatch.setattr(orchestrator, "estimate_chunk_bytes", lambda *_: next(costs))
    monkeypatch.setattr("hindsight_api.engine.memories.get_memories", lambda: store)
    config = MagicMock()
    config.retain_memory_budget_mb = 1
    chunks = ["chunk one", "chunk two"]

    with pytest.raises(_ReachedWrite):
        await asyncio.wait_for(
            orchestrator._streaming_retain_batch(
                pool=_WriteProbePool(),
                embeddings_model=MagicMock(),
                llm_config=MagicMock(),
                entity_resolver=MagicMock(),
                format_date_fn=str,
                bank_id="bank-5296",
                contents_dicts=[{"content": "\n".join(chunks)}],
                contents=[],
                config=config,
                document_id="doc-5296",
                is_first_batch=True,
                fact_type_override=None,
                document_tags=None,
                log_buffer=[],
                start_time=0.0,
                all_pre_chunks=list(chunks),
                chunk_to_content=[0, 0],
                chunk_batch_size=10,
            ),
            timeout=5,
        )
    assert calls == 2
