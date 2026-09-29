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

import pytest

from hindsight_api.engine.retain.memory_budget import (
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
