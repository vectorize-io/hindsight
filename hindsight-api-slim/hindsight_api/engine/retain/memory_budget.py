"""A byte budget for the extracted-but-unwritten state one retain holds (issue #3756).

The streaming retain pipeline has always bounded itself by a **count**: at most
``retain_chunk_batch_size`` chunks in a consumer batch, and twice that queued. A count is
only a memory bound if chunks cost a predictable amount, and they do not — a chunk carries
however many facts the extractor found in it, each with its own text, context and
embedding. Between a terse chunk yielding one fact and a dense one yielding fifty, the same
"100 chunks" spans two orders of magnitude. The limit was a number nobody could size a
worker against.

This makes it a number that means something. The producer reserves the estimated cost of a
chunk's facts before queueing them and the consumer releases it once they are committed, so
what the pipeline holds is bounded in bytes regardless of how the document chunked, how
verbose extraction was, or how large the embedding model's vectors are. Under the budget
nothing changes and extraction runs at full width; at the budget the producer waits, which
is the pipeline throttling itself rather than the worker being OOM-killed.

Two rules keep it from deadlocking. A reservation larger than the whole budget is admitted
anyway when nothing else is held, or a single unusually rich chunk would wait forever for
room that only it could free. And a consumer that has drained the queue writes its open
batch as soon as a producer is waiting for room (``RetainMemoryBudget.next_item_or_producer_waiting``), or a
small open batch plus a large next chunk would leave each side waiting on the other (#5296).
"""

import asyncio
import enum
import logging
from dataclasses import dataclass
from typing import TypeVar

from .types import ChunkMetadata, ExtractedFact, ProcessedFact

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


# A one-member enum rather than a bare `object()` so `item is PRODUCER_WAITING` narrows the
# type for the checker, and the consumer can index the real queue items without a cast.
class _ProducerWaiting(enum.Enum):
    TOKEN = "producer_waiting"


# Returned by `next_item_or_producer_waiting` instead of a queue item: the queue is empty and
# a producer is blocked on the budget, so the consumer must write what it holds.
PRODUCER_WAITING = _ProducerWaiting.TOKEN

# What one fact costs beyond its own text: the dataclass, its dict, the entity refs and
# datetimes hanging off it. Deliberately a flat number — this is a budget, not accounting,
# and paying for a precise walk of every object would cost more than it saves.
_PER_FACT_OVERHEAD_BYTES = 512

# What one Python string costs before its characters. Ordinary ASCII content is one byte
# per character on top of this; text outside Latin-1 is two or four, so the estimate reads
# low for such content. It is a floor, and callers should size the budget with that in mind.
_PER_STRING_OVERHEAD_BYTES = 49


def _text_bytes(text: str | None) -> int:
    """What a string costs, counting ``None`` and the empty string as the header alone."""
    return _PER_STRING_OVERHEAD_BYTES + len(text or "")


def estimate_chunk_bytes(
    processed: list[ProcessedFact],
    extracted: list[ExtractedFact],
    chunk_meta: list[ChunkMetadata],
) -> int:
    """Estimate what one extracted chunk holds while it waits to be written.

    Counts the three things that scale with the extraction — the processed facts (text,
    context and embedding), the raw extracted facts they came from, and the chunk text kept
    for the ``chunks`` row — and ignores the rest as noise against them.

    Every string is read through ``_text_bytes``, which tolerates ``None``. The fields are
    annotated ``str``, but a converted file upload retains with no context and puts ``None``
    in one — and an estimate that raises turns a heuristic into a failed retain, which is
    exactly what it did to the file-retain path before this guard.
    """
    total = 0
    for fact in processed:
        total += _PER_FACT_OVERHEAD_BYTES + _text_bytes(fact.fact_text) + _text_bytes(fact.context)
        total += len(fact.embedding) * fact.embedding.itemsize
    for raw in extracted:
        total += _PER_FACT_OVERHEAD_BYTES + _text_bytes(raw.fact_text) + _text_bytes(raw.context)
    for meta in chunk_meta:
        total += _text_bytes(meta.chunk_text)
    return total


@dataclass
class RetainMemoryBudget:
    """Bytes of in-flight extraction state one retain operation may hold at once.

    Not a semaphore over a count of things but over their measured size, which is what a
    worker's memory limit is actually denominated in. ``limit_bytes <= 0`` disables it: the
    pipeline then falls back to the chunk-count bound it had before, which is the right
    behaviour for a deployment that has tuned that count deliberately.

    Single-consumer by construction — one retain operation owns one instance — so the
    waiter set only ever holds the producer's extraction tasks.
    """

    limit_bytes: int

    def __post_init__(self) -> None:
        # Fail loudly on a limit that is not a number. Every comparison below would
        # otherwise "succeed" against a stand-in object and return something truthy, which
        # makes `reserve` wait for room that can never be reported — a retain that hangs
        # until its wall-clock ceiling instead of one that reports a bad config. A caller
        # handing this a mock rather than a resolved config is the realistic way to get
        # here, and a 300-second hang is a terrible way to find that out.
        if not isinstance(self.limit_bytes, int) or isinstance(self.limit_bytes, bool):
            raise TypeError(
                f"RetainMemoryBudget needs an int limit_bytes, got {type(self.limit_bytes).__name__}; "
                "pass config.retain_memory_budget_mb * 1024 * 1024 from a resolved config"
            )
        self._held_bytes = 0
        self._room = asyncio.Event()
        self._room.set()
        # Set while at least one `reserve` is blocked, so the consumer can tell "the producer
        # is still extracting" from "the producer is waiting for me to free room".
        self._waiters = 0
        self._producer_waiting = asyncio.Event()

    @property
    def held_bytes(self) -> int:
        """Bytes currently reserved: queued facts plus the consumer's open batch."""
        return self._held_bytes

    @property
    def enabled(self) -> bool:
        return self.limit_bytes > 0

    async def reserve(self, nbytes: int) -> None:
        """Wait until ``nbytes`` fits, then hold it.

        Returns immediately when the budget is disabled, and when nothing is held — a
        reservation bigger than the entire budget has to be admitted by someone, and the
        only safe someone is the caller that finds the pipeline empty.
        """
        if not self.enabled:
            return
        if self._held_bytes > 0 and self._held_bytes + nbytes > self.limit_bytes:
            self._waiters += 1
            self._producer_waiting.set()
            try:
                while self._held_bytes > 0 and self._held_bytes + nbytes > self.limit_bytes:
                    self._room.clear()
                    await self._room.wait()
            finally:
                self._waiters -= 1
                if self._waiters == 0:
                    self._producer_waiting.clear()
        self._held_bytes += nbytes
        if self._held_bytes >= self.limit_bytes:
            self._room.clear()

    def release(self, nbytes: int) -> None:
        """Give back a reservation once its facts are committed and dropped."""
        if not self.enabled:
            return
        self._held_bytes = max(0, self._held_bytes - nbytes)
        if self._held_bytes < self.limit_bytes:
            self._room.set()

    def should_flush(self, batch_bytes: int) -> bool:
        """Whether the consumer should write now rather than take another chunk.

        True once an open batch has grown past half the budget, which leaves the other half
        for the producer to keep extracting into while that batch is being written. Without
        this the consumer would happily accumulate to its chunk-count trigger and the
        producer would simply stall against a full budget — correct, but serialised.
        """
        if not self.enabled:
            return False
        return batch_bytes * 2 >= self.limit_bytes

    async def next_item_or_producer_waiting(self, queue: "asyncio.Queue[_T]") -> "_T | _ProducerWaiting":
        """The next queue item, or ``PRODUCER_WAITING`` if a producer blocks on the budget first.

        For a consumer holding an open batch. Its reservation counts against the budget, so a
        producer waiting for room can be waiting on that very batch — and the consumer waiting
        for the producer's next item would then wait forever (#5296). Only an empty queue races
        the signal: with items queued the consumer keeps its normal batching, so this changes
        nothing until the alternative is a hang.

        ``Queue.get`` is cancellation-safe — an item only leaves the queue once the getter
        resumes — so abandoning the pending get when the signal wins loses nothing.
        """
        if not queue.empty() or not self.enabled:
            return await queue.get()
        get = asyncio.ensure_future(queue.get())
        waiting = asyncio.ensure_future(self._producer_waiting.wait())
        try:
            await asyncio.wait({get, waiting}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiting.cancel()
            if not get.done():
                get.cancel()
        if get.done() and not get.cancelled():
            return get.result()
        return PRODUCER_WAITING
