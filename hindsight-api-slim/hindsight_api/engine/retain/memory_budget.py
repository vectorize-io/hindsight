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

The one rule that keeps it from deadlocking: a reservation larger than the whole budget is
admitted anyway when nothing else is held. Otherwise a single unusually rich chunk would
wait forever for room that only it could free.
"""

import asyncio
import logging
from dataclasses import dataclass

from .types import ChunkMetadata, ExtractedFact, ProcessedFact

logger = logging.getLogger(__name__)

# What one fact costs beyond its own text: the dataclass, its dict, the entity refs and
# datetimes hanging off it. Deliberately a flat number — this is a budget, not accounting,
# and paying for a precise walk of every object would cost more than it saves.
_PER_FACT_OVERHEAD_BYTES = 512

# What one Python string costs before its characters. Ordinary ASCII content is one byte
# per character on top of this; text outside Latin-1 is two or four, so the estimate reads
# low for such content. It is a floor, and callers should size the budget with that in mind.
_PER_STRING_OVERHEAD_BYTES = 49


def estimate_chunk_bytes(
    processed: list[ProcessedFact],
    extracted: list[ExtractedFact],
    chunk_meta: list[ChunkMetadata],
) -> int:
    """Estimate what one extracted chunk holds while it waits to be written.

    Counts the three things that scale with the extraction — the processed facts (text,
    context and embedding), the raw extracted facts they came from, and the chunk text kept
    for the ``chunks`` row — and ignores the rest as noise against them.
    """
    total = 0
    for fact in processed:
        total += _PER_FACT_OVERHEAD_BYTES
        total += len(fact.fact_text) + _PER_STRING_OVERHEAD_BYTES
        total += len(fact.context) + _PER_STRING_OVERHEAD_BYTES
        total += len(fact.embedding) * fact.embedding.itemsize
    for raw in extracted:
        total += _PER_FACT_OVERHEAD_BYTES
        total += len(raw.fact_text) + _PER_STRING_OVERHEAD_BYTES
        total += len(raw.context) + _PER_STRING_OVERHEAD_BYTES
    for meta in chunk_meta:
        total += len(meta.chunk_text) + _PER_STRING_OVERHEAD_BYTES
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
        self._held_bytes = 0
        self._room = asyncio.Event()
        self._room.set()

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
        while self._held_bytes > 0 and self._held_bytes + nbytes > self.limit_bytes:
            self._room.clear()
            await self._room.wait()
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
