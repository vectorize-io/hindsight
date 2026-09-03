"""Collect the streamed retain sub-batches, for tests that assert over a whole split.

Retain consumes ``iter_sub_batches`` one sub-batch at a time and never holds the rest —
that is the point of it, since the slices are the document cut up (#3756). Tests, though,
mostly want to assert about the split as a whole: how many sub-batches a body produced,
which inputs each came from, how the chunk counts add up.

So the eager view lives here rather than in the engine. Keeping it in the engine would
mean shipping a function nothing in production calls, and CI's dead-code check would be
right to flag it.
"""

from hindsight_api.config import HindsightConfig
from hindsight_api.engine.memory_engine import (
    RetainContentDict,
    ScreenedDocumentBody,
    _iter_raw_sub_batches,
    _RawSubBatch,
    iter_sub_batches,
)


def collect_sub_batches(
    contents: list[RetainContentDict],
    tokens_per_batch: int,
    *,
    chunk_size: int,
    structured_chunk_size: int | None = None,
) -> list[_RawSubBatch]:
    """Drain the raw splitter, so a test can index into the split it produced."""
    return list(
        _iter_raw_sub_batches(
            contents,
            tokens_per_batch,
            chunk_size=chunk_size,
            structured_chunk_size=structured_chunk_size,
        )
    )


def collect_screened_bodies(
    contents: list[RetainContentDict],
    tokens_per_batch: int,
    *,
    chunk_size: int,
    structured_chunk_size: int | None = None,
    config: HindsightConfig,
) -> list[ScreenedDocumentBody | None]:
    """The screened, hashed body override of each sub-batch, in order.

    Goes through the full ``iter_sub_batches`` rather than the raw splitter, so it exercises
    the caching that screens an oversized item's identical body once however many slices it
    produced.
    """
    return [
        sub.document_body
        for sub in iter_sub_batches(
            contents,
            tokens_per_batch,
            chunk_size=chunk_size,
            structured_chunk_size=structured_chunk_size,
            config=config,
        )
    ]
