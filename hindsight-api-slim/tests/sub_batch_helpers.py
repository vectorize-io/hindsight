"""Collect the streamed retain sub-batches, for tests that assert over a whole split.

Retain consumes ``iter_sub_batches`` one sub-batch at a time and never holds the rest —
that is the point of it, since the slices are the document cut up (#3756). Tests, though,
mostly want to assert about the split as a whole: how many sub-batches a body produced,
which inputs each came from, how the chunk counts add up.

So the eager view lives here rather than in the engine. Keeping it in the engine would
mean shipping a function nothing in production calls, and CI's dead-code check would be
right to flag it.
"""

from dataclasses import dataclass, field

from hindsight_api.config import HindsightConfig
from hindsight_api.engine.memory_engine import (
    RetainContentDict,
    ScreenedDocumentBody,
    _iter_raw_sub_batches,
    iter_sub_batches,
)


@dataclass
class CollectedSplit:
    """Every sub-batch of one split, as parallel lists indexed together.

    ``sub_batches[i]`` is the content items of sub-batch ``i``; ``origin_indices[i]`` the
    indices into the submitted contents that fed it; ``document_body_overrides[i]`` the full
    original body when ``i`` is a slice of an oversized item, else ``None``; and
    ``chunk_counts[i]`` how many native chunks it holds.
    """

    sub_batches: list[list[RetainContentDict]] = field(default_factory=list)
    origin_indices: list[list[int]] = field(default_factory=list)
    document_body_overrides: list[str | None] = field(default_factory=list)
    chunk_counts: list[int] = field(default_factory=list)


def collect_sub_batches(
    contents: list[RetainContentDict],
    tokens_per_batch: int,
    *,
    chunk_size: int,
    structured_chunk_size: int | None = None,
) -> CollectedSplit:
    """Drain the raw splitter into a ``CollectedSplit``."""
    collected = CollectedSplit()
    for raw in _iter_raw_sub_batches(
        contents,
        tokens_per_batch,
        chunk_size=chunk_size,
        structured_chunk_size=structured_chunk_size,
    ):
        collected.sub_batches.append(raw.contents)
        collected.origin_indices.append(raw.origins)
        collected.document_body_overrides.append(raw.body_override)
        collected.chunk_counts.append(raw.chunk_count)
    return collected


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
