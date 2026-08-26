"""A document's body is written from an accumulator, not rebuilt from the store each sub-batch.

`put_document` REPLACES a document's chunk list — it takes the ordered texts whole. A sub-batched
retain therefore used to restore the prefix by reading `[0, chunk_index_offset)` back out of the
store on every sub-batch. That read is linear in the offset, so summed over a document it is
quadratic in its chunks, and the write beside it re-sent the same body text every time.

The sub-batches now hand their slices to an accumulator instead, keyed by offset so completion
order cannot reorder the document, and the accumulator writes:

- when the unwritten text has at least doubled (so the number of writes is logarithmic in the
  document and the total bytes written stay near the document's own size), and
- once more at the end, so a finished retain has a complete body.

These pin the three properties that matter: nothing is read back, the order is right whatever order
the slices arrive in, and a gap is never written across.
"""

import asyncio

import pytest

from hindsight_api.engine.retain.orchestrator import (
    _contiguous_prefix,
    _flush_document_body,
    flush_document_bodies,
)


def _acc(meta=True):
    return {
        "slices": {},
        "meta": (
            {
                "bank_id": "b",
                "content_hash": "h",
                "combined_content": "body",
                "merged_tags": [],
                "config": None,
                "retain_params": None,
                "expect_watermark": None,
            }
            if meta
            else None
        ),
        "flushed_bytes": 0,
        "lock": asyncio.Lock(),
    }


class _Recorder:
    """Captures what would be written, in place of the store call."""

    def __init__(self):
        self.writes: list[list[str]] = []
        self.offsets: list[int] = []
        self.watermarks: list[object] = []

    async def __call__(self, **kw):
        self.writes.append(list(kw["chunk_texts"]))
        self.offsets.append(kw["chunk_index_offset"])
        self.watermarks.append(kw["expect_watermark"])


@pytest.fixture
def recorder(monkeypatch):
    r = _Recorder()
    monkeypatch.setattr("hindsight_api.engine.retain.orchestrator._store_document_bodies", r)
    return r


# --- ordering -------------------------------------------------------------------------------


def test_slices_are_ordered_by_offset_not_by_arrival():
    """The property that makes concurrent sub-batches safe."""
    slices = {66: ["g", "h"], 0: ["a", "b", "c"], 3: ["d", "e", "f"]}
    # 0..2 then 3..5; 66 is past the end, so it is not contiguous and must be held back.
    assert _contiguous_prefix(slices) == ["a", "b", "c", "d", "e", "f"]


def test_a_gap_stops_the_prefix():
    """A positional list must never be written across a hole.

    Sub-batches can finish out of order, so the accumulator may hold slice 0 and slice 2 while
    slice 1 is still running. Writing them concatenated would shift every chunk after the hole.
    """
    assert _contiguous_prefix({0: ["a"], 5: ["z"]}) == ["a"]
    assert _contiguous_prefix({1: ["b"]}) == []
    assert _contiguous_prefix({}) == []


# --- when it writes -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_small_document_is_written_once_at_the_end(recorder):
    """Under the flush threshold, the whole body is one write — not one per sub-batch."""
    acc = _acc()
    for offset in range(0, 40, 10):
        acc["slices"][offset] = [f"chunk{offset + i}" for i in range(10)]
        await _flush_document_body(acc, "doc", force=False)

    assert recorder.writes == [], "nothing should be written before the threshold or the end"

    await flush_document_bodies({"doc": acc})
    assert len(recorder.writes) == 1, "a finished retain writes its body"
    assert len(recorder.writes[0]) == 40
    assert recorder.offsets == [0], "the accumulator holds the document from index 0"


@pytest.mark.asyncio
async def test_a_large_document_flushes_as_it_grows(recorder):
    """Progress is durable: a long retain does not hold the whole body until the end.

    This is what stops an interrupted retain leaving memories with no body at all. Doubling keeps
    the number of writes logarithmic rather than one per sub-batch.
    """
    acc = _acc()
    big = "x" * (5 * 1024 * 1024)  # over the flush minimum on its own
    offset = 0
    for _ in range(8):
        acc["slices"][offset] = [big]
        offset += 1
        await _flush_document_body(acc, "doc", force=False)

    assert recorder.writes, "a document this size must flush before the end"
    assert len(recorder.writes) < 8, "flushing per sub-batch is what this replaces"
    # Every write is a prefix of the next: the document only ever grows.
    for earlier, later in zip(recorder.writes, recorder.writes[1:]):
        assert later[: len(earlier)] == earlier
    assert all(o == 0 for o in recorder.offsets)


@pytest.mark.asyncio
async def test_the_append_watermark_rides_only_the_first_write(recorder):
    """An append's compare-and-set belongs to the write derived from the stored base."""
    acc = _acc()
    acc["meta"]["expect_watermark"] = 7
    big = "y" * (5 * 1024 * 1024)
    for offset in range(4):
        acc["slices"][offset] = [big]
        await _flush_document_body(acc, "doc", force=False)
    await flush_document_bodies({"doc": acc})

    assert recorder.watermarks[0] == 7
    assert all(w is None for w in recorder.watermarks[1:]), (
        "later writes build on what the first wrote, not on the old document"
    )


@pytest.mark.asyncio
async def test_nothing_is_written_for_a_document_that_produced_no_slices(recorder):
    await flush_document_bodies({"doc": _acc()})
    assert recorder.writes == []


@pytest.mark.asyncio
async def test_a_document_with_no_metadata_is_skipped(recorder):
    """No sub-batch reached the body write, so there is nothing to describe the document."""
    acc = _acc(meta=False)
    acc["slices"][0] = ["a"]
    await flush_document_bodies({"doc": acc})
    assert recorder.writes == []


@pytest.mark.asyncio
async def test_concurrent_flushes_of_one_document_do_not_interleave(recorder):
    """Slices arriving from concurrent sub-batches still produce a consistent body."""
    acc = _acc()

    async def add(offset: int):
        acc["slices"][offset] = [f"c{offset}"]
        await _flush_document_body(acc, "doc", force=False)

    await asyncio.gather(*(add(o) for o in reversed(range(20))))
    await flush_document_bodies({"doc": acc})

    assert len(recorder.writes) == 1
    assert recorder.writes[0] == [f"c{o}" for o in range(20)]


@pytest.mark.asyncio
async def test_the_accumulating_path_never_asks_the_store_for_the_prefix(recorder):
    """The quadratic, pinned.

    `_store_document_bodies` reads `[0, chunk_index_offset)` back out of the store — and ONLY when
    that offset is above zero (see the `if chunk_index_offset > 0` guard there). The accumulator
    holds the document from index 0, so it always writes at offset 0 and the read-back can never
    fire, whatever the sub-batch's own position in the document was.
    """
    acc = _acc()
    big = "z" * (5 * 1024 * 1024)
    # Slices deliberately start at a high offset once the first is in, mimicking a document deep
    # into its sub-batches — the case that used to read thousands of chunks back per write.
    for offset in range(6):
        acc["slices"][offset] = [big]
        await _flush_document_body(acc, "doc", force=False)
    await flush_document_bodies({"doc": acc})

    assert recorder.writes, "the fixture must actually write"
    assert set(recorder.offsets) == {0}, (
        f"every write must be at offset 0 so the prefix read-back cannot fire; got {sorted(set(recorder.offsets))}"
    )
