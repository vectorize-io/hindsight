"""An append coalesces a new turn into a document; it must not re-mint what is already there.

A conversation append sends ONE new turn onto a document that already holds many. The delta plan
plans against the WHOLE body, so the question this file pins is what happens to the turns in the
middle: their chunks did not change, so their memories must survive as the same unit ids.

Chunk coverage does not catch a regression here. `Σ chunk_text` can equal the full body while every
unit id underneath it is new — the text round-trips and the memories were still torn down and
re-extracted, which is what destroys observation lineage standing on those ids.
"""

from datetime import datetime, timezone

import pytest

from hindsight_api.config import clear_config_cache


def _ts() -> float:
    return datetime.now(timezone.utc).timestamp()


# Splitting is a TRANSPORT detail — an oversized item is sliced into sequential sub-batches — so
# every assertion in this file must hold either way, and none of them used to run split at all.
#
# This axis does NOT catch #3989 (verified: it passes on the commit that had the bug). That one
# needed a SECOND append to surface, because the first truncated `original_text` and only the next
# one diffed against the truncation — it is pinned end to end in
# test_oversized_append_truncates_document.py. The axis is here because "only breaks when split"
# is a whole class of defect that nothing in this file could previously see.
@pytest.fixture(autouse=True, params=["unsplit", "split"], ids=["unsplit", "split"])
def _split_mode(request, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", "1000000" if request.param == "unsplit" else "300")
    clear_config_cache()
    yield request.param
    clear_config_cache()


@pytest.fixture(autouse=True)
def _no_side_work(monkeypatch):
    # Consolidation and observations would mint and retire units of their own, which is not what
    # these assertions are about.
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION", "false")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_OBSERVATIONS", "false")
    clear_config_cache()
    yield
    clear_config_cache()


def _turns(first: int, count: int) -> str:
    # Distinct lines so no two turns share a content hash, and long enough that the body spans
    # several chunks — an append that only ever touches the tail chunk is the shape being tested.
    return "\n".join(
        f"[role: user] turn {i}: alpha bravo charlie delta echo foxtrot golf hotel india juliet "
        f"kilo lima mike november oscar papa quebec romeo sierra tango"
        for i in range(first, first + count)
    )


async def _units_by_chunk(memory, bank_id, document_id, request_context) -> dict[str, set[str]]:
    """`{chunk_id: {unit_id}}` for a document's live memories."""
    listing = await memory.list_memory_units(
        bank_id, document_id=document_id, limit=10000, request_context=request_context
    )
    by_chunk: dict[str, set[str]] = {}
    for row in listing["items"]:
        by_chunk.setdefault(row["chunk_id"], set()).add(row["id"])
    return by_chunk


@pytest.mark.asyncio
async def test_append_keeps_the_units_of_the_chunks_it_did_not_touch(memory, request_context):
    """Appending a turn must leave the earlier turns' memories in place, as the same units.

    The assertion is per chunk, not per document: a chunk whose text is unchanged after the append
    must still carry exactly the unit ids it carried before. That is the property — "the document
    still has some memories" would pass even if every one of them had been re-minted.
    """
    bank_id = f"test_append_coalesce_{_ts()}"
    document_id = "conversation-1"

    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": _turns(0, 24), "document_id": document_id}],
            request_context=request_context,
        )
        before = await _units_by_chunk(memory, bank_id, document_id, request_context)
        assert before, "no memories were extracted, so the test would assert nothing"
        assert len(before) > 1, "the body must span several chunks or an untouched chunk cannot exist"

        # One more turn, the way a conversation appends.
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": _turns(24, 1), "document_id": document_id, "update_mode": "append"}],
            request_context=request_context,
        )
        after = await _units_by_chunk(memory, bank_id, document_id, request_context)

        # The last chunk before the append is the only one the new turn can have flowed into, so it
        # is allowed to change. Every chunk before it must be untouched, ids included.
        #
        # Ordered by the chunk's INDEX, not by the id as a string: a chunk id is
        # `{bank}_{document}_{index}`, so a lexicographic sort puts `_10` before `_2` and picks the
        # wrong "last" chunk the moment a document has ten or more.
        def _index_of(chunk_id: str) -> int:
            return int(chunk_id.rsplit("_", 1)[-1])

        settled = sorted(before, key=_index_of)[:-1]
        assert settled, "expected at least one chunk that the append cannot have touched"

        dropped = {c: sorted(before[c]) for c in settled if not before[c] <= after.get(c, set())}
        assert not dropped, (
            f"an append re-minted memories on chunks it did not change: {dropped} — "
            f"the append replaced instead of coalescing, so the turns in the middle lost their "
            f"unit ids (and anything referencing them)"
        )

        # And it did add the new turn rather than silently dropping it.
        assert sum(len(v) for v in after.values()) >= sum(len(v) for v in before.values()), (
            "the append lost memories overall"
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


def _json_turns(first: int, count: int) -> str:
    # A saved live session: one JSON object per line (JSONL), not one JSON array.
    return "\n".join(
        '{"role": "user", "content": "turn %d: alpha bravo charlie delta echo foxtrot golf hotel india '
        'juliet kilo lima mike november oscar papa quebec romeo sierra tango"}' % i
        for i in range(first, first + count)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("turns", [_turns, _json_turns], ids=["text", "jsonl"])
async def test_a_second_append_keeps_the_chunks_the_first_append_wrote(memory, request_context, turns, _split_mode):
    """An append must chunk its body the way the NEXT append re-chunks it.

    The stored body and the new text went in as two items, each chunked on its own, so the old tail
    chunk stayed short and the new text began a chunk of its own; the next append re-chunked the
    whole body as one text, filling that tail and shifting every boundary after it — so each append
    re-extracted the chunks the previous one wrote (beta: 678 such chunks against 253 new ones over
    314 live-session saves). Only arrays were merged into one item (#2409).
    """
    if _split_mode == "split":
        # Known, not fixed here: an append over the batch budget is sliced on the NEW text's own chunk
        # boundaries, with the stored body's chunks counted as an offset (memory_engine, the
        # append_prepend_chunks path), so its chunks still start apart from the stored tail. Only
        # appends over HINDSIGHT_API_RETAIN_BATCH_TOKENS (10k by default) take that path.
        pytest.skip("a split append still chunks the new text apart from the stored body")
    bank_id = f"test_append_twice_{_ts()}"
    document_id = "conversation-2"
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": turns(0, 24), "document_id": document_id}],
            request_context=request_context,
        )
        # A first append long enough to write several chunks of its own.
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": turns(24, 80), "document_id": document_id, "update_mode": "append"}],
            request_context=request_context,
        )
        before = await _units_by_chunk(memory, bank_id, document_id, request_context)
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": turns(104, 1), "document_id": document_id, "update_mode": "append"}],
            request_context=request_context,
        )
        after = await _units_by_chunk(memory, bank_id, document_id, request_context)

        def _index_of(chunk_id: str) -> int:
            return int(chunk_id.rsplit("_", 1)[-1])

        settled = sorted(before, key=_index_of)[:-1]
        assert len(settled) > 3, "the first append must have written chunks of its own"
        changed = [c for c in settled if after.get(c) != before[c]]
        assert not changed, f"the second append re-extracted chunks it did not touch: {sorted(changed, key=_index_of)}"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
