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
