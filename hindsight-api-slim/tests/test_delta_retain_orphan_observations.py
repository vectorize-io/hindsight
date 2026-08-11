"""End-to-end regression for issue #3294: delta retain must not orphan observations.

The reporter's sequence, driven through the public engine API rather than by calling
the storage helpers directly:

    retain(document) -> consolidate -> retain(same document_id, edited) -> consolidate

Before the fix, the second retain took the delta path, which deletes the changed
chunks and lets the FK cascade drop their facts — with no observation sweep in
between. The observations derived from those facts stayed behind, still valid and
still recallable, pointing at ``source_memory_ids`` that no longer resolved. Nothing
could reach them afterwards: consolidation batches are built from facts, so an
observation whose sources are all gone is never selected into a batch again.

These tests assert the invariant the lifecycle documents ("removing a document: all
observations derived from the document's memories are deleted") on the paths delta
retain actually takes: an edit, a removal, and a no-op re-ingest. A rewrite of *every*
chunk is deliberately not covered here — with no unchanged chunk left, delta declines
and the full-replace path (already covered in ``test_observation_invalidation.py``)
handles it.
"""

import uuid

import pytest

from hindsight_api import RequestContext
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.memories import FactRecord, get_memories
from hindsight_api.engine.memory_engine import MemoryEngine, fq_table

# Delta retain works per chunk, so the document has to be big enough to produce several
# (chunk size is 3000 chars) and at least one of them must come out unchanged — with no
# unchanged chunk delta declines and falls back to a full replace. Keeping the FIRST
# block byte-identical across a re-ingest is what guarantees that: chunking is greedy
# from the start of the text, so an edit after chunk 0's boundary cannot move it.
_BLOCK_A = " ".join(
    f"Alice shipped the Alpha{i} milestone at Google in the search infrastructure group." for i in range(40)
)
_BLOCK_B = " ".join(f"Bob reviewed the Beta{i} rollout at Microsoft in the Azure networking group." for i in range(40))
_BLOCK_B_EDITED = " ".join(
    f"Bob reviewed the Beta{i} rollout at Amazon in the AWS networking group." for i in range(40)
)

_DOCUMENT_V1 = f"{_BLOCK_A} {_BLOCK_B}"
_DOCUMENT_V2_PARTIAL_EDIT = f"{_BLOCK_A} {_BLOCK_B_EDITED}"


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


async def _scan(memory: MemoryEngine, bank_id: str, fact_types: list[str]) -> list[FactRecord]:
    """Every stored memory of these types, read through whichever store holds them."""
    store = get_memories()
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        page = await store.scan_memories(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            fact_types=fact_types,
            limit=1_000_000,
        )
    return list(page.memories)


async def _facts(memory: MemoryEngine, bank_id: str) -> list[FactRecord]:
    return await _scan(memory, bank_id, ["experience", "world"])


async def _observations(memory: MemoryEngine, bank_id: str) -> list[FactRecord]:
    return await _scan(memory, bank_id, ["observation"])


def _broken_source_refs(observations: list[FactRecord], live_fact_ids: set[str]) -> list[tuple[str, list[str]]]:
    """The reporter's diagnostic: observations whose sources no longer resolve.

    Returns ``(observation_id, unresolvable_source_ids)`` per affected row — what the
    bug report counted as "broken references" on their bank.
    """
    broken = []
    for obs in observations:
        missing = [sid for sid in obs.source_memory_ids if sid not in live_fact_ids]
        if missing:
            broken.append((obs.unit_id, missing))
    return broken


async def _assert_no_orphans(memory: MemoryEngine, bank_id: str, when: str) -> None:
    facts = await _facts(memory, bank_id)
    observations = await _observations(memory, bank_id)
    broken = _broken_source_refs(observations, {f.unit_id for f in facts})
    assert broken == [], (
        f"{when}: {len(broken)} of {len(observations)} observation(s) reference deleted source "
        f"memories (issue #3294 — delta retain cascaded the facts away without sweeping the "
        f"observations derived from them): {broken[:5]}"
    )


async def _retain_document(
    memory: MemoryEngine, bank_id: str, document_id: str, content: str, request_context: RequestContext
) -> None:
    await memory.retain_async(
        bank_id=bank_id,
        content=content,
        context="team roster",
        document_id=document_id,
        request_context=request_context,
    )


def _facts_by_chunk(facts: list[FactRecord]) -> dict[str, set[str]]:
    by_chunk: dict[str, set[str]] = {}
    for fact in facts:
        if fact.chunk_id:
            by_chunk.setdefault(fact.chunk_id, set()).add(fact.unit_id)
    return by_chunk


@pytest.mark.asyncio
async def test_delta_retain_partial_edit_leaves_no_orphan_observations(
    memory: MemoryEngine, request_context: RequestContext
):
    """Editing the tail of a consolidated document orphans nothing.

    Also pins the precision of the sweep: the untouched first chunk keeps its facts
    AND the observations derived only from them, so a small edit does not
    re-consolidate the whole document — the case delta retain exists for.
    """
    bank_id = f"test_delta_orphan_partial_{uuid.uuid4().hex[:8]}"
    document_id = "roster-doc"

    try:
        await _retain_document(memory, bank_id, document_id, _DOCUMENT_V1, request_context)
        await memory.run_consolidation(bank_id=bank_id, request_context=request_context)

        facts_v1 = await _facts(memory, bank_id)
        observations_v1 = await _observations(memory, bank_id)
        by_chunk_v1 = _facts_by_chunk(facts_v1)
        assert len(by_chunk_v1) >= 2, f"Setup: the document should span several chunks, got {list(by_chunk_v1)}"
        assert observations_v1, "Setup: consolidation should have produced observations to orphan"
        await _assert_no_orphans(memory, bank_id, "after the first retain")

        first_chunk = sorted(by_chunk_v1)[0]
        kept_fact_ids = by_chunk_v1[first_chunk]
        edited_fact_ids = {fid for chunk, ids in by_chunk_v1.items() if chunk != first_chunk for fid in ids}
        assert kept_fact_ids and edited_fact_ids

        obs_over_edited = {o.unit_id for o in observations_v1 if edited_fact_ids.intersection(o.source_memory_ids)}
        obs_only_over_kept = {
            o.unit_id
            for o in observations_v1
            if o.source_memory_ids and set(o.source_memory_ids).issubset(kept_fact_ids)
        }
        assert obs_over_edited, "Setup: the chunks being edited should have observations derived from them"
        assert obs_only_over_kept, "Setup: the unchanged chunk should have observations of its own"

        # Re-ingest with only the tail changed — this is the delta path.
        await _retain_document(memory, bank_id, document_id, _DOCUMENT_V2_PARTIAL_EDIT, request_context)

        surviving_fact_ids = {f.unit_id for f in await _facts(memory, bank_id)}
        # Delta really applied: the unchanged chunk's facts were preserved rather than
        # re-extracted under new ids (a full replace would have changed all of them).
        assert kept_fact_ids.issubset(surviving_fact_ids), (
            "Unchanged chunk's facts should survive the delta re-ingest — if they did not, this "
            "test fell back to the full-replace path and no longer covers the bug"
        )
        assert not edited_fact_ids.intersection(surviving_fact_ids), "The edited chunks' facts should be gone"

        await _assert_no_orphans(memory, bank_id, "after the delta re-ingest")

        observation_ids_v2 = {o.unit_id for o in await _observations(memory, bank_id)}
        assert not observation_ids_v2.intersection(obs_over_edited), (
            "Observations derived from the edited chunks' facts should have been invalidated"
        )
        assert obs_only_over_kept.issubset(observation_ids_v2), (
            "Observations derived only from the unchanged chunk must survive a partial edit"
        )

        # And the follow-up consolidation the reporter ran — still no orphans.
        await memory.run_consolidation(bank_id=bank_id, request_context=request_context)
        await _assert_no_orphans(memory, bank_id, "after re-consolidating the edited document")

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delta_retain_removed_chunks_leave_no_orphan_observations(
    memory: MemoryEngine, request_context: RequestContext
):
    """Shortening a document orphans nothing either.

    Delta deletes removed chunks through the same call as changed ones, so this
    covers the ``removed_indices`` half of that list — a document that shrinks loses
    facts without any replacement being extracted for them.
    """
    bank_id = f"test_delta_orphan_shrink_{uuid.uuid4().hex[:8]}"
    document_id = "roster-doc"

    try:
        await _retain_document(memory, bank_id, document_id, _DOCUMENT_V1, request_context)
        await memory.run_consolidation(bank_id=bank_id, request_context=request_context)

        facts_v1 = await _facts(memory, bank_id)
        observations_v1 = await _observations(memory, bank_id)
        by_chunk_v1 = _facts_by_chunk(facts_v1)
        assert len(by_chunk_v1) >= 2, f"Setup: the document should span several chunks, got {list(by_chunk_v1)}"
        assert observations_v1, "Setup: consolidation should have produced observations to orphan"

        first_chunk = sorted(by_chunk_v1)[0]
        kept_fact_ids = by_chunk_v1[first_chunk]
        dropped_fact_ids = {fid for chunk, ids in by_chunk_v1.items() if chunk != first_chunk for fid in ids}
        obs_over_dropped = {o.unit_id for o in observations_v1 if dropped_fact_ids.intersection(o.source_memory_ids)}
        assert obs_over_dropped, "Setup: the chunks being dropped should have observations derived from them"

        # Re-ingest only the first block: every later chunk is removed outright.
        await _retain_document(memory, bank_id, document_id, _BLOCK_A, request_context)

        surviving_fact_ids = {f.unit_id for f in await _facts(memory, bank_id)}
        assert kept_fact_ids.issubset(surviving_fact_ids), (
            "The retained chunk's facts should survive — if they did not, this test fell back "
            "to the full-replace path and no longer covers the bug"
        )
        assert not dropped_fact_ids.intersection(surviving_fact_ids), "The removed chunks' facts should be gone"

        await _assert_no_orphans(memory, bank_id, "after shrinking the document")
        assert not {o.unit_id for o in await _observations(memory, bank_id)}.intersection(obs_over_dropped), (
            "Observations derived from the removed chunks' facts should have been invalidated"
        )

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delta_retain_unchanged_content_keeps_observations(memory: MemoryEngine, request_context: RequestContext):
    """Re-submitting identical content deletes no chunk, so it sweeps no observation
    and requeues nothing for consolidation."""
    bank_id = f"test_delta_orphan_noop_{uuid.uuid4().hex[:8]}"
    document_id = "roster-doc"

    try:
        await _retain_document(memory, bank_id, document_id, _DOCUMENT_V1, request_context)
        await memory.run_consolidation(bank_id=bank_id, request_context=request_context)
        observation_ids_v1 = {o.unit_id for o in await _observations(memory, bank_id)}
        assert observation_ids_v1

        await _retain_document(memory, bank_id, document_id, _DOCUMENT_V1, request_context)

        assert {o.unit_id for o in await _observations(memory, bank_id)} == observation_ids_v1, (
            "A no-op delta re-ingest must not touch existing observations"
        )
        assert all(f.consolidated_at is not None for f in await _facts(memory, bank_id)), (
            "A no-op delta re-ingest must not requeue facts for consolidation"
        )
        await _assert_no_orphans(memory, bank_id, "after a no-op re-ingest")

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
