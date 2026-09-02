"""Re-retaining a document under a NARROWER tag set must invalidate its observations.

The reporter's shape: one document ("mobile room key is available") is tagged with the
18 hotels it applies to. One of those hotels stops offering it, so the document is
re-ingested — same body, ``tags`` now listing 17 hotels. The world facts are relabelled
correctly, but the mental model for the dropped hotel keeps citing the *observation*
built over those facts, because that observation still carries the old tag.

``MemoryEngine.update_document`` (the tags PATCH) already runs the cascade this needs —
delete the observations built over the document's memories and requeue their surviving
co-sources so consolidation rebuilds them under the new tags. Retain does not: the delta
path relabels surviving memory units through
``fact_storage.update_memory_units_metadata_and_tags`` and stops there. So the same tag
change invalidates observations when it arrives as a PATCH and does not when it arrives
as an upsert.

These tests drive the public retain API on the three shapes an upsert can take:
tags-only (identical body), tags plus a partial edit (unchanged chunks survive and get
relabelled), and a change to ``observation_scopes`` rather than to the tags themselves.
"""

import uuid

import pytest

from hindsight_api import RequestContext
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.memories import FactRecord, get_memories
from hindsight_api.engine.memory_engine import MemoryEngine, fq_table

# Two chunk-sized blocks (chunk size is 3000 chars). Keeping BLOCK_A byte-identical
# across a re-ingest is what keeps the second retain on the delta path: chunking is
# greedy from the start of the text, so an edit after chunk 0's boundary cannot move it.
_BLOCK_A = " ".join(
    f"The Grandview property offers mobile room key access on floor {i} of the north tower." for i in range(40)
)
_BLOCK_B = " ".join(f"Front desk staff issue physical keycards at kiosk {i} during check-in." for i in range(40))
_BLOCK_B_EDITED = " ".join(f"Front desk staff issue wristbands at kiosk {i} during check-in." for i in range(40))

_DOCUMENT_V1 = f"{_BLOCK_A} {_BLOCK_B}"
_DOCUMENT_V2_PARTIAL_EDIT = f"{_BLOCK_A} {_BLOCK_B_EDITED}"

_KEPT_HOTEL = "hotel-1234"
_DROPPED_HOTEL = "hotel-5720"


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


async def _retain(
    memory: MemoryEngine,
    bank_id: str,
    document_id: str,
    content: str,
    tags: list[str],
    request_context: RequestContext,
    observation_scopes: str | None = None,
) -> None:
    item: dict = {"content": content, "context": "hotel amenities", "document_id": document_id, "tags": list(tags)}
    if observation_scopes is not None:
        item["observation_scopes"] = observation_scopes
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[item],  # type: ignore[list-item]
        document_tags=list(tags),
        request_context=request_context,
    )


def _facts_by_chunk(facts: list[FactRecord]) -> dict[str, set[str]]:
    by_chunk: dict[str, set[str]] = {}
    for fact in facts:
        if fact.chunk_id:
            by_chunk.setdefault(fact.chunk_id, set()).add(fact.unit_id)
    return by_chunk


def _stale_scoped(observations: list[FactRecord], dropped_tag: str) -> list[tuple[str, list[str]]]:
    """Observations still visible to a tag no live fact carries any more."""
    return [(o.unit_id, list(o.tags or [])) for o in observations if dropped_tag in (o.tags or [])]


@pytest.mark.asyncio
async def test_tags_only_re_retain_invalidates_observations(memory: MemoryEngine, request_context: RequestContext):
    """The reporter's case: same body, one tag removed.

    The body is byte-identical, so retain takes the "no chunks changed" path and does
    nothing but relabel. The facts end up scoped to the 1 remaining hotel while the
    observation built over them is still scoped to the dropped one — which is exactly
    what that hotel's mental model keeps reading.
    """
    bank_id = f"test_retag_obs_tags_only_{uuid.uuid4().hex[:8]}"
    document_id = "mobile-room-key"

    try:
        await _retain(memory, bank_id, document_id, _DOCUMENT_V1, [_KEPT_HOTEL, _DROPPED_HOTEL], request_context)
        await memory.run_consolidation(bank_id=bank_id, request_context=request_context)

        observations_v1 = await _observations(memory, bank_id)
        assert observations_v1, "Setup: consolidation should have produced observations"
        assert any(_DROPPED_HOTEL in (o.tags or []) for o in observations_v1), (
            "Setup: the observations should be scoped to the hotel that is about to be dropped"
        )

        # The upsert: identical body, one tag fewer.
        await _retain(memory, bank_id, document_id, _DOCUMENT_V1, [_KEPT_HOTEL], request_context)

        facts_v2 = await _facts(memory, bank_id)
        assert facts_v2, "Setup: the facts should have survived a tags-only re-ingest"
        assert all(_DROPPED_HOTEL not in (f.tags or []) for f in facts_v2), (
            "Setup: the world facts should have been relabelled to the narrower tag set"
        )

        stale = _stale_scoped(await _observations(memory, bank_id), _DROPPED_HOTEL)
        assert stale == [], (
            f"{len(stale)} observation(s) are still scoped to {_DROPPED_HOTEL} after the document "
            f"was re-retained without it. The facts underneath them are no longer tagged with it, "
            f"so that hotel's mental model keeps citing an observation nothing supports: {stale[:5]}"
        )

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_tag_change_with_partial_edit_invalidates_unchanged_chunks_observations(
    memory: MemoryEngine, request_context: RequestContext
):
    """A narrower tag set arriving alongside a body edit.

    Delta sweeps the observations of the chunks it *deletes*, so a tag change that
    happens to land on an edited chunk is covered by accident. The unchanged chunk is
    the gap: its facts are relabelled in place and their observations are left alone.
    """
    bank_id = f"test_retag_obs_partial_{uuid.uuid4().hex[:8]}"
    document_id = "mobile-room-key"

    try:
        await _retain(memory, bank_id, document_id, _DOCUMENT_V1, [_KEPT_HOTEL, _DROPPED_HOTEL], request_context)
        await memory.run_consolidation(bank_id=bank_id, request_context=request_context)

        facts_v1 = await _facts(memory, bank_id)
        by_chunk_v1 = _facts_by_chunk(facts_v1)
        assert len(by_chunk_v1) >= 2, f"Setup: the document should span several chunks, got {list(by_chunk_v1)}"
        first_chunk = sorted(by_chunk_v1)[0]
        kept_fact_ids = by_chunk_v1[first_chunk]

        observations_v1 = await _observations(memory, bank_id)
        obs_only_over_kept = {
            o.unit_id
            for o in observations_v1
            if o.source_memory_ids and set(o.source_memory_ids).issubset(kept_fact_ids)
        }
        assert obs_only_over_kept, "Setup: the unchanged chunk should have observations of its own"

        # Edit the tail AND narrow the tags in the same upsert.
        await _retain(memory, bank_id, document_id, _DOCUMENT_V2_PARTIAL_EDIT, [_KEPT_HOTEL], request_context)

        surviving_fact_ids = {f.unit_id for f in await _facts(memory, bank_id)}
        assert kept_fact_ids.issubset(surviving_fact_ids), (
            "The unchanged chunk's facts should survive — if they did not, this test fell back to "
            "the full-replace path and no longer covers the retag gap"
        )

        stale = _stale_scoped(await _observations(memory, bank_id), _DROPPED_HOTEL)
        assert stale == [], (
            f"{len(stale)} observation(s) over the UNCHANGED chunk are still scoped to "
            f"{_DROPPED_HOTEL}; delta relabelled their source facts but never invalidated them: "
            f"{stale[:5]}"
        )

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_observation_scopes_change_re_retain_invalidates_observations(
    memory: MemoryEngine, request_context: RequestContext
):
    """Changing how a document's facts are scoped is a change to the observations too.

    ``observation_scopes`` decides which passes a fact is consolidated under: "combined"
    folds all its tags into one observation, "per_tag" produces one per tag. Switching it
    on a re-retain has to rebuild the observations, the same way narrowing the tags does —
    otherwise the bank keeps the observations of the OLD scoping alongside the new ones.
    """
    bank_id = f"test_retag_obs_scopes_{uuid.uuid4().hex[:8]}"
    document_id = "mobile-room-key"

    try:
        await _retain(
            memory,
            bank_id,
            document_id,
            _DOCUMENT_V1,
            [_KEPT_HOTEL, _DROPPED_HOTEL],
            request_context,
            observation_scopes="combined",
        )
        await memory.run_consolidation(bank_id=bank_id, request_context=request_context)

        observations_v1 = {o.unit_id for o in await _observations(memory, bank_id)}
        assert observations_v1, "Setup: consolidation should have produced observations"

        # Same body, same tags, different scoping.
        await _retain(
            memory,
            bank_id,
            document_id,
            _DOCUMENT_V1,
            [_KEPT_HOTEL, _DROPPED_HOTEL],
            request_context,
            observation_scopes="per_tag",
        )

        facts_v2 = await _facts(memory, bank_id)
        assert facts_v2, "Setup: the facts should have survived a metadata-only re-ingest"
        assert all(f.observation_scopes == "per_tag" for f in facts_v2), (
            "Surviving facts should carry the observation_scopes the re-retain supplied, "
            f"got {[f.observation_scopes for f in facts_v2[:5]]}"
        )

        surviving_obs = {o.unit_id for o in await _observations(memory, bank_id)}
        assert not surviving_obs.intersection(observations_v1), (
            "Observations built under the previous observation_scopes should have been "
            "invalidated so consolidation can rebuild them under the new scoping"
        )

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
