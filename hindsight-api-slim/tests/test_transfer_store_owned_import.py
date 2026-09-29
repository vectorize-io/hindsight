"""Import into a bank whose memories live in the memories store rather than in Postgres.

The SQL import path reads source liveness from, and writes observations into, ``memory_units``.
A store that owns its memories leaves that table empty, so every source read as missing: every
observation was skipped, its writes matched no rows, and no imported fact kept the consolidation
markers the archive carried. These pin the store-backed halves that replace it.

They are also the only coverage those two functions get. CI runs against Postgres, where
``store_owned_for`` is False and neither is ever reached, so a defect in them would otherwise
ship green.
"""

from datetime import datetime, timezone

from hindsight_api.engine.memories.base import StoredMemory
from hindsight_api.engine.retain.types import ProcessedFact, pack_embedding
from hindsight_api.engine.transfer.importer import (
    _ObservationOutcome,
    _import_observations_via_store,
    _restore_fact_lifecycle_via_store,
)
from hindsight_api.engine.transfer.schema import TransferFact, TransferObservation
from tests.test_memories_extension import InMemoryMemories

BANK = "store-owned-bank"
SRC_A = "00000000-0000-0000-0000-0000000000a1"
SRC_B = "00000000-0000-0000-0000-0000000000b2"
OBS_TEXT = "Alice and Bob are colleagues."
# Well before anything the import stamps, so "kept its own marker" and "stamped now" are
# distinguishable by value rather than by merely being non-None.
EARLIER = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _source(unit_id: str, *, consolidated_at: datetime | None = None) -> StoredMemory:
    return StoredMemory(unit_id=unit_id, text="a source fact", fact_type="world", consolidated_at=consolidated_at)


def _store(*sources: StoredMemory) -> InMemoryMemories:
    store = InMemoryMemories()
    for row in sources:
        store.rows[row.unit_id] = row
    return store


def _processed() -> ProcessedFact:
    """The re-embedded observation, as the import pass hands it over."""
    return ProcessedFact(
        fact_text=OBS_TEXT,
        fact_type="observation",
        embedding=pack_embedding([0.1, 0.2]),
        occurred_start=None,
        occurred_end=None,
        mentioned_at=None,
        context="",
        metadata={},
    )


# -- observations ------------------------------------------------------------


async def test_the_observation_is_written_whole_with_its_sources():
    """The bug itself: against the SQL path every source read as missing and this was skipped."""
    store = _store(_source(SRC_A), _source(SRC_B))
    observation = TransferObservation(text=OBS_TEXT, source_id="src-obs", proof_count=4, tags=["team"])

    outcome = await _import_observations_via_store(
        store, BANK, [(observation, [SRC_A, SRC_B])], [_processed()], _ObservationOutcome()
    )

    assert outcome.imported == 1
    assert outcome.skipped == 0
    written = store.rows[outcome.remapped_unit_ids["src-obs"]]
    assert written.fact_type == "observation"
    assert written.text == OBS_TEXT
    assert written.proof_count == 4
    assert written.tags == ["team"]
    assert set(written.source_memory_ids) == {SRC_A, SRC_B}


async def test_an_observation_whose_source_is_missing_is_skipped():
    """The liveness check still refuses an observation the archive cannot stand up."""
    store = _store(_source(SRC_A))  # SRC_B was never imported

    outcome = await _import_observations_via_store(
        store, BANK, [(TransferObservation(text=OBS_TEXT), [SRC_A, SRC_B])], [_processed()], _ObservationOutcome()
    )

    assert outcome.imported == 0
    assert outcome.skipped == 1
    assert [row for row in store.rows.values() if row.fact_type == "observation"] == []


async def test_only_a_source_with_no_marker_of_its_own_is_stamped():
    """The SQL path's COALESCE: an archived marker wins over the one import would stamp."""
    store = _store(_source(SRC_A, consolidated_at=EARLIER), _source(SRC_B))

    await _import_observations_via_store(
        store, BANK, [(TransferObservation(text=OBS_TEXT), [SRC_A, SRC_B])], [_processed()], _ObservationOutcome()
    )

    assert store.rows[SRC_A].consolidated_at == EARLIER
    assert store.rows[SRC_B].consolidated_at is not None
    assert store.rows[SRC_B].consolidated_at > EARLIER


# -- fact consolidation markers ----------------------------------------------


async def test_fact_markers_are_restored_grouped_by_value():
    """Each marker value reaches the store, so an imported fact is never re-consolidated.

    ``InMemoryMemories.mark_consolidated`` writes ``when`` whichever marker it is, so the two
    groups are told apart here by their timestamps.
    """
    store = _store(_source(SRC_A), _source(SRC_B))
    done = datetime(2026, 2, 2, tzinfo=timezone.utc)
    failed = datetime(2026, 3, 3, tzinfo=timezone.utc)
    facts = [
        TransferFact(text="a", fact_type="world", consolidated_at=done),
        TransferFact(text="b", fact_type="world", consolidation_failed_at=failed),
    ]

    await _restore_fact_lifecycle_via_store(store, BANK, facts, [0, 1], [SRC_A, SRC_B])

    assert store.rows[SRC_A].consolidated_at == done
    assert store.rows[SRC_B].consolidated_at == failed


async def test_a_fact_the_import_dropped_is_skipped():
    """A degenerate fact filtered out on import has no target unit to stamp."""
    store = _store(_source(SRC_A))
    facts = [TransferFact(text="dropped", fact_type="world", consolidated_at=EARLIER)]

    await _restore_fact_lifecycle_via_store(store, BANK, facts, [None], [SRC_A])

    assert store.rows[SRC_A].consolidated_at is None
