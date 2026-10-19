"""Document change history (#5638), rebuilt from the operations that made each change.

Observations are rebuilt, not versioned, so the history is the only place a removed one is still
named. These drive the real engine paths — an async retain, a fact edit, consolidation — and read
the history back through the engine, plus the consolidation attribution as a pure transform.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.document_history import (
    MAX_IDS,
    ObservationChanges,
    ObservationIdSample,
    note_observation_written,
    deletion_events,
    note_observations_removed,
    track_observation_changes,
)
from hindsight_api.engine.memory_engine import MemoryEngine

pytestmark = pytest.mark.memory_backend_incompatible


def test_id_sample_counts_everything_and_caps_the_ids():
    sample = ObservationIdSample.of([f"o{i:04d}" for i in range(MAX_IDS + 5)] + ["o0000"])
    assert sample.count == MAX_IDS + 5
    assert len(sample.ids) == MAX_IDS


def test_consolidation_changes_are_filed_under_every_source_document():
    changes = ObservationChanges(
        created={"new": ["f1", "f2"]},
        updated={"upd": ["f2", "f3"]},
        removed={"gone": ["f3"]},
    )
    events = changes.by_document({"f1": "doc-a", "f2": "doc-a", "f3": "doc-b"})

    assert set(events) == {"doc-a", "doc-b"}
    assert events["doc-a"].document_id == "doc-a"
    assert events["doc-a"].observations_created.ids == ["new"]
    assert events["doc-a"].observations_updated.ids == ["upd"]
    assert events["doc-a"].observations_removed.count == 0
    assert events["doc-b"].observations_created.count == 0
    assert events["doc-b"].observations_updated.ids == ["upd"]
    assert events["doc-b"].observations_removed.ids == ["gone"]


def test_a_delete_files_each_document_under_the_observations_its_own_facts_held():
    changes = ObservationChanges(removed={"o1": ["f1", "x"], "o2": ["f2"], "o3": ["f3"]})
    events = {e.document_id: e for e in deletion_events({"f1": "a", "f2": "a", "f3": "b", "f4": None}, changes)}

    assert set(events) == {"a", "b"}, "a fact with no document is not filed anywhere"
    assert (events["a"].facts_removed, events["a"].fact_id) == (2, None)
    assert events["a"].observations_removed.ids == ["o1", "o2"]
    assert (events["b"].facts_removed, events["b"].fact_id) == (1, "f3")
    assert events["b"].observations_removed.ids == ["o3"]


def test_notes_outside_a_tracked_operation_are_dropped():
    note_observations_removed({"x": ["f"]}, ["f"])  # must not raise
    with track_observation_changes() as changes:
        note_observations_removed({"x": ["f"]}, ["f"])
        note_observation_written("y", ["f"], created=True)
    note_observation_written("z", ["f"], created=False)
    assert changes.removed == {"x": ["f"]}
    assert changes.created == {"y": ["f"]}
    assert changes.updated == {}
    assert changes.swept_facts == {"f"}


async def _history(memory: MemoryEngine, bank_id: str, doc_id: str, request_context: RequestContext):
    result = await memory.get_document_history(bank_id, doc_id, request_context=request_context)
    assert result is not None
    return result


async def _ids(memory: MemoryEngine, bank_id: str, request_context: RequestContext, **filters) -> list[str]:
    page = await memory.list_memory_units(bank_id, request_context=request_context, **filters)
    return [m["id"] for m in page["items"]]


@pytest.mark.asyncio
async def test_history_is_rebuilt_from_the_operations_that_made_each_change(
    memory: MemoryEngine, request_context: RequestContext
):
    bank_id = f"test-doc-history-{uuid.uuid4().hex[:8]}"
    doc_id = "call-2026-10-01"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)

    # The test engine runs queued operations inline: this retains, then consolidates.
    await memory.submit_async_retain(
        bank_id,
        [{"content": "Alice called about her knee surgery recovery. She starts physio Monday.", "document_id": doc_id}],
        request_context=request_context,
    )
    facts = await _ids(memory, bank_id, request_context, document_id=doc_id, fact_type=["world", "experience"])
    first_obs = set(await _ids(memory, bank_id, request_context, fact_type="observation"))
    assert facts and first_obs

    history = await _history(memory, bank_id, doc_id, request_context)
    consolidated, retained = history.items
    # On the operation that ran the retain (a batch submit's child, not the id the submit returns).
    assert (retained.kind, retained.facts_added) == ("retain", len(facts))
    assert retained.operation_id is not None
    assert consolidated.kind == "consolidation"
    assert set(consolidated.observations_created.ids) == first_obs

    with patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()):
        await memory.update_memory_unit(
            bank_id, facts[0], occurred_start="2026-10-01T10:00:00Z", request_context=request_context
        )

    history = await _history(memory, bank_id, doc_id, request_context)
    assert [e.kind for e in history.items] == ["consolidation", "fact_edited", "consolidation", "retain"]
    rebuild, edited = history.items[0], history.items[1]
    assert edited.fact_id == facts[0]
    [removed] = edited.observations_removed.ids
    assert removed in first_obs, "the edit names the observation it deleted"
    # The edit is recorded on the consolidation it queued, next to that rebuild.
    assert edited.operation_id == rebuild.operation_id
    assert rebuild.observations_created.count == 1 and removed not in rebuild.observations_created.ids
    assert history.total == 3, "total counts operations: retain, consolidation, edit + its rebuild"

    assert (await _history(memory, bank_id, "some-other-doc", request_context)).items == []
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_a_retain_with_no_operation_leaves_no_retain_event(memory: MemoryEngine, request_context: RequestContext):
    """The stated limit: a synchronous retain runs without an operation, so only the consolidation
    it queues — which does have one — shows up."""
    bank_id = f"test-doc-history-sync-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)

    await memory.retain_async(
        bank_id, "Bob moved to Lisbon in March.", document_id="notes", request_context=request_context
    )

    assert [e.kind for e in (await _history(memory, bank_id, "notes", request_context)).items] == ["consolidation"]
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_append_invalidation_and_retag_are_each_recorded(memory: MemoryEngine, request_context: RequestContext):
    bank_id = f"test-doc-history-kinds-{uuid.uuid4().hex[:8]}"
    doc_id = "session-1"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)

    await memory.submit_async_retain(
        bank_id,
        [{"content": "Alice started physio on Monday.", "document_id": doc_id}],
        request_context=request_context,
    )
    await memory.submit_async_retain(
        bank_id,
        [{"content": "Alice also started swimming on Friday.", "document_id": doc_id, "update_mode": "append"}],
        request_context=request_context,
    )
    facts = await _ids(memory, bank_id, request_context, document_id=doc_id, fact_type=["world", "experience"])

    with patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()):
        await memory.update_memory_unit(bank_id, facts[0], state="invalidated", request_context=request_context)
    # A tag change moves every fact's observation scope, so it drops what was built on them.
    await memory.update_document(doc_id, bank_id, tags=["patient:alice"], request_context=request_context)

    history = await _history(memory, bank_id, doc_id, request_context)
    by_kind = {e.kind: e for e in reversed(history.items)}  # newest of each kind wins
    assert {"retain", "append", "fact_invalidated", "retag"} <= set(by_kind)
    assert by_kind["append"].facts_added > 0
    assert by_kind["fact_invalidated"].fact_id == facts[0]
    assert by_kind["retag"].observations_removed.count > 0
    # Both are recorded on the consolidation they queued, where its rebuild (if any) lands too.
    for kind in ("fact_invalidated", "retag"):
        op = await memory.get_operation_status(bank_id, by_kind[kind].operation_id, request_context=request_context)
        assert op["operation_type"] == "consolidation", (kind, op)

    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_deletes_and_clears_are_recorded_on_the_consolidation_they_queue(
    memory: MemoryEngine, request_context: RequestContext
):
    bank_id = f"test-doc-history-delete-{uuid.uuid4().hex[:8]}"
    doc_id = "visit-notes"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    await memory.submit_async_retain(
        bank_id,
        [
            {
                "content": "Bob moved to Lisbon. Bob adopted a dog. Bob started learning Portuguese.",
                "document_id": doc_id,
            }
        ],
        request_context=request_context,
    )
    facts = await _ids(memory, bank_id, request_context, document_id=doc_id, fact_type=["world", "experience"])
    assert len(facts) >= 3

    with patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()):
        await memory.clear_observations_for_memory(bank_id, facts[0], request_context=request_context)
        await memory.delete_memory_unit(facts[1], bank_id=bank_id, request_context=request_context)
        await memory.delete_memory_units(facts[2:], bank_id=bank_id, request_context=request_context)

    history = await _history(memory, bank_id, doc_id, request_context)
    changes = [e for e in history.items if e.kind in ("observations_cleared", "fact_deleted")]
    # Newest first: the bulk delete, the single delete, then the clear.
    assert [(e.kind, e.fact_id, e.facts_removed) for e in changes] == [
        ("fact_deleted", facts[2] if len(facts) == 3 else None, len(facts) - 2),
        ("fact_deleted", facts[1], 1),
        ("observations_cleared", facts[0], 0),
    ]
    for event in changes:
        assert event.observations_removed.count > 0
        op = await memory.get_operation_status(bank_id, event.operation_id, request_context=request_context)
        assert op["operation_type"] == "consolidation"

    await memory.delete_bank(bank_id, request_context=request_context)
