"""Retain ingestion must not store null metadata values (issue #3209).

The retain API accepts arbitrary JSON metadata; a null value (e.g.
{"ocr_engine": null}) stored verbatim poisons the read path, which validates
MemoryFact.metadata as dict[str, str] and made every recall fail for the
affected rows. RetainContent drops null-valued keys at construction, so facts
extracted from it (metadata=content.metadata in fact_extraction) stay canonical
on the write side; the read path drops nulls again for legacy rows.
Non-string values are preserved as-is here and coerced by the read path.

The delta paths, which sync document metadata onto units they preserve rather
than re-extracting them, are covered on a real database by
test_delta_retain.py::test_delta_retain_drops_null_metadata_values.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine import memories as memories_module
from hindsight_api.engine.memories.base import META_METADATA_JSON, ScanPage, StoredMemory
from hindsight_api.engine.metadata_utils import as_string_metadata, drop_null_values
from hindsight_api.engine.retain import fact_storage
from hindsight_api.engine.retain.orchestrator import _build_contents
from hindsight_api.engine.retain.types import RetainContent
from hindsight_api.engine.temporal_precision import OCCURRENCE_PRECISION_METADATA_KEY


def test_drop_null_values_keeps_everything_else_untouched():
    assert drop_null_values({"a": None, "b": "x", "c": 5, "d": ""}) == {"b": "x", "c": 5, "d": ""}


def test_drop_null_values_normalizes_absent_metadata_to_empty_dict():
    assert drop_null_values(None) == {}


def test_as_string_metadata_drops_nulls_and_stringifies_the_rest():
    assert as_string_metadata({"a": None, "n": 348}) == {"n": "348"}


def test_retain_content_drops_null_metadata_values():
    content = RetainContent(content="hi", metadata={"ocr_engine": None, "source": "slack"})
    assert content.metadata == {"source": "slack"}


def test_retain_content_keeps_non_null_values_as_given():
    content = RetainContent(content="hi", metadata={"n": 5, "source": "slack"})
    assert content.metadata == {"n": 5, "source": "slack"}


def test_retain_content_normalizes_null_metadata_to_empty_dict():
    """``"metadata": null`` in the request reaches the dataclass as None; the
    field is declared dict[str, str], so it must not stay None."""
    assert RetainContent(content="hi", metadata=None).metadata == {}


def test_build_contents_normalizes_null_metadata_from_api():
    """The ingestion path accepts JSON null metadata; stored facts must not
    carry null values (regression for the reported retain-with-null case)."""
    contents = _build_contents(
        [{"content": "hi", "metadata": {"ocr_engine": None, "n": 5, "source": "slack"}}],
        None,
    )
    assert contents[0].metadata == {"n": 5, "source": "slack"}


@pytest.mark.asyncio
async def test_delta_sql_metadata_replacement_preserves_engine_precision_and_strips_spoof(monkeypatch):
    store = MagicMock()
    store.writes_memory_rows_in_sql_for.return_value = True
    monkeypatch.setattr(memories_module, "get_memories", lambda: store)

    conn = MagicMock()
    conn.fetch = AsyncMock(
        return_value=[
            {
                "id": "year-fact",
                "metadata": json.dumps({"source": "old", OCCURRENCE_PRECISION_METADATA_KEY: "year"}),
            },
            {
                "id": "exact-fact",
                "metadata": {"source": "old", OCCURRENCE_PRECISION_METADATA_KEY: "day"},
            },
            {
                "id": "month-fact",
                "metadata": {"source": "old", OCCURRENCE_PRECISION_METADATA_KEY: "month"},
            },
            {
                "id": "instant-fact",
                "metadata": {"source": "old", OCCURRENCE_PRECISION_METADATA_KEY: "instant"},
            },
            {
                "id": "range-fact",
                "metadata": {"source": "old", OCCURRENCE_PRECISION_METADATA_KEY: "range"},
            },
            {
                "id": "invalid-fact",
                "metadata": {"source": "old", OCCURRENCE_PRECISION_METADATA_KEY: "spoofed"},
            },
            {"id": "undated-fact", "metadata": {"source": "old"}},
        ]
    )
    conn.parse_json.side_effect = lambda value: json.loads(value) if isinstance(value, str) else value
    conn.executemany = AsyncMock()
    caller_metadata = {"source": "new", "drop_me": None, OCCURRENCE_PRECISION_METADATA_KEY: "instant"}

    updated = await fact_storage.update_memory_units_metadata_and_tags(
        conn,
        "bank",
        "document",
        ["tag"],
        caller_metadata,
    )

    assert updated == 7
    assert caller_metadata == {
        "source": "new",
        "drop_me": None,
        OCCURRENCE_PRECISION_METADATA_KEY: "instant",
    }
    assert "WHERE bank_id = $1 AND document_id = $2" in conn.fetch.await_args.args[0]
    assert "WHERE bank_id = $3 AND id = $4" in conn.executemany.await_args.args[0]
    updates = conn.executemany.await_args.args[1]
    metadata_by_id = {unit_id: json.loads(stored_metadata) for _tags, stored_metadata, _bank, unit_id in updates}
    assert metadata_by_id == {
        "year-fact": {"source": "new", OCCURRENCE_PRECISION_METADATA_KEY: "year"},
        "exact-fact": {"source": "new", OCCURRENCE_PRECISION_METADATA_KEY: "day"},
        "month-fact": {"source": "new", OCCURRENCE_PRECISION_METADATA_KEY: "month"},
        "instant-fact": {"source": "new", OCCURRENCE_PRECISION_METADATA_KEY: "instant"},
        "range-fact": {"source": "new", OCCURRENCE_PRECISION_METADATA_KEY: "range"},
        "invalid-fact": {"source": "new"},
        "undated-fact": {"source": "new"},
    }
    assert {bank for _tags, _metadata, bank, _unit_id in updates} == {"bank"}
    assert all(tags == ["tag"] for tags, _metadata, _bank, _unit_id in updates)


@pytest.mark.asyncio
async def test_delta_external_store_metadata_replacement_preserves_engine_precision_and_strips_spoof(monkeypatch):
    store = MagicMock()
    store.writes_memory_rows_in_sql_for.return_value = False
    store.scan_memories = AsyncMock(
        return_value=ScanPage(
            memories=[
                StoredMemory(
                    unit_id="year-fact",
                    text="event",
                    fact_type="world",
                    metadata={"source": "old", OCCURRENCE_PRECISION_METADATA_KEY: "year"},
                ),
                StoredMemory(
                    unit_id="month-fact",
                    text="event",
                    fact_type="world",
                    metadata={"source": "old", OCCURRENCE_PRECISION_METADATA_KEY: "month"},
                ),
                StoredMemory(
                    unit_id="day-fact",
                    text="event",
                    fact_type="world",
                    metadata={"source": "old", OCCURRENCE_PRECISION_METADATA_KEY: "day"},
                ),
                StoredMemory(
                    unit_id="instant-fact",
                    text="event",
                    fact_type="world",
                    metadata={"source": "old", OCCURRENCE_PRECISION_METADATA_KEY: "instant"},
                ),
                StoredMemory(
                    unit_id="range-fact",
                    text="event",
                    fact_type="world",
                    metadata={"source": "old", OCCURRENCE_PRECISION_METADATA_KEY: "range"},
                ),
                StoredMemory(unit_id="undated-fact", text="fact", fact_type="world", metadata={"source": "old"}),
            ]
        )
    )
    store.update_memories = AsyncMock()
    monkeypatch.setattr(memories_module, "get_memories", lambda: store)

    caller_metadata = {"source": "new", OCCURRENCE_PRECISION_METADATA_KEY: "range"}
    updated = await fact_storage.update_memory_units_metadata_and_tags(
        MagicMock(),
        "bank",
        "document",
        ["tag"],
        caller_metadata,
    )

    assert updated == 6
    assert caller_metadata == {"source": "new", OCCURRENCE_PRECISION_METADATA_KEY: "range"}
    bank_id, patches = store.update_memories.await_args.args
    assert bank_id == "bank"
    assert store.scan_memories.await_args.kwargs["bank_id"] == "bank"
    assert store.scan_memories.await_args.kwargs["document_id"] == "document"
    assert all(patch.tags == ["tag"] for patch in patches)
    metadata_by_id = {
        patch.unit_id: json.loads(patch.metadata[META_METADATA_JSON]) for patch in patches if patch.metadata is not None
    }
    assert metadata_by_id == {
        "year-fact": {"source": "new", OCCURRENCE_PRECISION_METADATA_KEY: "year"},
        "month-fact": {"source": "new", OCCURRENCE_PRECISION_METADATA_KEY: "month"},
        "day-fact": {"source": "new", OCCURRENCE_PRECISION_METADATA_KEY: "day"},
        "instant-fact": {"source": "new", OCCURRENCE_PRECISION_METADATA_KEY: "instant"},
        "range-fact": {"source": "new", OCCURRENCE_PRECISION_METADATA_KEY: "range"},
        "undated-fact": {"source": "new"},
    }
