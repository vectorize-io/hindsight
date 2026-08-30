"""Storage contracts for precision-aware curation writes."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest

from hindsight_api.engine.memories.pg import reads, writes
from hindsight_api.engine.temporal_precision import OCCURRENCE_PRECISION_METADATA_KEY


def _fq_table(name: str) -> str:
    return name


@pytest.mark.asyncio
async def test_postgres_edit_replaces_metadata_and_text_signals_in_the_field_update():
    conn = AsyncMock()
    conn.execute_rows_affected.return_value = 1
    config = SimpleNamespace(
        text_search_extension="native",
        text_search_extension_native_language="english",
    )
    occurred = datetime(2026, 8, 30, tzinfo=UTC)
    expected_updated_at = datetime(2026, 8, 29, tzinfo=UTC)
    metadata = {"source": "fixture", OCCURRENCE_PRECISION_METADATA_KEY: "day"}

    with patch.object(writes, "get_config", return_value=config):
        applied = await writes.apply_edit(
            conn=conn,
            fq_table=_fq_table,
            bank_id="bank",
            unit_id="00000000-0000-0000-0000-000000000001",
            text="The exact summit date",
            context="curated",
            fact_type="world",
            occurred_start=occurred,
            occurred_end=occurred,
            event_date=occurred,
            mentioned_at=None,
            entity_ids=None,
            metadata=metadata,
            text_signals="August 30 2026",
            expected_updated_at=expected_updated_at,
        )

    assert applied is True
    field_update = conn.execute_rows_affected.await_args.args
    sql = field_update[0]
    assert "metadata = $9::jsonb, text_signals = $10" in sql
    assert "COALESCE($10, '')" in sql
    assert "AND updated_at = $11" in sql
    assert json.loads(field_update[9]) == metadata
    assert field_update[10] == "August 30 2026"
    assert field_update[11] == expected_updated_at


@pytest.mark.asyncio
async def test_postgres_edit_conflict_does_not_delete_derived_links():
    conn = AsyncMock()
    conn.execute_rows_affected.return_value = 0
    config = SimpleNamespace(text_search_extension="none")
    occurred = datetime(2026, 8, 30, tzinfo=UTC)

    with patch.object(writes, "get_config", return_value=config):
        applied = await writes.apply_edit(
            conn=conn,
            fq_table=_fq_table,
            bank_id="bank",
            unit_id="00000000-0000-0000-0000-000000000001",
            text="stale edit",
            context=None,
            fact_type="world",
            occurred_start=occurred,
            occurred_end=occurred,
            event_date=occurred,
            mentioned_at=None,
            entity_ids=None,
            metadata={},
            text_signals="August 30 2026",
            expected_updated_at=datetime(2026, 8, 29, tzinfo=UTC),
        )

    assert applied is False
    conn.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_addressed_read_carries_updated_at_as_the_curation_version():
    conn = AsyncMock()
    updated_at = datetime(2026, 8, 30, 12, tzinfo=UTC)
    unit_id = UUID("00000000-0000-0000-0000-000000000001")
    conn.fetch.return_value = [
        {
            "id": unit_id,
            "text": "versioned memory",
            "fact_type": "world",
            "updated_at": updated_at,
        }
    ]

    memories = await reads.get_memories(
        conn=conn,
        fq_table=_fq_table,
        bank_id="bank",
        unit_ids=[str(unit_id)],
    )

    assert memories[0].updated_at == updated_at
    assert "updated_at" in conn.fetch.await_args.args[0]


def test_archived_memory_decodes_json_metadata_for_precision_aware_reembedding():
    occurred = datetime(2026, 1, 1, tzinfo=UTC)
    row = {
        "id": "00000000-0000-0000-0000-000000000001",
        "text": "Summit | When: 2026",
        "fact_type": "world",
        "context": None,
        "document_id": None,
        "chunk_id": None,
        "tags": [],
        "metadata": json.dumps({OCCURRENCE_PRECISION_METADATA_KEY: "year"}),
        "proof_count": 1,
        "event_date": occurred,
        "occurred_start": occurred,
        "occurred_end": occurred,
        "mentioned_at": occurred,
        "created_at": occurred,
        "consolidated_at": None,
        "entity_ids": [],
    }

    restored = writes._archived_stored(row)

    assert restored.metadata == {OCCURRENCE_PRECISION_METADATA_KEY: "year"}
