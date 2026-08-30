"""Storage contracts for precision-aware curation writes."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.memories.pg import writes
from hindsight_api.engine.temporal_precision import OCCURRENCE_PRECISION_METADATA_KEY


def _fq_table(name: str) -> str:
    return name


@pytest.mark.asyncio
async def test_postgres_edit_replaces_metadata_and_text_signals_in_the_field_update():
    conn = AsyncMock()
    config = SimpleNamespace(
        text_search_extension="native",
        text_search_extension_native_language="english",
    )
    occurred = datetime(2026, 8, 30, tzinfo=UTC)
    metadata = {"source": "fixture", OCCURRENCE_PRECISION_METADATA_KEY: "day"}

    with patch.object(writes, "get_config", return_value=config):
        await writes.apply_edit(
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
        )

    field_update = conn.execute.await_args_list[0].args
    sql = field_update[0]
    assert "metadata = $9::jsonb, text_signals = $10" in sql
    assert "COALESCE($10, '')" in sql
    assert json.loads(field_update[9]) == metadata
    assert field_update[10] == "August 30 2026"


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
