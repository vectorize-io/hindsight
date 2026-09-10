"""A store-owned bank's attachment lookup must not touch Postgres.

Recall resolves the attachments behind each fact through ``attachments_for_memories``, which
reads ``memory_units.attachment_ids``. For a bank whose memories store owns its rows, that table
holds none of them, so the read can only come back empty -- and it is not a cheap empty read. The
table carries partial vector indexes per bank, and the planner opens and locks every index on a
table to plan any statement against it: in a tenant with a few thousand banks, ~15k locks and
hundreds of milliseconds of planning, on every recall.

So the property asserted is that the lookup returns before it asks for the bank profile or a
connection, not merely that it returns ``{}``: an empty result is what the expensive read
produced too.
"""

from types import SimpleNamespace

import pytest

import hindsight_api.engine.memories as memories_module
from hindsight_api.engine.memory_engine import MemoryEngine


class _NoPostgres:
    """Stands in for the engine: any attempt to reach Postgres fails the test."""

    async def get_bank_profile(self, *a, **k):
        raise AssertionError("a store-owned bank's attachment lookup read the bank profile")

    async def _get_backend(self):
        raise AssertionError("a store-owned bank's attachment lookup took a connection")


def _memories(store_owned: bool):
    return SimpleNamespace(store_owned_for=lambda bank_id: store_owned)


@pytest.mark.asyncio
async def test_a_store_owned_bank_resolves_no_attachments_without_touching_postgres(monkeypatch):
    monkeypatch.setattr(memories_module, "get_memories", lambda: _memories(store_owned=True))

    result = await MemoryEngine.attachments_for_memories(
        _NoPostgres(), "bank-1", ["00000000-0000-0000-0000-000000000001"], request_context=None
    )

    assert result == {}


@pytest.mark.asyncio
async def test_a_bank_whose_rows_live_in_sql_still_reads_them(monkeypatch):
    """The guard must not swallow the Postgres-backed case: there the read is the feature."""
    monkeypatch.setattr(memories_module, "get_memories", lambda: _memories(store_owned=False))

    with pytest.raises(AssertionError, match="read the bank profile"):
        await MemoryEngine.attachments_for_memories(
            _NoPostgres(), "bank-1", ["00000000-0000-0000-0000-000000000001"], request_context=None
        )
