"""Regression test: deleting a bank that was never created is a no-op, not a 500.

Bug: ``delete_bank`` locked the bank row with ``FOR NO KEY UPDATE`` and threw the result
away, so it carried on as if the bank existed. For a store that owns its storage, a bank
nobody created has no namespace, and counting in it is a fault rather than "zero" — the
store will not report missing storage as an empty bank, because empty and absent are not
the same answer. The count raised, the transaction's ``except`` wrapped it, and
``DELETE /v1/default/banks/{bank_id}`` 500'd with "namespace ... has no manifest".
Only the store-owned backend showed it: on SQL the same counts are ``SELECT COUNT(*)``s
that happily answer 0.

The fake below is faithful to that: its storage methods raise unless
``ensure_bank_storage`` created the namespace first. So a regression here fails loudly
instead of quietly counting zero, and the existing-bank case is the positive control that
keeps the gate from being "never ask the store".

Runs via: uv run pytest tests/test_delete_missing_bank_store_owned.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hindsight_api.engine.memories as memories_mod
from hindsight_api.models import RequestContext


class _NoManifest(Exception):
    """What a store that owns its storage raises for a namespace nobody created."""


class _StoreOwnedStore:
    """A store that keeps memories outside SQL and has no namespace until asked for one."""

    def __init__(self):
        self.namespaces: set[str] = set()
        self.touched: list[str] = []

    def _require(self, bank_id: str, what: str) -> None:
        self.touched.append(f"{what}:{bank_id}")
        if bank_id not in self.namespaces:
            raise _NoManifest(f"namespace test/{bank_id} has no manifest")

    def store_owned_for(self, bank_id: str) -> bool:
        return True

    async def ensure_bank_storage(self, bank_id: str) -> None:
        self.namespaces.add(bank_id)

    async def drop_bank_storage(self, bank_id: str) -> None:
        self._require(bank_id, "drop_bank_storage")
        self.namespaces.discard(bank_id)

    async def delete_where(self, bank_id: str, predicate) -> int:
        self._require(bank_id, "delete_where")
        return 0

    async def count_memories(self, *, conn, fq_table, bank_id: str) -> dict:
        self._require(bank_id, "count_memories")
        return {}

    async def count_documents(self, *, bank_id: str) -> int:
        self._require(bank_id, "count_documents")
        return 0

    async def list_entities(self, *, conn, fq_table, bank_id: str, search=None, limit=100, offset=0) -> dict:
        self._require(bank_id, "list_entities")
        return {"items": [], "total": 0, "limit": limit, "offset": offset}

    async def scan_memories(self, *, conn, fq_table, bank_id: str, fact_types=None, limit=100):
        """Reached by a fact_type-scoped delete of a bank the store owns. Nothing stored here."""
        self._require(bank_id, "scan_memories")
        return SimpleNamespace(memories=[])


@pytest.fixture
def store_owned(monkeypatch):
    store = _StoreOwnedStore()
    monkeypatch.setattr(memories_mod, "get_memories", lambda: store)
    return store


@pytest.fixture
def request_context():
    return RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # DELETE /v1/default/banks/{id}
        ({}, {"memory_units_deleted": 0, "entities_deleted": 0, "documents_deleted": 0}),
        # DELETE /v1/default/banks/{id}/memories
        ({"delete_bank_profile": False}, {"memory_units_deleted": 0, "entities_deleted": 0, "documents_deleted": 0}),
        # DELETE /v1/default/banks/{id}/memories?type=world
        ({"fact_type": "world"}, {"memory_units_deleted": 0, "entities_deleted": 0}),
    ],
    ids=["delete_bank", "clear_memories", "clear_memories_by_type"],
)
async def test_delete_missing_bank_never_asks_the_store(memory, store_owned, request_context, kwargs, expected):
    """A bank nobody created reports zero, and the store is never asked about storage it lacks."""
    bank_id = "delete_missing_bank_never_created"

    result = await memory.delete_bank(bank_id, request_context=request_context, **kwargs)

    for key, value in expected.items():
        assert result[key] == value, f"{key}: {result}"
    assert store_owned.touched == [], f"store consulted for a bank it has no namespace for: {store_owned.touched}"


@pytest.mark.asyncio
async def test_delete_existing_bank_still_goes_through_the_store(memory, store_owned, request_context):
    """The positive control: the gate is "no bank row", not "never ask the store"."""
    bank_id = "delete_missing_bank_control"

    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    assert bank_id in store_owned.namespaces

    result = await memory.delete_bank(bank_id, request_context=request_context)

    assert result["bank_deleted"] is True
    # Counted through the store, and the namespace dropped with the row.
    assert f"count_memories:{bank_id}" in store_owned.touched
    assert f"drop_bank_storage:{bank_id}" in store_owned.touched
    assert bank_id not in store_owned.namespaces
