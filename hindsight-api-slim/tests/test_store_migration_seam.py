"""A store can be asked to migrate its own derived state, per tenant.

The defaults matter more than they look: this hook is called from a migration,
for EVERY tenant, including the overwhelming majority whose store has no derived
state at all. A default that did anything but nothing would make every such
tenant pay for a feature it does not use.

Runs via: uv run pytest tests/test_store_migration_seam.py -v
"""

from __future__ import annotations

import pytest

from hindsight_api.engine.memories import MemoriesExtension
from hindsight_api.engine.memories.postgres import PostgresMemories


@pytest.mark.asyncio
async def test_a_store_with_no_derived_state_migrates_nothing():
    store = PostgresMemories({})
    assert await store.run_store_migration(migration_id="anything") == (0, 0)
    # (0, 0) from status reads as "finished", which is the right answer for a
    # store that never started anything — a caller polling this must not hang.
    assert await store.store_migration_status(migration_id="anything") == (0, 0)


@pytest.mark.asyncio
async def test_an_overriding_store_is_the_one_that_answers():
    """The property that makes the hook reachable at all.

    A deployment may put a router in front of several stores, and such a router
    can only forward methods the INTERFACE declares — it has no way to know about
    one that exists solely on a concrete store. So a migration hook that lived
    only on the implementation would be silently dropped for exactly the
    deployments that have a store to migrate: the call would return the default,
    report success, and migrate nothing.
    """

    class _Owning(PostgresMemories):
        async def run_store_migration(self, *, migration_id: str) -> tuple[int, int]:
            return (7, 9)

    assert await _Owning({}).run_store_migration(migration_id="x") == (7, 9)
    assert "run_store_migration" in vars(MemoriesExtension), (
        "the hook must be declared on the interface, not only on a concrete store, or a routing store cannot forward it"
    )
