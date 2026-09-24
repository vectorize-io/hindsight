"""Trigger the memories store's bank-catalog migration for this tenant

Revision ID: d9e8f7a6b5c4
Revises: b8d3f1a6c2e4
Create Date: 2026-09-24

A store that keeps derived state of its own needs its EXISTING data brought up to
what a new build expects. Nothing else reaches it: a bank nobody writes to is
never revisited by the store's own maintenance, so it can never acquire state a
new build introduced. That is how the bank catalog shipped — every pre-existing
bank invisible to the ordering it feeds.

This revision does not do the work. It asks the store to, and returns: the
trigger is an enqueue, so it is O(1) whatever the tenant's size, and the store
drains it afterwards on whichever component owns writing its index. That is what
makes this safe in the pre-upgrade hook, where a migration that took its time
would turn "slow" into "failed release".

Re-running is free. Every store migration is required to be idempotent and to
detect per namespace whether it has already been applied, so there is no progress
state here to keep or repair — which is also why this revision records nothing
beyond alembic's own version row.

A store with no derived state of its own no-ops (the interface defaults do
nothing), so this is inert for a Postgres-owned deployment.
"""

import asyncio
from collections.abc import Sequence

from alembic import context

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "d9e8f7a6b5c4"
down_revision: str | Sequence[str] | None = "b8d3f1a6c2e4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: The store's id for this migration. Named, not numbered: asking a store that
#: does not have it fails loudly here, rather than a version comparison quietly
#: deciding there was nothing to do.
MIGRATION_ID = "bank-catalog"


def _pg_upgrade() -> None:
    """Ask this tenant's store to apply `MIGRATION_ID`."""
    from hindsight_api.engine.memories import get_memories
    from hindsight_api.engine.memory_engine import _current_schema

    schema = context.config.get_main_option("target_schema")
    if not schema:
        # The base schema holds no tenant's banks; there is nothing to migrate.
        return

    async def go() -> tuple[int, int]:
        # The store resolves the tenant from the schema in context, the same way
        # every other store call in this process does.
        _current_schema.set(schema)
        return await get_memories().run_store_migration(migration_id=MIGRATION_ID)

    enqueued, namespaces = asyncio.run(go())
    if namespaces:
        print(f"store migration {MIGRATION_ID!r}: {enqueued} of {namespaces} namespace(s) enqueued for {schema}")


def _pg_downgrade() -> None:
    """Nothing to undo.

    The migration only ADDS derived state the store rebuilds for itself anyway;
    removing it would make the store worse at no one's request, and a downgrade
    that destroys derived data is how a rollback becomes an outage.
    """


#: Both dialect slots run the SAME function, and that is the honest answer rather
#: than an exemption from the convention: this revision issues no SQL. It asks the
#: store to migrate its own data, which is identical whatever dialect the control
#: plane happens to run on.
_oracle_upgrade = _pg_upgrade
_oracle_downgrade = _pg_downgrade


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
