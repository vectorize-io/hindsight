"""Startup migrates the base schema even when the tenant extension does not list it (#5400).

The cross-tenant maintenance routines live in the base schema and only a migration
run targeting it creates them. The step must never block startup.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hindsight_api.config import get_config
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.extensions import Tenant


def _base() -> str:
    return get_config().database_schema or "public"


def _engine(run: AsyncMock) -> SimpleNamespace:
    return SimpleNamespace(_run_pg_migrations=run)


@pytest.mark.asyncio
async def test_base_schema_migrated_when_extension_lists_only_tenants():
    run = AsyncMock()
    await MemoryEngine._migrate_base_schema(_engine(run), "postgresql://x", [Tenant(schema="user_a")])  # type: ignore[arg-type]
    run.assert_awaited_once_with("postgresql://x", [_base()])


@pytest.mark.asyncio
async def test_base_schema_migrated_when_extension_lists_nothing_yet():
    # Supabase's list_tenants() is empty at startup.
    run = AsyncMock()
    await MemoryEngine._migrate_base_schema(_engine(run), "postgresql://x", [])  # type: ignore[arg-type]
    run.assert_awaited_once_with("postgresql://x", [_base()])


@pytest.mark.asyncio
async def test_base_schema_not_migrated_twice_when_already_listed():
    run = AsyncMock()
    await MemoryEngine._migrate_base_schema(_engine(run), "postgresql://x", [Tenant(schema=_base())])  # type: ignore[arg-type]
    run.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_schema_failure_does_not_block_startup(caplog):
    run = AsyncMock(side_effect=RuntimeError("permission denied for schema public"))
    with caplog.at_level(logging.WARNING):
        await MemoryEngine._migrate_base_schema(_engine(run), "postgresql://x", [Tenant(schema="user_a")])  # type: ignore[arg-type]
    assert "permission denied for schema public" in caplog.text
