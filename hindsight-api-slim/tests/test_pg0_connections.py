"""Embedded PostgreSQL must leave connection capacity outside the application pool."""

import uuid
from contextlib import AsyncExitStack
from unittest.mock import patch

import asyncpg
import pytest

from hindsight_api.config import DEFAULT_DB_POOL_MAX_SIZE, HindsightConfig, get_config
from hindsight_api.pg0 import EmbeddedPostgres, resolve_database_url


@pytest.mark.parametrize("configured_limit", [None, "450"])
def test_embedded_postgres_connection_limit(monkeypatch: pytest.MonkeyPatch, configured_limit: str | None) -> None:
    if configured_limit is None:
        monkeypatch.delenv("HINDSIGHT_API_PG0_MAX_CONNECTIONS", raising=False)
    else:
        monkeypatch.setenv("HINDSIGHT_API_PG0_MAX_CONNECTIONS", configured_limit)

    with patch("pg0.Pg0") as pg0:
        EmbeddedPostgres(name="connection-budget")._get_pg0()

    assert pg0.call_args.kwargs["config"]["max_connections"] == (configured_limit or "300")


def test_explicit_pg0_config_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HINDSIGHT_API_PG0_MAX_CONNECTIONS", "450")
    explicit_config = {"max_connections": "500", "shared_buffers": "64MB"}
    with patch("pg0.Pg0") as pg0:
        EmbeddedPostgres(config=explicit_config)._get_pg0()

    assert pg0.call_args.kwargs["config"] == explicit_config
    assert explicit_config == {"max_connections": "500", "shared_buffers": "64MB"}


def test_other_pg0_settings_preserve_connection_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HINDSIGHT_API_PG0_MAX_CONNECTIONS", "450")
    explicit_config = {"shared_buffers": "64MB"}
    with patch("pg0.Pg0") as pg0:
        EmbeddedPostgres(config=explicit_config)._get_pg0()

    assert pg0.call_args.kwargs["config"] == {"max_connections": "450", "shared_buffers": "64MB"}
    assert explicit_config == {"shared_buffers": "64MB"}


def test_pg0_connection_limit_is_static(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HINDSIGHT_API_PG0_MAX_CONNECTIONS", "450")
    assert get_config().pg0_max_connections == 450
    assert "pg0_max_connections" in HindsightConfig.get_static_fields()


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_pg0_connection_limit_must_be_positive(monkeypatch: pytest.MonkeyPatch, limit: str) -> None:
    monkeypatch.setenv("HINDSIGHT_API_PG0_MAX_CONNECTIONS", limit)
    with pytest.raises(ValueError, match="HINDSIGHT_API_PG0_MAX_CONNECTIONS must be >= 1"):
        HindsightConfig.from_env()


@pytest.mark.asyncio
async def test_external_database_does_not_start_pg0() -> None:
    url = "postgresql://localhost/hindsight"
    with patch("pg0.Pg0") as pg0:
        assert await resolve_database_url(url) == url
    pg0.assert_not_called()


@pytest.mark.asyncio
async def test_full_default_pool_leaves_room_for_direct_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    """A saturated application pool must not block index maintenance or diagnostics."""
    monkeypatch.delenv("HINDSIGHT_API_PG0_MAX_CONNECTIONS", raising=False)
    postgres = EmbeddedPostgres(name=f"hindsight-connection-budget-{uuid.uuid4().hex[:12]}")
    try:
        url = await postgres.start(max_retries=1)
        async with asyncpg.create_pool(url, min_size=1, max_size=DEFAULT_DB_POOL_MAX_SIZE) as pool:
            async with AsyncExitStack() as connections:
                for _ in range(DEFAULT_DB_POOL_MAX_SIZE):
                    await connections.enter_async_context(pool.acquire())
                direct = await asyncpg.connect(url)
                try:
                    assert await direct.fetchval("SELECT 1") == 1
                finally:
                    await direct.close()
    finally:
        await postgres.stop()
