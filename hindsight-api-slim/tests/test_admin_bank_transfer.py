"""Tests for the admin CLI whole-bank transfer boundary."""

from collections.abc import Awaitable, Callable
from pathlib import Path
from types import TracebackType
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hindsight_api.admin import cli
from hindsight_api.engine.transfer.archive import TransferArchive


class _FakeConnection:
    def __init__(self) -> None:
        self.codecs: list[str] = []

    async def set_type_codec(self, type_name: str, **kwargs: Any) -> None:
        self.codecs.append(type_name)


class _FakeAcquire:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> _FakeConnection:
        return self.connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class _FakePool:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection
        self.closed = False

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self.connection)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_run_export_bank_declares_decoded_json_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The codec-enabled admin producer must identify its rows as decoded."""
    connection = _FakeConnection()
    pool = _FakePool(connection)
    source = tmp_path / "source.zip"
    source.write_bytes(b"archive")
    archive = TransferArchive(path=str(source), size_bytes=source.stat().st_size)
    export_bank = AsyncMock(return_value=archive)
    file_storage = object()

    async def fake_create_pool(
        db_url: str,
        *,
        min_size: int,
        max_size: int,
        init: Callable[[_FakeConnection], Awaitable[None]],
    ) -> _FakePool:
        assert db_url == "postgresql://example"
        assert (min_size, max_size) == (1, 2)
        await init(connection)
        return pool

    monkeypatch.setattr(cli.asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(cli, "create_file_storage", lambda **kwargs: file_storage)
    monkeypatch.setattr(cli, "export_bank", export_bank)

    output = tmp_path / "bank.zip"
    size = await cli._run_export_bank(
        "postgresql://example",
        "source-bank",
        output,
        "tenant_schema",
        include_history=True,
    )

    export_bank.assert_awaited_once_with(
        connection,
        "source-bank",
        include_history=True,
        bank_rows_json_encoding="decoded",
        file_storage=file_storage,
    )
    assert output.read_bytes() == b"archive"
    assert size == len(b"archive")
    assert connection.codecs == ["json", "jsonb"]
    assert pool.closed is True
    assert not source.exists()
