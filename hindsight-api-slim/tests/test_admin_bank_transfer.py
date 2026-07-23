"""Tests for the admin CLI whole-bank transfer boundary."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from hindsight_api.admin import cli


class _FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_run_export_bank_declares_decoded_json_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The codec-enabled admin producer must identify its rows as decoded."""
    connection = _FakeConnection()
    export_bank = AsyncMock(return_value=b"archive")

    async def fake_admin_connect(db_url: str) -> _FakeConnection:
        assert db_url == "postgresql://example"
        return connection

    monkeypatch.setattr(cli, "_admin_connect", fake_admin_connect)
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
    )
    assert output.read_bytes() == b"archive"
    assert size == len(b"archive")
    assert connection.closed is True
