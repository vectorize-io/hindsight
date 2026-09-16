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
    # The CLI resolves the configured memories store and hands it to the export: a bank whose
    # memories live outside SQL cannot be read from this connection, and without the store the
    # archive would come out well-formed and empty.
    store = object()
    monkeypatch.setattr(cli, "get_memories", lambda: store)

    output = tmp_path / "bank.zip"
    size = await cli._run_export_bank(
        "postgresql://example",
        "source-bank",
        output,
        "tenant_schema",
        include_history=True,
    )

    from hindsight_api.engine.transfer import TransferScope

    call = export_bank.await_args
    assert call.args == (connection, "source-bank")
    assert call.kwargs["bank_rows_json_encoding"] == "decoded"
    assert call.kwargs["memories"] is store
    # --include-history maps onto the scope's third flag; the other two are what a
    # migration always carries.
    assert call.kwargs["scope"] == TransferScope(data=True, bank_config=True, history=True)
    # Attachment bytes live in file storage rather than in a column, so the CLI has
    # to hand the export a storage client or a bank with attachments exports rows
    # pointing at blobs the archive does not carry.
    assert call.kwargs["file_storage"] is not None
    assert output.read_bytes() == b"archive"
    assert size == len(b"archive")
    assert connection.closed is True
