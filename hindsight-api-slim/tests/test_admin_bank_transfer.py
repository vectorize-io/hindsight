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
    call_record: dict = {}

    async def fake_stream_export_bank(conn, bank_id, **kwargs):
        call_record["args"] = (conn, bank_id)
        call_record["kwargs"] = kwargs
        yield b"arch"
        yield b"ive"

    async def fake_admin_connect(db_url: str) -> _FakeConnection:
        assert db_url == "postgresql://example"
        return connection

    monkeypatch.setattr(cli, "_admin_connect", fake_admin_connect)
    monkeypatch.setattr(cli, "stream_export_bank", fake_stream_export_bank)
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

    assert call_record["args"] == (connection, "source-bank")
    assert call_record["kwargs"]["bank_rows_json_encoding"] == "decoded"
    assert call_record["kwargs"]["memories"] is store
    # --include-history maps onto the scope's third flag; the other two are what a
    # migration always carries.
    assert call_record["kwargs"]["scope"] == TransferScope(data=True, bank_config=True, history=True)
    # Attachment bytes live in file storage rather than in a column, so the CLI has
    # to hand the export a storage client or a bank with attachments exports rows
    # pointing at blobs the archive does not carry.
    assert call_record["kwargs"]["file_storage"] is not None
    assert output.read_bytes() == b"archive"
    assert size == len(b"archive")
    assert connection.closed is True


@pytest.mark.asyncio
async def test_export_of_missing_bank_raises_before_any_chunk(memory) -> None:
    """A typo'd bank id must not export as a well-formed empty archive (#6776)."""
    from hindsight_api.engine.transfer import BankNotFoundError, stream_export_bank

    backend = await memory._get_backend()
    chunks: list[bytes] = []
    with pytest.raises(BankNotFoundError):
        async for chunk in stream_export_bank(backend, "zzz-no-such-bank"):
            chunks.append(chunk)
    assert chunks == []


def test_export_bank_command_exits_nonzero_for_missing_bank(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from hindsight_api.engine.transfer import BankNotFoundError

    async def fake_run_export_bank(*args, **kwargs):
        raise BankNotFoundError("Bank 'zzz' not found")

    monkeypatch.setenv("HINDSIGHT_API_DATABASE_URL", "postgresql://user:secret@db.example:5432/hs")
    monkeypatch.setattr(cli, "_run_export_bank", fake_run_export_bank)
    output = tmp_path / "out.zip"
    result = CliRunner().invoke(cli.app, ["export-bank", "--bank", "zzz", "--output", str(output)])

    assert result.exit_code == 1
    assert "not found" in result.output
    assert "db.example" in result.output and "secret" not in result.output
    assert not output.exists()
