"""Unit tests for the high-level bank export/import convenience wrappers.

These mock the generated sub-clients so no server is needed — they verify the
orchestration the wrapper adds (submit -> poll -> download for an export, and
forwarding the scope flags for both halves), not the HTTP behaviour, which is
covered by the API's own test_document_transfer.py.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_client import Hindsight

OPERATION_ID = "123e4567-e89b-12d3-a456-426614174000"
STORAGE_KEY = f"banks/my-bank/exports/{OPERATION_ID}/transfer.zip"
DOWNLOAD_URL = f"/v1/default/files/download/{STORAGE_KEY}"
ARCHIVE_BYTES = b"PK\x03\x04 fake zip bytes"


def _make_client():
    return Hindsight(base_url="http://localhost:8888")


def _status(status, result_metadata=None, error_message=None):
    return MagicMock(status=status, result_metadata=result_metadata, error_message=error_message)


def _mock_download(client):
    client._api_client.param_serialize = MagicMock(return_value=("GET", DOWNLOAD_URL, {}, None, []))
    response = MagicMock()
    response.read = AsyncMock(return_value=ARCHIVE_BYTES)
    client._api_client.call_api = AsyncMock(return_value=response)


async def test_export_bank_submits_polls_and_downloads():
    client = _make_client()
    client._bank_transfer_api.export_bank_transfer = AsyncMock(return_value=MagicMock(operation_id=OPERATION_ID))
    client._operations_api.get_operation_status = AsyncMock(
        side_effect=[
            _status("processing"),
            _status("completed", result_metadata={"download_url": DOWNLOAD_URL, "storage_key": STORAGE_KEY}),
        ]
    )
    _mock_download(client)

    result = await client.aexport_bank("my-bank", poll_interval=0)

    assert result == ARCHIVE_BYTES
    kwargs = client._bank_transfer_api.export_bank_transfer.call_args.kwargs
    assert kwargs["include_data"] is True
    assert kwargs["include_bank_config"] is True
    assert kwargs["include_history"] is False
    assert client._operations_api.get_operation_status.await_count == 2


async def test_export_bank_forwards_a_narrowed_scope():
    """A caller asking for the memories only must not silently get the config too —
    webhooks ride with the config, and a copy that inherits them starts calling the
    source's consumer."""
    client = _make_client()
    client._bank_transfer_api.export_bank_transfer = AsyncMock(return_value=MagicMock(operation_id=OPERATION_ID))
    client._operations_api.get_operation_status = AsyncMock(
        return_value=_status("completed", result_metadata={"download_url": DOWNLOAD_URL})
    )
    _mock_download(client)

    await client.aexport_bank("my-bank", include_bank_config=False, poll_interval=0)

    kwargs = client._bank_transfer_api.export_bank_transfer.call_args.kwargs
    assert kwargs["include_bank_config"] is False
    assert kwargs["include_data"] is True


async def test_export_bank_raises_on_failed_operation():
    client = _make_client()
    client._bank_transfer_api.export_bank_transfer = AsyncMock(return_value=MagicMock(operation_id=OPERATION_ID))
    client._operations_api.get_operation_status = AsyncMock(
        return_value=_status("failed", error_message="disk full")
    )

    with pytest.raises(RuntimeError, match="disk full"):
        await client.aexport_bank("my-bank", poll_interval=0)


async def test_import_bank_returns_the_operation_id_and_forwards_the_target():
    client = _make_client()
    client._bank_transfer_api.import_bank_transfer = AsyncMock(return_value=MagicMock(operation_id=OPERATION_ID))

    operation_id = await client.aimport_bank("my-bank", ARCHIVE_BYTES, target_bank_id="my-bank-copy")

    assert operation_id == OPERATION_ID
    args, kwargs = client._bank_transfer_api.import_bank_transfer.call_args
    assert args[0] == "my-bank"
    assert args[1] == ARCHIVE_BYTES
    assert kwargs["target_bank_id"] == "my-bank-copy"
