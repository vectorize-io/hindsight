"""Unit tests for the high-level export_documents convenience wrapper.

These mock the generated sub-clients so no server is needed — they verify the
submit -> poll -> download orchestration (the value the wrapper adds on top of
the raw async export operation), not the HTTP behaviour (covered in the API's
test_document_transfer.py).
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
    """Stub the low-level download path to return ARCHIVE_BYTES."""
    client._api_client.param_serialize = MagicMock(return_value=("GET", DOWNLOAD_URL, {}, None, []))
    response = MagicMock()
    response.read = AsyncMock(return_value=ARCHIVE_BYTES)
    client._api_client.call_api = AsyncMock(return_value=response)


async def test_export_documents_submits_polls_and_downloads():
    client = _make_client()
    client._document_transfer_api.export_documents = AsyncMock(
        return_value=MagicMock(operation_id=OPERATION_ID)
    )
    # First poll still processing, second poll completed with the archive location.
    client._operations_api.get_operation_status = AsyncMock(
        side_effect=[
            _status("processing"),
            _status("completed", result_metadata={"download_url": DOWNLOAD_URL, "storage_key": STORAGE_KEY}),
        ]
    )
    _mock_download(client)

    result = await client.aexport_documents("my-bank", poll_interval=0)

    assert result == ARCHIVE_BYTES
    # Submitted with the whole-bank defaults.
    submit_kwargs = client._document_transfer_api.export_documents.call_args.kwargs
    assert submit_kwargs["document_id"] is None
    assert submit_kwargs["include_observations"] is False
    # Polled until completed (twice), then downloaded the server-provided URL.
    assert client._operations_api.get_operation_status.await_count == 2
    assert client._api_client.param_serialize.call_args.kwargs["resource_path"] == DOWNLOAD_URL


async def test_export_documents_forwards_subset_and_observations():
    client = _make_client()
    client._document_transfer_api.export_documents = AsyncMock(
        return_value=MagicMock(operation_id=OPERATION_ID)
    )
    client._operations_api.get_operation_status = AsyncMock(
        return_value=_status("completed", result_metadata={"download_url": DOWNLOAD_URL})
    )
    _mock_download(client)

    await client.aexport_documents("my-bank", document_ids=["doc-1", "doc-2"], poll_interval=0)

    kwargs = client._document_transfer_api.export_documents.call_args.kwargs
    assert kwargs["document_id"] == ["doc-1", "doc-2"]


async def test_export_documents_raises_on_failed_operation():
    client = _make_client()
    client._document_transfer_api.export_documents = AsyncMock(
        return_value=MagicMock(operation_id=OPERATION_ID)
    )
    client._operations_api.get_operation_status = AsyncMock(
        return_value=_status("failed", error_message="boom")
    )

    with pytest.raises(RuntimeError, match="boom"):
        await client.aexport_documents("my-bank", poll_interval=0)


async def test_export_documents_times_out():
    client = _make_client()
    client._document_transfer_api.export_documents = AsyncMock(
        return_value=MagicMock(operation_id=OPERATION_ID)
    )
    client._operations_api.get_operation_status = AsyncMock(return_value=_status("processing"))

    with pytest.raises(TimeoutError):
        await client.aexport_documents("my-bank", poll_interval=0, timeout=0)


async def test_export_documents_raises_without_download_url():
    client = _make_client()
    client._document_transfer_api.export_documents = AsyncMock(
        return_value=MagicMock(operation_id=OPERATION_ID)
    )
    client._operations_api.get_operation_status = AsyncMock(
        return_value=_status("completed", result_metadata={})
    )

    with pytest.raises(RuntimeError, match="download_url"):
        await client.aexport_documents("my-bank", poll_interval=0)
