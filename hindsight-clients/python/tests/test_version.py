"""
Tests for the get_version()/aget_version() convenience wrappers.

These mock the underlying MonitoringApi so no running server is required —
they verify the wrapper delegates to the generated client and returns the
typed VersionResponse (parity with the TypeScript client's getVersion helper).
"""

from unittest.mock import AsyncMock

from hindsight_client import Hindsight, VersionResponse
from hindsight_client_api.models.features_info import FeaturesInfo


def _make_client() -> Hindsight:
    return Hindsight(base_url="http://localhost:8888")


def _version_response() -> VersionResponse:
    features = FeaturesInfo(
        observations=True,
        mcp=True,
        worker=True,
        bank_config_api=True,
        bank_llm_health=True,
        file_upload_api=True,
        document_export_api=True,
        document_import_api=True,
        audit_log=True,
        llm_trace=True,
        store_document_text=True,
    )
    return VersionResponse(api_version="0.8.2", features=features)


async def test_aget_version_delegates_to_monitoring_api():
    """aget_version() should call MonitoringApi.get_version and return its result."""
    client = _make_client()
    client._monitoring_api.get_version = AsyncMock(return_value=_version_response())

    version = await client.aget_version()

    assert version.api_version == "0.8.2"
    assert version.features.observations is True
    client._monitoring_api.get_version.assert_awaited_once()


def test_get_version_delegates_to_monitoring_api():
    """get_version() (sync) should call MonitoringApi.get_version and return its result."""
    client = _make_client()
    # AsyncMock returns an awaitable when called, which the sync wrapper awaits via _run_async.
    client._monitoring_api.get_version = AsyncMock(return_value=_version_response())

    version = client.get_version()

    assert version.api_version == "0.8.2"
    client._monitoring_api.get_version.assert_called_once()
