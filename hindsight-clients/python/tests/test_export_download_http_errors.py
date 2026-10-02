"""Export archive downloads must validate real HTTP responses."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from hindsight_client import Hindsight
from hindsight_client_api.exceptions import ApiException

OPERATION_ID = "123e4567-e89b-12d3-a456-426614174000"
ARCHIVE = b"PK\x03\x04archive"


@pytest.mark.parametrize("export_method", ["aexport_documents", "aexport_bank"])
@pytest.mark.parametrize("download_status", [200, 404, 500])
async def test_relative_export_download_validates_http_status(export_method: str, download_status: int) -> None:
    download_path = "/v1/default/files/download/banks/test/exports/transfer.zip"

    async def handler(request: web.Request) -> web.Response:
        if request.method == "POST":
            return web.json_response({"operation_id": OPERATION_ID}, status=202)
        if "/operations/" in request.path:
            return web.json_response(
                {
                    "operation_id": OPERATION_ID,
                    "status": "completed",
                    "result_metadata": {"download_url": download_path},
                }
            )
        assert request.path == download_path
        assert request.headers["Authorization"] == "Bearer test-key"
        return web.Response(status=download_status, body=ARCHIVE if download_status == 200 else b"archive unavailable")

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    async with TestServer(app) as server:
        client = Hindsight(base_url=str(server.make_url("")).rstrip("/"), api_key="test-key")
        try:
            if download_status == 200:
                assert await getattr(client, export_method)("test", poll_interval=0) == ARCHIVE
            else:
                with pytest.raises(ApiException) as exc:
                    await getattr(client, export_method)("test", poll_interval=0)
                assert exc.value.status == download_status
        finally:
            await client.aclose()
