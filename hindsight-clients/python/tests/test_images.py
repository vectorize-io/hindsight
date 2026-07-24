import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import hindsight_client_api
from hindsight_client import Hindsight
from hindsight_client_api.models.recall_response import RecallResponse


@pytest.mark.asyncio
async def test_image_multipart_order_headers_and_management(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    client = Hindsight("http://unused")
    api = SimpleNamespace(
        image_retain=AsyncMock(return_value=object()),
        list_image_assets=AsyncMock(return_value=object()),
        get_image_asset_with_http_info=AsyncMock(
            return_value=SimpleNamespace(
                raw_data=b"image-bytes",
                headers={
                    "Content-Type": "image/jpeg",
                    "Content-Length": "11",
                    "X-Hindsight-Image-SHA256": "a" * 64,
                    "X-Hindsight-Image-Width": "8",
                    "X-Hindsight-Image-Height": "6",
                    "X-Hindsight-Asset-Created-At": "2026-01-01T00:00:00+00:00",
                    "X-Hindsight-Asset-Updated-At": "2026-01-01T00:00:01+00:00",
                },
            )
        ),
        delete_image_asset=AsyncMock(return_value=None),
    )
    client._images_api = api
    try:
        await client.aretain_images(
            "bank",
            [first, second],
            images=[{"asset_id": "a/1"}, {"asset_id": "b/2"}],
            idempotency_key="idem",
        )
        kwargs = api.image_retain.await_args.kwargs
        assert kwargs["files"] == [("first.jpg", b"first"), ("second.png", b"second")]
        assert json.loads(kwargs["request"])["images"] == [{"asset_id": "a/1"}, {"asset_id": "b/2"}]
        assert kwargs["idempotency_key"] == "idem"

        await client.alist_image_assets("bank", document_id="doc/one", status="ready")
        assert api.list_image_assets.await_args.kwargs["document_id"] == "doc/one"
        downloaded = await client.aget_image_asset("bank", "a/1")
        assert api.get_image_asset_with_http_info.await_args.kwargs["asset_id"] == "a/1"
        assert downloaded.content == b"image-bytes"
        assert downloaded.descriptor.document_ids == []
        await client.adelete_image_asset("bank", "a/1")
        assert api.delete_image_asset.await_args.kwargs["asset_id"] == "a/1"
        api.image_retain.side_effect = hindsight_client_api.ApiException(status=404, reason="image API disabled")
        with pytest.raises(hindsight_client_api.ApiException) as exc:
            await client.aretain_images("bank", [first])
        assert exc.value.status == 404
    finally:
        await client.aclose()


def test_recall_image_map_and_old_server_failure_are_explicit():
    response = RecallResponse.from_dict(
        {
            "results": [],
            "image_assets": {
                "doc": [
                    {
                        "asset_id": "asset",
                        "mime_type": "image/jpeg",
                        "size_bytes": 1,
                        "sha256": "a" * 64,
                        "width": 1,
                        "height": 1,
                        "status": "ready",
                        "document_ids": ["doc"],
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-01T00:00:00Z",
                        "ordinal": 0,
                    }
                ]
            },
        }
    )
    assert response.image_assets["doc"][0].asset_id == "asset"
    error = hindsight_client_api.ApiException(status=404, reason="image API is disabled")
    assert error.status == 404
    assert "disabled" in str(error)


@pytest.mark.asyncio
async def test_transfer_helpers_use_streams_and_caller_owned_file_handles(monkeypatch):
    client = Hindsight("http://unused", api_key="token")

    class Response:
        def __init__(self, status, payload=b""):
            self.status = status
            self.payload = payload
            self.content = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def iter_chunked(self, _size):
            yield self.payload[:3]
            yield self.payload[3:]

        async def text(self):
            return self.payload.decode()

    class Session:
        responses = [Response(200, b"archive"), Response(202, b'{"operation_id":"op"}')]
        posts = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get(self, *_args, **_kwargs):
            return self.responses.pop(0)

        def post(self, *_args, **kwargs):
            self.posts.append(kwargs)
            return self.responses.pop(0)

    session = Session()
    client._api_client.rest_client._pool_manager = session
    client._api_client.rest_client.proxy = "http://proxy.example"
    client._api_client.rest_client.proxy_headers = {"X-Proxy": "configured"}
    client._api_client.set_default_header("X-Custom", "configured")
    target = io.BytesIO()
    source = io.BytesIO(b"archive")
    try:
        await client.aexport_documents_to("bank/one", target)
        assert target.getvalue() == b"archive"
        result = await client.aimport_documents_from("bank/one", source)
        assert result.operation_id == "op"
        assert not source.closed
        assert Session.posts[0]["data"].is_multipart
        assert Session.posts[0]["headers"]["X-Custom"] == "configured"
        assert Session.posts[0]["proxy"] == "http://proxy.example"
        assert Session.posts[0]["proxy_headers"] == {"X-Proxy": "configured"}
    finally:
        client._api_client.rest_client._pool_manager = None
        await client.aclose()
