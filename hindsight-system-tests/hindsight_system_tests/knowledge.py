"""Knowledge-bank calls for the stories, made through the published client's own transport.

Knowledge banks are hidden: their routes are left out of the OpenAPI spec, so the generated client
has no methods for them yet. The stories still go through the client — its base URL, headers and
aiohttp session — by serializing each call with the client's ``ApiClient``, exactly as a generated
method does. Only the method is missing, not the transport. When the routes are published, these
become the generated calls and the stories do not change.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from hindsight_client import Hindsight

KB = "/v1/default/knowledge-banks"


@dataclass
class Response:
    status: int
    #: The JSON body, or None when the body is not JSON (a file, an image).
    body: Any
    raw: bytes


@dataclass
class UploadFile:
    filename: str
    data: bytes


class KnowledgeBanks:
    def __init__(self, client: Hindsight) -> None:
        self._api = client._api_client

    async def call(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        query: dict[str, Any] | None = None,
        files: list[UploadFile] | None = None,
        form: dict[str, str] | None = None,
    ) -> Response:
        headers = {"Accept": "application/json"}
        if files is not None:
            headers["Content-Type"] = "multipart/form-data"
        elif json_body is not None:
            headers["Content-Type"] = "application/json"
        serialized = self._api.param_serialize(
            method=method,
            resource_path=path,
            query_params=list((query or {}).items()) or None,
            header_params=headers,
            body=json_body,
            post_params=list((form or {}).items()) or None,
            files={"files": [(f.filename, f.data) for f in files]} if files else None,
        )
        response = await self._api.call_api(*serialized)
        raw = await response.read()
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = None
        return Response(response.status, body, raw)

    # --- the calls the stories make, each asserting nothing beyond what it returns ---------------

    async def create(self, bank_id: str) -> Response:
        return await self.call("POST", KB, json_body={"bank_id": bank_id})

    async def get(self, bank_id: str) -> Response:
        return await self.call("GET", f"{KB}/{bank_id}")

    async def delete(self, bank_id: str) -> Response:
        return await self.call("DELETE", f"{KB}/{bank_id}")

    async def ingest(self, bank_id: str, items: list[dict[str, Any]], *, wait: bool = True) -> Response:
        return await self.call("POST", f"{KB}/{bank_id}/ingest", json_body={"items": items, "async": not wait})

    async def ingest_files(
        self, bank_id: str, files: list[UploadFile], options: list[dict[str, Any]] | None = None
    ) -> Response:
        form = {"request": json.dumps({"files": options})} if options else None
        return await self.call("POST", f"{KB}/{bank_id}/files/ingest", files=files, form=form)

    async def document(self, bank_id: str, document_id: str) -> Response:
        return await self.call("GET", f"{KB}/{bank_id}/documents/{document_id}")

    async def documents(self, bank_id: str, **query: Any) -> Response:
        return await self.call("GET", f"{KB}/{bank_id}/documents", query=query)

    async def delete_document(self, bank_id: str, document_id: str) -> Response:
        return await self.call("DELETE", f"{KB}/{bank_id}/documents/{document_id}")

    async def file(self, bank_id: str, document_id: str) -> Response:
        return await self.call("GET", f"{KB}/{bank_id}/documents/{document_id}/file")

    async def search(self, bank_id: str, **body: Any) -> Response:
        return await self.call("POST", f"{KB}/{bank_id}/search", json_body=body)

    async def wait_until_processed(self, bank_id: str, count: int, timeout: float = 30.0) -> None:
        """Until ``count`` documents of the bank are ready or failed: an async ingest's worker is done."""
        deadline = time.monotonic() + timeout
        while True:
            page = (await self.documents(bank_id, limit=200)).body
            done = [d for d in page["documents"] if d["status"] in ("ready", "failed")]
            if len(done) >= count:
                return
            if time.monotonic() > deadline:
                raise AssertionError(f"only {len(done)} of {count} documents processed: {page['documents']}")
            await asyncio.sleep(0.2)
