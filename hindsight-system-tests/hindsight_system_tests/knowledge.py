"""Knowledge-bank calls for the stories, made through the published client's generated API.

Each call returns the status and the body rather than raising on an error status, so a story
asserts a refusal (a 400, a 404) the same way it asserts a success.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any

from hindsight_client import Hindsight
from hindsight_client_api.api.knowledge_banks_api import KnowledgeBanksApi
from hindsight_client_api.models import (
    CreateKnowledgeBankRequest,
    KnowledgeBankConfigUpdate,
    KnowledgeIngestRequest,
    KnowledgeSearchRequest,
)
from hindsight_client_api.rest import RESTResponseType


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


async def _read(pending: Awaitable[RESTResponseType]) -> Response:
    response = await pending
    raw = await response.read()
    try:
        body = json.loads(raw) if raw else None
    except ValueError:
        body = None
    return Response(response.status, body, raw)


class KnowledgeBanks:
    def __init__(self, client: Hindsight) -> None:
        self._api = KnowledgeBanksApi(client._api_client)

    # --- the calls the stories make, each asserting nothing beyond what it returns ---------------

    async def list(self) -> Response:
        return await _read(self._api.list_knowledge_banks_without_preload_content())

    async def create(self, bank_id: str) -> Response:
        request = CreateKnowledgeBankRequest(bank_id=bank_id)
        return await _read(self._api.create_knowledge_bank_without_preload_content(request))

    async def get(self, bank_id: str) -> Response:
        return await _read(self._api.get_knowledge_bank_without_preload_content(bank_id))

    async def delete(self, bank_id: str) -> Response:
        return await _read(self._api.delete_knowledge_bank_without_preload_content(bank_id))

    async def config(self, bank_id: str) -> Response:
        return await _read(self._api.get_knowledge_bank_config_without_preload_content(bank_id))

    async def update_config(self, bank_id: str, updates: dict[str, Any]) -> Response:
        update = KnowledgeBankConfigUpdate(updates=updates)
        return await _read(self._api.update_knowledge_bank_config_without_preload_content(bank_id, update))

    async def reset_config(self, bank_id: str) -> Response:
        return await _read(self._api.reset_knowledge_bank_config_without_preload_content(bank_id))

    async def ingest(self, bank_id: str, items: list[dict[str, Any]], *, wait: bool = True) -> Response:
        request = KnowledgeIngestRequest.from_dict({"items": items, "async": not wait})
        return await _read(self._api.ingest_knowledge_texts_without_preload_content(bank_id, request))

    async def ingest_files(
        self, bank_id: str, files: list[UploadFile], options: list[dict[str, Any]] | None = None
    ) -> Response:
        return await _read(
            self._api.ingest_knowledge_files_without_preload_content(
                bank_id,
                [(f.filename, f.data) for f in files],
                request=json.dumps({"files": options}) if options else None,
            )
        )

    async def document(self, bank_id: str, document_id: str) -> Response:
        return await _read(self._api.get_knowledge_document_without_preload_content(bank_id, document_id))

    async def documents(self, bank_id: str, **query: Any) -> Response:
        return await _read(self._api.list_knowledge_documents_without_preload_content(bank_id, **query))

    async def delete_document(self, bank_id: str, document_id: str) -> Response:
        return await _read(self._api.delete_knowledge_document_without_preload_content(bank_id, document_id))

    async def file(self, bank_id: str, document_id: str) -> Response:
        return await _read(self._api.get_knowledge_document_file_without_preload_content(bank_id, document_id))

    async def tags(self, bank_id: str, **query: Any) -> Response:
        return await _read(self._api.list_knowledge_bank_tags_without_preload_content(bank_id, **query))

    async def search(self, bank_id: str, **body: Any) -> Response:
        request = KnowledgeSearchRequest.from_dict(body)
        return await _read(self._api.search_knowledge_bank_without_preload_content(bank_id, request))

    async def operations(self, bank_id: str, **query: Any) -> Response:
        """The bank's background jobs: a knowledge bank's own, never ``/banks/{id}/operations``."""
        return await _read(self._api.list_knowledge_bank_operations_without_preload_content(bank_id, **query))

    async def operation(self, bank_id: str, operation_id: str) -> Response:
        return await _read(self._api.get_knowledge_bank_operation_without_preload_content(bank_id, operation_id))

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
