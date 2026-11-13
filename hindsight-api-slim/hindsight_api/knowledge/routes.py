"""HTTP routes for knowledge banks: ``/v1/default/knowledge-banks``."""

from __future__ import annotations

import base64
import binascii
import json
import mimetypes
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal
from urllib.parse import quote

import aiohttp
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ..config import get_config
from ..engine.aiohttp_session import UpstreamHTTPError
from ..extensions.operation_validator import OperationValidationError
from ..models import RequestContext
from .bank_config import KnowledgeBankOverrides, KnowledgeBankSettings
from .service import (
    IngestItem,
    IngestResult,
    KnowledgeBankConfig,
    KnowledgeBankError,
    KnowledgeBankInfo,
    KnowledgeService,
    SearchRequest,
    SearchResultItem,
    get_knowledge_service,
)
from .store import KnowledgeDocument, KnowledgePassage, SearchFilters, SearchHit

PREFIX = "/v1/default/knowledge-banks"


# --- models ---------------------------------------------------------------------------------


class KnowledgeBankConfigResponse(BaseModel):
    bank_id: str
    config: KnowledgeBankSettings = Field(
        description="Every per-bank ingest setting as the bank splits files with it: its override, else "
        "the server's default."
    )
    overrides: KnowledgeBankOverrides = Field(description="Only the settings this bank overrides.")


class KnowledgeBankConfigUpdate(BaseModel):
    updates: dict[str, Any] = Field(
        description="Settings to override (named as in `config`); a null value clears that override. "
        "Each change applies to files ingested after it: re-uploading an unchanged file re-splits it."
    )


class CreateKnowledgeBankRequest(BaseModel):
    bank_id: str = Field(description="Id of the new knowledge bank. It shares the id space of memory banks.")


class KnowledgeBankResponse(BaseModel):
    bank_id: str
    created_at: datetime | None
    document_count: int
    passage_count: int
    ready_count: int
    pending_count: int
    failed_count: int
    total_bytes: int
    documents_by_media_type: dict[str, int]


class KnowledgeBankListResponse(BaseModel):
    banks: list[KnowledgeBankResponse]
    total: int = Field(description="Knowledge banks matching `q`, before paging")
    limit: int
    offset: int


class KnowledgeDeletedResponse(BaseModel):
    deleted: bool = True


_STRATEGY_FIELD = Field(
    default=None,
    description="Ingestion strategy for this item (one of the server's `HINDSIGHT_API_INGEST_STRATEGIES`). Omit "
    "it and the server's default applies. A file whose type the strategy does not map is chunked as it is.",
)


class KnowledgeIngestTextItem(BaseModel):
    document_id: str | None = Field(default=None, description="Stable id; ingesting it again replaces the document.")
    text: str = Field(min_length=1)
    filename: str | None = Field(default=None, description="Defaults to <document_id>.md.")
    tags: list[str] = Field(default_factory=list)
    strategy: str | None = _STRATEGY_FIELD


class KnowledgeIngestRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    items: list[KnowledgeIngestTextItem] = Field(min_length=1, max_length=100)
    async_: bool = Field(
        default=False,
        alias="async",
        description="If true, queue each document and return at once (202), as the file ingest always does. "
        "If false, return once every document is searchable (default: false).",
    )


class KnowledgeIngestFileOptions(BaseModel):
    document_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    strategy: str | None = _STRATEGY_FIELD


class KnowledgeIngestFilesOptions(BaseModel):
    """The multipart ``request`` field: one entry per file, in order (or none)."""

    files: list[KnowledgeIngestFileOptions] = Field(default_factory=list)


class KnowledgeIngestResponse(BaseModel):
    document_ids: list[str] = Field(description="Every document ingested, in request order.")
    operation_ids: list[str] = Field(
        description="For each document that was queued, the one background operation that processes the whole "
        "ingest (the same id for all of them). Empty for a sync ingest."
    )
    unchanged_document_ids: list[str] = Field(
        default_factory=list, description="Documents identical to what is stored: nothing was queued for them."
    )


class KnowledgePassageResponse(BaseModel):
    index: int
    media_type: Literal["document", "image", "audio", "video", "text"] = Field(title="Knowledge Media Type")
    text: str | None = None
    page: int | None = Field(default=None, description="1-based page number (PDF pages and images).")
    start_ms: int | None = Field(default=None, description="Window start, for audio and video.")
    end_ms: int | None = None
    passage_url: str | None = Field(
        default=None, description="API path that streams this passage's page image or audio/video clip."
    )


class KnowledgeDocumentResponse(BaseModel):
    document_id: str
    filename: str | None
    content_type: str
    media_type: Literal["document", "image", "audio", "video", "text"] = Field(title="Knowledge Media Type")
    size_bytes: int
    status: Literal["pending", "processing", "ready", "failed"] = Field(title="Knowledge Document Status")
    error: str | None
    passage_count: int
    tags: list[str]
    created_at: datetime | None
    updated_at: datetime | None
    document_url: str = Field(description="API path that streams the original file.")
    source_document_id: str | None = Field(
        default=None, description="The document a pre-processing strategy split this one out of."
    )


class KnowledgeDocumentDetailResponse(KnowledgeDocumentResponse):
    passages: list[KnowledgePassageResponse]


class KnowledgeDocumentListResponse(BaseModel):
    documents: list[KnowledgeDocumentResponse]
    total: int


class KnowledgeTimeseriesBucket(BaseModel):
    time: datetime
    documents: int = Field(description="Documents first written in this bucket.")
    passages: int = Field(description="Passages those documents hold now.")


class KnowledgeTimeseriesResponse(BaseModel):
    bank_id: str
    period: Literal["1d", "7d", "30d", "90d"]
    trunc: Literal["hour", "day"]
    buckets: list[KnowledgeTimeseriesBucket]


class KnowledgeSearchRequest(BaseModel):
    query: str | None = Field(default=None, description="Text query.")
    query_image: str | None = Field(default=None, description="Base64 image to search with (alone or with text).")
    query_video: str | None = Field(
        default=None,
        description="Base64 MP4 clip to search with. Text, image and video can be combined: they are "
        "embedded together as one query.",
    )
    budget: Literal["low", "mid", "high"] = Field(
        default="mid",
        description="How deep the search ranks before `max_tokens` picks from it: low is fastest, high reaches "
        "the most candidates (as memory-bank recall's budget).",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=1_000_000,
        description="What the results may weigh in all, as the tokens of their text. Hits come best first, each "
        "text hit with the passages around it joined into its `text`, as many as fit; the rest of the budget widens the "
        "best hits' context. An image, page or clip costs a small fixed amount plus its text.",
    )
    tags: list[str] | None = None
    tags_match: Literal["any", "all"] = Field(default="any", title="Knowledge Tags Match")
    document_ids: list[str] | None = None
    media_types: list[Literal["document", "image", "audio", "video", "text"]] | None = Field(
        default=None, title="Knowledge Media Types"
    )


class KnowledgeSearchResult(KnowledgePassageResponse):
    document_id: str
    filename: str | None
    tags: list[str]
    document_url: str


class KnowledgeSearchResponse(BaseModel):
    """Results best first. Ranks are what a caller compares: scores are fused per query and mean nothing
    across responses, so the response does not carry them."""

    results: list[KnowledgeSearchResult]


# --- mapping --------------------------------------------------------------------------------


def _doc_url(bank_id: str, document_id: str) -> str:
    return f"{PREFIX}/{quote(bank_id, safe='')}/documents/{quote(document_id, safe='')}"


def bank_response(info: KnowledgeBankInfo) -> KnowledgeBankResponse:
    s = info.stats
    return KnowledgeBankResponse(
        bank_id=info.bank_id,
        created_at=info.created_at,
        document_count=s.document_count,
        passage_count=s.passage_count,
        ready_count=s.ready_count,
        pending_count=s.pending_count,
        failed_count=s.failed_count,
        total_bytes=s.total_bytes,
        documents_by_media_type=s.by_media_type,
    )


def document_response(bank_id: str, d: KnowledgeDocument) -> KnowledgeDocumentResponse:
    return KnowledgeDocumentResponse(
        document_id=d.document_id,
        filename=d.filename,
        content_type=d.content_type,
        media_type=d.media_type,
        size_bytes=d.size_bytes,
        status=d.status,
        error=d.error,
        passage_count=d.passage_count,
        tags=d.tags,
        created_at=d.created_at,
        updated_at=d.updated_at,
        document_url=f"{_doc_url(bank_id, d.document_id)}/file",
        source_document_id=d.source_document_id,
    )


def passage_response(bank_id: str, d: KnowledgeDocument, p: KnowledgePassage) -> KnowledgePassageResponse:
    return KnowledgePassageResponse(
        index=p.index,
        media_type=d.media_type,
        text=p.text,
        page=p.page,
        start_ms=p.start_ms,
        end_ms=p.end_ms,
        passage_url=f"{_doc_url(bank_id, d.document_id)}/passages/{p.index}/media" if p.media_key else None,
    )


def search_result(bank_id: str, hit: SearchHit) -> KnowledgeSearchResult:
    d = hit.document
    return KnowledgeSearchResult(
        **passage_response(bank_id, d, hit.passage).model_dump(),
        document_id=d.document_id,
        filename=d.filename,
        tags=d.tags,
        document_url=f"{_doc_url(bank_id, d.document_id)}/file",
    )


def item_result(bank_id: str, item: SearchResultItem) -> KnowledgeSearchResult:
    """A hit as the API returns it: a text hit's ``text`` is the run of passages the budget gave it, the
    matched one (``index``) and the text around it; a context field beside it said the same thing twice."""
    result = search_result(bank_id, item.hit)
    if item.context is not None:
        result.text = item.context
    return result


def config_response(state: KnowledgeBankConfig) -> KnowledgeBankConfigResponse:
    return KnowledgeBankConfigResponse(bank_id=state.bank_id, config=state.settings, overrides=state.overrides)


def _ingest_response(result: IngestResult) -> KnowledgeIngestResponse:
    return KnowledgeIngestResponse(
        document_ids=result.document_ids,
        operation_ids=[op for op in result.operation_ids if op is not None],
        unchanged_document_ids=result.unchanged_document_ids,
    )


def _content_type(upload: UploadFile) -> str:
    """The upload's type; a generic one is guessed from the filename, so a download comes back typed."""
    declared = (upload.content_type or "").split(";")[0].strip()
    if declared and declared != "application/octet-stream":
        return declared
    return mimetypes.guess_type(upload.filename or "")[0] or "application/octet-stream"


def _http_error(exc: Exception) -> HTTPException:
    # A store shedding the call (indexing behind its writers) is "not now": 503 with Retry-After, the
    # memory banks' own rule, not a 500 that tells a bulk ingest its request is broken. Imported here:
    # api/http.py builds this router, so a module-level import would load the whole of it first.
    from ..api.http import _store_unavailable

    unavailable = _store_unavailable(exc)
    if unavailable is not None:
        return unavailable
    if isinstance(exc, KnowledgeBankError):
        return HTTPException(status_code=exc.status_code, detail=str(exc))
    if isinstance(exc, OperationValidationError):
        return HTTPException(status_code=exc.status_code, detail=exc.reason)
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, aiohttp.ClientError | UpstreamHTTPError | TimeoutError):
        # The embedder is a remote server here: its outage is ours to report, not a crash.
        return HTTPException(status_code=503, detail=f"knowledge embedder unavailable: {exc}")
    raise exc


def _content_disposition(filename: str | None) -> dict[str, str]:
    if not filename:
        return {}
    return {"Content-Disposition": f"inline; filename*=UTF-8''{quote(filename)}"}


# --- router ---------------------------------------------------------------------------------


def _require_enabled() -> None:
    # Checked per request rather than at mount time, so a test (or an operator) can flip the flag
    # without rebuilding the app; with it off the server answers as if the routes did not exist.
    if not get_config().enable_knowledge_banks:
        raise HTTPException(status_code=404, detail="Not Found")


def _require_config_writes() -> None:
    # The memory banks' switch for per-bank config writes covers these too: an operator who froze
    # bank settings froze all of them.
    if not get_config().enable_bank_config_api:
        raise HTTPException(
            status_code=404,
            detail="Bank configuration API is disabled. Set HINDSIGHT_API_ENABLE_BANK_CONFIG_API=true to re-enable.",
        )


def build_router(get_request_context: Callable[..., RequestContext]) -> APIRouter:
    # Out of the OpenAPI spec while knowledge banks are a hidden feature: no generated client,
    # CLI command or API reference page names them. Back in the spec at launch.
    router = APIRouter(
        prefix=PREFIX,
        tags=["Knowledge Banks"],
        dependencies=[Depends(_require_enabled)],
        include_in_schema=False,
    )

    def service(request: Request) -> KnowledgeService:
        return get_knowledge_service(request.app.state.memory)

    @router.post(
        "",
        response_model=KnowledgeBankResponse,
        summary="Create a knowledge bank",
        operation_id="create_knowledge_bank",
    )
    async def create_bank(
        body: CreateKnowledgeBankRequest, request: Request, rc: RequestContext = Depends(get_request_context)
    ) -> KnowledgeBankResponse:
        try:
            return bank_response(await service(request).create_bank(body.bank_id, rc))
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.get(
        "",
        response_model=KnowledgeBankListResponse,
        summary="List knowledge banks",
        operation_id="list_knowledge_banks",
    )
    async def list_banks(
        request: Request,
        limit: int = Query(default=100, ge=0),
        offset: int = Query(default=0, ge=0),
        q: str | None = Query(default=None, description="Part of the bank id, case-insensitive"),
        rc: RequestContext = Depends(get_request_context),
    ) -> KnowledgeBankListResponse:
        try:
            page = await service(request).list_banks(rc, limit=limit, offset=offset, q=q)
        except Exception as exc:
            raise _http_error(exc) from exc
        return KnowledgeBankListResponse(
            banks=[bank_response(b) for b in page.banks], total=page.total, limit=limit, offset=offset
        )

    @router.get(
        "/{bank_id}",
        response_model=KnowledgeBankResponse,
        summary="Get a knowledge bank",
        operation_id="get_knowledge_bank",
    )
    async def get_bank(
        bank_id: str, request: Request, rc: RequestContext = Depends(get_request_context)
    ) -> KnowledgeBankResponse:
        try:
            return bank_response(await service(request).get_bank(bank_id, rc))
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.delete(
        "/{bank_id}",
        response_model=KnowledgeDeletedResponse,
        summary="Delete a knowledge bank and everything in it",
        operation_id="delete_knowledge_bank",
    )
    async def delete_bank(
        bank_id: str, request: Request, rc: RequestContext = Depends(get_request_context)
    ) -> KnowledgeDeletedResponse:
        try:
            await service(request).delete_bank(bank_id, rc)
        except Exception as exc:
            raise _http_error(exc) from exc
        return KnowledgeDeletedResponse()

    # `overrides` lists only what the bank sets: its unset fields are left out, not sent as null.
    @router.get(
        "/{bank_id}/config",
        response_model=KnowledgeBankConfigResponse,
        response_model_exclude_none=True,
        summary="Get a knowledge bank's ingest settings",
        operation_id="get_knowledge_bank_config",
    )
    async def get_bank_config(
        bank_id: str, request: Request, rc: RequestContext = Depends(get_request_context)
    ) -> KnowledgeBankConfigResponse:
        try:
            return config_response(await service(request).get_bank_config(bank_id, rc))
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.patch(
        "/{bank_id}/config",
        response_model=KnowledgeBankConfigResponse,
        response_model_exclude_none=True,
        summary="Override a knowledge bank's ingest settings",
        operation_id="update_knowledge_bank_config",
        dependencies=[Depends(_require_config_writes)],
    )
    async def update_bank_config(
        bank_id: str,
        body: KnowledgeBankConfigUpdate,
        request: Request,
        rc: RequestContext = Depends(get_request_context),
    ) -> KnowledgeBankConfigResponse:
        try:
            return config_response(await service(request).update_bank_config(bank_id, body.updates, rc))
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.delete(
        "/{bank_id}/config",
        response_model=KnowledgeBankConfigResponse,
        response_model_exclude_none=True,
        summary="Reset a knowledge bank's ingest settings to the server's",
        operation_id="reset_knowledge_bank_config",
        dependencies=[Depends(_require_config_writes)],
    )
    async def reset_bank_config(
        bank_id: str, request: Request, rc: RequestContext = Depends(get_request_context)
    ) -> KnowledgeBankConfigResponse:
        try:
            return config_response(await service(request).reset_bank_config(bank_id, rc))
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.get(
        "/{bank_id}/stats/timeseries",
        response_model=KnowledgeTimeseriesResponse,
        summary="Documents and passages added over time",
        operation_id="get_knowledge_bank_timeseries",
    )
    async def get_timeseries(
        bank_id: str,
        request: Request,
        period: Literal["1d", "7d", "30d", "90d"] = Query(default="7d"),
        media_type: Literal["document", "image", "audio", "video", "text"] | None = Query(
            default=None, description="Only documents of this media type."
        ),
        rc: RequestContext = Depends(get_request_context),
    ) -> KnowledgeTimeseriesResponse:
        trunc: Literal["hour", "day"] = "hour" if period == "1d" else "day"
        try:
            buckets = await service(request).timeseries(bank_id, period, trunc, rc, media_type=media_type)
        except Exception as exc:
            raise _http_error(exc) from exc
        return KnowledgeTimeseriesResponse(
            bank_id=bank_id,
            period=period,
            trunc=trunc,
            buckets=[
                KnowledgeTimeseriesBucket(time=b.time, documents=b.documents, passages=b.passages) for b in buckets
            ],
        )

    @router.post(
        "/{bank_id}/files/ingest",
        response_model=KnowledgeIngestResponse,
        status_code=202,
        summary="Ingest files (PDF, images, audio, video, text, Office)",
        description="Each file becomes one document, chunked into passages (or pre-processed first by the "
        "extension its ingestion strategy maps its type to) and embedded in the background. Poll the operation, or the document's status, to "
        "know when it is searchable.",
        operation_id="ingest_knowledge_files",
    )
    async def ingest_files(
        bank_id: str,
        request: Request,
        files: list[UploadFile] = File(...),
        options: str | None = Form(default=None, alias="request"),
        rc: RequestContext = Depends(get_request_context),
    ) -> KnowledgeIngestResponse:
        try:
            parsed = (
                KnowledgeIngestFilesOptions.model_validate(json.loads(options))
                if options
                else KnowledgeIngestFilesOptions()
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid 'request' field: {exc}") from exc
        if parsed.files and len(parsed.files) != len(files):
            raise HTTPException(status_code=400, detail="'request.files' must have one entry per file")
        items = []
        for i, upload in enumerate(files):
            opts = parsed.files[i] if parsed.files else KnowledgeIngestFileOptions()
            items.append(
                IngestItem(
                    data=await upload.read(),
                    filename=upload.filename,
                    content_type=_content_type(upload),
                    document_id=opts.document_id,
                    tags=opts.tags,
                    strategy=opts.strategy,
                )
            )
        try:
            result = await service(request).ingest(bank_id, items, rc)
        except Exception as exc:
            raise _http_error(exc) from exc
        return _ingest_response(result)

    @router.post(
        "/{bank_id}/ingest",
        response_model=KnowledgeIngestResponse,
        summary="Ingest text documents",
        description="Synchronous by default: returns once every document is searchable. With `async: true` "
        "each document is queued instead (202) and its operation returned.",
        responses={202: {"model": KnowledgeIngestResponse, "description": "Queued (`async: true`)"}},
        operation_id="ingest_knowledge_texts",
    )
    async def ingest_texts(
        bank_id: str,
        body: KnowledgeIngestRequest,
        request: Request,
        response: Response,
        rc: RequestContext = Depends(get_request_context),
    ) -> KnowledgeIngestResponse:
        items = [
            IngestItem(
                data=d.text.encode(),
                filename=d.filename,
                content_type="text/markdown",
                document_id=d.document_id,
                tags=d.tags,
                strategy=d.strategy,
            )
            for d in body.items
        ]
        try:
            result = await service(request).ingest(bank_id, items, rc, wait=not body.async_)
        except Exception as exc:
            raise _http_error(exc) from exc
        if body.async_:
            response.status_code = 202
        return _ingest_response(result)

    @router.get(
        "/{bank_id}/documents",
        response_model=KnowledgeDocumentListResponse,
        summary="List documents",
        operation_id="list_knowledge_documents",
    )
    async def list_documents(
        bank_id: str,
        request: Request,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        q: str | None = Query(default=None, description="Substring of the id or filename."),
        status: str | None = Query(default=None),
        media_type: str | None = Query(default=None),
        tags: list[str] | None = Query(default=None),
        rc: RequestContext = Depends(get_request_context),
    ) -> KnowledgeDocumentListResponse:
        try:
            page = await service(request).list_documents(
                bank_id, rc, limit=limit, offset=offset, query=q, status=status, media_type=media_type, tags=tags
            )
        except Exception as exc:
            raise _http_error(exc) from exc
        return KnowledgeDocumentListResponse(
            documents=[document_response(bank_id, d) for d in page.documents], total=page.total
        )

    @router.get(
        "/{bank_id}/documents/{document_id}",
        response_model=KnowledgeDocumentDetailResponse,
        summary="Get a document and its passages",
        operation_id="get_knowledge_document",
    )
    async def get_document(
        bank_id: str, document_id: str, request: Request, rc: RequestContext = Depends(get_request_context)
    ) -> KnowledgeDocumentDetailResponse:
        try:
            detail = await service(request).get_document(bank_id, document_id, rc)
        except Exception as exc:
            raise _http_error(exc) from exc
        return KnowledgeDocumentDetailResponse(
            **document_response(bank_id, detail.document).model_dump(),
            passages=[passage_response(bank_id, detail.document, p) for p in detail.passages],
        )

    @router.delete(
        "/{bank_id}/documents/{document_id}",
        response_model=KnowledgeDeletedResponse,
        summary="Delete a document",
        operation_id="delete_knowledge_document",
    )
    async def delete_document(
        bank_id: str, document_id: str, request: Request, rc: RequestContext = Depends(get_request_context)
    ) -> KnowledgeDeletedResponse:
        try:
            await service(request).delete_document(bank_id, document_id, rc)
        except Exception as exc:
            raise _http_error(exc) from exc
        return KnowledgeDeletedResponse()

    @router.get(
        "/{bank_id}/documents/{document_id}/file",
        summary="Download a document's original file",
        operation_id="get_knowledge_document_file",
        response_class=StreamingResponse,
    )
    async def get_file(
        bank_id: str, document_id: str, request: Request, rc: RequestContext = Depends(get_request_context)
    ) -> StreamingResponse:
        try:
            opened = await service(request).open_file(bank_id, document_id, rc)
        except Exception as exc:
            raise _http_error(exc) from exc
        return StreamingResponse(
            opened.stream, media_type=opened.content_type, headers=_content_disposition(opened.filename)
        )

    @router.get(
        "/{bank_id}/documents/{document_id}/passages/{index}/media",
        summary="Stream a passage's media (page image, image, audio or video window)",
        operation_id="get_knowledge_passage_media",
        response_class=StreamingResponse,
    )
    async def get_passage_media(
        bank_id: str, document_id: str, index: int, request: Request, rc: RequestContext = Depends(get_request_context)
    ) -> StreamingResponse:
        try:
            opened = await service(request).open_passage_media(bank_id, document_id, index, rc)
        except Exception as exc:
            raise _http_error(exc) from exc
        return StreamingResponse(
            opened.stream, media_type=opened.content_type, headers={"Cache-Control": "private, max-age=3600"}
        )

    @router.post(
        "/{bank_id}/search",
        response_model=KnowledgeSearchResponse,
        summary="Search passages by meaning: a text, image or video query, alone or combined",
        operation_id="search_knowledge_bank",
    )
    async def search(
        bank_id: str, body: KnowledgeSearchRequest, request: Request, rc: RequestContext = Depends(get_request_context)
    ) -> KnowledgeSearchResponse:
        try:
            image = base64.b64decode(body.query_image, validate=True) if body.query_image else None
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="query_image must be base64") from exc
        try:
            video = base64.b64decode(body.query_video, validate=True) if body.query_video else None
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="query_video must be base64") from exc
        filters = SearchFilters(
            tags=body.tags,
            tags_match=body.tags_match,
            document_ids=body.document_ids,
            media_types=list(body.media_types) if body.media_types else None,
        )
        try:
            hits = await service(request).search(
                bank_id,
                SearchRequest(
                    query=body.query,
                    query_image=image,
                    query_video=video,
                    filters=filters,
                    max_tokens=body.max_tokens,
                    budget=body.budget,
                ),
                rc,
            )
        except Exception as exc:
            raise _http_error(exc) from exc
        return KnowledgeSearchResponse(results=[item_result(bank_id, h) for h in hits])

    return router
