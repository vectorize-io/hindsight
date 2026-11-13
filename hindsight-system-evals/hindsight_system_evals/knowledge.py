"""Knowledge-bank retrieval, end to end through the HTTP API, and how it is scored.

A suite writes its corpus into a knowledge bank (file and text ingest), waits until every
document is ready, runs every query through search, and scores what came back. Nothing calls an LLM; the only model
is the server's knowledge embedder, so a run measures that embedder and the pipeline around
it (page rendering and text layers, the passage splitter, neighbour expansion, the store).

Scoring is the leaderboards' own: documents in the order of their best passage, scored
against graded qrels with nDCG@10 (trec_eval's linear gain, as MTEB/BEIR and ViDoRe report it)
and recall@k (how Flickr30k is reported).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import httpx
from pydantic import BaseModel

from hindsight_system_evals.knowledge_data import FileItem, RankedQuery, TextItem
from hindsight_system_evals.report import KnowledgeIngest, KnowledgeLatency
from hindsight_system_evals.retrieval_metrics import alpha_ndcg_at_k, ndcg_at_k, recall_at_k

#: Searches in flight at once. The embedder dominates a search, and a remote one is mostly
#: round trip, so a few in parallel keep a run in minutes; latency is reported with it.
SEARCH_CONCURRENCY = int(os.getenv("HINDSIGHT_EVAL_KNOWLEDGE_SEARCH_CONCURRENCY", "4"))
SEARCH_RETRIES = 3
#: The search budget the ranked scoring asks for: the API's largest, so every ranked candidate comes back.
SCORING_BUDGET = 1_000_000
INGEST_TIMEOUT = float(os.getenv("HINDSIGHT_EVAL_KNOWLEDGE_INGEST_TIMEOUT", str(6 * 3600)))
FILES_PER_UPLOAD = 16
#: The text ingest's ceiling per request.
TEXTS_PER_UPLOAD = 100
#: Documents loaded per batch: a multiple of both ceilings, so batching the loads adds no
#: short upload requests.
LOAD_BATCH = 400
#: The search API's ceiling. Text corpora return several passages per document, so the
#: document list can be shorter than 100 and recall@100 there is a floor.
RANKED_TOP_K = 200
PREFIX = "/v1/default/knowledge-banks"


# --- through the API ----------------------------------------------------------------------


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    page: int | None


# The routes' responses, parsed at the boundary. Only the fields a suite reads: the server's
# models carry more, and pydantic ignores the rest.


class BankStats(BaseModel):
    passage_count: int
    ready_count: int
    failed_count: int


class FailedDocument(BaseModel):
    document_id: str
    error: str | None


class _DocumentList(BaseModel):
    documents: list[FailedDocument]


class _SearchResult(BaseModel):
    document_id: str
    page: int | None = None


class _SearchResponse(BaseModel):
    results: list[_SearchResult]


class KnowledgeApi:
    """The knowledge-bank routes over plain HTTP: they are hidden from the OpenAPI spec, so the
    published client has no methods for them."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=httpx.Timeout(600.0))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _call(self, method: str, path: str, **kwargs: object) -> bytes:
        r = await self._http.request(method, f"{PREFIX}{path}", **kwargs)  # type: ignore[arg-type]
        if r.status_code >= 300:
            raise httpx.HTTPStatusError(
                f"{method} {path}: {r.status_code} {r.text[:500]}", request=r.request, response=r
            )
        return r.content

    async def create_bank(self, bank_id: str) -> None:
        await self._call("POST", "", json={"bank_id": bank_id})

    async def delete_bank(self, bank_id: str) -> None:
        await self._call("DELETE", f"/{bank_id}")

    async def bank(self, bank_id: str) -> BankStats:
        return BankStats.model_validate_json(await self._call("GET", f"/{bank_id}"))

    async def ingest_files(self, bank_id: str, items: Sequence[FileItem]) -> None:
        files = [("files", (i.path.name, i.path.read_bytes())) for i in items]
        options = json.dumps({"files": [{"document_id": i.id} for i in items]})
        # A large upload now and then gets a 500 from a Postgres connection the file save broke
        # ("invalid frontend message type"); the same request again succeeds, and with the same
        # document ids it is an upsert. Retried so one such failure does not cost a whole run.
        for attempt in range(3):
            try:
                await self._call("POST", f"/{bank_id}/files/ingest", files=files, data={"request": options})
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500 or attempt == 2:
                    raise
            await asyncio.sleep(5)

    async def ingest_texts(self, bank_id: str, items: Sequence[TextItem]) -> None:
        body = {
            "items": [
                {"document_id": i.id, "text": i.text, "filename": i.filename, "tags": list(i.tags)} for i in items
            ],
            "async": True,
        }
        await self._call("POST", f"/{bank_id}/ingest", json=body)

    async def failed(self, bank_id: str, limit: int = 20) -> list[FailedDocument]:
        # The list endpoint caps a page at 500, so walk pages until `limit` (or the end) is reached.
        docs: list[FailedDocument] = []
        while len(docs) < limit:
            page_size = min(500, limit - len(docs))
            params = {"status": "failed", "limit": page_size, "offset": len(docs)}
            raw = await self._call("GET", f"/{bank_id}/documents", params=params)
            page = _DocumentList.model_validate_json(raw).documents
            docs.extend(page)
            if len(page) < page_size:
                break
        return docs

    async def search(self, bank_id: str, query: str, top_k: int) -> list[SearchHit]:
        """The ``top_k`` best hits. The API takes a token budget, not a count: the scoring asks for its
        deepest one (every candidate fits) and keeps the ranks it scores."""
        raw = await self._call("POST", f"/{bank_id}/search", json={"query": query, "max_tokens": SCORING_BUDGET})
        return [SearchHit(r.document_id, r.page) for r in _SearchResponse.model_validate_json(raw).results][:top_k]


_MEDIA_TYPES = ("document", "image", "audio", "video", "text")


def embedder_name() -> str:
    """What a result is labelled with. The server does not tell (its embedder is its own business),
    so the run says: ``HINDSIGHT_EVAL_KNOWLEDGE_EMBEDDER``, else the embedder a server this package
    starts is given, else ``unknown``."""
    if named := os.getenv("HINDSIGHT_EVAL_KNOWLEDGE_EMBEDDER"):
        return named
    provider = os.getenv("HINDSIGHT_EVAL_SET_KNOWLEDGE_EMBEDDINGS_PROVIDER")
    model = os.getenv("HINDSIGHT_EVAL_SET_KNOWLEDGE_EMBEDDINGS_MODEL")
    return f"{provider}:{model or 'default'}" if provider else "unknown"


def embedder_media_types() -> list[str]:
    """The media types the run's embedder writes, from ``HINDSIGHT_EVAL_KNOWLEDGE_MEDIA_TYPES`` (comma
    separated); every type when unset. A case for any other type is skipped."""
    raw = os.getenv("HINDSIGHT_EVAL_KNOWLEDGE_MEDIA_TYPES")
    return [t.strip() for t in raw.split(",") if t.strip()] if raw else list(_MEDIA_TYPES)


#: Loads the items for a batch of document ids. Ingest asks for one batch at a time, so a corpus
#: read from disk (EnterpriseRAG's 1.26 GB zip) is never held in memory whole.
Loader = Callable[[Sequence[str]], Sequence[FileItem | TextItem]]


def preloaded(items: Sequence[FileItem | TextItem]) -> Loader:
    """A loader over items already in memory."""
    by_id = {i.id: i for i in items}
    return lambda ids: [by_id[i] for i in ids]


async def _upload(api: KnowledgeApi, bank_id: str, document_ids: Sequence[str], load: Loader) -> None:
    for n in range(0, len(document_ids), LOAD_BATCH):
        batch = load(document_ids[n : n + LOAD_BATCH])
        files = [i for i in batch if isinstance(i, FileItem)]
        texts = [i for i in batch if isinstance(i, TextItem)]
        for m in range(0, len(files), FILES_PER_UPLOAD):
            await api.ingest_files(bank_id, files[m : m + FILES_PER_UPLOAD])
        for m in range(0, len(texts), TEXTS_PER_UPLOAD):
            await api.ingest_texts(bank_id, texts[m : m + TEXTS_PER_UPLOAD])


async def ingest(
    api: KnowledgeApi,
    bank_id: str,
    document_ids: Sequence[str],
    load: Loader,
    *,
    reuse_bank: bool = False,
    timeout: float = INGEST_TIMEOUT,
) -> KnowledgeIngest:
    """Create the bank (or, with ``reuse_bank``, write into the one already there), upload every
    document, and wait until each is ready or failed."""
    if reuse_bank:
        with contextlib.suppress(httpx.HTTPStatusError):  # already there: re-ingest unchanged documents
            await api.create_bank(bank_id)
    else:
        await api.create_bank(bank_id)
    started = time.perf_counter()
    await _upload(api, bank_id, document_ids, load)
    uploaded = time.perf_counter() - started

    # One bank read per poll instead of one per document: waiting on 5,000 documents one at a
    # time would add minutes of polling to the ingest time being measured.
    deadline = started + timeout
    retried = False
    while True:
        bank = await api.bank(bank_id)
        if bank.ready_count + bank.failed_count >= len(document_ids):
            if not bank.failed_count or retried:
                break
            # A remote embedder behind a tunnel drops a long request now and then, and the server
            # fails that ingest's documents once its retry budget runs out. Writing them again once
            # (same ids, so an upsert; a failed document is never "unchanged") keeps a blip from
            # failing a run; a document that fails twice is reported.
            retried = True
            failed = {d.document_id for d in await api.failed(bank_id, limit=len(document_ids))}
            await _upload(api, bank_id, [d for d in document_ids if d in failed], load)
            await asyncio.sleep(5.0)
            continue
        if time.perf_counter() > deadline:
            raise TimeoutError(f"{bank_id}: {bank.ready_count}/{len(document_ids)} ready after {timeout:.0f}s")
        await asyncio.sleep(1.0)
    wall = time.perf_counter() - started
    return KnowledgeIngest(
        documents=len(document_ids),
        passages=bank.passage_count,
        failed=bank.failed_count,
        upload_s=round(uploaded, 2),
        wall_s=round(wall, 2),
        documents_per_s=round(len(document_ids) / wall, 3),
        passages_per_s=round(bank.passage_count / wall, 3),
    )


async def failed_documents(api: KnowledgeApi, bank_id: str) -> list[str]:
    return [f"{d.document_id}: {d.error}" for d in await api.failed(bank_id)]


@dataclass(frozen=True)
class SearchCall:
    query_id: str
    text: str
    top_k: int


@dataclass(frozen=True)
class Searched:
    query_id: str
    results: list[SearchHit]
    client_ms: float


async def search_all(api: KnowledgeApi, bank_id: str, calls: Sequence[SearchCall]) -> list[Searched]:
    gate = asyncio.Semaphore(SEARCH_CONCURRENCY)

    async def one(call: SearchCall) -> Searched:
        # A remote embedder (a vLLM endpoint behind a tunnel) drops a connection now and then,
        # and the server reports it as 503. A few retries keep one blip from failing a run;
        # anything else, or a 503 that persists, still raises.
        for attempt in range(SEARCH_RETRIES + 1):
            async with gate:
                t = time.perf_counter()
                try:
                    hits = await api.search(bank_id, call.text, call.top_k)
                except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", 503)
                    if attempt == SEARCH_RETRIES or status < 500:
                        raise
                    hits = None
                ms = (time.perf_counter() - t) * 1000
            if hits is not None:
                return Searched(call.query_id, hits, ms)
            await asyncio.sleep(2.0 * (attempt + 1))
        raise AssertionError("unreachable")

    return list(await asyncio.gather(*(one(c) for c in calls)))


def percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


def latency(searched: Sequence[Searched]) -> KnowledgeLatency:
    client = [s.client_ms for s in searched]
    return KnowledgeLatency(
        searches=len(searched),
        concurrency=SEARCH_CONCURRENCY,
        client_p50_ms=round(percentile(client, 50), 1),
        client_p95_ms=round(percentile(client, 95), 1),
    )


# --- ranked scoring -----------------------------------------------------------------------


def ranked_documents(results: Sequence[SearchHit], page_ids: Mapping[str, Sequence[str]] | None = None) -> list[str]:
    """Each document once, at the rank of its best passage (max-sim). With ``page_ids``, a hit on
    page n of a multi-page document stands for that page's id instead."""

    def judged(r: SearchHit) -> str:
        pages = (page_ids or {}).get(r.document_id)
        return pages[r.page - 1] if pages and r.page else r.document_id

    return list(dict.fromkeys(judged(r) for r in results))


def score_ranked(queries: Sequence[RankedQuery], runs: dict[str, list[str]]) -> dict[str, float]:
    """nDCG@10 and recall@1/5/10/100 (x100) overall, plus nDCG@10 and R@1 per query group.

    With several groups (ViDoRe's six languages) the group mean of nDCG@10 is also given:
    that is MTEB's per-task score.
    """

    def mean(values: list[float]) -> float:
        return round(100 * statistics.fmean(values), 2) if values else 0.0

    def block(qs: Sequence[RankedQuery], prefix: str, ks: tuple[int, ...]) -> dict[str, float]:
        out = {f"{prefix}ndcg@10": mean([ndcg_at_k(runs[q.id], q.relevant, 10) for q in qs])}
        out |= {f"{prefix}recall@{k}": mean([recall_at_k(runs[q.id], q.relevant, k) for q in qs]) for k in ks}
        return out

    metrics = block(queries, "", (1, 5, 10, 100))
    if any(q.aspect_of for q in queries):  # BRIGHT-Pro's headline: aspect coverage in the top 25
        metrics["alpha_ndcg@25"] = mean(
            [alpha_ndcg_at_k(runs[q.id], q.aspect_of, q.aspect_weight, 25) for q in queries]
        )
    groups = sorted({q.group for q in queries})
    if len(groups) > 1:
        per = {g: block([q for q in queries if q.group == g], f"{g}:", (1,)) for g in groups}
        for values in per.values():
            metrics |= values
        metrics["ndcg@10_group_mean"] = round(statistics.fmean(per[g][f"{g}:ndcg@10"] for g in groups), 2)
    return metrics
