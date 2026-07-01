"""Ingestion worker — drains pending jobs through the governed pipeline.

State machine: pending → downloading → parsing → chunking → embedding → indexed
(or → skipped / failed). Each transition writes an audit event and advances the
document's sync_status. The embed/index step is delegated to the memory/vector
layer via ``embed_and_index``; v0.1 uses a no-op stub so nothing is written to a
real vector store, but the governed flow and audit trail are exercised.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.connectors.google_drive import audit_actions as A
from app.connectors.google_drive.normalize import is_supported
from app.db import repositories as repo
from app.db import tables as t
from app.db.ids import utcnow

# Delegated embed/index hook. Real impl will call the embedding + vector adapter;
# the worker only governs and audits. Returns a chunk count.
EmbedIndexFn = Callable[[dict], Awaitable[int]]


async def _noop_embed_index(job: dict) -> int:  # pragma: no cover - trivial
    return 0


async def process_job(session: AsyncSession, job_id: str, *,
                      embed_and_index: EmbedIndexFn = _noop_embed_index) -> dict:
    job = await repo.get_job(session, job_id)
    if not job:
        return {"error": "job_not_found"}
    if job["status"] != "pending":
        return {"job_id": job_id, "status": job["status"], "skipped": True}

    ws = job["workspace_id"]
    fid = job["external_id"]

    async def audit(action: str, status: str = "success", **meta) -> None:
        await repo.write_audit(session, action=action, workspace_id=ws, source=job["source"],
                               source_file_id=fid, status=status, metadata=meta)

    async def advance(doc_status: str, job_status: str) -> None:
        await repo.update_job_status(session, job_id, job_status)
        if job.get("document_id"):
            await session.execute(sa.update(t.source_documents)
                                  .where(t.source_documents.c.id == job["document_id"])
                                  .values(sync_status=doc_status, updated_at=utcnow()))

    if not is_supported(job.get("mime_type")):
        await repo.update_job_status(session, job_id, "skipped")
        await audit(A.FILE_SKIPPED, status="skipped", reason="unsupported_mime")
        return {"job_id": job_id, "status": "skipped"}

    try:
        await advance("downloading", "downloading")
        await audit(A.FILE_DOWNLOADED)

        await advance("parsing", "parsing")
        await audit(A.FILE_PARSED)

        await advance("chunking", "chunking")
        await audit(A.FILE_CHUNKED)

        await advance("embedding", "embedding")
        chunks = await embed_and_index(job)
        await audit(A.FILE_EMBEDDED, chunks=chunks)

        await advance("indexed", "indexed")
        await audit(A.FILE_INDEXED, chunks=chunks)
        return {"job_id": job_id, "status": "indexed", "chunks": chunks}
    except Exception as exc:  # noqa: BLE001 - record failure, never crash the drain
        await repo.update_job_status(session, job_id, "failed", error=str(exc))
        await audit(A.FILE_INDEX_FAILED, status="failed", error=str(exc))
        return {"job_id": job_id, "status": "failed", "error": str(exc)}


async def process_pending(session: AsyncSession, *, workspace_id: str,
                          embed_and_index: EmbedIndexFn = _noop_embed_index) -> dict:
    jobs = await repo.list_jobs(session, workspace_id=workspace_id, status="pending")
    results = [await process_job(session, j["id"], embed_and_index=embed_and_index) for j in jobs]
    return {"processed": len(results), "results": results}
