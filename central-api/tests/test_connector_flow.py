"""End-to-end Google Drive connector flow (isolated SQLite, fake Drive)."""

import asyncio

from app.connectors.google_drive import service as gd
from app.connectors.google_drive.oauth import token_store
from app.db import repositories as repo
from app.db.engine import init_models, session_scope
from app.ingestion import worker

SEED = {
    "files": {
        "fld1": [
            {"id": "f1", "name": "Plan", "mimeType": "application/vnd.google-apps.document",
             "owners": [{"emailAddress": "a@x.com"}], "webViewLink": "http://v/f1",
             "parents": ["fld1"]},
            {"id": "f2", "name": "image.png", "mimeType": "image/png", "parents": ["fld1"]},
        ]
    },
    "permissions": {
        "f1": [{"id": "p1", "type": "user", "role": "reader", "emailAddress": "a@x.com"}],
        "f2": [],
    },
}


async def _bootstrap(session):
    user = await repo.create_user(session, email="op@x.com", is_operator=True)
    ws = await repo.create_workspace(session, name="W", owner_id=user["id"])
    return user, ws


def test_connect_sync_process_disconnect():
    async def scenario():
        await init_models()
        async with session_scope() as s:
            user, ws = await _bootstrap(s)
        async with session_scope() as s:
            await gd.connect(s, workspace_id=ws["id"], actor_id=user["id"], folder_ids=["fld1"],
                             account_email="op@x.com", refresh_token="rt", seed=SEED)
        async with session_scope() as s:
            sync_result = await gd.sync(s, workspace_id=ws["id"], actor_id=user["id"])
        async with session_scope() as s:
            docs = await repo.list_documents(s, workspace_id=ws["id"])
            jobs = await repo.list_jobs(s, workspace_id=ws["id"])
            perms_f1 = next(d for d in docs if d["external_id"] == "f1")
            perms = await repo.list_document_permissions(s, perms_f1["id"])
            audit = await repo.list_audit(s, workspace_id=ws["id"])
        async with session_scope() as s:
            wr = await worker.process_pending(s, workspace_id=ws["id"])
        async with session_scope() as s:
            jobs_after = await repo.list_jobs(s, workspace_id=ws["id"])
            docs_after = await repo.list_documents(s, workspace_id=ws["id"])
        connector_id = (await _connector_id(ws["id"]))
        async with session_scope() as s:
            dc = await gd.disconnect(s, workspace_id=ws["id"], actor_id=user["id"])
            status = await gd.status(s, workspace_id=ws["id"])
        return {
            "sync": sync_result, "docs": docs, "jobs": jobs, "perms": perms, "audit": audit,
            "wr": wr, "jobs_after": jobs_after, "docs_after": docs_after, "dc": dc,
            "status": status, "connector_id": connector_id,
        }

    async def _connector_id(ws_id):
        async with session_scope() as s:
            c = await repo.get_connector_by_provider(s, workspace_id=ws_id, provider="google_drive")
            return c["id"]

    r = asyncio.run(scenario())
    # discovery: 2 found, 1 queued (gdoc), 1 skipped (png, unsupported — not failed)
    assert r["sync"]["discovered"] == 2
    assert r["sync"]["queued"] == 1
    assert r["sync"]["skipped"] == 1
    assert len(r["docs"]) == 2
    assert [j["external_id"] for j in r["jobs"]] == ["f1"]
    assert len(r["perms"]) == 1 and r["perms"][0]["email_address"] == "a@x.com"
    actions = {e["action"] for e in r["audit"]}
    assert {"google_drive_connected", "file_discovered", "file_permission_snapshot_created",
            "file_ingestion_queued", "file_skipped"} <= actions
    # worker drains the pending job to indexed
    assert r["wr"]["processed"] == 1 and r["wr"]["results"][0]["status"] == "indexed"
    assert r["jobs_after"][0]["status"] == "indexed"
    assert any(d["external_id"] == "f1" and d["sync_status"] == "indexed" for d in r["docs_after"])
    # disconnect clears status and token
    assert r["dc"]["status"] == "disconnected"
    assert r["status"]["status"] == "disconnected"
    assert token_store.has(r["connector_id"]) is False


def test_sync_refused_when_not_connected():
    async def scenario():
        await init_models()
        async with session_scope() as s:
            user, ws = await _bootstrap(s)
        async with session_scope() as s:
            return await gd.sync(s, workspace_id=ws["id"], actor_id=user["id"])

    assert asyncio.run(scenario()) == {"error": "connector_not_connected"}
