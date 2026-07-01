"""Google Drive connector orchestration (v0.1: manual sync only).

connect / disconnect / status / sync. Sync discovers files in the configured
folders, upserts source_documents, snapshots permissions, and **emits ingestion
jobs** (it does not embed). Every step writes a source_audit_events row.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.connectors.google_drive import audit_actions as A
from app.connectors.google_drive.client import DriveClient, FakeDriveClient
from app.connectors.google_drive.normalize import is_supported, normalize_file, normalize_permission
from app.connectors.google_drive.oauth import StoredToken, token_store
from app.db import repositories as repo

PROVIDER = "google_drive"


def default_drive_client(connector: dict) -> DriveClient:
    """Return a client for the connector. v0.1: a seeded fake unless a real client
    is injected by the caller. Real Google access is wired by passing a
    ``GoogleDriveClient`` built from the stored token.
    """
    seed = (connector.get("config") or {}).get("seed_files")
    seed_perms = (connector.get("config") or {}).get("seed_permissions")
    return FakeDriveClient(files=seed, permissions=seed_perms)


async def connect(session: AsyncSession, *, workspace_id: str, actor_id: str,
                  folder_ids: list[str], account_email: str | None = None,
                  refresh_token: str | None = None,
                  seed: dict | None = None) -> dict:
    """Establish a connection. ``seed`` lets dev/tests preload a fake Drive."""
    config: dict[str, Any] = {"folder_ids": folder_ids}
    if seed:
        config["seed_files"] = seed.get("files")
        config["seed_permissions"] = seed.get("permissions")
    connector = await repo.upsert_connector(
        session, workspace_id=workspace_id, provider=PROVIDER, status="connected",
        connected_by=actor_id, account_email=account_email, config=config,
    )
    if refresh_token:
        token_store.save(StoredToken(connector_id=connector["id"], refresh_token=refresh_token,
                                     account_email=account_email))
    await repo.write_audit(session, action=A.GOOGLE_DRIVE_CONNECTED, actor_id=actor_id,
                           workspace_id=workspace_id, source=PROVIDER,
                           metadata={"folder_ids": folder_ids, "account_email": account_email})
    return connector


async def disconnect(session: AsyncSession, *, workspace_id: str, actor_id: str) -> dict | None:
    connector = await repo.get_connector_by_provider(session, workspace_id=workspace_id,
                                                      provider=PROVIDER)
    if not connector:
        return None
    await repo.set_connector_status(session, connector["id"], "disconnected")
    token_store.delete(connector["id"])
    await repo.write_audit(session, action=A.GOOGLE_DRIVE_DISCONNECTED, actor_id=actor_id,
                           workspace_id=workspace_id, source=PROVIDER)
    return await repo.get_connector(session, connector["id"])


async def status(session: AsyncSession, *, workspace_id: str) -> dict:
    connector = await repo.get_connector_by_provider(session, workspace_id=workspace_id,
                                                      provider=PROVIDER)
    if not connector:
        return {"provider": PROVIDER, "status": "not_connected"}
    return {
        "provider": PROVIDER,
        "status": connector["status"],
        "account_email": connector.get("account_email"),
        "folder_ids": (connector.get("config") or {}).get("folder_ids", []),
        "has_token": token_store.has(connector["id"]),
        "connector_id": connector["id"],
    }


async def sync(session: AsyncSession, *, workspace_id: str, actor_id: str,
               client: DriveClient | None = None) -> dict:
    """Manual sync: discover → store metadata → snapshot permissions → queue jobs."""
    connector = await repo.get_connector_by_provider(session, workspace_id=workspace_id,
                                                      provider=PROVIDER)
    if not connector or connector["status"] != "connected":
        return {"error": "connector_not_connected"}

    drive = client or default_drive_client(connector)
    folder_ids = (connector.get("config") or {}).get("folder_ids", [])
    discovered = 0
    queued = 0
    skipped = 0

    for folder_id in folder_ids:
        for raw in await drive.list_files(folder_id):
            fields = normalize_file(raw)
            doc = await repo.upsert_document(
                session, connector_id=connector["id"], workspace_id=workspace_id,
                provider=PROVIDER, external_id=fields["external_id"], name=fields["name"],
                mime_type=fields["mime_type"], size=fields["size"],
                web_view_link=fields["web_view_link"], checksum=fields["checksum"],
                trashed=fields["trashed"], metadata=fields["metadata"],
                sync_status="discovered",
            )
            discovered += 1
            await repo.write_audit(session, action=A.FILE_DISCOVERED, actor_id=actor_id,
                                   workspace_id=workspace_id, source=PROVIDER,
                                   source_file_id=fields["external_id"],
                                   metadata={"name": fields["name"], "mime": fields["mime_type"]})

            perms = [normalize_permission(p) for p in await drive.list_permissions(
                fields["external_id"])]
            count = await repo.replace_document_permissions(session, doc["id"], perms)
            await repo.write_audit(session, action=A.FILE_PERMISSION_SNAPSHOT_CREATED,
                                   actor_id=actor_id, workspace_id=workspace_id, source=PROVIDER,
                                   source_file_id=fields["external_id"],
                                   metadata={"permission_count": count})

            if not is_supported(fields["mime_type"]):
                skipped += 1
                await repo.write_audit(session, action=A.FILE_SKIPPED, actor_id=actor_id,
                                       workspace_id=workspace_id, source=PROVIDER,
                                       source_file_id=fields["external_id"], status="skipped",
                                       metadata={"reason": "unsupported_mime",
                                                 "mime": fields["mime_type"]})
                continue

            await repo.create_job(
                session, workspace_id=workspace_id, connector_id=connector["id"],
                document_id=doc["id"], source=PROVIDER, external_id=fields["external_id"],
                mime_type=fields["mime_type"], operation="index", status="pending",
                metadata={"name": fields["name"]},
            )
            queued += 1
            await repo.write_audit(session, action=A.FILE_INGESTION_QUEUED, actor_id=actor_id,
                                   workspace_id=workspace_id, source=PROVIDER,
                                   source_file_id=fields["external_id"])

    stats = {"discovered": discovered, "queued": queued, "skipped": skipped}
    await repo.upsert_sync_state(session, connector_id=connector["id"], last_status="ok",
                                 stats=stats)
    return {"connector_id": connector["id"], **stats}
