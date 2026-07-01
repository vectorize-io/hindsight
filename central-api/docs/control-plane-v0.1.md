# Control Plane v0.1 — Central API + Operator GUI + Google Drive Connector

The first **governed control layer** for CollabMind. It proves the path
`controlled source connection → metadata capture → permission snapshot → manual ingestion →
operator visibility → audit trail → MCP-safe access` — **without** chat/retrieval over Drive yet.

## Core rule

MCP tools, agents, and the GUI **never** access Google Drive, the vector DB, or the database
directly. The only path is:

```
Operator GUI / MCP Agent / External Agent
        ↓
CollabMind Central API / Router        ← the control plane (this service)
        ↓
Google Drive Connector (read-only)
        ↓
Ingestion Queue → Worker (parse/chunk/embed/index — embed delegated/stubbed in v0.1)
        ↓
Control-plane DB (source of truth) + Governed Retrieval
```

## Persistence

Control-plane DB is the **source of truth** for ownership, permissions, source metadata, sync
status, audit, and operator governance — Drive state never lives only in vector payloads.

- **Stack:** SQLAlchemy Core (no ORM relationships) + Alembic; async (`asyncpg` for Postgres,
  `aiosqlite` for dev/test). Portable column types — the schema runs unchanged on both.
- **Tables** (`app/db/tables.py`): `users`, `workspaces`, `workspace_members`, `source_connectors`,
  `source_documents`, `source_document_permissions`, `source_ingestion_jobs`,
  `source_audit_events`, `source_sync_state`, `agent_activity`, `operator_approvals`,
  `memory_source_mappings`.
- **Migrations:** `alembic upgrade head` (baseline `0001` builds the schema from metadata). Dev
  bootstraps via the app lifespan (`init_models`).

## Endpoints (Central API)

| Area | Endpoints |
|------|-----------|
| Identity / workspaces | `GET /me`, `GET/POST /workspaces` |
| Documents | `GET /source-documents`, `POST /source-documents/{id}/disable` |
| Jobs / audit / agents | `GET /ingestion-jobs`, `GET /audit-events`, `GET/POST /agent-activity`, `POST /ingestion/process` |
| Connectors | `GET /connectors`, `GET /connectors/{id}/status` |
| Google Drive | `…/google-drive/{oauth-config, connect, disconnect, status, sync, files, files/{id}, files/{id}/permissions, audit}` |

All workspace-scoped routes check membership against the DB (never trust the request body).

## Google Drive connector (read-only)

- **Scopes:** `drive.metadata.readonly`, `drive.readonly` **only**. Write/delete/admin scopes are
  rejected by `oauth.validate_scopes`. No write-back, no deletion from Drive.
- **Sync (manual, v0.1):** discover files in configured folder IDs → upsert `source_documents` →
  snapshot permissions → emit **ingestion jobs** (the connector never embeds directly) → audit each
  step. Unsupported MIME types are **skipped**, not failed.
- **Export-before-ingest:** Google-native types map to export formats (Docs→text, Sheets→CSV,
  Slides→text); PDF/DOCX/TXT/MD/CSV ingest as-is.
- **Client:** `FakeDriveClient` (dev/tests, in-memory) vs `GoogleDriveClient` (real, lazily imports
  the optional `gdrive` extra).

## Governance

- **Retrieval fails closed** (`app/governance/retrieval_policy.py`): unknown permission, missing
  metadata, disabled document, inactive connector, or non-membership → **deny**. A chunk is
  retrievable only if the principal is permitted on the original Drive file (user/group/domain/anyone,
  read role, non-expired).
- **Agent actions** (`agent_control.py`): high-impact actions (bulk_sync, source_deletion,
  permission_override, connector_disconnect, retrieval_across_restricted_sources) →
  `requires_approval`; unknown → `denied`. Every request is recorded as `agent_activity` + audited.
- **MCP tools** (`app/mcp/tools.py`): `list_workspaces`, `list_connected_sources`,
  `list_source_documents`, `sync_source`, `get_source_audit`, `list_ingestion_jobs`,
  `search_governed_documents`. Each routes through the service/repository layer, enforces
  membership, and writes a `mcp_tool_called` audit event. No path to Drive/Qdrant/DB/tokens.
- **Audit:** canonical action vocabulary in `app/connectors/google_drive/audit_actions.py`; every
  important operation writes a `source_audit_events` row.

## Secrets policy

No real client secret, access token, or refresh token in the repo, logs, or tests. Config comes
from the environment (`.env`, see `.env.example`). `oauth.redact()` masks token-bearing fields as
`[REDACTED]`. Token persistence is modeled by `TokenStore` (in-process for v0.1; an encrypted
backend later).

## Operator GUI

`operator-ui/` (Next.js). Calls **only** the Central API via the single `lib/api.ts` client. Pages:
Login, Workspaces, Connectors, Google Drive Connection, Indexed Files, Ingestion Queue, Audit Log,
Agent Activity, Permissions View.

## Verify

```bash
# Backend
pip install -e '.[dev]'
ruff check app tests && pytest -q          # 61 tests, mock-safe, no network/tokens
alembic upgrade head                        # builds all tables on a scratch URL
uvicorn app.main:app --reload               # dev SQLite by default

# GUI
cd operator-ui && npm install && npm run build
```

## Out of scope (v0.1)

Chat/RAG over Drive, real embeddings/vectors, webhooks/background sync, Drive write/delete,
multi-IdP login, Authentik wiring, Nginx Proxy Manager.
