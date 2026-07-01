# Index — CollabMind Control Plane Documentation

## How to use this knowledge base

This index is the primary file to include in context. It contains sufficient metadata to determine which file to read for any question. Only load additional files when you need implementation-level detail.

**Quick lookup:**
- "Where is X implemented?" → `components.md`
- "What does the auth flow look like?" → `interfaces.md` (Auth Flow Summary section)
- "What is the write-gate?" → `architecture.md` (Core Architectural Patterns §2) or `components.md` (write-gate row)
- "What DB tables exist?" → `data_models.md`
- "How does agent action approval work?" → `workflows.md` §4
- "What are the MCP tools?" → `interfaces.md` (MCP Tools section)
- "What external services are needed?" → `dependencies.md`
- "What is stub vs real?" → `codebase_info.md` (central-api layout) or `architecture.md` (Two Parallel Implementations table)

---

## Files

### `codebase_info.md`
Full directory layouts for all 4 repos, key env vars, and which repo is running vs scaffold. Read first for orientation. Contains: central-api layout, collabmind-api layout, collabmind-memory monorepo layout, collabmind-console layout, env var table.

### `architecture.md`
System overview diagram, the two-implementation model (TS running / Python scaffold), and the 6 core patterns: verify-once-at-edge, write-gate pipeline, RRF fusion, fail-closed retrieval governance, adapter/interface pattern, confidentiality ceiling. Also: DB schema summary and agent action control table.

### `components.md`
Per-component responsibility tables for all 4 repos. The most useful file for "where does X live?" questions. Covers: collabmind-api (auth, proxy, routes), memory-controller (write-gate, unifier, adapters), packages/core (HSG, embeddings, temporal), packages/documents (ingestion, Qdrant, S3), central-api modules, collabmind-console routes.

### `interfaces.md`
All HTTP API surfaces: collabmind-api routes (:3050), central-api contract routes (:8000), MCP tool signatures, X-CM-Context header format, adapter interface (Python + TS), auth flow diagram.

### `data_models.md`
`RequestContext`, `AdapterHit`, `GovernanceResult`/`GateResult`, `ContextPack`, `RankedHit`, `AuditEvent`. Full DB table schemas for central-api (SQLAlchemy Core). Identity hierarchy. Sensitivity level table.

### `workflows.md`
Step-by-step sequence/flow diagrams for: memory write, memory search + context-pack build, document ingestion pipeline, agent action request (requires_approval), retrieval governance check (per-document), auth resolution order, operator review queue.

### `dependencies.md`
Python and TS dependency lists, external services (Authentik, Postgres, Qdrant, Ollama, Valkey, S3, MemLord, OpenMemory, Google Drive), MemLord API contract details, Authentik token claims.

### `review_notes.md`
Consistency and completeness findings. Identified gaps. Read before making any cross-repo changes.

---

## Key Facts (inline summary)

**Repos and their role:**
- `central-api` (:8000) — Python/FastAPI. **Scaffold/contract only.** Adapters are stubs. No real DB yet. This is the migration target.
- `collabmind-api` (:3050) — TypeScript/Express. **Running.** Governance edge proxy. Verify-once-at-edge, X-CM-Context signing.
- `collabmind-memory` (:3020 memory-controller) — TypeScript monorepo. **Running.** HSG engine, write-gate, RRF search, Qdrant docs, evaluation, model gateway.
- `collabmind-console` (:3000) — Next.js. **Running.** Operator GUI. Single API client → collabmind-api only.

**Auth:**
- Human users: Authentik OIDC → JWT RS256 (`Authorization: Bearer eyJ…`)
- Agents/machines: API key with `mem11_sk_` prefix (`X-Api-Key` or `Bearer mem11_sk_…`)
- Internal hops: `x-cm-context` HMAC-SHA256 header (requires `INTERNAL_CONTEXT_SECRET` set on both edge and controller)

**Governance non-negotiables:**
- Write-gate runs before every store. `secret`-sensitivity content is **rejected** (not stored).
- Retrieval is fail-closed: unknown permission = deny.
- `tenant_id` is always from `req.context`, never from request body.
- Encrypted settings: `value_json` is nulled in API responses.
- `MemlordAdapter.read_only = true` — the federation never writes to MemLord.

**RRF constant:** `RRF_K = 60` in both repos. Do not change without updating both.

**MCP tools (v0.1 allowed):** list_workspaces, list_connected_sources, list_source_documents, sync_source, get_source_audit, list_ingestion_jobs, search_governed_documents.

**High-impact actions (require operator approval):** bulk_sync, source_deletion, permission_override, connector_disconnect, retrieval_across_restricted_sources.
