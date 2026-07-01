# AGENTS.md — CollabMind Control Plane

> Navigation guide for AI agents. Start here, then pull specific files from `.agents/summary/` as needed.
> Full knowledge base index: `.agents/summary/index.md`

---

## System at a Glance

Four repos form the CollabMind memory control plane:

| Repo | Stack | Status | Port | Role |
|------|-------|--------|------|------|
| `central-api` | Python/FastAPI | **AI-002 Live** | 8000 | AI Gateway: provider registry, model inventory, route preview, router decisions |
| `collabmind-api` | TypeScript/Express | **Running** | 3050 | Public-facing governance edge |
| `collabmind-memory` | TypeScript monorepo | **Running** | 3020 | Memory backend (HSG, write-gate, RRF, docs) |
| `collabmind-console` | Next.js | **AI-002 Live** | 3000 | Operator GUI: Model Center + Router preview wired |

`collabmind-api` + `collabmind-memory` are the active system. `central-api` is the migration target — its governance logic (policy, RRF, context-pack) is real; its adapters and DB are stubs.

---

## Directory Map

```
central-api/
  app/
    main.py              Entry point (FastAPI, lifespan)
    config.py            Settings — env prefix CENTRAL_API_
    adapters/            Engine registry + BaseMemoryAdapter ABC
    auth/                RequestContext, scope checks, confidentiality ceiling
    policy/              Write-gate: redactor.py, classifier.py, rules.py
    memory/              routes, service (orchestration), schemas
    retrieval/           routes, rrf.py (RRF_K=60), context_pack.py
    governance/          retrieval_policy.py (fail-closed), agent_control.py
    mcp/tools.py         7 MCP tools
    connectors/          Google Drive connector (OAuth + manual sync)
    controlplane/        users, workspaces, members
    operator/            approval queue
    audit/               in-memory logger stub
    db/                  tables.py (11 SA Core tables), repositories.py
  .agents/summary/       Full documentation (start with index.md)

collabmind-api/
  src/
    index.ts             Express app, port 3050
    auth.ts              Authentik JWKS JWT + mem11_sk_ API key auth
    internal-context.ts  X-CM-Context HMAC-SHA256 sign/verify
    proxy.ts             Upstream proxy with signed context injection
    routes/              memory.ts (proxy), settings.ts (direct DB), graph.ts, mcp.ts, …

collabmind-memory/
  packages/
    memory-controller/src/
      middleware/auth.ts       X-CM-Context trust + JWT + API key
      governance/write-gate.ts redact → classify → decide
      search/unify.ts          RRF_K=60 + dedup + confidentiality filter
      adapters/                internal, openmemory, memlord (read-only), documents
      routes/                  memory.ts, source-resources.ts
    core/src/
      memory/hsg.ts            HSG memory engine (5 sectors, salience, decay)
      memory/embed.ts          Embedding factory (Ollama/pgvector/Valkey)
      temporal/                temporal facts store + query
      schema/00–15.sql         DB migrations
    documents/src/             Ingestion pipeline: chunker → S3 → Qdrant
    connectors/src/            Source connector framework (interface only)
    mcp-server/src/            MCP server
    evaluation/src/            Evaluation + correction
    model-gateway/src/         Model routing

collabmind-console/
  src/
    app/                       Next.js routes (dashboard, memory, mcp, settings, services)
    lib/api.ts                 Single API client → collabmind-api:3050 only
```

---

## Critical Patterns (must know before editing)

### Verify-Once-at-Edge
`collabmind-api` is the only token verifier. After auth it serializes `RequestContext`, signs it as `X-CM-Context: <base64url(json)>.<base64url(hmac)>`, and forwards it. `memory-controller` trusts this header. `INTERNAL_CONTEXT_SECRET` must match on both services.

### Write-Gate (both repos, identical logic)
Every write runs: `redactSecrets()` → `classifySensitivity()` → `policyDecision()`

- `secret` sensitivity → **reject** (not stored, even redacted)
- `restricted/private` → **quarantine** (stored, excluded from retrieval pending review)
- `public/internal` → **allow**

Python: `app/policy/rules.py` · TS: `memory-controller/src/governance/write-gate.ts`

### Fail-Closed Retrieval
`evaluate_retrieval()` denies on any unknown. Missing connector, missing snapshot, trashed document, or no matching Drive ACL all produce a deny + audit event. Never add "assume allowed" logic here.

### Tenant Isolation
`tenant_id` is always from `req.context`. Never read it from the request body or query string.

### Adapter Pattern
All backends implement `BaseMemoryAdapter` (Python) or the adapter interface (TS). 

**Primary:** `HindsightAdapter` (read-write) — full CRUD support for the `collabmind-control-plane` bank with 15 directives and 8 mental models.

**Legacy:** `MemlordAdapter.read_only = true` — federation searches MemLord but never writes to it. Used for backward compatibility during migration.

### RRF Constant
`RRF_K = 60` in both repos. Change both if you change either.

### Secrets in Settings
`collabmind-api/src/routes/settings.ts`: encrypted rows have `value_json` nulled in all API responses. Never change this — raw secrets must not leave the API.

---

## Auth Quick Reference

| Credential | Format | Verified by |
|-----------|--------|------------|
| Human JWT | `Authorization: Bearer eyJ…` | Authentik JWKS RS256 |
| Agent API key | `Authorization: Bearer mem11_sk_…` or `X-Api-Key: mem11_sk_…` | SHA-256 lookup in `api_keys` table |
| Internal hop | `x-cm-context: <payload>.<sig>` | HMAC-SHA256 with `INTERNAL_CONTEXT_SECRET` |

Dev fallback (non-production only): default tenant `00000000-0000-0000-0000-000000000001`, actor `00000000-0000-0000-0000-000000000002`.

---

## MCP Tools (v0.1)

Allowed: `list_workspaces`, `list_connected_sources`, `list_source_documents`, `sync_source`, `get_source_audit`, `list_ingestion_jobs`, `search_governed_documents`

High-impact (require operator approval): `bulk_sync`, `source_deletion`, `permission_override`, `connector_disconnect`, `retrieval_across_restricted_sources`

All other actions: denied (fail-closed).

---

## Known Gaps (read before changing)

- **`collabmind-memory/NO_TOUCH_FILES.md`** — lists files that must not be modified without reading first.
- **`collabmind-memory/DO_NOT_TOUCH_TUNNEL.md`** — Cloudflare tunnel constraints.
- **`central-api` DB schema conflict** — `app/models/__init__.py` (SQLModel) and `app/db/tables.py` (SA Core) overlap. Resolve before running migrations.
- **Sensitivity level naming** differs between Python (`private/sensitive/secret_blocked`) and TS (`restricted/secret`). See `.agents/summary/review_notes.md` §C1.
- **`central-api` MCP tools** (`app/mcp/tools.py`) are not wired to an HTTP route in the scaffold.

Full gap list: `.agents/summary/review_notes.md`

---

## Custom Instructions

### AI Gateway (AI-001 & AI-002)
- **Provider Registry**: `app/ai/providers.py` — reads from config defaults (LOCALAI_URL, OLLAMA_URL, etc.) + DB provider_endpoints table
- **Adapters**: `app/ai/localai.py`, `app/ai/ollama.py`, `app/ai/litellm.py` — all implement `list_models()` + `health()`
- **Route Preview**: `app/ai/router_preview.py` — stateless decision logic; privacy=internal prefers local; fallback_chain built from policy
- **Routes**: `app/ai/routes.py` — all `/api/ai/*` and `/api/router/decisions` endpoints live
- **Console Wiring** (AI-002): Model Center health refresh per provider + Router preview 7-control panel (request_type, mode, privacy_level, prefer_local, requires_*) with live result display

<!-- This section is for human and agent-maintained operational knowledge.
     Add repo-specific conventions, gotchas, and workflow rules here.
     This section is preserved exactly as-is when re-running codebase-summary. -->
