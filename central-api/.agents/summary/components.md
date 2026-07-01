# Components — CollabMind Control Plane

## collabmind-api — Governance Edge (TS, :3050)

**Role:** Single public-facing API. Verifies identity once, injects signed context, proxies to internal services. Also directly owns settings/secrets persistence.

| Component | File | Responsibility |
|-----------|------|----------------|
| Auth middleware | `src/auth.ts` | Authentik JWKS JWT (RS256) + `mem11_sk_` API-key lookup in Postgres. Attaches `req.context`. |
| Internal context | `src/internal-context.ts` | `signContext()` / `verifyContext()` — HMAC-SHA256 over `InternalContext` JSON. Format: `base64url(json).base64url(hmac)`. |
| Proxy | `src/proxy.ts` | `proxyTo()` / `proxyFetch()` — forwards request + injects `X-CM-Context` header. Returns 502 on network error. |
| Memory routes | `src/routes/memory.ts` | Wildcard proxy: all `/api/memory/**` → `memory-controller:3020/memory/**`. |
| Settings routes | `src/routes/settings.ts` | Direct Postgres: `platform_settings`, `secrets` tables. Encrypted secrets: `value_json` is nulled in responses. |
| Graph routes | `src/routes/graph.ts` | `/api/graph` → proxied upstream. |
| MCP routes | `src/routes/mcp.ts` | `/mcp` → proxied upstream. |

## collabmind-memory — Memory Backend (TS monorepo)

### memory-controller (:3020)

**Role:** Receives X-CM-Context from edge, runs write-gate, dispatches to adapters, unifies search results.

| Component | File | Responsibility |
|-----------|------|----------------|
| Auth middleware | `src/middleware/auth.ts` | Priority: (1) X-CM-Context HMAC verify, (2) API-key lookup, (3) JWT, (4) dev fallback. Attaches `req.context`. |
| Internal context | `src/internal-context.ts` | `contextFromHeader()` — verifier side. |
| Write-gate | `src/governance/write-gate.ts` | `runWriteGate(content)` → `redactSecrets` → `classifySensitivity` → `policyDecision`. Returns `GateResult`. |
| Search unifier | `src/search/unify.ts` | `unifySearch(perBackend, ctx, limit)` — RRF_K=60, metadata-first filter, confidentiality ceiling, content-hash dedup. Returns `RankedHit[]`. |
| Internal adapter | `src/adapters/internal.ts` | Reads/writes to core HSG via Postgres. |
| OpenMemory adapter | `src/adapters/openmemory.ts` | HTTP client to OpenMemory service. |
| MemLord adapter | `src/adapters/memlord.ts` | Read-only HTTP client to MemLord `/api-dev/*`. |
| Documents adapter | `src/adapters/documents.ts` | Delegates to documents service. |
| Memory routes | `src/routes/memory.ts` | CRUD + search endpoints under `/memory`. |
| Source-resources routes | `src/routes/source-resources.ts` | Source connector / document management endpoints. |

### packages/core

**Role:** Canonical memory engine — HSG architecture, temporal facts, embeddings.

| Component | File | Responsibility |
|-----------|------|----------------|
| HSG engine | `src/memory/hsg.ts` | Hierarchical Sectored Graph — 5 sectors (episodic, semantic, procedural, emotional, reflective), salience + time-decay. |
| Embeddings | `src/memory/embed.ts` | Multi-provider embedding (Ollama, pgvector, Bedrock stub). Embedding factory pattern. |
| Temporal store | `src/temporal/store.ts` | Subject-predicate-object facts with time validity. |
| Temporal query | `src/temporal/query.ts` | Fact retrieval with time filters. |
| DB layer | `src/core/db.ts` | Raw Postgres queries, no ORM. Connection pool. |
| Identity | `src/core/identity.ts` | Tenant/actor/source_app/project resolution. |
| Lifecycle | `src/core/lifecycle.ts` | Memory state transitions (active→paused→archived→deleted). |
| Audit | `src/core/audit.ts` | Audit event writes. Every operation logged. |
| Source registry | `src/connectors/source-resource-registry.ts` | Known source providers + their capabilities. |
| Vector stores | `src/memory/vector/postgres.ts`, `valkey.ts` | pgvector + Valkey (Redis-compatible) vector backends. |
| Migrations | `src/schema/00–15.sql` | Applied via `src/migrations/run.ts`. |

### packages/documents

**Role:** Document ingestion pipeline — parse → chunk → embed → index.

| Component | File | Responsibility |
|-----------|------|----------------|
| Document routes | `src/routes/documents.ts` | CRUD for source documents. |
| Ingestion jobs | `src/routes/ingestion-jobs.ts` | Job queue management. |
| Indexing routes | `src/routes/indexing.ts` | Trigger indexing, status checks. |
| Retrieval routes | `src/routes/retrieval.ts` | Semantic search over document chunks. |
| Collections routes | `src/routes/collections.ts` | Qdrant collection + alias management. |
| Chunker | `src/parsers/chunker.ts` | Text chunking with configurable strategy. |
| Text parser | `src/parsers/text.ts` | Plain text extraction. |
| S3 storage | `src/storage/s3.ts` | Object storage for raw documents. |
| Qdrant client | `src/vector/qdrant.ts` | Collection create/upsert/search. |
| Embedding factory | `src/embeddings/factory.ts` | Selects embedding provider by config. |
| DB — documents | `src/db/documents.ts` | Document/chunk persistence. |
| DB — retrieval | `src/db/retrieval.ts` | Retrieval trace writes. |

### packages/connectors

**Role:** Source connector framework.

| Component | File | Responsibility |
|-----------|------|----------------|
| Provider interface | `src/providers/interface.ts` | Abstract connector contract. |
| Context | `src/context.ts` | Connector execution context. |
| Types | `src/types.ts` | Shared connector types. |

### packages/mcp-server

MCP server exposing memory tools to MCP clients (Claude, Cursor, etc.). Entry: `src/`. See `README.md` for tool list and Claude Desktop config.

### packages/evaluation

Evaluation + correction pipeline. Schemas and failure-type registry documented in `EVALUATION_SCHEMA.md`, `FAILURE_TYPE_REGISTRY.md`, `CORRECTION_WORKFLOW.md`.

### packages/model-gateway

Model routing and provider registry. Documented in `MODEL_GATEWAY.md`, `MODEL_PROVIDER_REGISTRY.md`, `MODEL_ROUTE_POLICY.md`.

## central-api — Python Control-Plane Scaffold (:8000)

**Role:** Contract target. Defines the final Python service API. The TS stack is the running reference; this is the migration destination.

| Component | File | Responsibility |
|-----------|------|----------------|
| Auth context | `app/auth/context.py` | `RequestContext`, `get_context()`, `ContextDep`. Validates JWT or API key; dev fallback in non-prod. |
| Permissions | `app/auth/permissions.py` | `require_scope()`, `within_confidentiality()`. Confidentiality order: public→internal→private→sensitive→secret_blocked. |
| Adapter registry | `app/adapters/__init__.py` | `get_write_adapter()`, `get_memory_search_adapters()`, `get_code_adapter()`. MemLord only joins when `configured` (API key present). |
| MemLord adapter | `app/adapters/memlord.py` | Real HTTP client. `/api-dev/memories/search` (search), `/api-dev/memories` (paginated export). Requires `mlk_…` key. |
| Policy redactor | `app/policy/redactor.py` | `redact_secrets()` — 8 patterns. Returns `(content, count, kinds)`. |
| Policy rules | `app/policy/rules.py` | `run_write_gate()`. Enums: `Sensitivity`, `PolicyDecision`, `Status`. |
| Memory service | `app/memory/service.py` | Orchestration: write-gate → adapter write → audit. |
| RRF | `app/retrieval/rrf.py` | `reciprocal_rank_fusion()` — generic, type-parametric, `RRF_K=60`. |
| Context-pack | `app/retrieval/context_pack.py` | `build_context_pack()` — pure fn, fuses memory + code hits, deduplicates, cites. |
| Retrieval policy | `app/governance/retrieval_policy.py` | `evaluate_retrieval()` — fail-closed, Drive-ACL-aware. `Principal` dataclass. |
| Agent control | `app/governance/agent_control.py` | `decide(action)`, `request_action()`. Creates approval row for high-impact actions. |
| MCP tools | `app/mcp/tools.py` | 7 tools — all audit `mcp_tool_called` before acting. All require workspace membership. |
| Connector routes | `app/connectors/routes.py` | Google Drive OAuth flow, list connectors, manual sync. |
| Control plane routes | `app/controlplane/routes.py` | Users, workspaces, members CRUD. |
| Operator routes | `app/operator/routes.py` | Review queue, approve, reject. |
| Audit logger | `app/audit/logger.py` | In-memory stub. Target: `audit_events` table. |
| DB repositories | `app/db/repositories.py` | All SQL — no ORM, parameterized queries only. |

## collabmind-console — Operator GUI (Next.js, :3000)

**Role:** Operator-facing UI. Single source of truth for config, memory inspection, MCP tool browser, service status.

| Route | File | Content |
|-------|------|---------|
| `/dashboard` | `src/app/dashboard/page.tsx` | System overview |
| `/memory` | `src/app/memory/` | Memory explorer |
| `/mcp` | `src/app/mcp/` | MCP tool browser |
| `/settings` | `src/app/settings/` | Settings / secrets management |
| `/services` | `src/app/services/` | Service status |
| — | `src/lib/api.ts` | **Only** API client — all calls → `collabmind-api:3050`. No direct backend calls. |
