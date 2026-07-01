# CollabMind System Topology — 2026-07-01

> Canonical reference: every service, adapter, panel, API, port, URL, env var, and route.
> Source of truth for all agent alignment. Store to Memlord workspace `collabmind`.

---

## 1. Service Map (Active Processes)

| # | Service | Local Port | Public URL | Stack | Process | Status |
|---|---------|-----------|------------|-------|---------|--------|
| S1 | **Hindsight API** | `:8888` | `https://hindsight-api.collabmind.dev` | Python/FastAPI + Uvicorn | `python3.1 *:8888` | ✅ Healthy |
| S2 | **Hindsight API (2nd)** | `:8899` | — | Python/FastAPI | `Python *:8899` | ✅ Healthy |
| S3 | **Central API (Cognitive Router)** | `:8000` | — | Python/FastAPI + SQLite | `python3.1 127.0.0.1:8000` | ✅ Running |
| S4 | **Cockpit (Operator Panel)** | `:8999` | `https://cockpit.collabmind.dev` | Next.js 16 (production) | `next-server` | ✅ Running |
| S5 | **Main Control Plane** | `:9999` | — | Next.js 16 | `next-server` | ✅ Running |
| S6 | **Embedded Control Plane** | `:10000` | — | Next.js 16 | `next-server` | ✅ Running |
| S7 | **Memlord MCP** | `:8005` | — | Python/Docker | `memlord-service` (Docker) | ✅ Healthy |
| S8 | **Grafana LGTM** | `:3000` | — | Grafana + Loki + Tempo + Mimir | `hindsight-monitoring` (Docker) | ✅ Healthy |
| S9 | **Langfuse (LLM Obs)** | `:3002` | — | Next.js (Langfuse v2.95.11) | `collabmind-llm-obs` (Docker) | ✅ Running |
| S10 | **Jaeger Tracing** | `:16686` | — | Jaeger UI | `collabmind-traces` (Docker) | ✅ Running |
| S11 | **Ollama Embeddings** | `:11434` | — | Ollama (bare metal) | `ollama 127.0.0.1:11434` | ✅ Running |
| S12 | **Ollama LLM** | `:11435` | — | Ollama (Docker) | `ollama-llm` (Docker) | ✅ Running |
| S13 | **LM Studio** | `:1234` | — | LM Studio (192.168.1.144) | `LM Studio` | ✅ Running |
| S14 | **TEI Reranker** | `:8081` | — | HuggingFace TEI | `tei` (Docker) | ✅ Running |
| S15 | **Memlord PostgreSQL** | `:5435` | — | PostgreSQL | `memlord-postgres` (Docker) | ✅ Healthy |
| S16 | **VectorAdmin** | `:3006` | — | Next.js | `vector-admin` (Docker) | ✅ Healthy |
| S17 | **VectorAdmin PostgreSQL** | `:5433` | — | PostgreSQL | `vectoradmin-postgres-1` (Docker) | ✅ Healthy |
| S18 | **Docker Model Runner Proxy** | `:6000` | — | Docker Desktop | `docker-model-runner-proxy` (Docker) | ✅ Running |
| S19 | **Memlord Qdrant** | `:6233/6234` | — | Qdrant | `memlord-qdrant` (Docker) | ✅ Running |
| S20 | **FreeLLM API** | `:3001` | — | — | `freellm-api` (Docker) | ✅ Running |

---

## 2. Cockpit Operator Panels

### Navigation Registry: `src/lib/operator-nav-types.ts`

| Panel ID | Route | Icon | Page Status |
|----------|-------|------|------------|
| `cockpit` | `/cockpit` | `LayoutDashboard` | ✅ Built — overview/dashboard |
| `chat` | `/chat` | `MessageSquare` | ✅ Built — AI Chat Assistant |
| `runs` | `/runs` | `Timeline` | ✅ Built — Run timeline |
| `agents` | `/agents` | `Bot` | ✅ Built — Agent registry |
| `memories` | `/memories` | `Brain` | ✅ Built — Memory console |
| `vector` | `/vector` | `Database` | ✅ Built — Vector explorer |
| `evaluation` | `/evaluation` | `FlaskConical` | ✅ Built — Eval lab |
| `directives` | `/directives` | `ScrollText` | ✅ Built — Directives |
| `backplane` | `/backplane` | `Network` | ✅ Built — System map |
| `tools` | `/tools` | `Wrench` | ✅ Built — Tool registry |
| `config` | `/config` | `KeyRound` | ✅ Built — Secrets/config |
| `audit` | `/audit` | `Shield` | ✅ Built — Audit & provenance |
| `settings` | `/settings` | `Settings` | ✅ Built — System settings |
| `constitution` | `/constitution` | `BookOpen` | ✅ Built — Prime directives |
| `router` | `/router` | `GitCompare` | ✅ Built — Intelligent router |
| `api-center` | `/api-center` | `FileJson` | ✅ Built — Service catalog |
| `voice` | `/voice` | `Mic` | ❌ Not implemented |
| `connectors` | `/connectors` | `Plug` | ✅ Built — Data sources |
| `traces` | *(pending)* | `Activity` | ❌ **Package A** — Langfuse traces |
| `monitoring` | *(pending)* | `BarChart3` | ❌ **Package B** — Monitoring/Graphs |

**Page directory**: `src/app/[locale]/` — each panel has a corresponding directory with `page.tsx`

### Cockpit API Proxy Routes: `src/app/api/`

| Route | Purpose |
|-------|---------|
| `/api/auth/*` | Authentication |
| `/api/banks` | Bank listing |
| `/api/chat` | AI Chat API |
| `/api/chunks` | Document chunks |
| `/api/documents` | Document management |
| `/api/entities` | Entity management |
| `/api/extract` | Memory extraction |
| `/api/files` | File operations |
| `/api/graph` | Knowledge graph |
| `/api/health` | Health checks |
| `/api/list` | List operations |
| `/api/memories` | Memory operations |
| `/api/operations` | Async operation tracking |
| `/api/profile` | Bank profile |
| `/api/recall` | Memory recall |
| `/api/reflect` | Memory reflect |
| `/api/stats` | Statistics |
| `/api/system` | System/service management |
| `/api/vector-admin` | Vector admin proxy |
| `/api/version` | Version info |

### Key ENV for Cockpit

```bash
HINDSIGHT_CP_DATAPLANE_API_URL=https://hindsight-api.collabmind.dev  # REST API base
HINDSIGHT_CP_DATAPLANE_API_KEY=your-collabminds-api-key
HINDSIGHT_CP_ACCESS_KEY=your-collabminds-access-key
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3002  # Langfuse local
NEXT_PUBLIC_BRAND_NAME=CollabMinds
```

---

## 3. Central API — Cognitive Router

### Port: `:8000` | Stack: Python/FastAPI + SQLite

### ENV Configuration

```bash
CENTRAL_API_ENV=development
CENTRAL_API_DEBUG=true
CENTRAL_API_DATABASE_URL=                     # Empty → SQLite default
CENTRAL_API_MEMLORD_URL=http://localhost:8005  # Memlord MCP HTTP
CENTRAL_API_MEMLORD_API_KEY=                   # Not set yet
CENTRAL_API_HINDSIGHT_URL=http://localhost:8888/mcp  # ⚠️ Wrong — should be REST base
CENTRAL_API_HINDSIGHT_API_KEY=                 # Not set yet
CENTRAL_API_HINDSIGHT_BANK_ID=opencode
```

### Route Map (60+)

| Route Group | Base Path | Purpose |
|------------|-----------|---------|
| AI | `/api/ai/*` | Provider registry, model inventory, chat, router preview |
| Router | `/api/router/decisions` | Router decisions |
| Memory | `/api/memory/*` | Search, route, context-pack, governance |
| Retrieval | `/api/retrieval/*` | Code/document search |
| Governance | `/api/gov/*` | Approval queue, policy check |
| Operator | `/api/operator/*` | Approve/reject/review |
| Connectors | `/connectors/*` | Google Drive connector |
| MCP | `/mcp/tools` | MCP tool registry |
| Health | `/health`, `/api/health/*` | Service health, engine health |
| Dashboard | `/api/dashboard/*` | Agent activity, workspaces |
| Audit | `/api/audit/*` | Audit events |
| Control Plane | `/workspaces`, `/me`, `/source-documents` | User/workspace management |

### Adapter Registry: `app/adapters/__init__.py`

| Adapter | File | Backend | Read/Write | Configured? | Notes |
|---------|------|---------|------------|-------------|-------|
| `InternalAdapter` | `internal.py` | `internal` | RW | ✅ Always | Returns stub data |
| `MemlordAdapter` | `memlord.py` | `memlord` | Read-only | ⚠️ Only if API key set | Real Memlord MCP API calls |
| `OpenMemoryAdapter` | `openmemory.py` | `openmemory` | Read-only | ✅ Always | Returns stub data |
| `CodeRAGAdapter` | `coderag.py` | `coderag` | Read-only | ✅ Always | Returns stub data |
| **`HindsightAdapter`** | `hindsight.py` | `hindsight` | RW | ❌ Not registered | Calls fake MCP paths ⚠️ |

### Known Central API Gaps

| Gap | Impact | Fix |
|-----|--------|-----|
| `HindsightAdapter` not registered | No real Hindsight reads/writes | Add to `__init__.py` |
| Adapter calls fake MCP REST paths | `/mcp/hindsight/retrieve_memory` — doesn't exist | Use real REST endpoints |
| DB schema conflict (SQLModel vs SA Core) | Migration issues, testing false positives | Resolve models/__init__.py |
| MCP tools not wired to HTTP | `/mcp/tools` not serving | Wire `mcp/tools.py` to routes |
| `CENTRAL_API_HINDSIGHT_URL` has `/mcp` suffix | Wrong base URL | Change to `http://localhost:8888` |
| No Memlord API key set | MemlordAdapter not active | Add `CENTRAL_API_MEMLORD_API_KEY` |

---

## 4. Hindsight REST API

### Port: `:8888` | Base path: `/v1/default/`

### Key Endpoints for Adapter Wiring

| Method | Path | Adapter Method | Purpose |
|--------|------|---------------|---------|
| `GET` | `/v1/default/banks` | — | List all banks |
| `GET` | `/v1/default/banks/{bank_id}` | `health()` | Get bank profile |
| `POST` | `/v1/default/banks/{bank_id}/memories/recall` | `search()` | Hybrid semantic + full-text search |
| `POST` | `/v1/default/banks/{bank_id}/memories` | `store()` | Create new memory (sync_retain) |
| `PATCH` | `/v1/default/banks/{bank_id}/memories/{memory_id}` | `update()` | Update existing memory |
| `DELETE` | `/v1/default/banks/{bank_id}/memories/{memory_id}` | `delete()` | Delete a memory |
| `GET` | `/v1/default/banks/{bank_id}/memories/list` | `export()` | List/paginate memories |
| `GET` | `/v1/default/banks/{bank_id}/stats` | — | Bank statistics |
| `GET` | `/v1/default/banks/{bank_id}/llm-requests/stats` | — | LLM request metrics |
| `POST` | `/v1/default/banks/{bank_id}/reflect` | — | Memory reflection |
| `GET` | `/health` | — | Health check |

### MCP SSE Endpoint

```bash
# SSE transport for MCP clients (Claude, ChatGPT, etc.)
# NOT a REST endpoint — uses JSON-RPC over SSE
SSE: http://localhost:8888/mcp
Messages: POST http://localhost:8888/mcp/message
```

### ENV (relevant)

```bash
HINDSIGHT_API_LLM_PROVIDER=lmstudio
HINDSIGHT_API_LLM_BASE_URL=http://192.168.1.144:1234/v1
HINDSIGHT_API_LLM_MODEL=google/gemma-4-e4b
HINDSIGHT_API_LLM_MAX_CONCURRENT=2
HINDSIGHT_API_LLM_TIMEOUT=600
HINDSIGHT_API_EMBEDDINGS_PROVIDER=litellm-sdk
HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_API_BASE=http://localhost:11434
HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_MODEL=ollama/nomic-embed-text
HINDSIGHT_API_DATABASE_URL=postgresql://oliververmeulen@localhost:5432/hindsight17
HINDSIGHT_API_RERANKER_PROVIDER=rrf
```

---

## 5. Memlord MCP

### Port: `:8005` | Stack: Python/Docker + PostgreSQL + Qdrant

### REST API Surface (API-key only)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Health check |
| `POST` | `/api-dev/memories/search` | Hybrid BM25+vector search (RRF) |
| `POST` | `/api-dev/memories` | Paginated memory listing |
| `POST` | `/api-dev/memories/store` | Store a memory |
| `POST` | `/workspaces` | Workspace management |

### ENV

```bash
# Container: memlord-service
# Port mapping: 8005:8000
# Database: PostgreSQL :5435, Qdrant :6233/:6234
```

### Data
- **Workspace `collabmind`**: 128 memories (facts, preferences, decisions)
- **Workspace `personal`**: User-specific memories

---

## 6. Langfuse (LLM Observability)

### Port: `:3002` | Version: 2.95.11

### Connection

```bash
# SDK configuration (used in cockpit server)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3002

# Langfuse API (for direct REST calls)
GET http://localhost:3002/api/public/health  → {"status":"OK"}
GET http://localhost:3002/api/public/traces  → List traces
GET http://localhost:3002/api/public/observations → List observations
GET http://localhost:3002/api/public/sessions → List sessions
```

### Data Flow
- Cockpit server-side routes → Langfuse SDK → Langfuse API → PostgreSQL
- Hindsight API → Langfuse (via Langfuse SDK or OTEL export)

---

## 7. Cloudflare Tunnel

### Dashboard: Cloudflare Zero Trust → Networks → Tunnels

| Public Hostname | Target | Status |
|-----------------|--------|--------|
| `cockpit.collabmind.dev` → | `http://localhost:8999` | ✅ Working |
| `hindsight-api.collabmind.dev` → | `http://localhost:8888` | ✅ Working |
| `auth.oliverv.dev` → | `http://localhost:9000` | ❌ Dead (:9000 stopped) |
| `supermemory-tunnel` → | `http://localhost:3000` | ⚠️ Redirects |
| `obsidian-mcp` → | `http://localhost:27124` | ❌ Dead |
| `neuron-ai-controller` → | `http://localhost:9998` | ❌ Dead |

**Overall status**: Degraded (health-check aggregate of dead secondary routes, primary routes functional)

---

## 8. Architecture Diagram

```
                            Cloudflare Tunnel
                    ┌────────────────────────────┐
                    │ cockpit.collabmind.dev:8999 │
                    │ hindsight-api:8888          │
                    └──────────┬─────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                Operator Layer (Cockpit :8999)                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐ │
│  │ Operator │ │ AI Chat  │ │ Langfuse │ │ Monitoring/    │ │
│  │ Panels   │ │ Assist.  │ │ Traces A  │ │ Graphs B       │ │
│  └────┬─────┘ └──────────┘ └──────────┘ └────────────────┘ │
└───────┼─────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────────┐
│            Governance Layer (Central API :8000) C             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐ │
│  │Governance│ │ Policy   │ │ Router   │ │ Project/Scope  │ │
│  │(approve) │ │(write-   │ │ Decisons │ │ Management     │ │
│  │          │ │ gate)    │ │          │ │                │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘ │
└───────┼─────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────────┐
│                  Memory Layer                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ Hindsight    │ │ Memlord      │ │ Ollama       │        │
│  │ :8888 REST   │ │ :8005 MCP    │ │ :11434 embed │        │
│  │ opencode bk  │ │ collabmind   │ │ nomic-em text│        │
│  │ 1318 facts   │ │ 128 memories │ │ 768-dim      │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ LM Studio    │ │ Langfuse     │ │ Grafana LGTM │        │
│  │ :1234 LLM    │ │ :3002 traces │ │ :3000 metrics│        │
│  │ Gemma 4 E4B  │ │ v2.95.11     │ │ Loki/Tempo   │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. ENV Variable Map

| Env Var | Value | Used By | Purpose |
|---------|-------|---------|---------|
| `HINDSIGHT_API_LLM_PROVIDER` | `lmstudio` | Hindsight API | LLM provider selection |
| `HINDSIGHT_API_LLM_BASE_URL` | `http://192.168.1.144:1234/v1` | Hindsight API | LM Studio endpoint |
| `HINDSIGHT_API_LLM_MAX_CONCURRENT` | `2` | Hindsight API | LLM concurrency cap |
| `HINDSIGHT_API_LLM_TIMEOUT` | `600` | Hindsight API | LLM request timeout |
| `HINDSIGHT_API_EMBEDDINGS_PROVIDER` | `litellm-sdk` | Hindsight API | Embedding provider |
| `HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_API_BASE` | `http://localhost:11434` | Hindsight API | Ollama embeddings |
| `HINDSIGHT_API_DATABASE_URL` | `postgresql://oliververmeulen@localhost:5432/hindsight17` | Hindsight API | Production DB |
| `HINDSIGHT_CP_DATAPLANE_API_URL` | `https://hindsight-api.collabmind.dev` | Cockpit | Hindsight REST proxy |
| `HINDSIGHT_CP_ACCESS_KEY` | `your-collabminds-access-key` | Cockpit | Auth key |
| `LANGFUSE_PUBLIC_KEY` | `pk-lf-...` | Cockpit | Langfuse auth |
| `LANGFUSE_SECRET_KEY` | `sk-lf-...` | Cockpit | Langfuse auth |
| `LANGFUSE_BASE_URL` | `http://localhost:3002` | Cockpit | Langfuse host |
| `CENTRAL_API_MEMLORD_URL` | `http://localhost:8005` | Central API | Memlord MCP |
| `CENTRAL_API_HINDSIGHT_URL` | `http://localhost:8888/mcp` | Central API | ⚠️ Should be `http://localhost:8888` |
| `CENTRAL_API_HINDSIGHT_BANK_ID` | `opencode` | Central API | Default Hindsight bank |

---

## 10. Adapter Method → Real API Mapping (Package C Fix)

### HindsightAdapter Fix Plan

| Current (Broken) | Should Be | Method |
|-----------------|-----------|--------|
| `POST /mcp/hindsight/retrieve_memory` | `POST /v1/{bank}/memories/recall` | `search()` |
| `POST /mcp/hindsight/sync_retain` | `POST /v1/{bank}/memories` | `store()` |
| `POST /mcp/hindsight/update_memory` | `PATCH /v1/{bank}/memories/{id}` | `update()` |
| `POST /mcp/hindsight/delete_memory` | `DELETE /v1/{bank}/memories/{id}` | `delete()` |
| `POST /mcp/hindsight/list_memories` | `GET /v1/{bank}/memories/list` | `export()` |
| `POST /mcp/hindsight/get_bank` | `GET /v1/{bank}` | `health()` |

### MemlordAdapter Status
- ✅ Uses real Memlord REST API (`POST /api-dev/memories/search`, `POST /api-dev/memories`, `GET /health`)
- ✅ `.env` points to correct URL: `http://localhost:8005`
- ❌ No API key set → `configured=False` → adapter returns stubs
- Fix: Set `CENTRAL_API_MEMLORD_API_KEY` in `.env`

---

## 11. Package Interface Contract

### A → C (Cockpit → Central API)
- Cockpit `/api/*` proxy routes call Central API at `http://localhost:8000/api/*`
- Not yet wired — current cockpit proxies directly to Hindsight REST API

### C → Memory (Central API → Adapters)
- Central API routes → `get_write_adapter()` → `InternalAdapter` (stub)
- Should be: Central API → `HindsightAdapter` → `http://localhost:8888/v1/{bank}/...`

### Cockpit → Hindsight (current, bypassing Central API)
- Cockpit API routes → `fetch("http://localhost:8888/...")` via `HINDSIGHT_CP_DATAPLANE_API_URL`
- This works now; migration to Central API is future Phase 3

---

*Generated 2026-07-01. Store to Memlord workspace `collabmind` as `collabmind-topology-2026-07-01`.*
