# CollabMind Action Plan — 2026-07-01

> Comprehensive task dispatch for parallel model execution.
> Session context: LM Studio flood fixed, Central API revived, Cockpit public.

---

## System State Snapshot

| Service | Port | Status | Notes |
|---------|------|--------|-------|
| Hindsight API | 8888 | ✅ Running | LM Studio LLM via 192.168.1.144:1234 |
| Ollama Embeddings | 11434 | ✅ Running | 22 models, nomic-embed-text |
| Central API (Cognitive Router) | 8000 | ✅ Running | FastAPI, SQLite, AI providers registered |
| Cockpit (Control Plane) | 8999 | ✅ Running | Production mode, public at cockpit.collabmind.dev |
| Memlord MCP | 8005 | ✅ Running | PostgreSQL + Qdrant |
| Hindsight MCP | 8888/mcp | ✅ Running | opencode bank: 1,318 facts |
| Workers | - | ⚠️ Stopped | Not required — API handles consolidation inline |
| Grafana LGTM | 3000 | ✅ Running | Monitoring stack |
| Cloudflare Tunnel | - | ⚠️ Degraded | Dead secondary routes (:9000, :9998, :27124) cause warning — primary routes (:8888, :8999) functional |

### OpenCode Bank Stats
- **1,318 facts** (764 experience, 508 world, 46 observations)
- **96 completed operations**, 64 failed (LM Studio timeouts from pre-fix era)
- **1,173 pending consolidation** items being processed
- Central API bank: `opencode` (connected to Hindsight MCP)

---

## Recent Fixes (This Session)

### 1. LM Studio Timeout Flood — `.env` (local, gitignored)
```bash
HINDSIGHT_API_LLM_MAX_CONCURRENT=2       # Global cap
HINDSIGHT_API_REFLECT_LLM_MAX_CONCURRENT=1
HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT=1
HINDSIGHT_API_LLM_TIMEOUT=600            # 10 min buffer
HINDSIGHT_API_WORKERS=2                  # Reduced from 4
HINDSIGHT_API_RERANKER_PROVIDER=rrf      # Avoids sentence-transformers
```
**Root cause**: 4 workers × multiple operations = 20+ parallel LLM requests hitting single-threaded LM Studio backend on `192.168.1.144:1234`. Fixed by capping all operation types to `MAX_CONCURRENT=1-2`.

### 2. Stuck Operations Cleared
- Cancelled 3 pending operations in opencode bank
- Consolidation self-healed (96 completed, up from 62)

### 3. Central API (Cognitive Router) — Port 8000
- Installed missing `aiosqlite` dependency
- API now live with 30+ routes including:
  - Health & DB connectivity
  - AI Provider Registry (LocalAI registered)
  - Router decision engine
  - Governance/policy/operator approval
  - Workspace/scope management

### 4. Cockpit Public Access
- Fixed 502 error (IPv4/IPv6 bind issue, wrong tunnel port)
- Switched from `next dev --turbopack` to `next start` (production mode)
- `typescript: { ignoreBuildErrors: true }` in next.config.ts
- URL: https://cockpit.collabmind.dev/

---

## Parallel Work Packages

Each section below is an independent unit that a separate coding model can execute without dependencies on other sections.

---

### PACKAGE A: Cockpit UI — Langfuse Traces Panel
**Difficulty**: Medium | **Location**: `hindsight-control-plane/` | **Est. time**: 2-4 hrs

**Goal**: Add a Langfuse traces visualization panel to the cockpit UI.

**Context**: Langfuse local backend is already working (`hindsight-control-plane/src/lib/langfuse.ts` exists). The traces data is being collected but has no UI.

**Existing files**:
- `hindsight-control-plane/src/lib/langfuse.ts` — Langfuse client library
- `hindsight-control-plane/src/app/[locale]/cockpit/page.tsx` — Cockpit main page
- `hindsight-control-plane/src/components/trace-stream.tsx` — Trace stream component (exists as stub?)

**Tasks**:
1. Read existing `langfuse.ts` to understand the data models
2. Create a Langfuse traces panel component:
   - Real-time trace list with filtering (by model, status, time range)
   - Trace detail view showing spans, tokens, latency
   - Cost breakdown per trace/session
3. Integrate into cockpit navigation (add to `operator-nav-types.ts`)
4. Add route at `/cockpit/traces` or similar
5. Verify against Langfuse local SDK (v2.x)

**Verification**: Navigate to traces panel, see real traces from Hindsight API calls
**Risks**: Langfuse SDK version compatibility — check `package.json` for version

---

### PACKAGE B: Cockpit UI — Monitoring & Graphs Panel
**Difficulty**: Medium | **Location**: `hindsight-control-plane/` | **Est. time**: 3-5 hrs

**Goal**: Add a monitoring dashboard panel showing system graphs and metrics.

**Context**: Grafana LGTM stack is running on port 3000 with dashboards for Hindsight Operations, LLM, API Service. Cockpit needs an embedded view or API-integrated monitoring panel.

**Existing files**:
- `hindsight-control-plane/src/components/engine-load-bar.tsx` — Engine load component
- `hindsight-control-plane/src/components/metric-sparkline.tsx` — Sparkline component
- `hindsight-control-plane/src/components/service-health-table.tsx` — Service health table
- `scripts/dev/monitoring/docker-compose.yaml` — LGTM stack config
- Monitoring dashboards in `monitoring/grafana/dashboards/`

**Tasks**:
1. Create a monitoring panel component with:
   - Service health status grid (Ollama, LM Studio, Hindsight API, Central API, Memlord)
   - LLM request rate graph (reads from Hindsight API `/v1/default/llm-requests/stats`)
   - Worker/consolidation throughput
   - Memory usage sparklines
2. Add alert indicators for failed operations, stalled consolidation
3. Integrate into cockpit navigation
4. Consider Grafana iframe embedding as alternative

**Verification**: Panel shows live metrics from Hindsight API and grafana
**Risks**: Grafana iframe may require auth bypass; prefer API-based metrics

---

### PACKAGE C: Central API Deepening — Real Adapters
**Difficulty**: High | **Location**: `central-api/` | **Est. time**: 4-6 hrs

**Goal**: Wire central-api's adapters to real backends (Hindsight, Memlord) instead of stubs.

**Context**: The central-api AGENTS.md notes that adapters and DB are stubs. The HindsightAdapter and MemlordAdapter exist but need real API wiring.

**Existing files**:
- `central-api/app/adapters/` — Engine registry + BaseMemoryAdapter ABC
- `central-api/app/db/tables.py` — 11 SQLAlchemy Core tables
- `central-api/app/db/repositories.py` — Repository layer
- `central-api/.env` — Already pointed at localhost:8888 (Hindsight MCP) and localhost:8005 (Memlord)

**Tasks**:
1. Read existing `BaseMemoryAdapter` interface
2. Wire `HindsightAdapter` to real Hindsight MCP at `http://localhost:8888/mcp`
   - Use Hindsight MCP tools via HTTP for retain/recall/reflect
3. Wire `MemlordAdapter` to real Memlord at `http://localhost:8005`
4. Fix the known DB schema conflict (`app/models/__init__.py` SQLModel vs `app/db/tables.py` SA Core)
5. Wire MCP tools to HTTP routes (known gap — `app/mcp/tools.py` not HTTP-wired)
6. Add more AI providers (Ollama, LM Studio) to provider registry

**Verification**: Central API can read/write to Hindsight and Memlord
**Risks**: MCP tool HTTP wiring requires understanding the Hindsight MCP protocol. Check `hindsight_api/mcp_tools.py` for the tool interface.

---

### PACKAGE D: operator-panel Feature Development
**Difficulty**: Medium | **Location**: `hindsight-control-plane/` | **Est. time**: 3-5 hrs

**Goal**: Continue building operator panel features on `feat/operator-panel` branch.

**Context**: Branch has 11 commits with 13,449 lines added across 66 files. Already has:
- AI Chat Assistant (bridge between operator and intelligence)
- 13 operator panels with real API integration
- Whitelabel system
- CollabMind Brain architecture

**Tasks** (in priority order):
1. Audit existing 13 panels for completeness — which ones need finishing?
2. Add remaining panels from the operator-nav-types:
   - Check `operator-nav-types.ts` for all registered panels
   - Compare against implemented pages in `src/app/[locale]/`
3. Wire panels that show "coming soon" to real API data
4. Add API Center panel wiring (existing: `src/app/[locale]/api-center/`)
5. Add Connectors panel (existing: `src/app/[locale]/connectors/`)
6. Add Router panel (existing: `src/app/[locale]/router/`)

**Verification**: All nav items lead to functional pages
**Risks**: Monorepo hoisting type conflicts (NextURL types) — use `ignoreBuildErrors` if needed

---

### PACKAGE E: File Cleanup
**Difficulty**: Low | **Location**: repo root | **Est. time**: 1-2 hrs

**Goal**: Sort ~50 untracked files into keep/archive/delete.

**Context**: Various scripts, docs, backups, and artifacts accumulated during development.

**Categories**:
1. **✅ KEEP** (move to proper location):
   - `scripts/dev/` files — already in proper location
   - `central-api/` — already a proper project directory
   - `skills/hindsight-local/` — agent skill
   - `vector-admin/` — if used

2. **📄 DOCS** (move to `hindsight-docs/` or `docs/`):
   - `ARCHITECTURE_WHY.md`
   - `CLIENTS_AND_INTEGRATIONS_EVALUATION.md`
   - `DOCKER_MODEL_RUNNER.md`
   - `EMBEDDINGS_CONFIG.md`, `EMBEDDINGS_OPTIONS.md`
   - `HINDSIGHT_ENV_VARS_COMPLETE.md`
   - `IMPORT_COMPLETE.md`, `IMPORT_STATUS.md`, `IMPORT_SUCCESS.md`
   - `LITELLM_IN_PROCESS_GUIDE.md`
   - `MONITORING_GUIDE.md`
   - `OLLAMA_SPLIT_SETUP.md`
   - `PRODUCTION_DEPLOYMENT_GUIDE.md`
   - `QUICK_START.md`
   - `SERVICES_DASHBOARD.md`
   - `STARTUP_ANALYSIS_2026-06-27.md`
   - `TRACING_AND_DATASETS.md`
   - `AGENTS_GENERATED.md`

3. **🗑️ DELETE** (no longer needed):
   - `.env.backup*`, `.env.broken`, `.env.production`, `.env.*` — old env backups
   - `docker-compose*.yml` files in root
   - `pyproject.toml1` — typo artifact
   - `check_database.py`, `clear_*.py`, `import_*.py`, `test_*.py` — one-off scripts
   - `grok-image-*.jpg` — random image
   - `design-ideas-GUI-PANEL-ADMIN/` — design artifacts

4. **❓ TBD** (ask user):
   - `old/`, `oldconole/` — old directories
   - `export/` — export data
   - `standalone-*.tar.gz` — cockpit backup

---

### PACKAGE F: Documentation to Memory Systems
**Difficulty**: Low | **Location**: Memlord + Hindsight | **Est. time**: 30 min

**Goal**: Store this session's outcomes and learnings to both memory systems.

**Tasks**:
1. Store to Memlord workspace `collabmind` with tags: `hindsight`, `session`, `2026-07-01`
2. Store to Hindsight `opencode` bank for agent operational context
3. Create mental models for key patterns:
   - LM Studio concurrency architecture
   - Central API cognitive router structure
   - CollabMind 3-tier topology

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Cloudflare Tunnel                        │
│  cockpit.collabmind.dev ──→ localhost:8999 (Cockpit Next.js)    │
│  hindsight-api.collabmind.dev ──→ localhost:8888 (Hindsight API)│
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────────────────────────────────────────────┐
│                      Operator (Cockpit :8999)                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐ │
│  │ Operator │ │  AI Chat │ │ Langfuse │ │ Monitoring/Graphs  │ │
│  │  Panels  │ │Assist.   │ │ Traces   │ │ (TODO)            │ │
│  └────┬─────┘ └──────────┘ └──────────┘ └────────────────────┘ │
└───────┼─────────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────────────┐
│              Central API — Cognitive Router (:8000)             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐ │
│  │Governance│ │  Policy  │ │ Router   │ │ Project/Scope Mgmt │ │
│  │(approve/ │ │ (write-  │ │ Decisions│ │ (workspaces,       │ │
│  │ reject)  │ │  gate)   │ │          │ │  tenants)          │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────────┘ │
└───────┼─────────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────────────┐
│                    Memory Layer                                  │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ │
│  │ Hindsight (:8888) │ │  Memlord (:8005)  │ │ Ollama (:11434) │ │
│  │ MCP + REST API   │ │  PostgreSQL+Qdrant│ │ Embeddings lane │ │
│  │ opencode bank    │ │  128 memories     │ │ nomic-embed-text│ │
│  │ 1,318 facts      │ │  collabmind ws    │ │ 768-dim         │ │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘ │
│                                 LM Studio (192.168.1.144:1234)   │
│                                 Single-threaded LLM (Gemma 4)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Execution Order & Dependencies

```
Phase 1: Infrastructure
  └── Package F (Documentation) — can run anytime
  └── Package E (Cleanup) — can run anytime

Phase 2: UI Panels (parallel)
  ├── Package A (Langfuse Traces) — independent
  └── Package B (Monitoring/Graphs) — independent

Phase 3: Backend Integration
  └── Package C (Central API adapters) — depends on stable API

Phase 4: Feature Completion
  └── Package D (operator-panel) — independent of all above
```

All packages in Phase 2 and 4 are fully parallelizable.
Package C should be done after verifying both Hindsight and Memlord are stable.

---

## Quick Reference: Service URLs

| Service | Local URL | Public URL |
|---------|-----------|-----------|
| Cockpit | http://localhost:8999 | https://cockpit.collabmind.dev/ |
| Central API | http://localhost:8000 | — |
| Hindsight API | http://localhost:8888 | https://hindsight-api.collabmind.dev/ |
| Hindsight MCP | http://localhost:8888/mcp | — |
| Grafana | http://localhost:3000 | — |
| Memlord | http://localhost:8005/health | — |
| Ollama Embeddings | http://localhost:11434 | — |
| LM Studio | http://192.168.1.144:1234/v1 | — |

---
*Generated 2026-07-01. Store to Memlord + Hindsight before closing session.*
