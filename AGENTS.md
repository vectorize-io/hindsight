# AGENTS.md

See [CLAUDE.md](./CLAUDE.md) for project documentation and coding conventions.

## Hindsight Development Environment - Critical Information

### Split Ollama Configuration (2026-06-26)

**Problem Solved**: ReadTimeout/connection errors when LLM extraction and embeddings shared the same Ollama endpoint.

**Solution**: Split Ollama into two isolated execution lanes:
- **Embeddings Lane** (port 11434): Fast, frequent vector generation
- **LLM Lane** (port 11435): Heavy, long-running extraction/reasoning

**Configuration Location**: `/Users/oliververmeulen/hindsight/.env`
```bash
HINDSIGHT_API_LLM_OLLAMA_API_BASE=http://localhost:11435
HINDSIGHT_API_EMBEDDINGS_OLLAMA_API_BASE=http://localhost:11434
```

### Critical Rules for Agents

#### ❌ DO NOT:
1. **NEVER modify `.env` programmatically** - R&D mode with real credentials
2. **NEVER use raw `uv run` commands** - Always use `scripts/dev/start-api.sh` or `scripts/dev/start-worker.sh`
3. **NEVER mix environments** - Scripts handle environment isolation
4. **NEVER change Ollama ports** - 11434=embeddings, 11435=LLM is the tested configuration
5. **NEVER run multiple instances of `start-all.sh`** - Causes port conflicts
6. **NEVER delete `/tmp/hindsight-workers.state`** while workers are running

#### ✅ DO:
1. **Use production scripts** in `scripts/dev/`:
   - `start-all.sh` - Unified startup (--monitoring, --workers N)
   - `status.sh` - Read-only monitoring (--watch mode)
   - `dashboard.sh` - Interactive management
   - `scale-workers.sh` - Worker lifecycle
   - `start-ollama-split.sh` - Ollama lane management
2. **Check status before actions**: `./scripts/dev/status.sh`
3. **Review logs**: API → `/tmp/hindsight-api.log`, Workers → `logs/worker-N.log`
4. **Use monitoring**: Grafana at http://localhost:3000 (see MONITORING_GUIDE.md)

### Architecture

**Bare Metal** (for hot-reload during development):
- Hindsight API (port 8888)
- Workers (ports 9001, 9002, ... for metrics)
- Ollama lanes (ports 11434, 11435)

**Docker** (optional, can toggle):
- Grafana LGTM stack (Grafana + Loki + Tempo + Mimir)
- Ports: 3000 (Grafana UI), 4317 (OTLP gRPC), 4318 (OTLP HTTP)

### Quick Start

```bash
# Start everything (recommended)
./scripts/dev/start-services.sh

# Or use unified script with options
./scripts/dev/start-all.sh --monitoring --workers 4

# Monitor status (watch mode)
./scripts/dev/status.sh --watch

# Interactive dashboard
./scripts/dev/dashboard.sh

# Access services
open http://localhost:9999  # Control Plane UI
open http://localhost:3000  # Grafana
```

### Control Plane Access

**Production URLs** (via Cloudflare Tunnel):
- Control Plane: https://neuron-ai-controller.collabmind.dev → http://localhost:9998
- Hindsight API: https://hindsight-api.collabmind.dev → http://localhost:8888

**Local URLs**:
- Control Plane UI: http://localhost:9998
- Hindsight API: http://localhost:8888

The control plane UI requires authentication.

**To configure access** (edit `.env` manually):
```bash
# Add these lines to .env
HINDSIGHT_CP_ACCESS_KEY=your-secret-key-here  # Choose any secret for local dev
```

Then restart control plane:
```bash
pkill -f "next dev"
./scripts/dev/start-services.sh
```

### Monitoring & Tracing

**Full guide**: See `MONITORING_GUIDE.md`

**Quick access**:
- Grafana UI: http://localhost:3000 (no login required)
- Dashboards: Hindsight Operations, LLM, API Service
- Traces: Explore → Tempo
- Logs: Explore → Loki
- Metrics: Explore → Prometheus

**Enable tracing**: Uncomment OTEL settings in `.env` (see MONITORING_GUIDE.md)

### File Locations

**Configuration**:
- Main config: `/Users/oliververmeulen/hindsight/.env`
- Monitoring: `scripts/dev/monitoring/docker-compose.yaml`
- Dashboards: `monitoring/grafana/dashboards/`

**Logs**:
- API: `/tmp/hindsight-api.log`
- Workers: `logs/worker-N.log`
- Monitoring: `docker logs hindsight-monitoring`

**State**:
- Worker state: `/tmp/hindsight-workers.state`
- Ollama models: `/Volumes/Mac/Users/oliververmeulen/.ollama/models`

### Documentation

- `OLLAMA_SPLIT_SETUP.md` - Complete split Ollama guide
- `MONITORING_GUIDE.md` - Grafana, tracing, metrics, logs
- `scripts/dev/README.md` - Development scripts overview

### Future Phases

- **Phase 2**: External LLM APIs (OpenAI, Anthropic, Gemini, Groq)
- **Phase 3**: Cloud deployment (Google Cloud, AWS)

---

## MCP Services - Critical Operational Requirements (2026-06-27)

### Dual MCP Architecture

**CollabMind runs TWO independent MCP servers:**

1. **Memlord MCP** (port 8005) - User/workspace memory system
   - Purpose: Store cross-session memories for AI agents
   - Workspaces: personal (default), collabmind (shared)
   - OAuth-enabled for ChatGPT, Claude, other AI clients
   - Database: PostgreSQL (port 5435) + Qdrant vector DB
   - Docker containers: memlord-service, memlord-postgres, memlord-qdrant

2. **Hindsight MCP** (port 8888/mcp) - Agent memory engine
   - Purpose: Biomimetic memory system (retain/recall/reflect)
   - 40+ tools: memory management, mental models, directives, documents
   - Database: pg0 embedded (port 5434)
   - Used by CollabMind for correlation and agent intelligence

### Agent Memory Maintenance Rules

#### ✅ ALWAYS:
1. **Store critical learnings to BOTH systems**:
   - Memlord: Store session outcomes, fixes, architecture decisions
   - Hindsight: Store operational context for agent memory

2. **Use Memlord for cross-session persistence**:
   - Configuration changes and why they were made
   - Bug fixes with root cause analysis
   - Infrastructure decisions (ports, services, architecture)
   - Tag appropriately: hindsight, docker, ollama, collabmind, etc.

3. **Use Hindsight for agent operational memory**:
   - Banks for different contexts (opencode bank, default bank)
   - Mental models for synthesized knowledge
   - Directives for agent behavior rules

4. **Verify both MCP services are operational before major work**:
   ```bash
   curl http://localhost:8005/health  # Memlord
   curl http://localhost:8888/health  # Hindsight
   docker ps --filter "name=memlord"  # Memlord containers
   ```

#### ❌ NEVER:
1. **NEVER assume memories persist without verification**
2. **NEVER delete Docker volumes without explicit approval** (see Docker Desktop issues below)
3. **NEVER store secrets, credentials, or tokens in memory systems**

---

## Recent Critical Issues & Fixes (2026-06-27)

### Issue 1c: LM Studio Parallel Request Flooding (2026-06-27) ⚠️ IDENTIFIED

**Problem**: Hindsight workers flooding LM Studio with 20+ simultaneous requests

**Symptoms**:
- Multiple "Client disconnected. Stopping generation..." messages in LM Studio logs
- Workers timeout waiting for LLM response, retry, and send MORE requests
- LM Studio can only process one request at a time (single-threaded)
- Timeout-retry loop creates exponential request flood

**Root Cause**: 
- Multiple workers (4+) running batch_retain jobs concurrently
- Each worker sends LLM extraction requests to same LM Studio endpoint
- No request queuing or concurrency control configured
- Default timeout too aggressive for LM Studio's processing speed

**Current Configuration**:
```bash
HINDSIGHT_API_LLM_PROVIDER=lmstudio
HINDSIGHT_API_LLM_BASE_URL=http://192.168.1.144:1234/v1
HINDSIGHT_API_LLM_MODEL=openai/gpt-oss-20b
```

**Outstanding Fix Needed**:
- Configure worker concurrency limits (prevent 20+ parallel LLM calls)
- Increase LLM request timeout to accommodate LM Studio's processing time
- OR reduce number of workers from 4 to 1-2 for single-threaded LLM backend

**Status**: Identified but not yet fixed. System functional but suboptimal.

---

## Recent Critical Issues & Fixes (2026-06-27)

### Issue 1: Hindsight LLM Configuration Error ✅ FIXED

**Problem**: Workers stuck with ReadTimeout errors (941 tasks hung 900+ seconds)

**Root Cause**: Wrong environment variable in `.env`
- Used: `HINDSIGHT_API_LLM_OLLAMA_API_BASE=http://localhost:11435` ❌
- Should be: `HINDSIGHT_API_LLM_BASE_URL=http://localhost:11435/v1` ✅

**Why**: Hindsight uses generic `HINDSIGHT_API_LLM_BASE_URL` for ALL LLM providers (OpenAI, Anthropic, Ollama, Groq, etc.), NOT provider-specific URLs.

**Fix Applied**: 
- Line 26 in `.env` - Commented out wrong variable
- Added correct variable with `/v1` endpoint
- Documented in HINDSIGHT_ENV_VARS_COMPLETE.md

**Verification**: 
```bash
grep "HINDSIGHT_API_LLM_BASE_URL" /Users/oliververmeulen/hindsight/.env
```

**Stored in**: Memlord workspace 'collabmind' (memory: hindsight-env-llm-base-url-fix-2026-06-27)

---

### Issue 1b: LM Studio Migration (2026-06-27) ✅ COMPLETED

**Change**: Switched from Ollama (port 11435) to LM Studio for LLM inference

**Reason**: 
- Separate LLM processing from local Ollama instance
- LM Studio runs on dedicated machine (192.168.1.144:1234)
- Keeps LLM workload off local Ollama embedding lane
- Better resource isolation for benchmarking

**Configuration Applied**:
```bash
HINDSIGHT_API_LLM_PROVIDER=lmstudio
HINDSIGHT_API_LLM_API_KEY=lmstudio
HINDSIGHT_API_LLM_BASE_URL=http://192.168.1.144:1234/v1
HINDSIGHT_API_LLM_MODEL=openai/gpt-oss-20b
```

**Files Updated**:
- `/Users/oliververmeulen/hindsight/.env` (root - used by start-api.sh)
- `/Users/oliververmeulen/hindsight/hindsight-api-slim/.env` (backup)

**System Status**: 
- ✅ Hindsight API running and healthy (port 8888)
- ✅ Workers processing batch_retain jobs successfully
- ⚠️ LM Studio experiencing timeout issues from parallel request flooding (20+ simultaneous requests)

**Outstanding Issue**: Need to configure worker concurrency/timeout settings to prevent overwhelming single-threaded LM Studio

**Stored in**: Memlord workspace 'collabmind' (memory: hindsight-session-2026-06-27-lm-studio-config)

---

### Issue 2: Docker Desktop 4.79.0 Electron Crash Bug ⚠️ KNOWN ISSUE

**Problem**: Docker Desktop crashes unexpectedly (not manual quit)

**Root Cause**: Electron GUI bug in Docker Desktop 4.79.0 (released 2026-06-22)
```
[BUGSNAG] Uncaught exception…
AbortError: Request aborted
at I._destroy (createWindowManager.main.js:40:37480)
```

**Pattern**: 
- Backend starts shutdown (unknown trigger)
- GUI tries to clean up system monitoring streams
- Stream cleanup hits AbortError (backend already gone)
- Uncaught exception crashes entire Docker Desktop

**NOT caused by**:
- CleanMyMac or other utilities
- Memory pressure or OOM
- Auto-pause feature (already disabled: AutoPauseTimeoutSeconds=0)
- User error

**Current Workaround**: 
- Living with 4.79.0 (restart Docker when it crashes)
- Downgrade to 4.78.0 failed (unsigned app, Apple rejected)

**Protection Applied**:
- Volume protection: `external: true` in docker-compose.yml
- Critical volume: `collabmind-memory_postgres_data` (PostgreSQL 17, 73+ memories)
- Regular backups recommended

---

### Issue 3: Control Plane Production URL (neuron-ai-controller.collabmind.dev)

**Problem**: Production URL shows dev mode (webpack-hmr errors, white page)

**Root Cause**: Turbopack/webpack dev mode (`npm run dev --turbopack`) doesn't work through Cloudflare tunnel

**Why**: WebSocket HMR only functions on localhost, breaks on external domains

**Current State**:
- Local: http://localhost:9998 (working, dev mode OK)
- Production: Uses Cloudflare tunnel to localhost:9998
- Issue: HMR cross-origin errors in browser console

**Workaround Applied**:
- Added `allowedDevOrigins` to `next.config.ts`
- Allows WebSocket connections from production domain
- Stored in Memlord (memory: control-plane-dev-mode-external-fix)

**Future Fix**: 
- Use standalone production build: `node standalone/server.js`
- Script exists: `scripts/dev/start-control-plane-production.sh`
- Needs deployment automation

---

### Issue 4: Control Plane Unauthorized Errors

**Problem**: Control plane couldn't load banks (401 Unauthorized)

**Root Cause**: Control plane `.env` pointed to production API instead of local
- Was: `HINDSIGHT_CP_DATAPLANE_API_URL=https://hindsight-api.collabmind.dev`
- Fixed: `HINDSIGHT_CP_DATAPLANE_API_URL=http://localhost:8888`

**Fix Applied**: Line 241 in `/Users/oliververmeulen/hindsight/.env`

---

## Control Plane Architecture (CollabMind-Specific)

### Control Plane is OPTIONAL

**Key Understanding**:
- Workers, logging, tracing, audit logs work WITHOUT control plane UI
- Backend APIs are independent HTTP REST endpoints
- Control plane is just a viewer/dashboard for data that already exists
- Can build custom dashboards consuming the same APIs

### Development vs Embedded Control Plane

**NO DIFFERENCE** - Same code, same React UI, just different startup methods:
- Manual: `scripts/dev/start-control-plane.sh` (npm run dev, port 9998)
- Embedded: `hindsight-embed ui start` (same Next.js app, managed lifecycle)

### Backend APIs (Work Independently)

```bash
GET /v1/default/banks                       # List banks
GET /v1/default/banks/{id}/stats            # Statistics
GET /v1/default/banks/{id}/audit-logs       # Audit entries
GET /v1/default/banks/{id}/audit-logs/stats # Audit statistics
GET /v1/default/llm-requests                # LLM request traces
GET /v1/default/llm-requests/stats          # Trace statistics
GET /metrics                                # Prometheus metrics
GET /health                                 # Health check
```

### Existing Control Plane Features

- `/dashboard` - Bank selector
- `/system` - Service monitoring (status, logs, worker scaling)
- `/deployment` - Deployment management
- `/banks/[bankId]` - Bank management

**Control Plane Port**: 9998 (changed from 9999 to avoid Docker Desktop conflict)

**Access**: http://localhost:9998
- Requires: `HINDSIGHT_CP_ACCESS_KEY` in `.env`
- Default: `your-collabminds-access-key`

---

## CollabMind Vision & Roadmap

### Current Local Setup (Phase 1)
- ✅ Hindsight API with split Ollama (embeddings + LLM lanes)
- ✅ Memlord MCP for cross-session memory
- ✅ Hindsight MCP for agent memory operations
- ✅ Control plane for service visualization
- ✅ Grafana/LGTM monitoring stack
- ⚠️ Ollama LLM timeout issues (port 11435 not responding reliably)

### Phase 2: Production Cloud Deployment
- Google Cloud resources available (Gemini API, CPU, servers)
- Multi-LLM configuration (OpenAI, Anthropic, Gemini, Groq)
- Worker scaling (40 worker slots optimal for M1 Max 64GB)
- Kubernetes/Helm deployment
- Production monitoring and tracing

### Phase 3: CollabMind Correlation Features
- Custom dashboard on top of Hindsight APIs
- Agent correlation and intelligence features
- Multi-tenant support
- Advanced analytics

**Principle**: Perfect local setup FIRST, then scale to cloud with production APIs

---

## Session Recovery Protocol

### When Starting New Session

1. **Verify MCP services are healthy**:
   ```bash
   curl http://localhost:8005/health  # Memlord
   curl http://localhost:8888/health  # Hindsight
   docker ps --filter "name=memlord"
   ```

2. **Check for stored memories about recent issues**:
   - Query Memlord workspace 'collabmind' for recent fixes
   - Check Hindsight 'opencode' bank for operational context

3. **Review recent changes in this file** (AGENTS.md)

4. **Verify critical services**:
   - Ollama split (ports 11434, 11435)
   - Docker containers (monitoring, memlord)
   - Control plane (if needed)

### When Encountering Issues

1. **Document in Memlord FIRST** (before fixing)
2. **Test fix locally**
3. **Update this file with learnings**
4. **Store resolution in Memlord**

**Never repeat the same mistake twice - use memory systems!**


---

## Codebase Technical Architecture (2026-06-27 Analysis)

### Generated from Full Codebase Scan

**Documentation Files**: See `.agents/summary/` for detailed analysis
- `codebase_info.md` — Metrics & subsystem overview (163 lines)
- `DELTA_ANALYSIS.md` — Comparison with operational memory (258 lines)
- `README.md` — Index & how-to guide (156 lines)

### Core Subsystems

**1. Memory Engine** (`hindsight-api-slim/hindsight_api/engine/`)
- **Consolidation** (`consolidator.py`, 2,385 LOC): Atomic deduplication, delta reconciliation, scope-aware conflict resolution
- **Fact Extraction** (`retain/fact_extraction.py`, 2,675 LOC): LLM extraction with Chinese temporal parsing, causal relations, entity labels
- **Search & Retrieval** (`search/retrieval.py`, 873 LOC): 4-parallel strategies (semantic, keyword, graph, temporal) with RRF merge
- **Entity Resolution** (`entity_resolver.py`, 935 LOC): Trigram + fuzzy matching, canonical linking, cooccurrence tracking
- **Cross-Encoder Reranking** (`cross_encoder.py`, 1,750 LOC): Cohere, Google, Jina MLX, TEI, FlashRank, ZeroEntropy, SiliconFlow, LiteLLM

**2. LLM & Embedding Abstraction**
- **LLM Providers** (`engine/providers/`, 5,000+ LOC):
  - OpenAI-compatible (1,593 LOC) — Base implementation for Ollama, LiteLLM, others
  - Anthropic, Gemini, LiteLLM, Fireworks, Ollama, llama.cpp, Claude Code, Codex OAuth, Nous
  - Features: Tool calling, streaming, batch API support, cost tracking

- **Embeddings** (`engine/embeddings.py`, 1,706 LOC):
  - OpenAI, Gemini, LiteLLM SDK/CLI, ZeroEntropy, Cohere, CodexOAuth, Jina MLX, TEI, ONNX, local transformers
  - Prefix semantics (encode_query vs. encode_documents)
  - Batch processing, dimension handling

**3. Database Layer** (`engine/db/`)
- **PostgreSQL** (`ops_postgresql.py`, 1,285 LOC): Default with HNSW vector indexes
- **Oracle** (`oracle.py` + `ops_oracle.py`, 2,543 LOC): Enterprise support with SQL-to-Oracle rewrites
- **Abstract Backend Interface**: Extensible for custom implementations

**4. Mental Models & Directives**
- Mental model refresh triggers: Scheduled (CRON), post-consolidation, on-demand, stale detection
- Directive system: Priority ranking, active/inactive toggle, tag-based filtering, reflect agent integration
- Observation scopes: Named categories, per-scope observation limits, per-scope token budgets

**5. Worker Pool Architecture** (`worker/poller.py`, 1,379 LOC)
- Schema-aware task rotation, slot-based capacity per operation type
- Atomic task claiming with deadlock prevention
- Graceful shutdown, task recovery on crash
- Priority-based bank ordering

**6. HTTP API** (`api/http.py`, 7,211 LOC)
- REST endpoints for all memory operations
- Request validation, error handling, response formatting
- Audit logging, webhook delivery tracking
- Async operation status polling

**7. MCP Tools** (`mcp_tools.py`, 150 KB)
- 40+ tools for Claude, ChatGPT, other clients
- Memory tools (retain, recall, reflect), bank management, directives, mental models, documents
- Tool filtering by permissions, single-bank mode support
- Audit logging per tool call

**8. Configuration System** (`config.py`, 3,064 LOC)
- Hierarchical config with environment overrides
- Per-bank LLM settings, embedding dimensions, recall budgets, consolidation limits
- 40+ configurable parameters with validation

### Integrations Ecosystem (50+)

**IDE Agents**: Claude Code, Cursor, Cline, Windsurf, Zed, OpenCode  
**Frameworks**: LangGraph, CrewAI, Pydantic AI, Haystack, AutoGen, LlamaIndex, Langgraph  
**LLM Wrappers**: LiteLLM (unified multi-provider), Anthropic direct, OpenAI direct  
**No-Code**: Zapier, n8n, Flowise, Dify  
**Communication**: Vapi, Pipecat, OpenClaw (Telegram, Slack integration)  
**Knowledge Systems**: Obsidian plugin, RAG frameworks, document processing  

### Clients (4 Languages)

- **Python** (`hindsight-client`, 1,516 LOC): Sync & async, context managers
- **TypeScript** (`@vectorize-io/hindsight-client`, 1,123 LOC): Node.js & browser compatible
- **Rust** (auto-generated from OpenAPI, 1,400+ files): Full API coverage
- **Go** (auto-generated from OpenAPI, 159 files): Enterprise integration

### CLI Tool** (`hindsight-cli/`, Rust, 2,122 LOC)

Commands: bank management, memory operations, mental models, directives, webhooks, operations  
Features: Interactive TUI with gradient colors, config file management, batch operations

### Control Plane UI** (`hindsight-control-plane/`, Next.js, 40+ React components)

Pages: Bank dashboard, stats, mental models, webhooks, audit logs, LLM traces, operations  
Features: Real-time metrics, interactive graphs, service health monitoring

### Key New Insights (Not in Existing AGENTS.md)

1. **Memory Defense System** (`memory_defense.py`): Webhook-based review before retain/update, blocking policies with violation details, audit logging
2. **Observation Scopes**: Named categorization with per-scope observation limits and token budgets
3. **Consolidation Algorithm**: Atomic hashing + delta reconciliation (create/update/delete operations)
4. **Search Strategy Parallelization**: 4 independent strategies merged via reciprocal rank fusion
5. **Embedding Prefix Semantics**: Different handling for query encoding vs. document encoding
6. **Directive Priority Ranking**: Higher priority directives evaluated first in reflect agent
7. **Mental Model Staleness Detection**: Auto-refresh triggers based on observation changes
8. **Tenant Extension Architecture**: Multi-tenant with dynamic schema discovery via extension interface
9. **Oracle Database Support**: Full enterprise support with schema isolation and SQL translation
10. **Factorized LLM Config**: Per-operation type LLM selection (separate models for extraction, consolidation, reflect)

### Deployment Profiles

- **Bare Metal**: Python with PostgreSQL, hot-reload for development
- **Docker Standalone**: All-in-one container with embedded SQLite (pg0)
- **Docker Compose**: Multi-service stack with monitoring (Grafana/LGTM)
- **Kubernetes**: Helm charts available in `helm/hindsight/`
- **Embedded Mode** (`hindsight-embed`): Self-contained Python package, local profile management

### Configuration Recommendations

```bash
# LLM Provider Selection
HINDSIGHT_API_LLM_PROVIDER=ollama
HINDSIGHT_API_LLM_BASE_URL=http://localhost:11435/v1  # ← Correct (generic variable)
HINDSIGHT_API_LLM_MODEL=neural-chat

# Embeddings
HINDSIGHT_API_EMBEDDINGS_PROVIDER=ollama
HINDSIGHT_API_EMBEDDINGS_OLLAMA_API_BASE=http://localhost:11434
HINDSIGHT_API_EMBEDDINGS_DIMENSION=384

# Consolidation
HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE=10
HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS=8000

# Search & Retrieval
HINDSIGHT_API_RECALL_BUDGET_FUNCTION=adaptive  # or 'fixed'
HINDSIGHT_API_RECALL_BUDGET_ADAPTIVE_LOW=1000
HINDSIGHT_API_RECALL_BUDGET_ADAPTIVE_MID=3000
HINDSIGHT_API_RECALL_BUDGET_ADAPTIVE_HIGH=6000

# Cross-Encoder Reranking
HINDSIGHT_API_CROSS_ENCODER_PROVIDER=local-sentence-transformers
```

### Performance Characteristics

- **Consolidation**: Batch LLM extraction with atomic deduplication, conflict resolution
- **Retrieval**: Parallel 4-strategy search with RRF merge, typically <500ms for 1M facts
- **Entity Resolution**: Trigram indexing + fuzzy fallback, <100ms for canonical linking
- **Vector Indexing**: HNSW on PostgreSQL for sub-100ms semantic search
- **Worker Throughput**: Configurable slot-based capacity (typical: 40 slots for M1 Max)

---

## Integration with Memlord & Session Continuity (Updated 2026-06-27)

### Documented Patterns (Operational Memory Confirmed)

✅ Split Ollama setup (11434 embeddings, 11435 LLM)  
✅ Correct LLM config variable (HINDSIGHT_API_LLM_BASE_URL, not provider-specific)  
✅ MCP dual architecture (Memlord port 8005, Hindsight port 8888/mcp)  
✅ Worker slot management per operation type  
✅ Control plane as optional viewer (backend APIs work independently)

### New Technical Layers

✅ 10+ LLM providers with unified abstraction  
✅ 10+ embedding providers with prefix semantics  
✅ 10+ cross-encoder reranking models  
✅ Consolidation deduplication algorithm (atomic hashing + delta reconciliation)  
✅ 4-parallel search with RRF merge  
✅ Memory defense blocking policies  
✅ Observation scopes with per-scope budgets  
✅ Multi-tenant extension architecture  
✅ Oracle database enterprise support

### For Future Agent Sessions

1. **Reference Generated Docs**: `.agents/summary/codebase_info.md` for subsystem metrics
2. **Cross-Check Delta Analysis**: `.agents/summary/DELTA_ANALYSIS.md` tracks new insights
3. **Preserve This Section**: Contains session context & learnings
4. **Consult Original AGENTS.md**: Operational procedures, MCP setup, known issues

**AI Agents**: Use AGENTS_GENERATED.md (in repo root, created 2026-06-27) as navigation guide  
**Humans**: Use original AGENTS.md (this file) for operational procedures + generated docs for technical depth

## Durable Knowledge

**Project:** CollabMind / Hindsight  
**Title:** Hindsight as central memory/knowledge base engine — deployment phases  
**Scope:** system  
**Invariant / Standard:** Hindsight is the chosen thinking brain and knowledge base for CollabMind. All other memory models integrate into this structure. Do not enable external APIs until local benchmark is complete and stable.  
**Rationale:** Hindsight was evaluated against all major memory models and selected as the best fit. It supports distributed workers, external Postgres (pgvector), 50+ platform integrations, and full observability. The embedded pg0 is local-only; production uses external Postgres.  
**How to Apply:**  
- Phase 1: Deploy fully local on Mac Studio M1 Max. Test ingestion (memories, images, docs, text). Run multiple workers. Benchmark CPU/GPU per component. No external APIs.  
- Phase 2: Deploy on GCP. External Postgres (Neon/AlloyDB/pgvector) as shared vector store + job queue. GPU VMs run worker + Ollama + TEI reranker. API stays lightweight.  
- Phase 3: Enable external integrations (LiteLLM, cloud LLMs, 50+ connectors) only after local is stable.  
**Do Not:** Do not enable external APIs or cloud LLMs until Phase 1 local benchmark is complete. Do not mix embedding dimensions (current: nomic-embed-text 768-dim via Ollama :11434). Do not run embedding and LLM on the same Ollama instance (use split lanes).  
**References:** `hindsight/hindsight-api-slim/.env`, `hindsight/.env`, `hindsight/scripts/dev/start-ollama-split.sh`, `hindsight/scripts/dev/scale-workers.sh`  
**Change Log:**  
- 2026-06-27 - Initial architectural decision recorded (Oliver + Kiro)  
**README:** `hindsight/README.md`  
**CHANGELOG:** `hindsight/AGENTS.md`

---

## Hindsight Configuration Reference (2026-06-27)

### Environment Variable Naming Patterns

**Critical Difference**: LLM and Embeddings use different naming conventions:

1. **LLM Variables** (NO provider segment):
   ```bash
   HINDSIGHT_API_LLM_{PARAMETER}
   # Examples:
   HINDSIGHT_API_LLM_PROVIDER=openai
   HINDSIGHT_API_LLM_BASE_URL=http://localhost:11435/v1
   HINDSIGHT_API_LLM_MODEL=gpt-4o
   HINDSIGHT_API_LLM_API_KEY=sk-xxx
   ```

2. **Embeddings Variables** (INCLUDES provider segment):
   ```bash
   HINDSIGHT_API_EMBEDDINGS_{PROVIDER}_{PARAMETER}
   # Examples:
   HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL=http://localhost:12434/v1
   HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL=text-embedding-3-small
   HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_MODEL=cohere/embed-english-v3.0
   ```

3. **Reranker Variables** (INCLUDES provider segment):
   ```bash
   HINDSIGHT_API_RERANKER_{PROVIDER}_{PARAMETER}
   # Examples:
   HINDSIGHT_API_RERANKER_COHERE_MODEL=rerank-english-v3.0
   HINDSIGHT_API_RERANKER_LOCAL_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
   ```

**Common Pitfall**: Wrong variable name causes fallback to defaults and auth errors.

---

### LLM Providers (20+)

**Supported**: OpenAI, Anthropic, Gemini, Groq, Minimax, DeepSeek, z.ai, OpenCode-Go, Nous, Fireworks, Ollama, Ollama-Cloud, LM Studio, llama.cpp, Vertex AI, AWS Bedrock, LiteLLM, LiteLLM Router, Volcano, OpenRouter, OpenAI-Codex (OAuth), Claude-Code (OAuth), None (chunks mode)

**Default**: `openai` with `gpt-5-mini`

**DeepSeek Note**: ⚠️ DeepSeek does NOT provide embeddings endpoint - must use separate embedding provider

---

### Per-Operation LLM Configuration

Different memory operations can use different models/providers:

```bash
# Base LLM (fallback for all operations)
HINDSIGHT_API_LLM_PROVIDER=openai
HINDSIGHT_API_LLM_API_KEY=sk-xxx
HINDSIGHT_API_LLM_MODEL=gpt-4o

# Retain: Fact extraction (needs strong structured output)
HINDSIGHT_API_RETAIN_LLM_PROVIDER=openai
HINDSIGHT_API_RETAIN_LLM_MODEL=gpt-4o
HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT=4

# Reflect: Reasoning/response (can use cheaper/faster)
HINDSIGHT_API_REFLECT_LLM_PROVIDER=groq
HINDSIGHT_API_REFLECT_LLM_API_KEY=gsk-xxx
HINDSIGHT_API_REFLECT_LLM_MODEL=llama-3.3-70b-versatile
HINDSIGHT_API_REFLECT_LLM_MAX_CONCURRENT=8

# Consolidation: Observation synthesis
HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER=anthropic
HINDSIGHT_API_CONSOLIDATION_LLM_API_KEY=sk-ant-xxx
HINDSIGHT_API_CONSOLIDATION_LLM_MODEL=claude-sonnet-4-20250514
```

**Each operation supports**:
- `_PROVIDER`, `_API_KEY`, `_MODEL`, `_BASE_URL`
- `_MAX_CONCURRENT`, `_TIMEOUT`, `_MAX_RETRIES`
- `_INITIAL_BACKOFF`, `_MAX_BACKOFF`
- `_REASONING_EFFORT`, `_EXTRA_BODY`, `_DEFAULT_HEADERS`

**Concurrency Composition**: Per-operation `MAX_CONCURRENT` caps compose with global cap:
- Global: `HINDSIGHT_API_LLM_MAX_CONCURRENT=32` (default)
- Per-op: `HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT=4` adds extra cap
- Result: Retain limited to 4, but also counts against global 32

---

### Multi-LLM Strategies (Failover/Round-Robin)

Configure multiple LLMs for failover or load balancing:

```bash
# Primary (unindexed)
HINDSIGHT_API_LLM_PROVIDER=openai
HINDSIGHT_API_LLM_API_KEY=sk-xxx
HINDSIGHT_API_LLM_MODEL=gpt-4o

# Extra members (indexed 1, 2, 3...)
HINDSIGHT_API_LLM_1_PROVIDER=groq
HINDSIGHT_API_LLM_1_API_KEY=gsk-xxx
HINDSIGHT_API_LLM_1_MODEL=llama-3.3-70b-versatile

HINDSIGHT_API_LLM_2_PROVIDER=anthropic
HINDSIGHT_API_LLM_2_API_KEY=sk-ant-xxx
HINDSIGHT_API_LLM_2_MODEL=claude-sonnet-4-20250514

# Strategy
HINDSIGHT_API_LLM_STRATEGY='{"mode": "failover"}'
# or
HINDSIGHT_API_LLM_STRATEGY='{"mode": "round-robin", "weights": [3, 1, 1]}'
```

**Modes**:
- `failover`: Try in order (primary → 1 → 2), advance on failure
- `round-robin`: Rotate starting member, optional weights for unbalanced distribution

**Per-Operation Chains**: Use `HINDSIGHT_API_RETAIN_LLM_1_PROVIDER`, etc. for operation-specific chains

---

### Embedding Providers (8+)

**Supported**: local (SentenceTransformers), onnx (in-process ONNX Runtime), tei (HuggingFace TEI), openai, openai-codex (OAuth), openrouter, cohere, google (Gemini/Vertex AI), zeroentropy, litellm (proxy), litellm-sdk (direct)

**Default**: `local` with `BAAI/bge-small-en-v1.5` (384 dimensions)

**Docker Model Runner** (macOS/Linux with Docker Desktop):
```bash
HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai
HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL=http://localhost:12434/v1
HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL=docker.io/ai/nomic-embed-text-v1.5:latest
HINDSIGHT_API_EMBEDDINGS_DIMENSION=768
```

**Dimension Configuration**:
- Auto-detected from model at startup
- ⚠️ **Cannot change once data exists** (schema-locked)
- Common: 384 (BGE small), 768 (nomic-embed), 1536 (OpenAI), 3072 (Gemini/OpenAI large)

**Switching models with same dimension still breaks semantic search** - embedding spaces are NOT comparable even if dimensions match

---

### Reranker Providers (12+)

**Supported**: local (CrossEncoder), tei (HuggingFace TEI), cohere, openrouter, zeroentropy, siliconflow, alibaba (DashScope), google (Discovery Engine), flashrank, litellm (proxy), litellm-sdk (direct), jina-mlx (Apple Silicon), rrf (reciprocal rank fusion only)

**Default**: `local` with `cross-encoder/ms-marco-MiniLM-L-6-v2`

**RRF mode** (no neural reranking):
```bash
HINDSIGHT_API_RERANKER_PROVIDER=rrf
```

---

### Database Configuration

**Default**: `pg0` (embedded PostgreSQL on port 5433)

```bash
# External PostgreSQL
HINDSIGHT_API_DATABASE_URL=postgresql://user:pass@host:5432/dbname
HINDSIGHT_API_DATABASE_SCHEMA=public  # or custom schema

# Optional read replica
HINDSIGHT_API_READ_DATABASE_URL=postgresql://user:pass@replica:5432/dbname

# Migration URL (bypasses poolers like PgBouncer)
HINDSIGHT_API_MIGRATION_DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

**Vector Extensions**:
- `pgvector` (default) - HNSW, in-memory, good for <10M vectors
- `pgvectorscale` (recommended for scale) - DiskANN, disk-based, 28x better p95 latency
- `vchord` - High-dimensional embeddings (3000+ dims)
- `scann` - Google AlloyDB ScaNN

**Text Search Extensions**:
- `native` (default) - PostgreSQL tsvector + GIN
- `pg_search` (ParadeDB) - True BM25, only Citus-compatible option
- `vchord` - VectorChord BM25
- `pg_textsearch` - Timescale extension
- `pgroonga` - Multilingual/CJK support

---

### Worker Configuration

```bash
# Stable worker ID (survives restarts)
HINDSIGHT_API_WORKER_ID=hindsight-prod

# Number of uvicorn worker processes
HINDSIGHT_API_WORKERS=1
```

⚠️ **Critical**: Set stable `WORKER_ID` in production so tasks claimed before restart are recognized as own tasks after restart

---

### Retention Configuration

**Extraction Modes**:
- `concise` (default): Selective, fast
- `verbose`: Richer facts with full context
- `verbatim`: Store chunks as-is, LLM extracts metadata
- `chunks`: Store chunks as-is, zero LLM cost (embeddings only)
- `custom`: Full prompt override via `HINDSIGHT_API_RETAIN_CUSTOM_INSTRUCTIONS`

**Steering**:
```bash
HINDSIGHT_API_RETAIN_MISSION="Focus on technical decisions, architecture choices, and team expertise. Deprioritize social information."
```

**Batch APIs** (50% cost savings):
```bash
HINDSIGHT_API_RETAIN_BATCH_ENABLED=true
# Requires async retain (async=true in API call)
# Supported providers: openai, groq, gemini, fireworks
```

**Chunk Size**:
```bash
HINDSIGHT_API_RETAIN_CHUNK_SIZE=3000  # Default characters per chunk
HINDSIGHT_API_RETAIN_STRUCTURED_CHUNK_SIZE=12000  # For JSONL/conversation turns
```

---

### Recall Budget Mapping

**Fixed mode** (default):
```bash
HINDSIGHT_API_RECALL_BUDGET_FUNCTION=fixed
HINDSIGHT_API_RECALL_BUDGET_FIXED_LOW=100    # budget=low
HINDSIGHT_API_RECALL_BUDGET_FIXED_MID=300    # budget=mid (default)
HINDSIGHT_API_RECALL_BUDGET_FIXED_HIGH=1000  # budget=high
```

**Adaptive mode** (scales with max_tokens):
```bash
HINDSIGHT_API_RECALL_BUDGET_FUNCTION=adaptive
HINDSIGHT_API_RECALL_BUDGET_ADAPTIVE_LOW=0.025   # 2.5% of max_tokens
HINDSIGHT_API_RECALL_BUDGET_ADAPTIVE_MID=0.075   # 7.5% of max_tokens
HINDSIGHT_API_RECALL_BUDGET_ADAPTIVE_HIGH=0.25   # 25% of max_tokens
HINDSIGHT_API_RECALL_BUDGET_MIN=20               # Floor
HINDSIGHT_API_RECALL_BUDGET_MAX=2000             # Ceiling
```

---

### File Processing

**Parsers**:
- `markitdown` (default, local) - Microsoft's file-to-markdown converter
- `iris` (cloud) - Vectorize Iris for complex documents
- `llama_parse` (cloud) - LlamaIndex for complex layouts

**Parser Selection** (fallback chain):
```bash
HINDSIGHT_API_FILE_PARSER=iris,markitdown
HINDSIGHT_API_FILE_PARSER_ALLOWLIST=markitdown,iris
```

**Storage Backends**:
- `native` (default) - PostgreSQL BYTEA
- `s3` - AWS S3 or S3-compatible (MinIO, R2, Tigris)
- `gcs` - Google Cloud Storage
- `azure` - Azure Blob Storage

---

### Known Configuration Issues (Session 2026-06-27)

1. **pg0 vs External PostgreSQL**:
   - User's working instance: pg0 (embedded, port 5433)
   - External PostgreSQL attempts: Silent hangs during startup
   - **Root cause**: Likely resource exhaustion on bare metal (29 PostgreSQL processes from LiteLLM)

2. **Embedding Dimension Mismatch**:
   - Database had 384-dim schema
   - LM Studio returns 768-dim vectors
   - Config was set to 384
   - **Fix**: Update `HINDSIGHT_API_EMBEDDINGS_DIMENSION=768` and delete database or use matching model

3. **LM Studio Timeout Issues** (port 11435):
   - Single-threaded LLM backend
   - 20+ parallel worker requests caused timeout-retry loops
   - **Workaround**: Reduce worker count or switch to cloud LLM

4. **Docker Model Runner Discovery**:
   - Built-in Docker Desktop feature (macOS/Linux)
   - Accessible at `model-runner.docker.internal/v1/` (inside Docker network)
   - Exposed via port 12434 on host
   - Supports llama.cpp-metal, vllm-metal, diffusers backends

---

### Session 2026-06-27 Key Learnings

1. ✅ **pg0 is default** - Hindsight uses embedded PostgreSQL unless `DATABASE_URL` set
2. ✅ **LLM var pattern differs from embeddings** - No provider segment in `HINDSIGHT_API_LLM_*`
3. ✅ **Per-operation configs cascade** - Retain/Reflect/Consolidation fall back to base LLM config
4. ✅ **Dimension changes break existing data** - Must delete all memories or use matching dimensions
5. ✅ **Docker Model Runner built-in** - No custom setup needed, just Docker Desktop
6. ✅ **Batch APIs save 50%** - For async operations (OpenAI, Groq, Gemini, Fireworks)
7. ✅ **LiteLLM proxy vs SDK** - Proxy requires separate server, SDK is direct API access
8. ✅ **DeepSeek has NO embeddings** - Must use separate embedding provider (local, openai, cohere, google)
9. ✅ **llama.cpp provider** - Built-in local inference, auto-downloads Gemma 4 E2B (~3.5 GB)
10. ✅ **Worker concurrency composition** - `MAX_CONCURRENT` caps compose (per-operation + global)

---

### Quick Reference: Common Configurations

**Local Development** (pg0 + local models):
```bash
HINDSIGHT_API_DATABASE_URL=pg0
HINDSIGHT_API_LLM_PROVIDER=ollama
HINDSIGHT_API_LLM_BASE_URL=http://localhost:11435/v1
HINDSIGHT_API_LLM_MODEL=mistral:latest
HINDSIGHT_API_EMBEDDINGS_PROVIDER=local
HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=BAAI/bge-small-en-v1.5
HINDSIGHT_API_RERANKER_PROVIDER=local
```

**Production** (external PostgreSQL + cloud LLMs):
```bash
HINDSIGHT_API_DATABASE_URL=postgresql://user:pass@host:5432/hindsight
HINDSIGHT_API_LLM_PROVIDER=openai
HINDSIGHT_API_LLM_API_KEY=sk-xxx
HINDSIGHT_API_LLM_MODEL=gpt-4o
HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai
HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY=sk-xxx
HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL=text-embedding-3-small
HINDSIGHT_API_RERANKER_PROVIDER=cohere
HINDSIGHT_API_RERANKER_COHERE_API_KEY=xxx
```

**Hybrid** (local embeddings + cloud LLM):
```bash
HINDSIGHT_API_LLM_PROVIDER=gemini
HINDSIGHT_API_LLM_API_KEY=xxx
HINDSIGHT_API_LLM_MODEL=gemini-2.0-flash
HINDSIGHT_API_EMBEDDINGS_PROVIDER=local
HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=BAAI/bge-small-en-v1.5
HINDSIGHT_API_RERANKER_PROVIDER=rrf  # No neural reranking
```
