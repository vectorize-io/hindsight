# AGENTS.md — Hindsight Codebase Navigation for AI Agents

## Quick Start: 30-Second Overview

**Hindsight** is a biomimetic memory system for LLM agents. It stores facts, experiences, and learned mental models, then retrieves them intelligently during agent tasks.

**Entry Points**:
- **REST API**: `hindsight-api-slim/hindsight_api/main.py` → FastAPI server on port 8888
- **Worker Tasks**: `hindsight-api-slim/hindsight_api/worker/main.py` → Async task processor
- **Embedded Mode**: `hindsight-embed/hindsight_embed/cli.py` → Standalone Python package
- **UI Control Plane**: `hindsight-control-plane/src/app/` → Next.js React dashboard

---

## Directory Map & Component Navigation

### Core Subsystems

#### 1. **API Server** — `hindsight-api-slim/hindsight_api/`
- **Memory Engine** (`memory_engine.py`, 12,615 LOC)
  - Orchestrates all memory operations: retain, recall, reflect
  - Manages LLM interactions, embeddings, cross-encoder reranking
  - Handles multi-bank, multi-schema scenarios
  
- **Configuration** (`config.py`, 3,064 LOC)
  - Hierarchical config system with environment overrides
  - Per-bank LLM settings, embedding dimensions, recall budgets
  - 40+ configurable parameters

- **HTTP API** (`api/http.py`, 7,211 LOC)
  - REST endpoints for memory operations
  - Request validation, error handling, response formatting
  - Health checks, version endpoints, metrics

- **MCP Tools** (`mcp_tools.py`, 150 KB)
  - Tool definitions for Model Context Protocol (Claude, etc.)
  - Memory tools (retain, recall, reflect), bank management, directives
  - Tool filtering by permissions, single-bank mode

- **Consolidation** (`engine/consolidation/consolidator.py`, 2,385 LOC)
  - Fact consolidation & deduplication using LLM
  - Mental model generation from raw observations
  - Scope-aware conflict resolution

- **Embeddings** (`engine/embeddings.py`, 1,706 LOC)
  - 10+ embedding providers: OpenAI, Gemini, Ollama, ONNX, local
  - Batch processing, dimension handling, provider abstraction

- **LLM Providers** (`engine/providers/`)
  - OpenAI-compatible: `openai_compatible_llm.py` (1,593 LOC)
  - Anthropic, Gemini, LiteLLM, Fireworks, Ollama, llama.cpp
  - Tool calling, streaming, batch API support

- **Search & Retrieval** (`engine/search/retrieval.py`, 873 LOC)
  - 4-strategy parallel search: semantic, keyword (BM25), graph, temporal
  - RRF merging, cross-encoder reranking, token budgeting

- **Entity Resolution** (`engine/entity_resolver.py`, 935 LOC)
  - Canonical entity linking, fuzzy matching, trigram search
  - Cooccurrence tracking, entity statistics

- **Database Layer** (`engine/db/`)
  - PostgreSQL (default): `ops_postgresql.py`, HNSW vector indexes
  - Oracle (enterprise): `oracle.py`, `ops_oracle.py` with schema rewrites

#### 2. **Worker Pool** — `hindsight-api-slim/hindsight_api/worker/`
- **Poller** (`poller.py`, 1,379 LOC)
  - Async worker claiming tasks from database
  - Schema-aware rotation, slot-based capacity management
  - Graceful shutdown, task recovery on crash
  
- **Task Execution**
  - Submit tasks via `memory_engine.submit_async_*()`
  - Executor pattern: async workers, worker backend, or sync backend
  - Operation status tracking with child tasks

#### 3. **Embedded Mode** — `hindsight-embed/`
- **Profile Manager** (`profile_manager.py`, 686 LOC)
  - Local config storage, port allocation, profile switching
  
- **Daemon Manager** (`daemon_embed_manager.py`, 940 LOC)
  - Process lifecycle, health checks, port binding
  
- **CLI** (`cli.py`, 1,692 LOC)
  - User-facing commands: `start`, `stop`, `configure`, `profile`

#### 4. **Control Plane UI** — `hindsight-control-plane/`
- **Pages** (`src/app/[locale]/`):
  - `/banks/[bankId]` — Bank profile, stats, mental models
  - `/system` — Service monitoring, worker status
  - `/deployment` — Deployment settings
  - Dashboard views for operations, webhooks, audits

- **API Client** (`src/lib/api.ts`, 1,768 LOC)
  - Thin wrapper over HTTP API
  - Bank management, operation polling, audit logs

#### 5. **Clients** — `hindsight-clients/`
- **Python** (`python/hindsight_client/hindsight_client.py`, 1,516 LOC)
  - Sync & async methods for all API endpoints
  - Context managers, error handling
  
- **TypeScript** (`typescript/src/index.ts`, 1,123 LOC)
  - Node.js & browser compatible
  - Promise-based async API

#### 6. **Integrations** — `hindsight-integrations/` (50+ plugins)
- **IDE Agents**: Claude Code, Cursor, Cline, Windsurf
- **Frameworks**: LangGraph, CrewAI, Pydantic AI, Haystack
- **LLM Wrappers**: LiteLLM (OpenAI, Anthropic, Gemini wrappers)
- **No-Code**: Zapier, n8n, Dify, Flowise

#### 7. **CLI Tool** — `hindsight-cli/`
- **Rust binary** (`src/main.rs`, 2,122 LOC)
- Commands: `bank create`, `memory retain`, `memory recall`, `operation status`
- Compiled to `hindsight` executable

---

## Key Interfaces & Data Flows

### Memory Operations (3 Main Flows)

#### **Retain** — Store information
```
HTTP POST /v1/default/memory/retain
  → memory_engine.retain()
    → fact_extraction.extract_facts_from_contents() [LLM]
    → entity_resolver.resolve_entities_batch()
    → insert_facts_batch() to database
    → submit_async_consolidation() if triggered
    → emit callback with operation_id
```
**Key Types**: `RetainRequest`, `RetainResponse`, `MemoryItem`

#### **Recall** — Retrieve information
```
HTTP POST /v1/default/memory/recall
  → memory_engine.recall()
    → retrieval.retrieve_all_fact_types_parallel() [4 strategies]
    → cross_encoder.predict() [reranking]
    → RRF merge & token budgeting
    → return RecallResponse
```
**Key Types**: `RecallRequest`, `RecallResponse`, `RecallResult`

#### **Reflect** — Analyze and synthesize
```
HTTP POST /v1/default/memory/reflect
  → reflect_agent.run_reflect_agent()
    → memory_engine.recall() [gather context]
    → mental_model management
    → LLM reasoning with tools
    → delta operations (add/update/remove sections)
    → return ReflectResponse
```
**Key Types**: `ReflectRequest`, `ReflectResponse`

### Bank & Configuration Management

```
POST /v1/default/banks → memory_engine.ensure_bank_exists()
GET  /v1/default/banks → memory_engine.list_banks()
GET  /v1/default/banks/{id}/stats → memory_engine.get_bank_stats()
PUT  /v1/default/banks/{id} → memory_engine.update_bank()
```

### Mental Models (Learned Observations)

```
POST /v1/default/banks/{id}/mental-models → create_mental_model()
GET  /v1/default/banks/{id}/mental-models → list_mental_models()
POST /v1/default/banks/{id}/mental-models/{id}/refresh → refresh_mental_model()
```

---

## Configuration & Environment

**Main Config File**: `/Users/oliververmeulen/hindsight/.env` (29,426 bytes)

### Critical Variables
- `HINDSIGHT_API_LLM_BASE_URL` — LLM endpoint (e.g., `http://localhost:11435/v1`)
- `HINDSIGHT_API_EMBEDDINGS_OLLAMA_API_BASE` — Embeddings endpoint (e.g., `http://localhost:11434`)
- `HINDSIGHT_API_LLM_PROVIDER` — Provider: `openai`, `anthropic`, `gemini`, `ollama`
- `HINDSIGHT_API_LLM_MODEL` — Model name (e.g., `gpt-4-turbo`, `claude-3-5-sonnet`)
- `HINDSIGHT_DB_URL` — PostgreSQL or Oracle URL
- `HINDSIGHT_API_EMBEDDINGS_DIMENSION` — Embedding size (default: 1536)

### Split Ollama Setup (2026-06-27)
- **Embeddings Lane** (port 11434): Fast, frequent calls
- **LLM Lane** (port 11435): Heavy extraction/reasoning
- **Reason**: Prevents ReadTimeout errors when both share one endpoint

---

## Operational Patterns & Conventions

### 1. **Error Handling**
- HTTP endpoints return `{success: bool, error?: str}`
- Operation async tasks track `status` ("pending", "completed", "failed")
- Worker retry logic with exponential backoff

### 2. **Async Task Submission**
- Retain/consolidate/reflect return `{async: true, operation_id: uuid}`
- Poll via `GET /v1/default/operations/{operation_id}` for status
- Webhooks fire on operation completion

### 3. **Banking & Multi-Tenancy**
- Each bank has isolated memory, config, embeddings
- Banks identified by `bank_id` (string, no slashes)
- Single-bank mode available for CLI/embedded (filtered tool access)

### 4. **Request Context Propagation**
- `request_context` includes: bank_id, user_id, trace_id, timestamp
- Audit logs capture all operations with request context
- LLM tracing includes memory_ids for retrieval lineage

### 5. **Memory Scopes & Observation Types**
- Fact types: "world", "experience", "mental_model", "directive", "document"
- Observation scopes: Named categories for grouping observations
- Scope limits apply across consolidation & retrieval

---

## Common Development Tasks

### Add a New LLM Provider
1. Subclass `LLMProvider` in `engine/llm_wrapper.py`
2. Implement `call()`, `call_with_tools()`, `verify_connection()`
3. Register in `LLMProvider.create_llm_provider()`
4. Add env variables for API key/URL
5. Test in `tests/test_multi_llm_provider.py`

### Add a New Embedding Provider
1. Subclass `Embeddings` in `engine/embeddings.py`
2. Implement `encode()` method
3. Register in `create_embeddings_from_env()`
4. Set `dimension` property
5. Add provider-specific env variables

### Add a New MCP Tool
1. Define tool function in `mcp_tools.py` with `@tool` decorator
2. Implement tool logic using `memory_engine` or extensions
3. Add to `register_mcp_tools()` registry
4. Add filters/permissions in `operation_validator.py` if needed
5. Test in `tests/test_mcp_tools.py`

### Run Tests
```bash
cd hindsight-api-slim/
pytest tests/test_retain.py -v  # Specific test
pytest tests/ -k consolidation  # Filter by keyword
```

### Start Local Development
```bash
# Terminal 1: API
./scripts/dev/start-api.sh

# Terminal 2: Workers
./scripts/dev/start-worker.sh

# Terminal 3: Control Plane UI
./scripts/dev/start-control-plane.sh

# Monitor
./scripts/dev/status.sh --watch
```

---

## Custom Instructions

<!-- This section is for human and agent-maintained operational knowledge.
     Add repo-specific conventions, gotchas, and workflow rules here.
     This section is preserved exactly as-is when re-running codebase-summary. -->

### Critical Known Issues (As of 2026-06-27)

1. **Ollama LLM Configuration** 
   - ❌ WRONG: `HINDSIGHT_API_LLM_OLLAMA_API_BASE` (provider-specific variable)
   - ✅ RIGHT: `HINDSIGHT_API_LLM_BASE_URL=http://localhost:11435/v1` (generic variable)
   - Reason: Hindsight uses universal `LLM_BASE_URL` for all providers

2. **Docker Desktop Electron Crash (4.79.0)**
   - Bug: Unexpected shutdown during stream cleanup
   - Workaround: Restart Docker Desktop when it crashes
   - Status: Downgrade to 4.78.0 failed (Apple unsigned app rejection)

3. **Control Plane Dev Mode Over Tunnel**
   - Issue: WebSocket HMR fails cross-origin from Cloudflare tunnel
   - Temporary Fix: Added `allowedDevOrigins` in `next.config.ts`
   - Permanent Fix: Use production build `node standalone/server.js`

4. **Memlord MCP Service Requirements**
   - Must run on port 8005 (production URL: `neuron-ai-controller.collabmind.dev`)
   - PostgreSQL 17 backend (port 5435)
   - Qdrant vector DB integration
   - Critical volumes: `collabmind-memory_postgres_data` (73+ memories stored)

### Safe Operations & Guardrails

1. **NEVER** modify `.env` programmatically (R&D mode with real credentials)
2. **ALWAYS** use `scripts/dev/start-*.sh` wrappers (correct env isolation)
3. **DO NOT** delete `/tmp/hindsight-workers.state` while workers running
4. **CHECK** `./scripts/dev/status.sh` before making changes
5. **USE** `./scripts/dev/dashboard.sh` for interactive management

### Development Scripts Overview

| Script | Purpose | Usage |
|--------|---------|-------|
| `start-all.sh` | Unified startup (API + workers + monitoring) | `./start-all.sh --workers 4` |
| `status.sh` | Health check & monitoring (read-only) | `./status.sh --watch` |
| `dashboard.sh` | Interactive service management | `./ dashboard.sh` |
| `scale-workers.sh` | Add/remove workers on the fly | `./scale-workers.sh +2` |
| `start-ollama-split.sh` | Manage embeddings/LLM lanes | `./start-ollama-split.sh` |

### Documentation Files to Consult

- `HINDSIGHT_ENV_VARS_COMPLETE.md` — Full environment variable reference
- `OLLAMA_SPLIT_SETUP.md` — Detailed split Ollama configuration guide
- `MONITORING_GUIDE.md` — Grafana dashboards, OTEL tracing, logging
- `ARCHITECTURE_WHY.md` — Design decisions & performance rationale
- `QUICK_START.md` — End-user getting-started guide
- `CLAUDE.md` — Coding conventions, project structure, testing patterns
- `AGENTS.md` (original, in repo root) — Operational memory from previous sessions
