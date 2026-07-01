# Hindsight Codebase Information

**Project**: Hindsight - Agent Memory System for LLM-powered AI  
**Location**: `/Users/oliververmeulen/hindsight`  
**Repository**: github.com/vectorize-io/hindsight  
**License**: MIT

## Codebase Metrics

- **Total Files**: 3,708
- **Prioritized Files**: 1,845  
- **Total Lines of Code**: 465,166
- **Functions**: 18,723
- **Classes/Structs/Enums**: 1,863

## Language Support

| Language | Role | Status |
|----------|------|--------|
| **Python** | Core API, memory engine, consolidation, workers | ✅ Primary |
| **TypeScript** | Control plane UI, CLI client, integrations | ✅ Primary |
| **Rust** | CLI, performance-critical operations | ✅ Stable |
| **Go** | Generated SDK clients | ✅ Auto-generated |

## Core Subsystems

### 1. **Hindsight API** (`hindsight-api-slim/`)
- **Role**: Central REST API & memory management engine
- **Key Files**:
  - `hindsight_api/memory_engine.py` — Core memory orchestration (12,615 LOC)
  - `hindsight_api/config.py` — Configuration management (3,064 LOC)
  - `hindsight_api/mcp_tools.py` — MCP tool definitions (150 KB)
  - `hindsight_api/api/http.py` — HTTP routes (7,211 LOC)
- **Endpoints**: REST API on port 8888, MCP endpoint on port 8888/mcp
- **Dependencies**: FastAPI, SQLAlchemy, Pydantic, async/await architecture

### 2. **Worker Pool** (`hindsight-api-slim/hindsight_api/worker/`)
- **Role**: Asynchronous task processing (retain, consolidation, reflect, etc.)
- **Key Files**:
  - `poller.py` — Worker polling & task claim logic (1,379 LOC)
  - `main.py` — Worker CLI entry point (376 LOC)
- **Architecture**: Multi-worker pool with schema-aware task routing
- **Slot Reservation**: Per-operation type slot limits (consolidation, retain, etc.)

### 3. **Memory Engine** (`hindsight-api-slim/hindsight_api/engine/`)
- **Consolidation**: `consolidation/consolidator.py` — Fact consolidation & deduplication (2,385 LOC)
- **Fact Extraction**: `retain/fact_extraction.py` — LLM-based fact extraction (2,675 LOC)
- **Embeddings**: `embeddings.py` — 10+ embedding providers (1,706 LOC)
- **LLM Providers**: `providers/` — OpenAI, Anthropic, Gemini, Ollama, etc. (5,000+ LOC)
- **Search & Retrieval**: `search/retrieval.py` — Multi-strategy retrieval (873 LOC)
- **Entity Resolution**: `entity_resolver.py` — Canonical entity linking (935 LOC)
- **Cross-Encoder Reranking**: `cross_encoder.py` — 10+ reranking models (1,750 LOC)

### 4. **Database Layer** (`hindsight-api-slim/hindsight_api/engine/db/`)
- **PostgreSQL**: `ops_postgresql.py` — Default Postgres operations (1,285 LOC)
- **Oracle**: `oracle.py` + `ops_oracle.py` — Enterprise Oracle support (2,543 LOC)
- **Schema**: Alembic migrations in `alembic/versions/`
- **Vector Indexes**: HNSW indexes for semantic search

### 5. **Control Plane UI** (`hindsight-control-plane/`)
- **Framework**: Next.js + TypeScript + React
- **Components**: 40+ React components for bank management, visualization
- **Features**: Dashboard, stats, mental models, webhooks, operations
- **Ports**: 9998 (dev), 3000 (standalone), 9999 (control plane UI)

### 6. **Clients** (`hindsight-clients/`)
- **Python**: `hindsight_client` — Sync/async API client (1,516 LOC)
- **TypeScript**: `@vectorize-io/hindsight-client` — Node.js/browser client (1,123 LOC)
- **Rust**: Auto-generated from OpenAPI (1,400+ files)
- **Go**: Auto-generated from OpenAPI (159 files)

### 7. **Integrations** (`hindsight-integrations/` — 50+ integrations)
- **IDE**: Claude Code, Cursor, Cline, Windsurf, Zed, OpenCode
- **Frameworks**: LangGraph, CrewAI, Pydantic AI, OpenAI Agents, AutoGen, Haystack
- **LLM Providers**: LiteLLM (multi-provider wrapper)
- **No-Code**: Zapier, n8n, Flowise, Dify
- **Communication**: Vapi, Pipecat, OpenClaw
- **Knowledge**: Obsidian, LlamaIndex, RAG frameworks

### 8. **Embedded Mode** (`hindsight-embed/`)
- **Profile Manager**: `profile_manager.py` — Local profile/config management (686 LOC)
- **Daemon Manager**: `daemon_embed_manager.py` — Process lifecycle (940 LOC)
- **CLI**: `cli.py` — User-facing commands (1,692 LOC)
- **Control Center**: HTTP server for local UI

### 9. **CLI** (`hindsight-cli/`)
- **Language**: Rust  
- **Commands**: `main.rs` — Bank, memory, directive, operation management (2,122 LOC)
- **Binary**: Compiled to `hindsight` executable

### 10. **Documentation & Tools**
- **Docs**: Docusaurus site in `hindsight-docs/`
- **Benchmarks**: LongMemEval, Locomo, consolidation perf in `hindsight-dev/benchmarks/`
- **Testing**: 266 test files in `hindsight-api-slim/tests/`

## Architecture Patterns

### Three-Tier Design
1. **API Layer**: HTTP/REST + MCP, request routing & validation
2. **Business Logic**: Memory engine with consolidation, entity resolution, search
3. **Data Layer**: PostgreSQL + Oracle support with HNSW vector indexes

### Operational Patterns
- **LLM Providers**: Pluggable abstraction (OpenAI-compatible, Anthropic, Ollama, etc.)
- **Embeddings**: Multiple backends (OpenAI, Gemini, Ollama, ONNX, local transformers)
- **Cross-Encoders**: Pluggable reranking (Cohere, TEI, local, LiteLLM, etc.)
- **Task Backends**: Worker pool (async) + Sync (blocking) + Broker (queue-based)
- **Extensions**: Validator extensions, tenant extensions, MCP tool filtering

### Configuration
- **Environment-driven**: `.env` file (29,426 bytes with full config examples)
- **Hierarchical**: Per-bank overrides, per-operation LLM config
- **Banking**: Support for multi-bank, multi-tenant scenarios

## Performance Characteristics

- **Consolidation**: Batch LLM extraction with deduplication & conflict resolution
- **Retrieval**: Parallel 4-strategy search (semantic, keyword, graph, temporal)
- **Entity Resolution**: Trigram + fuzzy matching for canonical linking
- **Caching**: Gemini prompt caching, Cohere token cache support
- **Metrics**: Prometheus metrics, OpenTelemetry tracing, Grafana dashboards

## Key Dependency Insights

- **Async Foundation**: asyncio + FastAPI for concurrency
- **ORM**: SQLAlchemy for database abstraction
- **Type Safety**: Pydantic models for request/response validation
- **Vector DB**: PostgreSQL pgvector extension for embeddings
- **LLM Abstraction**: Custom wrapper supporting 12+ LLM providers
- **Testing**: pytest with 266 comprehensive test files

## Deployment Options

- **Docker**: Standalone image or docker-compose with monitoring
- **Bare Metal**: Python environment with PostgreSQL
- **Embedded**: Self-contained Python package with embedded SQLite (pg0)
- **Cloud**: Support for Google Cloud, AWS (via integrations)
- **Enterprise**: Oracle database support for on-prem deployments

## File Organization

```
/Users/oliververmeulen/hindsight/
├── hindsight-api-slim/          # Core API & memory engine
├── hindsight-embed/             # Embedded mode runtime
├── hindsight-control-plane/     # React UI dashboard
├── hindsight-cli/               # Rust CLI tool
├── hindsight-clients/           # Python, TypeScript, Rust, Go SDKs
├── hindsight-integrations/      # 50+ integrations
├── hindsight-dev/               # Development tools & benchmarks
├── hindsight-docs/              # Docusaurus documentation site
├── scripts/dev/                 # Local development scripts
├── docker/                      # Docker & compose configs
├── monitoring/                  # Grafana dashboards & OTEL config
└── AGENTS.md / CLAUDE.md        # Operational memory & coding guidelines
```

## Recent Development Notes

- **2026-06-27**: Split Ollama setup (embeddings port 11434, LLM port 11435)
- **2026-06-27**: Memlord MCP integration (port 8005) for cross-session memory
- **Known Issues**: Docker Desktop 4.79.0 Electron crash, control plane dev mode over tunnel
- **Active Focus**: Worker pool optimization, consolidation performance, multi-tenant scalability
