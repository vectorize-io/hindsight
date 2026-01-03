# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hindsight is an agent memory system that provides long-term memory for AI agents using biomimetic data structures. It stores memories as World facts, Experiences, Opinions, and Observations across memory banks.

## Development Commands

### API Server (Python/FastAPI)
```bash
# Start API server (loads .env automatically)
./scripts/dev/start-api.sh

# Run tests
cd hindsight-api && uv run pytest tests/

# Run specific test file
cd hindsight-api && uv run pytest tests/test_http_api_integration.py -v

# Lint
cd hindsight-api && uv run ruff check .
```

### Control Plane (Next.js)
```bash
./scripts/dev/start-control-plane.sh
# Or manually:
cd hindsight-control-plane && npm run dev
```

### Documentation Site (Docusaurus)
```bash
./scripts/dev/start-docs.sh
```

### Generating Clients/OpenAPI
```bash
# Regenerate OpenAPI spec after API changes
./scripts/generate-openapi.sh

# Regenerate all client SDKs (Python, TypeScript, Rust)
./scripts/generate-clients.sh
```

### Benchmarks
```bash
./scripts/benchmarks/run-longmemeval.sh
./scripts/benchmarks/run-locomo.sh
./scripts/benchmarks/start-visualizer.sh  # View results at localhost:8001
```

## Architecture

### Monorepo Structure
- **hindsight-api/**: Core FastAPI server with memory engine (Python, uv)
- **hindsight/**: Embedded Python bundle (hindsight-all package)
- **hindsight-control-plane/**: Admin UI (Next.js, npm)
- **hindsight-cli/**: CLI tool (Rust, cargo)
- **hindsight-clients/**: Generated SDK clients (Python, TypeScript, Rust)
- **hindsight-docs/**: Docusaurus documentation site
- **hindsight-integrations/**: Framework integrations (LiteLLM, OpenAI)
- **hindsight-dev/**: Development tools and benchmarks

### Core Engine (hindsight-api/hindsight_api/engine/)
- `memory_engine.py`: Main orchestrator for retain/recall/reflect operations
- `llm_wrapper.py`: LLM abstraction supporting OpenAI, Anthropic, Gemini, Groq, Ollama, LM Studio
- `embeddings.py`: Embedding generation (local or TEI)
- `cross_encoder.py`: Reranking (local or TEI)
- `entity_resolver.py`: Entity extraction and normalization
- `query_analyzer.py`: Query intent analysis
- `retain/`: Memory ingestion pipeline
- `search/`: Multi-strategy retrieval (semantic, BM25, graph, temporal)

### API Layer (hindsight-api/hindsight_api/api/)
FastAPI routers for all endpoints. Main operations:
- **Retain**: Store memories, extracts facts/entities/relationships
- **Recall**: Retrieve memories via parallel search strategies + reranking
- **Reflect**: Deep analysis forming new opinions/observations

### Database
PostgreSQL with pgvector. Schema managed via Alembic migrations in `hindsight-api/hindsight_api/alembic/`. Migrations run automatically on API startup.

Key tables: `banks`, `memory_units`, `documents`, `entities`, `entity_links`

### Database Backups (IMPORTANT)
**Before any operation that may affect the database, run a backup:**
```bash
docker exec hindsight /backups/backup.sh
```

Operations requiring backup:
- Running database migrations
- Modifying Alembic migration files
- Rebuilding Docker images
- Resetting or recreating containers
- Any schema changes
- Bulk data operations

Backups are stored in `~/hindsight-backups/` on the host.

To restore:
```bash
docker exec -it hindsight /backups/restore.sh <backup-file.sql.gz>
```

## Key Conventions

### Memory Banks
- Each bank is isolated (no cross-bank data access)
- Banks have dispositions (skepticism, literalism, empathy traits 1-5) affecting reflect
- Banks can have background context

### API Design
- All endpoints operate on a single bank per request
- Multi-bank queries are client responsibility
- Disposition traits only affect reflect, not recall

### Python Style
- Python 3.11+, type hints required
- Async throughout (asyncpg, async FastAPI)
- Pydantic models for request/response
- Ruff for linting (line-length 120)

### TypeScript Style
- Next.js App Router for control plane
- Tailwind CSS with shadcn/ui components

## Environment Setup

```bash
cp .env.example .env
# Edit .env with LLM API key

# Python deps
uv sync --directory hindsight-api/

# Node deps (workspace)
npm install
```

Required env vars:
- `HINDSIGHT_API_LLM_PROVIDER`: openai, anthropic, gemini, groq, ollama, lmstudio
- `HINDSIGHT_API_LLM_API_KEY`: Your API key
- `HINDSIGHT_API_LLM_MODEL`: Model name (e.g., o3-mini, claude-sonnet-4-20250514)
