# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hindsight is an agent memory system that provides long-term memory for AI agents using biomimetic data structures. Memories are organized as:
- **World facts**: General knowledge ("The sky is blue")
- **Experience facts**: Personal experiences ("I visited Paris in 2023")
- **Mental models**: Consolidated knowledge synthesized from facts ("User prefers functional programming patterns")

## Development Commands

### API Server (Python/FastAPI)
```bash
# Start API server (loads .env automatically)
./scripts/dev/start-api.sh

# Run all tests (parallelized with pytest-xdist)
cd hindsight-api && uv run pytest tests/

# Run specific test file
cd hindsight-api && uv run pytest tests/test_http_api_integration.py -v

# Run single test function
cd hindsight-api && uv run pytest tests/test_retain.py::test_retain_simple -v

# Lint and format
cd hindsight-api && uv run ruff check .
cd hindsight-api && uv run ruff format .

# Type checking (uses ty - extremely fast type checker from Astral)
cd hindsight-api && uv run ty check hindsight_api/
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
# Regenerate OpenAPI spec after API changes (REQUIRED after changing endpoints)
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
- **hindsight-cli/**: CLI tool (Rust, cargo, uses progenitor for API client)
- **hindsight-clients/**: Generated SDK clients (Python, TypeScript, Rust)
- **hindsight-docs/**: Docusaurus documentation site
- **hindsight-integrations/**: Framework integrations (LiteLLM, OpenAI)
- **hindsight-dev/**: Development tools and benchmarks

### Core Engine (hindsight-api/hindsight_api/engine/)
- `memory_engine.py`: Main orchestrator (~170KB) for retain/recall/reflect operations
- `llm_wrapper.py`: LLM abstraction supporting OpenAI, Anthropic, Gemini, Groq, Ollama, LM Studio
- `embeddings.py`: Embedding generation (local sentence-transformers or TEI)
- `cross_encoder.py`: Reranking (local or TEI)
- `entity_resolver.py`: Entity extraction and normalization
- `query_analyzer.py`: Query intent analysis

**retain/**: Memory ingestion pipeline
- `orchestrator.py`: Coordinates the retain flow
- `fact_extraction.py`: LLM-based fact extraction from content
- `link_utils.py`: Entity link creation and management

**search/**: Multi-strategy retrieval
- `retrieval.py`: Main retrieval orchestrator
- `graph_retrieval.py`: Entity/relationship graph traversal
- `mpfp_retrieval.py`: Multi-Path Fact Propagation retrieval
- `fusion.py`: Reciprocal rank fusion for combining results
- `reranking.py`: Cross-encoder reranking

### API Layer (hindsight-api/hindsight_api/api/)
- `http.py`: FastAPI HTTP routers (~80KB) for all REST endpoints
- `mcp.py`: Model Context Protocol server implementation

Main operations:
- **Retain**: Store memories, extracts facts/entities/relationships
- **Recall**: Retrieve memories via 4 parallel strategies (semantic, BM25, graph, temporal) + reranking
- **Reflect**: Disposition-aware reasoning using memories and mental models.

### Database
PostgreSQL with pgvector. Schema managed via Alembic migrations in `hindsight-api/hindsight_api/alembic/`. Migrations run automatically on API startup.

Key tables: `banks`, `memory_units`, `documents`, `entities`, `entity_links`

### Adding Database Migrations

1. **Create a new migration file** in `hindsight-api/hindsight_api/alembic/versions/`:
   - File name format: `<revision_id>_<description>.py` (e.g., `f1a2b3c4d5e6_add_new_index.py`)
   - Use a unique hex revision ID (12 chars)
   - Set `down_revision` to the previous migration's revision ID

2. **Migration template**:
   ```python
   """Description of the migration

   Revision ID: f1a2b3c4d5e6
   Revises: <previous_revision_id>
   Create Date: YYYY-MM-DD
   """
   from collections.abc import Sequence
   from alembic import context, op

   revision: str = "f1a2b3c4d5e6"
   down_revision: str | Sequence[str] | None = "<previous_revision_id>"
   branch_labels: str | Sequence[str] | None = None
   depends_on: str | Sequence[str] | None = None

   def _get_schema_prefix() -> str:
       """Get schema prefix for table names (required for multi-tenant support)."""
       schema = context.config.get_main_option("target_schema")
       return f'"{schema}".' if schema else ""

   def upgrade() -> None:
       schema = _get_schema_prefix()
       op.execute(f"CREATE INDEX ... ON {schema}table_name(...)")

   def downgrade() -> None:
       schema = _get_schema_prefix()
       op.execute(f"DROP INDEX IF EXISTS {schema}index_name")
   ```

3. **Run migrations locally**:
   ```bash
   # Set database URL and run migrations
   uv run hindsight-admin run-db-migration

   # Run on a specific tenant schema
   uv run hindsight-admin run-db-migration --schema tenant_xyz
   ```

## Key Conventions

### Code Quality
**Always run the lint script after making Python or TypeScript/Node changes:**
```bash
./scripts/hooks/lint.sh
```
This runs the same checks as the pre-commit hook (Ruff for Python, ESLint/Prettier for TypeScript).

### Memory Banks
- Each bank is an isolated memory store (like a "brain" for one user/agent)
- Banks have dispositions (skepticism, literalism, empathy traits 1-5) affecting reflect
- Banks can have background context
- Bank isolation is strict - no cross-bank data leakage

### API Design
- All endpoints operate on a single bank per request
- Multi-bank queries are client responsibility to orchestrate
- Disposition traits only affect reflect, not recall

### Control Plane API Routes

When adding or modifying parameters in the dataplane API (hindsight-api), you must also update the control plane routes that proxy to it:

1. **API Routes** (`hindsight-control-plane/src/app/api/`):
   - `recall/route.ts` - proxies to `/v1/default/banks/{bank_id}/memories/recall`
   - `reflect/route.ts` - proxies to `/v1/default/banks/{bank_id}/reflect`
   - `memories/retain/route.ts` - proxies to `/v1/default/banks/{bank_id}/memories/retain`
   - Other routes follow the same pattern

2. **Client types** (`hindsight-control-plane/src/lib/api.ts`):
   - Update the TypeScript type definitions for `recall()`, `reflect()`, `retain()` etc.

3. **Checklist when adding new API parameters**:
   - Add parameter extraction in the route handler (destructure from `body`)
   - Pass the parameter to the SDK call
   - Update the client type definition in `lib/api.ts`
   - Update any UI components that need to use the new parameter

### Python Style
- Python 3.11+, type hints required
- Async throughout (asyncpg, async FastAPI)
- Pydantic models for request/response
- Ruff for linting (line-length 120)
- No Python files at project root - maintain clean directory structure
- **Never use multi-item tuple return values** - prefer dataclass or Pydantic model for structured returns

### Type Safety with Pydantic Models
**NEVER use raw `dict` types for structured data.** Always use Pydantic models:
- Use Pydantic `BaseModel` for all data structures passed between functions
- Add `@field_validator` for type coercion (e.g., ensuring datetimes are timezone-aware)
- Avoid `dict.get()` patterns - use typed model attributes instead
- Parse external data (JSON, API responses) into Pydantic models at the boundary
- This catches type errors at parse time, not deep in business logic

```python
# BAD - error-prone dict access
def process(data: dict) -> str:
    return data.get("name", "")  # No validation, silent failures

# GOOD - typed and validated
class UserData(BaseModel):
    name: str
    created_at: datetime

    @field_validator("created_at", mode="before")
    @classmethod
    def ensure_tz_aware(cls, v):
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

def process(data: UserData) -> str:
    return data.name  # Type-safe, validated at construction
```

### TypeScript Style
- Next.js App Router for control plane
- Tailwind CSS with shadcn/ui components

### Adding New API Configuration Flags

When adding a new environment variable configuration:

1. **config.py** (`hindsight-api/hindsight_api/config.py`):
   - Add `ENV_*` constant for the environment variable name
   - Add `DEFAULT_*` constant for the default value
   - Add field to `HindsightConfig` dataclass
   - Add initialization in `from_env()` method

2. **main.py** (`hindsight-api/hindsight_api/main.py`):
   - Add field to the manual `HindsightConfig()` constructor call (search for "CLI override")

3. **Use the config** in code:
   ```python
   from ...config import get_config
   config = get_config()
   value = config.your_new_field
   ```

4. **Documentation** (`hindsight-docs/docs/developer/configuration.md`):
   - Add to appropriate section table with Variable, Description, Default

## Environment Setup

```bash
cp .env.example .env
# Edit .env with LLM API key

# Python deps
uv sync --directory hindsight-api/

# Node deps (uses npm workspaces)
npm install
```

Required env vars:
- `HINDSIGHT_API_LLM_PROVIDER`: openai, anthropic, gemini, groq, ollama, lmstudio
- `HINDSIGHT_API_LLM_API_KEY`: Your API key
- `HINDSIGHT_API_LLM_MODEL`: Model name (e.g., o3-mini, claude-sonnet-4-20250514)

Optional (uses local models by default):
- `HINDSIGHT_API_EMBEDDINGS_PROVIDER`: local (default) or tei
- `HINDSIGHT_API_RERANKER_PROVIDER`: local (default) or tei
- `HINDSIGHT_API_DATABASE_URL`: External PostgreSQL (uses embedded pg0 by default)
