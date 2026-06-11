# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hindsight is an agent memory system that provides long-term memory for AI agents using biomimetic data structures. Memories are organized as:
- **World facts**: General knowledge ("The sky is blue")
- **Experience facts**: Personal experiences ("I visited Paris in 2023")
- **Mental models**: Consolidated knowledge synthesized from facts ("User prefers functional programming patterns")

## Development Commands

### Local Development (API + UI)
```bash
# Start both API server and control plane UI
./scripts/dev/start.sh
```

### API Server (Python/FastAPI)
```bash
# Start API server only (loads .env automatically)
./scripts/dev/start-api.sh

# Run all tests (parallelized with pytest-xdist)
cd hindsight-api-slim && uv run pytest tests/

# Run specific test file
cd hindsight-api-slim && uv run pytest tests/test_http_api_integration.py -v

# Run single test function
cd hindsight-api-slim && uv run pytest tests/test_retain.py::test_retain_simple -v

# Lint and format
cd hindsight-api-slim && uv run ruff check .
cd hindsight-api-slim && uv run ruff format .

# Type checking (uses ty - extremely fast type checker from Astral)
cd hindsight-api-slim && uv run ty check hindsight_api/
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
# Accuracy benchmarks
./scripts/benchmarks/run-longmemeval.sh
./scripts/benchmarks/run-locomo.sh

# Performance benchmarks
./scripts/benchmarks/run-perf-test.sh                      # System perf (mock LLM + pg0)
./scripts/benchmarks/run-perf-test.sh --scale tiny          # Quick smoke test
./scripts/benchmarks/run-consolidation.sh

# Results viewer
./scripts/benchmarks/start-visualizer.sh  # View results at localhost:8001
```

## Architecture

### Monorepo Structure
- **hindsight-api-slim/**: Core FastAPI server with memory engine (Python, uv)
- **hindsight-control-plane/**: Admin UI (Next.js, npm)
- **hindsight-cli/**: CLI tool (Rust, cargo, uses progenitor for API client)
- **hindsight-clients/**: Generated SDK clients (Python, TypeScript, Rust)
- **hindsight-docs/**: Docusaurus documentation site
- **hindsight-integrations/**: Framework integrations (LiteLLM, CrewAI, LangGraph, Pydantic AI, AG2, Claude Code, etc.)
- **hindsight-dev/**: Development tools and benchmarks

### Core Engine (hindsight-api-slim/hindsight_api/engine/)
- `memory_engine.py`: Main orchestrator for retain/recall/reflect operations
- `llm_wrapper.py`: LLM abstraction supporting OpenAI, Anthropic, Gemini, VertexAI, Groq, MiniMax, Ollama, LM Studio, LiteLLM, Claude Code
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
- `graph_retrieval.py`: Graph retrieval abstract base class
- `link_expansion_retrieval.py`: Link expansion graph retrieval
- `fusion.py`: Reciprocal rank fusion for combining results
- `reranking.py`: Cross-encoder reranking

### API Layer (hindsight-api-slim/hindsight_api/api/)
- `http.py`: FastAPI HTTP routers for all REST endpoints
- `mcp.py`: Model Context Protocol server implementation

Main operations:
- **Retain**: Store memories, extracts facts/entities/relationships
- **Recall**: Retrieve memories via 4 parallel strategies (semantic, BM25, graph, temporal) + reranking
- **Reflect**: Disposition-aware reasoning using memories and mental models.

### Database
PostgreSQL with pgvector. Schema managed via Alembic migrations in `hindsight-api-slim/hindsight_api/alembic/`. Migrations run automatically on API startup.

Key tables: `banks`, `memory_units`, `documents`, `entities`, `entity_links`

### Adding Database Migrations

Hindsight runs the same Alembic tree against PostgreSQL and Oracle 23ai. Each
migration file dispatches through `run_for_dialect`, which calls either
`_pg_upgrade` or `_oracle_upgrade` based on the live connection. A pytest lint
(`tests/test_migration_shape.py`) fails CI if a migration omits the dispatcher.

1. **Create a new migration file** in `hindsight-api-slim/hindsight_api/alembic/versions/`:
   - File name format: `<revision_id>_<description>.py` (e.g., `f1a2b3c4d5e6_add_new_index.py`)
   - Use a unique hex revision ID (12 chars)
   - Set `down_revision` to the previous migration's revision ID

2. **Migration template** (the `script.py.mako` template scaffolds this; fill in the bodies):
   ```python
   """Description of the migration

   Revision ID: f1a2b3c4d5e6
   Revises: <previous_revision_id>
   Create Date: YYYY-MM-DD
   """
   from collections.abc import Sequence
   from alembic import context, op

   from hindsight_api.alembic._dialect import run_for_dialect

   revision: str = "f1a2b3c4d5e6"
   down_revision: str | Sequence[str] | None = "<previous_revision_id>"
   branch_labels: str | Sequence[str] | None = None
   depends_on: str | Sequence[str] | None = None


   def _pg_schema_prefix() -> str:
       """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
       schema = context.config.get_main_option("target_schema")
       return f'"{schema}".' if schema else ""


   def _pg_upgrade() -> None:
       schema = _pg_schema_prefix()
       op.execute(f"CREATE INDEX ... ON {schema}table_name(...)")


   def _pg_downgrade() -> None:
       schema = _pg_schema_prefix()
       op.execute(f"DROP INDEX IF EXISTS {schema}index_name")


   def _oracle_upgrade() -> None:
       # Oracle 23ai equivalent. Use op.get_bind().exec_driver_sql for forms
       # that Alembic core does not model (vector/text indexes, partitions).
       op.execute("CREATE INDEX ... ON table_name(...)")


   def _oracle_downgrade() -> None:
       op.execute("DROP INDEX IF EXISTS index_name")


   def upgrade() -> None:
       run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


   def downgrade() -> None:
       run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
   ```

   **Dialect-only migrations.** If a change genuinely doesn't apply to one
   dialect (e.g. enabling `pg_trgm` is PG-only), omit the unused slot:
   ```python
   def upgrade() -> None:
       run_for_dialect(pg=_pg_upgrade)  # oracle slot intentionally absent → no-op
   ```
   Make the asymmetry deliberate. Don't leave an Oracle slot empty just because
   you didn't think about it — copy-pasting a PG migration without the Oracle
   half is exactly how schemas drift.

3. **Run migrations locally**:
   ```bash
   # Set database URL and run migrations for the base schema plus all tenants
   uv run hindsight-admin run-db-migration

   # Run on a specific tenant schema
   uv run hindsight-admin run-db-migration --schema tenant_xyz
   ```

## Key Conventions

### Code Quality

**Before writing code, read `.claude/skills/code-review/SKILL.md`** for the full coding standards (Python style, type safety, TypeScript style, general principles).

**Always run the lint script after making Python or TypeScript/Node changes:**
```bash
./scripts/hooks/lint.sh
```

Dead-code detection runs in CI (the `check-unused-code` job) at two levels:
- **Blocking:** unused imports (ruff `F401`) and variables (`F841`) — `lint.sh` auto-removes
  them and `verify-generated-files` fails on any leftover diff; and **knip** for orphaned
  control-plane files / unused (or unlisted) `package.json` dependencies.
- **Advisory:** whole unused Python functions (vulture) and unused control-plane *exports*
  (the shadcn/ui surface is kept on purpose) — surfaced, not gated.

Run both locally with:
```bash
./scripts/hooks/check-unused.sh
```

**After completing any implementation work, run `/code-review`** to verify your changes against project standards (missing tests, dead code, type safety, etc.). Fix any "must fix" issues before considering the task done.

**MANDATORY: Run `/code-review` before pushing code or creating a pull request.** Do not push or create a PR until all "must fix" issues are resolved.

### Testing

Most tests are deterministic (MockLLM, pure functions) — assert directly.

**Tests that verify LLM behaviour use a real LLM + an LLM-as-judge.** When the thing under test is *how the model interprets a prompt* (classification, attribution, dimension preservation, instruction-following), MockLLM can't simulate it and exact string/enum asserts flake across providers and runs. Use this pattern instead:

1. Mark the test module `pytestmark = pytest.mark.hs_llm_core` (single-provider; CI runs it in the core-LLM job). Use `hs_llm_mat` only for provider-matrix acceptance tests.
2. Call the real pipeline (`LLMConfig.from_env()`, `_get_raw_config()`), e.g. `extract_facts_from_text(...)`.
3. Assert with the judge, not string matching:
   ```python
   from tests.llm_judge import assert_meets_criteria
   facts_summary = "\n".join(f"- [{f.fact_type}] {f.fact}" for f in facts)
   await assert_meets_criteria(
       response=facts_summary,
       criteria="The first-person user statements are classified 'world' and attributed to the user, not the agent.",
       context="What the input said and who was speaking.",
   )
   ```

Rules of thumb:
- **Judge anything non-deterministic** — including `fact_type` classification and speaker attribution. Do NOT hard-assert `fact_type == "..."`; pass a `[fact_type] fact` summary to the judge instead. Structural facts that ARE deterministic (counts, presence of a field, that a substring was injected into a prompt) stay as direct asserts in fast unit tests.
- **Split the test surface**: cover the deterministic mechanics (prompt assembly, suppression logic) with fast non-LLM unit tests, and the model-following behaviour with one `hs_llm_core` judge test. (Example pair: `test_narrator_resolution.py` + `test_narrator_context_override.py`.)
- The judge model is independent of the test provider (defaults to Gemini); never judge with the same call you're testing.

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

### Adding New Integrations

Every new integration in `hindsight-integrations/` must satisfy all of the following before it can be merged:

1. **Tests are required** — tests must simulate or exercise the external system (mock the framework's interfaces and verify the integration actually calls Hindsight correctly). Pure unit tests of helper functions are not sufficient.
2. **CI job** — add a test job in `.github/workflows/test.yml` following the existing pattern (e.g., `test-crewai-integration`). The job must build, install deps, and run `uv run pytest tests -v`. Also add the integration to `detect-changes` outputs so it only runs when its files change.
3. **Release process** — add the integration name to the `VALID_INTEGRATIONS` array in `scripts/release-integration.sh` so it can be released via the standard release workflow.
4. **Follow project code standards** — Python style, type safety, no raw dicts for structured data, no multi-item tuple returns (see `.claude/skills/code-review/SKILL.md`).

If any of these are missing, the integration is incomplete and must not be pushed or merged.

### Changelogs

Never add "Unreleased" entries to changelogs (e.g. `hindsight-docs/src/pages/changelog/**`). Changelog entries are written by the release script (`./scripts/release-integration.sh`) when a version is actually cut. If a bug fix or feature needs documenting before release, describe it in the PR/commit — the release tooling will surface it in the published changelog section.

### Adding New API Configuration Flags

Configuration follows a hierarchical system: **Global (env vars) → Tenant (via extension) → Bank (database)**.

Fields must be categorized as either **hierarchical** (can be overridden per-tenant/bank) or **static** (server-level only).

#### Adding a New Configuration Field

1. **config.py** (`hindsight-api-slim/hindsight_api/config.py`):
   - Add `ENV_*` constant for the environment variable name (e.g., `ENV_MY_SETTING = "HINDSIGHT_API_MY_SETTING"`)
   - Add `DEFAULT_*` constant for the default value
   - Add field to `HindsightConfig` dataclass with type annotation
   - **Mark as configurable** by adding to `_CONFIGURABLE_FIELDS` set if the field should be overridable per-tenant/bank via API
   - Add initialization in `from_env()` method

   ```python
   # Configurable field (can be overridden per-tenant/bank via API)
   _CONFIGURABLE_FIELDS = {
       ...,
       "my_setting",  # Add here for configurable
   }

   # Static field - just don't add to _CONFIGURABLE_FIELDS
   ```

2. **main.py** (`hindsight-api-slim/hindsight_api/main.py`):
   - Add field to the manual `HindsightConfig()` constructor call (search for "CLI override")

3. **Use hierarchical config in MemoryEngine**:
   ```python
   # Config is resolved automatically per bank via ConfigResolver
   config_dict = await self._config_resolver.get_bank_config(bank_id, context)
   value = config_dict["my_setting"]
   ```

4. **Use static config** (non-hierarchical):
   ```python
   from ...config import get_config
   config = get_config()
   value = config.my_static_field
   ```

5. **Documentation** (`hindsight-docs/docs/developer/configuration.md`):
   - Add to appropriate section table with Variable, Description, Default
   - Mark if it's hierarchical (can be overridden per-bank)

6. **Env template** (`.env.example`):
   - Add the variable to the appropriate section, commented if optional, with a
     short inline comment describing it (mirror the documentation entry).
   - This file is the single source of truth for the env template:
     `scripts/dev/setup.sh` copies it to `.env`, and `hindsight-embed` ships a
     bundled copy (`hindsight-embed/hindsight_embed/env.example`) that seeds
     embed/profile configs. After editing `.env.example`, re-copy it to the
     embed package (`cp .env.example hindsight-embed/hindsight_embed/env.example`)
     or the `test_bundled_template_matches_repo_root` sync test will fail.

#### Hierarchical vs Static Guidelines

**Hierarchical** (per-bank overridable):
- LLM settings (provider, model, API key, base URL)
- Operation-specific settings (retain mode, chunk size, etc.)
- Feature flags that vary by customer/bank

**Static** (server-level only):
- Infrastructure settings (database URL, port, host)
- Global limits (max concurrent operations)
- System-wide feature flags

## Environment Setup

```bash
cp .env.example .env
# Edit .env with LLM API key

# Python deps
uv sync --directory hindsight-api-slim/

# Node deps (uses npm workspaces)
npm install
```

Required env vars:
- `HINDSIGHT_API_LLM_PROVIDER`: openai, anthropic, gemini, groq, minimax, ollama, lmstudio
- `HINDSIGHT_API_LLM_API_KEY`: Your API key
- `HINDSIGHT_API_LLM_MODEL`: Model name (e.g., gpt-4o-mini, claude-sonnet-4-20250514)

Optional (uses local models by default):
- `HINDSIGHT_API_EMBEDDINGS_PROVIDER`: local (default) or tei
- `HINDSIGHT_API_RERANKER_PROVIDER`: local (default) or tei
- `HINDSIGHT_API_DATABASE_URL`: External PostgreSQL (uses embedded pg0 by default)
- `HINDSIGHT_API_ENABLE_BANK_CONFIG_API`: Enable per-bank config API (default: true)
