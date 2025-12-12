# AGENTS.md

This document captures architectural decisions and coding conventions for the Hindsight project.

## Documentation

- **Main documentation**: [hindsight-docs/docs/developer/](./hindsight-docs/docs/developer/)
- **Use case patterns**: [hindsight-docs/docs/cookbook/](./hindsight-docs/docs/cookbook/)
- **API reference**: Auto-generated from OpenAPI spec

## Project Structure

```
hindsight/              # Python package for embedded usage
hindsight-api/          # FastAPI server (core memory engine)
hindsight-cli/          # Rust CLI client
hindsight-control-plane/ # Next.js admin UI
hindsight-docs/         # Docusaurus documentation site
hindsight-dev/          # Development tools and benchmarks
hindsight-integrations/ # Framework integrations (LangChain, etc.)
hindsight-clients/      # Generated API clients (Python, TypeScript, Rust)
```

## Core Concepts

### Memory Banks
- Each bank is an isolated memory store (like a "brain" for one user/agent)
- Banks contain: memory units (facts), entities, documents, entity links
- Banks have a **disposition** (personality traits) and **background** (context)
- Bank isolation is strict - no cross-bank data leakage

### Memory Types
- **World facts**: General knowledge ("The sky is blue")
- **Experience facts**: Personal experiences ("I visited Paris in 2023")
- **Opinion facts**: Beliefs with confidence scores ("Paris is beautiful" - 0.9 confidence)

### Operations
- **Retain**: Store new memories (extracts facts, entities, relationships)
- **Recall**: Retrieve memories (semantic, BM25, graph, temporal search)
- **Reflect**: Deep analysis to form new insights/opinions

## API Design Decisions

### Single Bank Per Request
- All API endpoints (`recall`, `reflect`, `retain`) operate on a single bank
- Multi-bank queries are the **client/agent's responsibility** to orchestrate
- This keeps the API simple and the isolation model clear

### Disposition Traits (3-trait system)
- **Skepticism** (1-5): How skeptical vs trusting when forming opinions
- **Literalism** (1-5): How literally to interpret information
- **Empathy** (1-5): How much to consider emotional context
- These influence the `reflect` operation, not `recall`
- Background info also only affects `reflect` (opinion formation)

## Multi-Bank Architecture Patterns

See [hindsight-docs/docs/cookbook/](./hindsight-docs/docs/cookbook/) for detailed guides:

- **Per-User Memory**: One bank per user, simplest pattern
- **Support Agent + Shared Knowledge**: User bank + shared docs bank, client orchestrates

## Developer Guide

### Running the API Server

```bash
# From project root
./scripts/dev/start-api.sh

# With options
./scripts/dev/start-api.sh --reload --port 8888 --log-level debug
```

### Running Tests

```bash
# API tests
cd hindsight-api
uv run pytest tests/

# Specific test
uv run pytest tests/test_http_api_integration.py -v
```

### Generating OpenAPI Spec

After changing API endpoints, regenerate the OpenAPI spec and docs:

```bash
./scripts/generate-openapi.sh
```

This will:
1. Generate `openapi.json` at project root
2. Copy to `hindsight-docs/openapi.json`
3. Regenerate API reference documentation

### Generating API Clients

After updating the OpenAPI spec, regenerate all clients:

```bash
./scripts/generate-clients.sh
```

This generates:
- **Rust client**: `hindsight-clients/rust/` (via progenitor in build.rs)
- **Python client**: `hindsight-clients/python/` (via openapi-generator Docker)
- **TypeScript client**: `hindsight-clients/typescript/` (via @hey-api/openapi-ts)

Note: The maintained wrapper `hindsight_client.py` and `README.md` are preserved during regeneration.

### Running the Documentation Site

```bash
./scripts/dev/start-docs.sh
```

### Running the Control Plane

```bash
./scripts/dev/start-control-plane.sh
```

## Code Style

### Python (hindsight-api)
- Use `uv` for package management
- Async throughout (asyncpg, async FastAPI endpoints)
- Pydantic models for request/response validation
- No py files at project root - maintain clean directory structure

### TypeScript (control-plane, clients)
- Next.js with App Router for control plane
- Tailwind CSS with shadcn/ui components

### Rust (CLI)
- Async with tokio
- reqwest for HTTP client
- progenitor for API client generation

## Database

- PostgreSQL with pgvector extension
- Schema managed via Alembic migrations in `hindsight-api/alembic/`, db migrations happen during api startup, no manual commands
- Key tables: `banks`, `memory_units`, `documents`, `entities`, `entity_links`

# Branding
## Colors
- Primary: gradient from #0074d9 to #009296  
