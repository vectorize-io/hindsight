# Dependencies — CollabMind Control Plane

## central-api (Python)

Source: `central-api/pyproject.toml`

| Package | Purpose |
|---------|---------|
| `fastapi` | HTTP framework, dependency injection |
| `uvicorn` | ASGI server |
| `pydantic-settings` | Config via env vars (prefix `CENTRAL_API_`) |
| `sqlalchemy[asyncio]` | Async DB (Core only, no ORM) |
| `aiosqlite` | SQLite async driver (dev/test) |
| `asyncpg` | Postgres async driver (production) |
| `alembic` | DB migrations |
| `sqlmodel` | SQLModel table defs (Phase-1 schema, alongside Core tables) |
| `httpx` | Async HTTP client (MemLord adapter) |
| `python-jose` / `PyJWT` | JWT decode/verify |
| `pytest` | Testing |
| `ruff` | Linting |
| `mypy` | Type checking |

Runtime Python: **3.12** (uses `type[T]`, `StrEnum`, `X | Y` union syntax).

## collabmind-api (TypeScript/Node)

Source: `collabmind-api/package.json`

| Package | Purpose |
|---------|---------|
| `express` | HTTP server |
| `jsonwebtoken` | JWT verify (HS256 + RS256) |
| `jwks-rsa` | Authentik JWKS key fetching + cache |
| `pg` | Postgres client (api_keys, platform_settings) |
| `typescript` | Build |

Minimal surface by design — no router framework beyond Express.

## collabmind-memory (TypeScript monorepo)

Source: `package.json` + per-package `package.json` files

| Package | Used by | Purpose |
|---------|---------|---------|
| `express` | memory-controller | HTTP server |
| `pg` | core, memory-controller | Postgres client |
| `pgvector` | core | pgvector extension types |
| `ioredis` | core | Valkey / Redis vector store |
| `@qdrant/js-client-rest` | documents | Qdrant vector DB client |
| `@aws-sdk/client-s3` | documents | S3 object storage (stub in dev) |
| `@aws-sdk/client-bedrock-runtime` | core/stubs | Bedrock embeddings (stubbed) |
| `jsonwebtoken` | memory-controller, auth-middleware | JWT verify |
| `typescript` | all packages | Build |
| `vitest` | memory-controller | Unit tests (write-gate, unify) |

## collabmind-console (Next.js)

Source: `collabmind-console/package.json`

| Package | Purpose |
|---------|---------|
| `next` | App framework |
| `react`, `react-dom` | UI |
| `typescript` | Build |

The console has **no direct dependency on memory backends**. All calls go through `src/lib/api.ts` → `collabmind-api:3050`.

## External Service Dependencies

| Service | Used by | Notes |
|---------|---------|-------|
| **Authentik** | collabmind-api (JWKS), central-api (JWKS config) | OIDC provider for JWT auth. JWKS endpoint cached. |
| **PostgreSQL** | collabmind-api (api_keys), memory-controller (core), central-api (prod) | Single shared DB in the memory stack. |
| **Qdrant** | packages/documents | Document chunk vector search. |
| **Ollama** | packages/core (embeddings) | Local embedding model (nomic-embed-text, 768-dim). |
| **Valkey** | packages/core (vector store) | Redis-compatible, high-perf vector cache. |
| **S3** | packages/documents | Raw document storage (AWS S3 or compatible). |
| **MemLord** | central-api (MemlordAdapter), memory-controller (memlord.ts) | External memory engine. Read-only from federation. API key: `mlk_…`. Surface: `/api-dev/*`. |
| **OpenMemory** | central-api (OpenMemoryAdapter), memory-controller (openmemory.ts) | External memory engine. |
| **Google Drive** | central-api (connectors), memory-controller (source-resources) | Read-only. OAuth2 (drive.metadata.readonly + drive.readonly scopes). |

## Key External API Contracts

### MemLord `/api-dev/*`

- `POST /api-dev/memories/search` — `{query, limit}` → `{results: [{id, content, memory_type, workspace_id, workspace, score}]}`
- `POST /api-dev/memories` — `{page, page_size}` → `{memories: [...]}` (paginated export)
- `GET /health` — liveness

Auth: `X-API-Key: mlk_…`

Note: MemLord's dedicated export (`GET /api/workspaces/{id}/export`) requires cookie/OAuth — the API key cannot reach it. The pagination workaround is intentional.

### Authentik JWKS

- `GET /.well-known/jwks.json` — standard JWKS endpoint
- Token claims used: `sub` (actor_id), `groups` (roles), `scope` (scopes), `aud` (source_app_id)

## Infrastructure (collabmind-memory)

Caddy is used as the edge reverse proxy in the deployed stack (see `config/Caddyfile`). Docker Compose files: `docker-compose.yml` (core stack) + `docker-compose.collabmind-control.yml` (control plane additions).

The Cloudflare tunnel state and edge route policy are documented in `EDGE_TUNNEL_STATE.md` and `AUTH_ROUTE_POLICY.md` in the `collabmind-memory` repo root — those files are manually maintained and should not be modified without reading them first.
