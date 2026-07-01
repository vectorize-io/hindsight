# Codebase Info — CollabMind Control Plane

## Scope

Four repos collectively form the CollabMind memory control plane:

| Repo | Language | Status | Port |
|------|----------|--------|------|
| `central-api` | Python 3.12, FastAPI | Scaffold / contract target | 8000 |
| `collabmind-api` | TypeScript, Express | **Running** — governance-edge proxy | 3050 |
| `collabmind-memory` | TypeScript monorepo | **Running** — memory + retrieval backend | 3020 (memory-controller) |
| `collabmind-console` | Next.js 14 | **Running** — operator GUI | 3000 |

`central-api` captures the final Python contract. `collabmind-api` + `collabmind-memory` are the working implementation. The two are kept in sync intentionally — same governance logic, same RRF constant, same sensitivity levels.

## Repo Roots

```
/Users/oliververmeulen/XAI/central-api/
/Users/oliververmeulen/XAI/collabmind-api/
/Users/oliververmeulen/XAI/collabmind-memory/
/Users/oliververmeulen/XAI/collabmind-console/
```

## central-api layout

```
app/
  main.py              FastAPI app, lifespan, router registration
  config.py            Settings (pydantic-settings, env prefix CENTRAL_API_)
  adapters/            BaseMemoryAdapter ABC + engine registry
    base.py            AdapterHit, EngineHealth, abstract methods
    __init__.py        Registry: get_write_adapter, get_memory_search_adapters, get_code_adapter
    internal.py        InternalAdapter (scaffold stub)
    memlord.py         MemlordAdapter — read-only, real HTTP client, /api-dev/* surface
    coderag.py         CodeRAGAdapter stub
    openmemory.py      OpenMemoryAdapter stub
  auth/
    context.py         RequestContext, get_context(), ContextDep
    permissions.py     has_scope(), require_scope(), within_confidentiality()
    jwt.py             validate_jwt() — module boundary
  policy/
    redactor.py        redact_secrets() — 8 regex patterns, returns (content, count, kinds)
    classifier.py      classify_sensitivity()
    rules.py           run_write_gate(), PolicyDecision, Sensitivity, Status
  memory/
    routes.py          /api/memory/* — store/search/update/delete/export
    service.py         orchestration: write-gate → adapter → audit
    schemas.py         Pydantic I/O schemas
  retrieval/
    routes.py          /api/retrieval/code/search, /api/retrieval/docs/search, /api/context/build, /api/context/preview
    rrf.py             reciprocal_rank_fusion() — RRF_K=60
    context_pack.py    build_context_pack() — pure function
    schemas.py         RetrievalHit, ContextPack, ContextItem
  governance/
    retrieval_policy.py  evaluate_retrieval() — fail-closed, Principal, Decision
    agent_control.py     decide(), request_action() — allowed|denied|requires_approval
  audit/
    logger.py          log_event(), new_trace_id() — in-memory stub
    schemas.py         AuditEvent
    routes.py          GET /api/audit/events
  mcp/
    tools.py           7 MCP tools (list_workspaces, list_connected_sources,
                       list_source_documents, sync_source, get_source_audit,
                       list_ingestion_jobs, search_governed_documents)
  connectors/
    routes.py          /connectors — Google Drive OAuth, list, status, sync
    google_drive/      service.py, oauth.py, audit_actions.py
  controlplane/
    routes.py          /controlplane — users, workspaces, workspace_members
    deps.py            PrincipalDep, require_member()
  operator/
    routes.py          /api/operator/review, /approve, /reject
    schemas.py         Approval schemas
  health/
    routes.py          GET /health, GET /api/health/engines
  db/
    tables.py          11 SQLAlchemy Core tables
    engine.py          async engine, init_models()
    repositories.py    All DB queries (no ORM relations)
    ids.py             new_id(), utcnow()
  models/
    __init__.py        SQLModel Phase-1 table set (9 tables)
tests/
  test_health.py
  test_policy_redaction.py
  test_memory_contract.py
  test_retrieval_contract.py
  test_memlord_adapter.py
alembic/               DB migrations
migrations/            Alternate migration path (versions/0001_initial_control_plane.py)
pyproject.toml         Python 3.12, FastAPI, pydantic-settings, sqlalchemy, httpx, pytest
```

## collabmind-api layout

```
src/
  index.ts             Express app, port 3050, route wiring
  auth.ts              requireAuth() — Authentik JWKS JWT + mem11_sk_ API-key, DB lookup
  internal-context.ts  signContext() / verifyContext() — HMAC-SHA256 X-CM-Context header
  proxy.ts             proxyTo() / proxyFetch() — context-injecting upstream proxy
  routes/
    memory.ts          /api/memory → proxy to memory-controller:3020/memory/*
    documents.ts       /api/documents → proxy
    settings.ts        /api/settings, /api/secrets, /api/services — direct Postgres queries
    graph.ts           /api/graph
    mcp.ts             /mcp
    evaluation.ts      /api/evaluation
    admin.ts           /api/admin
    model.ts           /api/model
    auth.ts            /auth
    health.ts          /health
```

## collabmind-memory layout

```
packages/
  core/src/
    core/              db.ts, identity.ts, lifecycle.ts, audit.ts, operations.ts, models.ts
    memory/            hsg.ts (HSG engine), embed.ts, decay.ts, identifiers.ts, user_summary.ts
    memory/vector/     postgres.ts (pgvector), valkey.ts
    temporal/          store.ts, query.ts, timeline.ts, types.ts
    connectors/        source-resource-registry.ts
    ops/               dynamics.ts
    utils/             chunking.ts, keyword.ts, text.ts
    schema/            SQL migrations 00–15
    migrations/        run.ts, seed.ts
  memory-controller/src/
    index.ts           Express app, port 3020
    router.ts          Route registration
    middleware/        auth.ts (X-CM-Context + JWT + API-key), validation.ts
    internal-context.ts  contextFromHeader() — verifier side
    governance/        write-gate.ts (redactSecrets, classifySensitivity, runWriteGate)
    search/            unify.ts (unifySearch — RRF + dedup + confidentiality filter)
    adapters/          interface.ts, internal.ts, openmemory.ts, memlord.ts, documents.ts
    routes/            memory.ts (CRUD+search), source-resources.ts
  documents/src/
    index.ts           Express app
    routes/            documents.ts, retrieval.ts, indexing.ts, collections.ts, ingestion-jobs.ts
    db/                documents.ts, retrieval.ts
    parsers/           chunker.ts, text.ts
    storage/           s3.ts
    vector/            qdrant.ts
    embeddings/        factory.ts, interface.ts, mock.ts
  connectors/src/
    providers/         interface.ts
    context.ts
    types.ts
  mcp-server/src/      MCP server
  evaluation/src/      Evaluation + correction system
  model-gateway/src/   Model routing
  auth-middleware/src/ Shared auth middleware package
```

## collabmind-console layout

```
src/app/
  page.tsx             Root — redirects to /dashboard
  layout.tsx           Shell layout
  dashboard/page.tsx   Dashboard view
  memory/              Memory explorer
  mcp/                 MCP tool browser
  settings/            Settings UI
  services/            Services status
src/lib/
  api.ts               Single API client → collabmind-api:3050
```

## Key env vars

| Var | Used by |
|-----|---------|
| `CENTRAL_API_DATABASE_URL` | central-api DB |
| `CENTRAL_API_AUTHENTIK_JWKS_URL` | central-api JWT validation |
| `CENTRAL_API_MEMLORD_API_KEY` | central-api MemlordAdapter (mlk_…) |
| `AUTHENTIK_JWKS_URL` | collabmind-api JWT |
| `AUTHENTIK_ISSUER_URL` | collabmind-api JWT |
| `DATABASE_URL` | collabmind-api (api_keys table), memory-controller |
| `INTERNAL_CONTEXT_SECRET` | collabmind-api + memory-controller HMAC signing |
| `MEMORY_CONTROLLER_URL` | collabmind-api → memory-controller (default: http://memory-controller:3020) |
| `NEXT_PUBLIC_COLLABMIND_API_URL` | collabmind-console → API (default: http://localhost:3050) |
| `NEXT_PUBLIC_COLLABMIND_API_KEY` | collabmind-console dev key |
