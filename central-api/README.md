# CollabMind Central API (FastAPI)

Contract-first scaffold of the **memory control plane** Central API. This is the
target Python service described in
[`../docs/architecture/memory-control-plane.md`](../docs/architecture/memory-control-plane.md).

> **Status: scaffold only.** Adapters are stubs, auth is a module boundary (no real
> JWKS calls), and no engine traffic is moved. The Node/TS prototype
> (`../collabmind-api` + `../collabmind-memory`) remains the working reference and is
> untouched. This service captures the **final contract and module boundaries**.

## What's real vs stub

| Area | State |
|------|-------|
| Phase 3 route contract + Pydantic schemas | ✅ real |
| Adapter abstract base + registry | ✅ real |
| Engine adapters (memlord/coderag/openmemory/internal) | 🔶 stubs (controlled mocks / NotImplementedError) |
| Policy redactor / classifier / rules | ✅ real (pattern-based) |
| Context-pack builder + RRF | ✅ real (pure functions) |
| Authentik JWT validation | 🔶 module boundary + config only (no JWKS) |
| Audit logger | 🔶 in-memory stub (no DB yet) |
| DB / Qdrant / pgvector | ⛔ not implemented |

## Run

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
# docs at http://localhost:8000/docs
```

## Test & lint

```bash
ruff check .
pytest -q
```

## Contract (Phase 3)

```text
GET  /health                       GET  /api/health/engines
POST /api/memory/search            POST /api/memory/store
POST /api/memory/update            POST /api/memory/delete
POST /api/memory/export
POST /api/retrieval/code/search    POST /api/retrieval/docs/search
POST /api/context/build            POST /api/context/preview
GET  /api/operator/review          POST /api/operator/approve
POST /api/operator/reject
GET  /api/audit/events
```

## Layout

```text
app/
  main.py config.py
  auth/      jwt.py permissions.py context.py
  adapters/  base.py internal.py memlord.py coderag.py openmemory.py
  policy/    classifier.py redactor.py rules.py
  memory/    routes.py schemas.py service.py
  retrieval/ routes.py schemas.py context_pack.py rrf.py
  operator/  routes.py schemas.py
  audit/     routes.py schemas.py logger.py
  health/    routes.py
tests/
```
