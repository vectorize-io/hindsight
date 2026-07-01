# Review Notes — CollabMind Control Plane Docs

**Generated:** 2026-06-09  
**Scope:** central-api, collabmind-api, collabmind-memory, collabmind-console

---

## Consistency Findings

### C1 — Sensitivity level naming mismatch

| Location | Levels |
|----------|--------|
| `central-api/app/policy/rules.py` | `public, internal, private, sensitive, secret_blocked` |
| `memory-controller/src/governance/write-gate.ts` | `public, internal, restricted, secret` |
| `memory-controller/src/search/unify.ts` | `public, internal, restricted, secret` |

The Python `private` maps to TS `restricted`; Python `sensitive`+`secret_blocked` maps to TS `secret`. These are semantically equivalent in behavior (both quarantine `restricted/private`, both reject `secret/secret_blocked`) but the enum string values differ. Any code path that compares sensitivity labels cross-language will break.

**Gap:** No canonical shared enum or schema document defines the authoritative names.

### C2 — RRF_K constant defined in two places

Both `central-api/app/retrieval/rrf.py` and `memory-controller/src/search/unify.ts` hardcode `RRF_K = 60`. No single source of truth. A change in one won't propagate.

### C3 — Audit logger is a stub in central-api

`central-api/app/audit/logger.py` stores events in process memory (`_EVENTS: list`). The TS stack writes to `audit_events` in Postgres. This means the Python scaffold produces no durable audit trail. The mismatch is intentional (scaffold) but must not be confused with production behavior.

### C4 — DB schema duplication

`central-api` has two parallel table definitions:
- `app/models/__init__.py` — SQLModel ORM-style tables
- `app/db/tables.py` — SQLAlchemy Core tables

Both define overlapping tables (`source_documents`, `connector_accounts` vs `source_connectors`, etc.) with different column names in some cases. `db/tables.py` is newer and more complete. The SQLModel set may be an older draft. This needs reconciliation before any migration is run.

### C5 — MemLord export workaround documented in two places

`central-api/app/adapters/memlord.py` has an extensive docstring explaining the pagination workaround. `collabmind-memory/packages/memory-controller/src/adapters/memlord.ts` likely has similar handling. Both should be checked for consistency when the MemLord export endpoint changes.

---

## Completeness Gaps

### G1 — No documented connectors other than Google Drive

`central-api/app/config.py` references a `google_client_id` / `google_client_secret`. The connector schema supports `github`, `slack`, `gmail`, `local` types (from `ConnectorRow.connector_type`) but none are implemented. The `packages/connectors` framework in `collabmind-memory` only has an `interface.ts` — no concrete provider implementations beyond the scaffold.

### G2 — `packages/mcp-server` not deeply analyzed

The MCP server in `collabmind-memory/packages/mcp-server` was not read. The tool list and tool signatures may differ from what `central-api/app/mcp/tools.py` defines (the Python contract). This gap affects any agent relying on MCP tool parity.

### G3 — `packages/evaluation` and `packages/model-gateway` not documented

Both packages exist with source code but were not analyzed in depth. `EVALUATION_SCHEMA.md`, `FAILURE_TYPE_REGISTRY.md`, `MODEL_GATEWAY.md`, `MODEL_PROVIDER_REGISTRY.md` in the `collabmind-memory` root contain their documentation — read those files for details.

### G4 — `collabmind-console` route pages not read

Only `src/lib/api.ts` and `src/app/layout.tsx` were read. The page components under `/dashboard`, `/memory`, `/mcp`, `/settings`, `/services` were not analyzed. Their exact API calls and UI behavior are undocumented here.

### G5 — Auth-middleware package role unclear

`collabmind-memory/packages/auth-middleware` exists as a separate package but only the Dockerfile was confirmed. Its relationship to `memory-controller/src/middleware/auth.ts` (shared code? separate deployment?) is not documented.

### G6 — Central-api MCP endpoint not wired

`central-api/app/main.py` does not include a `/mcp` route. `app/mcp/tools.py` defines the 7 tools as plain async functions but there is no MCP server / route registration visible in the scaffold. The tools are callable from tests or internal code only — not via HTTP in the current scaffold.

### G7 — Google Drive webhook / delta sync not implemented

The connector is manual-sync only (v0.1). `source_sync_state.cursor` column exists for future delta sync support, but no webhook or background-sync code exists. Documented in `central-api/app/connectors/routes.py` docstring.

### G8 — `collabmind-memory` `NO_TOUCH_FILES.md` and `DO_NOT_TOUCH_TUNNEL.md` contents not read

These files likely contain operational constraints. An agent modifying `collabmind-memory` should read them before changing infra-related files.

---

## Recommendations

1. **Resolve sensitivity level naming** (C1): Define a canonical mapping document or adopt one set of names across both repos.
2. **Reconcile SQLModel vs Core tables** (C4): Determine which definition is authoritative in `central-api` before running Alembic.
3. **Read `NO_TOUCH_FILES.md`** before modifying anything in `collabmind-memory` infra.
4. **Read `DEPLOYMENT_GAPS.md`** for known deployment blockers (hardcoded secrets, missing Authentik config).
5. **MCP tool parity check** (G2): Compare `packages/mcp-server/src/` with `central-api/app/mcp/tools.py` before assuming tool signatures match.
