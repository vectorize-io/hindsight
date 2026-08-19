---
name: code-review
description: Review changed code against project standards. Checks for missing tests, dead code, type safety, lint issues, and coding conventions. Run after completing any implementation work.
user_invocable: true
---

# Code Review

Review all changed code against the project's quality standards and coding conventions.

## Code Standards

Read and internalize these standards before writing code. The review steps below verify compliance.

### Python Style
- Python 3.11+, type hints required
- Async throughout (asyncpg, async FastAPI)
- Pydantic models for request/response
- Ruff for linting (line-length 120)
- No Python files at project root - maintain clean directory structure
- **Never use multi-item tuple return values** — not even for internal/private functions. Always use a dataclass or Pydantic model. No exceptions, no "it's just two values" shortcuts. If a function returns more than one value, define a named type for it.

### Type Safety with Pydantic Models
**NEVER use raw `dict` types for structured data** — this applies to all code, including internal helpers and private functions. If the dict has known keys, it must be a dataclass or Pydantic model:
- Use Pydantic `BaseModel` for all data structures passed between functions
- Use `@dataclass` for lightweight internal data containers when Pydantic validation isn't needed
- Add `@field_validator` for type coercion (e.g., ensuring datetimes are timezone-aware)
- Avoid `dict.get()` patterns - use typed model attributes instead
- Parse external data (JSON, API responses) into Pydantic models at the boundary
- This catches type errors at parse time, not deep in business logic
- The only acceptable `dict` usage is for truly dynamic/unknown keys (e.g., arbitrary metadata, JSON blobs with no fixed schema)

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

### Code Comments
- **Always comment non-trivial technical decisions** with the reasoning behind the choice. If someone would ask "why is it done this way?", there should be a comment.
- **Keep comments up to date with history** — when changing an approach, update the comment to explain what was tried before and why it was changed. Comments serve as a tracker of previous implementations that likely had problems.
- Don't comment obvious code — only where the "why" isn't self-evident from the code itself.

```python
# BAD - no context for future readers
results = await asyncio.gather(*tasks, return_exceptions=True)

# GOOD - explains the non-obvious choice
# Use return_exceptions=True to avoid cancelling sibling tasks on failure.
# Previously we used TaskGroup but it cancelled all tasks when one failed,
# causing partial writes that left orphaned entity links (see #412).
results = await asyncio.gather(*tasks, return_exceptions=True)
```

### API Layer & Data Access
- **No direct database access in `api/http.py`** (or any API router). HTTP handlers must not build SQL, call `acquire_with_retry` / `conn.fetch` / `conn.fetchrow` / `conn.execute`, or reference `fq_table(...)`. All persistence and queries live in `MemoryEngine` (the engine layer). A handler parses/validates the request, calls an engine method, shapes the HTTP response, and maps domain results to status codes (e.g. a `None` return → 404).
- **Authentication/tenancy is enforced inside each engine method, not assumed by the handler.** Every engine method that touches bank-scoped data must authenticate via `request_context` — typically `await self._authenticate_tenant(request_context)` (often indirectly through `get_bank_profile(...)`) — so the correct tenant schema is resolved before any query runs. Handlers must thread `request_context` through to the engine method; never query a tenant-scoped table assuming the schema is already set.
- Engine methods return typed models (Pydantic/dataclass), not raw dicts (see Type Safety).
- **Every list endpoint paginates, following the existing ones.** A `GET` that returns a collection whose size grows with the data (banks, documents, memories, entities, operations, webhook deliveries, audit logs, …) must take `limit`/`offset` and bound its result — an unbounded list is an unbounded payload plus unbounded per-row work (per-item counts, config resolution, embedding hydration). Copy the shape `list_documents` uses, don't invent a new one: `limit: int = Query(default=100, ge=0)` and `offset: int = Query(default=0, ge=0)` on the handler, matching keyword args on the engine method, and a response carrying the page **plus `total`, `limit`, `offset`** so a client knows when to stop. Add a `q` search param when the collection is something a user picks from in a UI — client-side filtering only ever sees the loaded page. Bounded-by-construction endpoints are the exception, not the rule: a tree/export that is whole-structure by design, or a table capped at write time (e.g. `observation_history` / `mental_model_history`, trimmed to `*_max_entries` on insert). If it isn't bounded, paginate it.

### Bank/Tenant Isolation in Queries
- **Bank isolation is a hard security invariant: no query may read, count, update, or delete another bank's rows.** Tenant isolation is enforced at the schema level (the resolved `search_path` / `fq_table(...)` qualifier, gated by `_authenticate_tenant`); bank isolation is enforced *within* a schema by a `bank_id` predicate on every statement that touches a multi-bank table.
- **Every SQL statement against a multi-bank table must be constrained by `bank_id`** — directly in the `WHERE`, or transitively (see below). Multi-bank tables carry a `bank_id` column: `memory_units`, `documents`, `entities`, `entity_links`, `mental_models`, `knowledge_pages`, `memory_links`, `observation_history`, and similar.
- **The trap: filtering by a caller-supplied, non-globally-unique key without `bank_id`.** Keys like `document_id` and `mental_models.id` are unique only *per bank* (their PK is composite, e.g. `(id, bank_id)`), so the *same* id legally exists in every bank. A statement like `UPDATE memory_units SET tags = $1 WHERE document_id = $2` — no `bank_id` — silently reads/writes **every** bank's rows that share the id. This is the exact defect from #3429/#3430. Adding `AND bank_id = $n` fixes it.
- **Three ways a statement is legitimately scoped** (accept these; flag anything that fits none):
  1. **Explicit** `WHERE ... AND bank_id = $n`.
  2. **Globally-unique single-column PK.** Filtering by a global uuid PK (`memory_units.id`, `entities.id`, `knowledge_pages.id`) or a bank-encoded key (`chunks.chunk_id` is `{bank_id}_{document_id}_{idx}`) cannot collide across banks. Contrast the *composite*-PK ids (`documents.id`/`document_id`, `mental_models.id`) — those are dangerous and MUST carry `bank_id`.
  3. **Transitive.** Junction tables without a `bank_id` column (`unit_entities`, `entity_cooccurrences`, `observation_sources`) are safe only when reached through globally-unique unit/entity ids that were themselves selected from a bank-scoped query in the same call, and edges are intra-bank by construction. If the id set could contain another bank's ids, it is not scoped.
- **Watch two smells:** (a) a caller-supplied id used in the `WHERE` with no adjacent `bank_id`, while a *neighbouring* statement in the same method does carry `bank_id` (asymmetry is the tell); (b) a `bank_id` predicate applied only under `if bank_id:` with a `bank_id: str | None = None` default — latent even if all current callers pass one.
- **Cross-bank by design must rewrite `bank_id` to the destination.** The transfer/import path is the only one that legitimately crosses banks; verify every write pins the *destination* `bank_id` and never inherits a source row's `bank_id`.

### Database Locking
- **Never use PostgreSQL advisory locks** (`pg_advisory_lock`, `pg_try_advisory_lock`, `pg_advisory_xact_lock`, `pg_advisory_unlock`, …) in migrations, engine code, or anything else. Hindsight runs against connection poolers and managed/PG-compatible services where advisory locks are unreliable or unsupported: session-level locks silently leak or vanish when a pooler hands the session to another client, and callers can block forever on a lock the server never grants. Reject any new occurrence, including ones that look "safe" because they are transaction-scoped.
- The pre-existing usage in `hindsight_api/migrations.py` is grandfathered, not a precedent — it is tracked for removal. Don't copy it.
- Design the concurrency out instead of locking around it: give each process its own object to write (e.g. per-schema DDL rather than a shared `public.` object), make the operation idempotent, or use a real row/table constraint (`INSERT ... ON CONFLICT`, `SELECT ... FOR UPDATE` in a fixed order). See #2690 for a migration that reached for `pg_advisory_xact_lock` and had to be reverted.

### Branch Hygiene
- **Always start new feature branches from `origin/main`** — rebase to ensure a clean base.
- **Only include commits relevant to the PR/branch/feature** — no unrelated changes. If the branch contains commits that don't belong, they must be removed before merging.

### General Principles
- Don't add features, refactor code, or make "improvements" beyond what was asked
- Don't add unnecessary error handling for impossible scenarios
- Don't create helpers or abstractions for one-time operations
- No backwards-compatibility hacks (unused vars, re-exports, "removed" comments)
- Three similar lines of code is better than a premature abstraction

## Review Steps

### 1. Check branch hygiene

- Run `git log --oneline main..HEAD` to list all commits on the branch.
- Verify every commit is relevant to the feature/PR. Flag any unrelated commits.
- Check the branch is based on a recent `origin/main` (no stale base).

### 2. Identify changed files

Run `git diff --name-only HEAD` (unstaged) and `git diff --cached --name-only` (staged) to get all changed files. If there are no local changes, diff against the base branch using `git diff main...HEAD --name-only` and `git diff main...HEAD` to review all commits on the current branch.

### 3. Run linters

```bash
./scripts/hooks/lint.sh
```

Report any failures. Do NOT fix them yourself — just report.

### 4. Check for dead code

For each changed Python file, check for:
- Unused imports (Ruff should catch these, but verify)
- Functions/methods/classes that were added but are never called from anywhere
- Variables assigned but never read
- Commented-out code blocks that should be removed

For each changed TypeScript file, check for:
- Unused imports
- Unused variables or functions
- Commented-out code

### 5. Check type safety (Python)

For each changed Python file, check for violations:
- **No raw `dict` for structured data** — must use Pydantic model or dataclass, even for internal/private functions (only exception: truly dynamic/unknown keys)
- **No multi-item tuple returns** — must use dataclass or Pydantic model, even for internal/private functions (no exceptions)
- **Missing type hints** on function parameters and return types
- **Missing `@field_validator`** for datetime fields that should be timezone-aware

### 6. Check for missing tests

For each new or significantly changed function/endpoint/class:
- Check if there is a corresponding test addition or update
- New API endpoints MUST have integration tests
- New utility functions MUST have unit tests
- Bug fixes SHOULD have a regression test

Flag any new logic that lacks test coverage.

**LLM-behaviour changes need a real-LLM judge test, not MockLLM.** If the change alters how the model interprets a prompt — fact/observation extraction, `fact_type` (world/experience) classification, speaker attribution, instruction-following, prompt wording — there MUST be a test marked `pytest.mark.hs_llm_core` that runs the real pipeline and asserts via `tests.llm_judge.assert_meets_criteria` (not string/enum matching). Flag these as findings:
- A prompt/classification change verified only by MockLLM or string assertions (MockLLM echoes input — such tests pass spuriously). **Should fix.**
- A test that hard-asserts `fact_type == "world"/"experience"` (or other model-decided output) instead of judging it — non-deterministic, will flake across providers/runs. **Should fix** (move the classification check into the judge `criteria`; keep only genuinely deterministic structural asserts direct).
- Deterministic mechanics (prompt assembly, suppression/branching logic) that are covered *only* by a slow LLM test — these should also have fast non-LLM unit tests. **Note.**

See CLAUDE.md → Key Conventions → Testing for the full pattern.

### 6a. Check tests assert memory state via the engine API, not raw SQL

Tests must verify what a retain / recall / consolidation produced by calling the public
`MemoryEngine` read API — `list_memory_units` (units and their `metadata` / `tags`; counts via
`total`; `document_id` / `fact_type` / `entity_id` filters), `list_entities` (canonical names,
mention counts), `get_graph_data` (nodes/edges), `get_bank_stats`, `recall_async` — **not** by
reaching into the memory tables (`memory_units`, `memory_links`, `unit_entities`) with raw SQL via
`pool.acquire()` / `conn.fetch*`. Asserting on those tables couples the test to a storage-layer
detail and checks a proxy instead of the observable property (see **General Principles** → tests
assert the property, and the handler rule in **7b**).

**Flag as should fix** any added or changed test whose assertion runs a `SELECT` / `COUNT` against
`memory_units` / `memory_links` / `unit_entities` where an engine read method returns the same
fact. Prime tell: `async with pool.acquire() as conn:` followed by `SELECT ... FROM memory_units`
inside a test body; a `fetchval("SELECT count(*) FROM memory_units ...")` that `list_memory_units`
`["total"]` would return; a `canonical_name` query that `list_entities` covers.

Direct SQL on those tables is legitimate **only** when it forces or inspects internal state the
public API cannot express — e.g. an `UPDATE documents SET updated_at` that forges a race, or a
raw `memory_links` row-count that the deduped `get_graph_data` edge list cannot reproduce. Those
must carry a comment saying why the direct access is necessary; flag any that do not.

### 7. Check API consistency

If any files in `hindsight-api-slim/hindsight_api/api/` were changed:
- Were the OpenAPI specs regenerated? (`./scripts/generate-openapi.sh`)
- Were the client SDKs regenerated? (`./scripts/generate-clients.sh`)
- Were the control plane proxy routes updated? (`hindsight-control-plane/src/app/api/`)

### 7a. Check TS/Python wrapper-client parity

Two of the generated SDKs ship a **hand-written, maintained convenience wrapper** on top of the auto-generated low-level client — and *only* these two:
- **TypeScript**: `hindsight-clients/typescript/src/index.ts` (`HindsightClient`)
- **Python**: `hindsight-clients/python/hindsight_client/hindsight_client.py` (`Hindsight`)

(The Rust/Go/etc. clients are generated-only — no wrapper to keep in sync.)

These wrappers are what most third-party consumers actually call, and they must expose the same surface. **If a change touches one wrapper's method — adds/removes a parameter, changes a default, forwards a new query/body field — the equivalent method in the *other* wrapper must get the same change in the same (or an immediately-following) PR.** A parameter that exists in the generated SDK but is dropped by one wrapper silently strips it for every consumer of that language (this is exactly what #2975 / #3042 fixed for `detail`/`tags_match`/`limit`/`offset` on `listMentalModels`/`getMentalModel`). **Should fix** — flag any wrapper method that gains capabilities in one language but not the other, and add a matching mapping regression test on both sides.

Note: the `client-coverage-check` CI tool only validates **request-body** fields, not GET **query** parameters — so query-param parity gaps are *not* caught automatically and must be checked by hand here.

### 7b. Check API-layer data-access boundary

For each changed handler in `hindsight-api-slim/hindsight_api/api/` (e.g. `http.py`, `mcp.py`):
- **Flag any direct DB access in the handler** — `acquire_with_retry`, `conn.fetch` / `fetchrow` / `execute`, raw SQL strings, or `fq_table(...)`. These are a **must fix**: the query must be moved into a `MemoryEngine` method that returns a typed model, and the handler must call that method.
- **Verify authentication is enforced in the engine** — the handler must delegate to an engine method that authenticates via `request_context` (`_authenticate_tenant`, typically through `get_bank_profile`). A handler that reads/writes tenant-scoped data without an engine method enforcing auth is a **must fix** (tenant data could leak across schemas).

### 7c. Check bank/tenant query scoping

For **every SQL statement added or changed** in the diff (grep the diff for `conn.fetch`, `conn.fetchrow`, `conn.fetchval`, `conn.execute`, `executemany`, and any raw `SELECT`/`INSERT`/`UPDATE`/`DELETE` f-strings, including multi-line ones), verify it cannot touch another bank's rows — see **Bank/Tenant Isolation in Queries** above.

For each statement against a multi-bank table (`memory_units`, `documents`, `entities`, `entity_links`, `mental_models`, `knowledge_pages`, `memory_links`, `observation_history`, …), confirm it is scoped by one of the three legitimate mechanisms:
1. explicit `AND bank_id = $n`;
2. a globally-unique single-column PK (`*.id` uuid, or the bank-encoded `chunks.chunk_id`) — **not** a composite-PK id like `documents.id`/`document_id` or `mental_models.id`;
3. transitively, through a globally-unique id set that was itself selected from a bank-scoped query in the same call.

**Flag as a must fix** any statement filtering a multi-bank table by a caller-supplied, non-globally-unique key (`document_id`, `mental_models.id`, an entity name, …) with **no** `bank_id` predicate — construct the concrete two-bank scenario (two banks share the id; the statement reads/counts/updates/deletes the wrong bank's rows or over-reports) to confirm it's real before flagging. Prime tells: a `bank_id`-carrying sibling statement right next to a `bank_id`-less one; a `WHERE bank_id` guarded by `if bank_id:` with a `None` default; an import/transfer write that inherits a source `bank_id` instead of pinning the destination.

### 7d. Check list endpoints paginate

For every added or changed `GET` handler that returns a collection, confirm it takes `limit`/`offset` and returns `total` — see **API Layer & Data Access** above for the exact shape. Then check the fix is real end to end, since a param that nothing enforces is worse than none:

- **The bound reaches the work, not just the response.** Verify the page size actually limits the expensive part — the SQL `LIMIT`/`OFFSET`, or (when paging must happen after an in-process filter, as in `list_banks` where the `filter_bank_list` extension hook can drop any bank) an explicit slice with the per-item work — live store counts, `get_bank_configs`, re-embedding — done for the page only. Paging in SQL *before* a filter that can drop rows is a **must fix**: it hands back short or empty pages and a `total` that counts rows the caller can't see.
- **Every in-repo consumer pages.** A new default `limit` silently truncates callers that used to get everything: the control plane (`src/lib/api.ts` + the `src/app/api/` proxy route + any context/selector that holds the full list), the CLI (`hindsight-cli/src/api.rs`), MCP tools, and the Zapier dynamic dropdowns. Each must either page through to completion or expose paging in its UI — flag any consumer left on a single default-sized page.
- **Search moves server-side with it.** A picker that filtered client-side over the full list now only filters the loaded page. If the endpoint gained `q`, the UI must send it (and disable its local filtering, e.g. cmdk's `shouldFilter={false}`); if it didn't, say why the collection is small enough not to need it.
- **Tests that look up their own row must not depend on landing on page 1** — they should search or pass an explicit `limit`, not rely on default ordering.

### 8. Check code comments

For each non-trivial change:
- **New non-obvious logic** — is there a comment explaining the reasoning?
- **Changed approach** — does the comment include what was done before and why it changed?
- **Stale comments** — do existing comments near the changed code still accurately describe the behavior?

### 9. Check integration completeness

If any files in `hindsight-integrations/` were added or changed, verify:
- **Tests exist** — the integration must have tests that simulate/exercise the external framework (not just pure unit tests of helpers). Check for a `tests/` directory with meaningful test files.
- **CI job exists** — check `.github/workflows/test.yml` for a corresponding `test-<name>-integration` job. If missing, flag it.
- **Release process** — check that the integration name is in the `VALID_INTEGRATIONS` array in `scripts/release-integration.sh` AND in the `INTEGRATIONS` dict in `hindsight-dev/hindsight_dev/generate_changelog.py` (the changelog generator keeps its own list; a release fails at the changelog step if the name is missing there). If either is missing, flag it.
- **Docs gallery + sidebar entry** — the integration must have an entry in `hindsight-docs/src/data/integrations.json`. This file is the **single source of truth** that drives both the integrations gallery and the docs sidebar (the sidebar category is injected from it at render time across all docs versions). The entry needs an internal `/sdks/integrations/<slug>` `link` and a matching page at `hindsight-docs/docs-integrations/<slug>.md(x)`. The `hindsight-docs/scripts/check-integrations.mjs` build step enforces both directions — forward: every internal JSON entry has a doc page; reverse: every released tag (`integrations/<name>/vX.Y.Z`) appears in the JSON (private infra like `cloudflare-oauth-proxy` is in the script's `EXCLUDED` set). Flag any integration that is released (or being released) but missing from `integrations.json`, and any JSON entry without a doc page. Do **not** hand-edit `versioned_sidebars/*.json` to add integration links — they are positional placeholders filled from the JSON.
- **Code standards** — the integration code must follow all Python style rules (type hints, no raw dicts, no tuple returns, etc.).

### 9a. Check parity across sibling implementations

Whenever the same capability is implemented once per *variant* — one per harness, per language, per
dialect, per provider — the new or changed variant is where a capability silently goes missing. The
defect never looks like a bug in the diff: the code that's wrong is the code that **isn't there**,
and every existing test still passes because the sibling that forgot is by definition the one nobody
wrote a test for. That is how dsh shipped in daemon mode without ever starting a daemon (#3524):
`ensureDaemon` sat in the hook-only wrappers, so all five persistent-plugin harnesses lacked it.

Known sibling families in this repo (this is not the whole list — the rule is about the *shape*):

| Family | Where |
|---|---|
| Coding-agent harnesses | `hindsight-integrations/coding-agents/src/` (hook harnesses vs. persistent-plugin harnesses: dsh, opencode, Kilo, Cline, Prime Agent) |
| Wrapper SDK clients | TypeScript + Python wrappers — see step 7a |
| Alembic migrations | `_pg_upgrade` / `_oracle_upgrade` in every migration |
| Dataplane ↔ control plane | `api/http.py` params vs `hindsight-control-plane/src/app/api/**` proxy routes + `lib/api.ts` |
| LLM providers | per-provider branches in `engine/llm_wrapper.py` |

**Procedure — do this by hand; no linter catches it.** When the diff adds a new sibling, or changes
one sibling of a family:

1. **Enumerate the family.** List every existing sibling (`ls` the directory, grep the registry).
2. **Diff the capability list, not the code.** For each capability the *other* siblings have —
   lifecycle hooks called, setup/teardown performed, config flags honoured, opt-outs respected,
   registry/installer/docs entries — confirm the changed sibling has it, or that its absence is
   deliberate and commented. Grep is the tool: `grep -rn ensureDaemon src` proves who calls it.
3. **Prefer hoisting over copying.** If the capability now exists in N places, the fix is usually to
   move it into the one path every sibling already shares (e.g. `RuntimeCore`, `buildHookOutput`),
   not to paste an Nth copy that the N+1th sibling will forget again.
4. **Demand a structural guard, not just a unit test.** A test for the sibling that forgot doesn't
   exist by construction, so ask for a test that asserts *over the whole family*: enumerate the
   siblings from the filesystem/registry and assert each satisfies the contract, with an explicit,
   commented exemption list. Precedents: `registry covers every installable harness`
   (`harness/registry.test.ts`), `every harness entrypoint reaches a daemon` (`core/daemon.test.ts`),
   `test_backup_tables_covers_entire_schema`, `test_migration_shape.py`.

Flag a capability present in every sibling but one as a **must fix** — state which siblings have it,
which doesn't, and what the user-visible symptom is (for #3524: every `hindsight_*` tool call fails
with ECONNREFUSED and nothing ever starts the daemon). A new sibling family member landing with no
family-wide guard test is a **should fix**.

### 10. Check MCP tool registration completeness

If any new MCP tools were added or existing tools renamed in `hindsight-api-slim/hindsight_api/mcp_tools.py`:
- **`_ALL_TOOLS` set** in `mcp_tools.py` — must include the new tool name
- **`tools_to_register` default set** in `register_mcp_tools()` in `mcp_tools.py` — must include the new tool name
- **`_SINGLE_BANK_TOOLS` set** in `hindsight-api-slim/hindsight_api/api/mcp.py` — must include the new tool if it is bank-scoped (not a bank-management tool like `list_banks`/`create_bank`)
- **`MCP_TOOL_GROUPS`** in `hindsight-control-plane/src/components/bank-config-view.tsx` — must include the new tool in the appropriate group for the UI tool selector
- **Tool count assertions** in tests (e.g., `test_mcp_tools.py`) — must be updated to reflect the new count

### 11. Check backup/restore table coverage

If a migration adds a new PostgreSQL table (look for `CREATE TABLE` / `op.create_table` in `hindsight-api-slim/hindsight_api/alembic/versions/`):
- **`BACKUP_TABLES`** in `hindsight-api-slim/hindsight_api/admin/cli.py` — must include the new table, placed after any table it references via foreign key (parents before children). A missing entry is silent data loss: the table is never backed up, and restore's `TRUNCATE banks CASCADE` wipes any FK-to-banks child (e.g. `mental_models`, `directives`) on restore even though it was never saved.
- The guard test `test_backup_tables_covers_entire_schema` in `tests/test_admin_backup_restore.py` enforces this — flag it as a **must fix** if a new table is absent from `BACKUP_TABLES`.
- Oracle-only tables (e.g. `observation_sources`) are intentionally excluded — admin backup/restore is PostgreSQL-only.

### 11b. Check new config flags update the env template

If the diff adds a new configuration field (a new `ENV_*` / `HINDSIGHT_*` env var
in `hindsight-api-slim/hindsight_api/config.py`):
- **`.env.example`** (repo root) — must add the variable (commented if optional)
  alongside the docs entry in `hindsight-docs/docs/developer/configuration.md`.
  A flag added to `config.py` but absent from `.env.example` is a **should fix**.
- **`hindsight-embed/hindsight_embed/env.example`** — the bundled copy must stay
  byte-identical to the repo-root `.env.example` (it seeds embed/profile configs).
  The `test_bundled_template_matches_repo_root` sync test fails on drift; if the
  root file changed without re-copying, flag it as a **must fix**.

### 11c. Check for advisory locks

Grep the diff for `advisory` (`git diff main...HEAD | grep -in advisory`). Any new
`pg_advisory_lock` / `pg_try_advisory_lock` / `pg_advisory_xact_lock` /
`pg_advisory_unlock` call is a **must fix** — see Database Locking above. Point the
author at the alternatives (per-process objects, idempotent DDL, row-level
constraints) rather than just asking them to drop the lock.

### 12. Review against other coding standards

Check the diff for violations of the standards listed above:
- Python files at project root (not allowed)
- Missing async patterns (should be async throughout)
- Pydantic models for request/response
- Line length > 120 chars
- New features/code beyond what was asked (over-engineering)
- Unnecessary error handling for impossible scenarios
- Premature abstractions or speculative helpers
- Backwards-compatibility hacks (unused vars, re-exports, "removed" comments)

### 13. Report findings

Present a clear summary organized by severity:

**Must fix** — issues that will break CI or violate hard project rules:
- Unrelated commits on the branch
- Lint failures
- Missing type hints on public functions
- Raw dict usage for structured data (including internal code)
- Multi-item tuple returns (including internal code)
- Missing tests for new endpoints
- Direct DB access (raw SQL / `acquire_with_retry` / `fq_table`) in an `api/` handler instead of a `MemoryEngine` method
- Tenant-scoped data accessed without authentication enforced in the engine (`_authenticate_tenant` / `get_bank_profile`)
- A SQL statement against a multi-bank table filtered by a caller-supplied, non-globally-unique key without a `bank_id` predicate (cross-bank read/write leak — see step 7c)
- New integration missing tests, CI job, or release-integration.sh entry
- Released/added integration missing from `hindsight-docs/src/data/integrations.json`, or a JSON entry with no `docs-integrations/<slug>` page (fails the docs build via `check-integrations.mjs`)
- New PostgreSQL table missing from `BACKUP_TABLES` in `admin/cli.py` (silent data loss on restore)
- A capability every sibling implementation has except the one in the diff (see step 9a) — a
  harness, dialect, provider or language variant that skips a lifecycle step the others perform

**Should fix** — issues that hurt code quality:
- Dead code / unused imports missed by linter
- Missing tests for non-trivial utility functions
- Over-engineering beyond the task scope

**Note** — observations that may or may not need action:
- API changes that might need client regeneration
- Patterns that deviate from nearby code style

For each finding, include the file path, line number, and a brief explanation.

Do NOT auto-fix any issues. Report all findings and let the user decide what to address. If there are no findings, confirm the code looks good.
