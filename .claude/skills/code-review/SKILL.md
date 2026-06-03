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

### 7. Check API consistency

If any files in `hindsight-api-slim/hindsight_api/api/` were changed:
- Were the OpenAPI specs regenerated? (`./scripts/generate-openapi.sh`)
- Were the client SDKs regenerated? (`./scripts/generate-clients.sh`)
- Were the control plane proxy routes updated? (`hindsight-control-plane/src/app/api/`)

### 7b. Check API-layer data-access boundary

For each changed handler in `hindsight-api-slim/hindsight_api/api/` (e.g. `http.py`, `mcp.py`):
- **Flag any direct DB access in the handler** — `acquire_with_retry`, `conn.fetch` / `fetchrow` / `execute`, raw SQL strings, or `fq_table(...)`. These are a **must fix**: the query must be moved into a `MemoryEngine` method that returns a typed model, and the handler must call that method.
- **Verify authentication is enforced in the engine** — the handler must delegate to an engine method that authenticates via `request_context` (`_authenticate_tenant`, typically through `get_bank_profile`). A handler that reads/writes tenant-scoped data without an engine method enforcing auth is a **must fix** (tenant data could leak across schemas).

### 8. Check code comments

For each non-trivial change:
- **New non-obvious logic** — is there a comment explaining the reasoning?
- **Changed approach** — does the comment include what was done before and why it changed?
- **Stale comments** — do existing comments near the changed code still accurately describe the behavior?

### 9. Check integration completeness

If any files in `hindsight-integrations/` were added or changed, verify:
- **Tests exist** — the integration must have tests that simulate/exercise the external framework (not just pure unit tests of helpers). Check for a `tests/` directory with meaningful test files.
- **CI job exists** — check `.github/workflows/test.yml` for a corresponding `test-<name>-integration` job. If missing, flag it.
- **Release process** — check that the integration name is in the `VALID_INTEGRATIONS` array in `scripts/release-integration.sh`. If missing, flag it.
- **Code standards** — the integration code must follow all Python style rules (type hints, no raw dicts, no tuple returns, etc.).

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
- New integration missing tests, CI job, or release-integration.sh entry
- New PostgreSQL table missing from `BACKUP_TABLES` in `admin/cli.py` (silent data loss on restore)

**Should fix** — issues that hurt code quality:
- Dead code / unused imports missed by linter
- Missing tests for non-trivial utility functions
- Over-engineering beyond the task scope

**Note** — observations that may or may not need action:
- API changes that might need client regeneration
- Patterns that deviate from nearby code style

For each finding, include the file path, line number, and a brief explanation.

Do NOT auto-fix any issues. Report all findings and let the user decide what to address. If there are no findings, confirm the code looks good.
