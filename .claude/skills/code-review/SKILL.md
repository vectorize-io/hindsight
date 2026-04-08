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

### 7. Check API consistency

If any files in `hindsight-api-slim/hindsight_api/api/` were changed:
- Were the OpenAPI specs regenerated? (`./scripts/generate-openapi.sh`)
- Were the client SDKs regenerated? (`./scripts/generate-clients.sh`)
- Were the control plane proxy routes updated? (`hindsight-control-plane/src/app/api/`)

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

### 11. Review against other coding standards

Check the diff for violations of the standards listed above:
- Python files at project root (not allowed)
- Missing async patterns (should be async throughout)
- Pydantic models for request/response
- Line length > 120 chars
- New features/code beyond what was asked (over-engineering)
- Unnecessary error handling for impossible scenarios
- Premature abstractions or speculative helpers
- Backwards-compatibility hacks (unused vars, re-exports, "removed" comments)

### 12. Report findings

Present a clear summary organized by severity:

**Must fix** — issues that will break CI or violate hard project rules:
- Unrelated commits on the branch
- Lint failures
- Missing type hints on public functions
- Raw dict usage for structured data (including internal code)
- Multi-item tuple returns (including internal code)
- Missing tests for new endpoints
- New integration missing tests, CI job, or release-integration.sh entry

**Should fix** — issues that hurt code quality:
- Dead code / unused imports missed by linter
- Missing tests for non-trivial utility functions
- Over-engineering beyond the task scope

**Note** — observations that may or may not need action:
- API changes that might need client regeneration
- Patterns that deviate from nearby code style

For each finding, include the file path, line number, and a brief explanation.

Do NOT auto-fix any issues. Report all findings and let the user decide what to address. If there are no findings, confirm the code looks good.
