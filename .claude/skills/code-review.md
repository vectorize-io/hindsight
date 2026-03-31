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

### General Principles
- Don't add features, refactor code, or make "improvements" beyond what was asked
- Don't add unnecessary error handling for impossible scenarios
- Don't create helpers or abstractions for one-time operations
- No backwards-compatibility hacks (unused vars, re-exports, "removed" comments)
- Three similar lines of code is better than a premature abstraction

## Review Steps

### 1. Identify changed files

Run `git diff --name-only HEAD` (unstaged) and `git diff --cached --name-only` (staged) to get all changed files. If there are no local changes, diff against the base branch using `git diff main...HEAD --name-only` and `git diff main...HEAD` to review all commits on the current branch.

### 2. Run linters

```bash
./scripts/hooks/lint.sh
```

Report any failures. Do NOT fix them yourself — just report.

### 3. Check for dead code

For each changed Python file, check for:
- Unused imports (Ruff should catch these, but verify)
- Functions/methods/classes that were added but are never called from anywhere
- Variables assigned but never read
- Commented-out code blocks that should be removed

For each changed TypeScript file, check for:
- Unused imports
- Unused variables or functions
- Commented-out code

### 4. Check type safety (Python)

For each changed Python file, check for violations:
- **No raw `dict` for structured data** — should use Pydantic models
- **No multi-item tuple returns** — should use dataclass or Pydantic model
- **Missing type hints** on function parameters and return types
- **Missing `@field_validator`** for datetime fields that should be timezone-aware

### 5. Check for missing tests

For each new or significantly changed function/endpoint/class:
- Check if there is a corresponding test addition or update
- New API endpoints MUST have integration tests
- New utility functions MUST have unit tests
- Bug fixes SHOULD have a regression test

Flag any new logic that lacks test coverage.

### 6. Check API consistency

If any files in `hindsight-api-slim/hindsight_api/api/` were changed:
- Were the OpenAPI specs regenerated? (`./scripts/generate-openapi.sh`)
- Were the client SDKs regenerated? (`./scripts/generate-clients.sh`)
- Were the control plane proxy routes updated? (`hindsight-control-plane/src/app/api/`)

### 7. Review against coding standards

Check the diff for violations of the standards listed above:
- Python files at project root (not allowed)
- Missing async patterns (should be async throughout)
- Pydantic models for request/response
- Line length > 120 chars
- New features/code beyond what was asked (over-engineering)
- Unnecessary error handling for impossible scenarios
- Premature abstractions or speculative helpers
- Backwards-compatibility hacks (unused vars, re-exports, "removed" comments)

### 8. Report findings

Present a clear summary organized by severity:

**Must fix** — issues that will break CI or violate hard project rules:
- Lint failures
- Missing type hints on public functions
- Raw dict usage for structured data
- Missing tests for new endpoints

**Should fix** — issues that hurt code quality:
- Dead code / unused imports missed by linter
- Missing tests for non-trivial utility functions
- Over-engineering beyond the task scope

**Note** — observations that may or may not need action:
- API changes that might need client regeneration
- Patterns that deviate from nearby code style

For each finding, include the file path, line number, and a brief explanation.

Do NOT auto-fix any issues. Report all findings and let the user decide what to address. If there are no findings, confirm the code looks good.
