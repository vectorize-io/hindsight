---
name: code-review
description: Review changed code against project standards. Checks for missing tests, dead code, type safety, lint issues, and CLAUDE.md conventions. Run after completing any implementation work.
user_invocable: true
---

# Code Review

Review all changed code against the project's quality standards. This skill enforces the conventions in CLAUDE.md and catches common issues.

## Steps

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

For each changed Python file, check for violations of project conventions:
- **No raw `dict` for structured data** — should use Pydantic models (see CLAUDE.md)
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

### 7. Review against CLAUDE.md conventions

Check the diff for violations of:
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
