# Prompt Structure

## Core Principle

Every task prompt should contain four elements. If any are missing, ask the user before proceeding.

## The Four Elements

### 1. Goal
What to build, change, or fix.

- "Add user authentication to the API"
- "Fix the 500 error on /api/users"
- "Refactor the payment module to use the new gateway"

### 2. Context
Relevant files, folders, documentation, errors, or examples.

- "See `src/auth/` for existing auth patterns"
- "Error log: `TypeError: Cannot read property 'id' of undefined`"
- "Similar implementation exists in `services/orders/`"

### 3. Constraints
Standards, architecture rules, safety requirements, or project conventions.

- "Must follow the existing Repository → Service → Router pattern"
- "No new external dependencies"
- "Must be backward-compatible with v2 API"

### 4. Done When
How to verify the task is complete using testable, observable criteria.

- "All existing tests pass + new tests for auth endpoints"
- "The 500 error no longer occurs and returns 200"
- "Lint, type-check, and build pass"

## Usage

### For PM Agent / Orchestrator
When receiving a user request, decompose it into these four elements. If the user's prompt is missing any element:

| Missing | Action |
|---------|--------|
| Goal | Ask: "What specifically should change?" |
| Context | Search codebase for relevant files and patterns |
| Constraints | Check AGENTS.md, docs/constraints/, taste.yaml for project rules |
| Done When | Propose verification criteria and confirm with user |

### For Implementation Agents
Before starting work, verify you have all four elements. If not, check:
1. `docs/` knowledge base for constraints and conventions
2. Existing tests and patterns for implicit criteria
3. AGENTS.md for project-level rules

### For QA / Review Agents
Use "Done When" criteria as the primary review checklist. A task is not complete until all completion criteria are met and verified.

## Anti-patterns

- Starting implementation with only a Goal (no constraints or done-when)
- Inventing constraints the user didn't specify
- Accepting vague done-when like "it works"; push for testable criteria
