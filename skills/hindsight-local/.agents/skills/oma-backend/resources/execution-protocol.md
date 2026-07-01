# Backend Agent - Execution Protocol

## Step 0: Prepare
1. **Assess difficulty**: see `../../_shared/core/difficulty-guide.md`
   - **Simple**: Skip to Step 3 | **Medium**: All 4 steps | **Complex**: All steps + checkpoints
2. **Check lessons**: read your domain section in `../../_shared/core/lessons-learned.md`
3. **Clarify requirements**: follow `../../_shared/core/clarification-protocol.md`
   - Check **Uncertainty Triggers**: business logic, security/auth, existing code conflicts?
   - Determine level: LOW → proceed | MEDIUM → present options | HIGH → ask immediately
4. **Budget context**: follow `../../_shared/core/context-budget.md` (read symbols, not whole files)

**Intelligent Escalation**: When uncertain, escalate early. Don't blindly proceed.

Follow these steps in order (adjust depth by difficulty).

## Step 1: Analyze
- Read the task requirements carefully
- Identify which endpoints, models, and services are needed
- Check existing code with Serena: `get_symbols_overview("app/api")`, `find_symbol("existing_function")`
- If the task is ORM-heavy, load `resources/orm-reference.md` before deciding on loading strategy, transaction scope, or client/session lifecycle
- List assumptions; ask if unclear

## Step 2: Plan
- Decide on file structure: models, schemas, routes, services
- Define API contracts (method, path, request/response types)
- Plan database schema changes (tables, columns, indexes, migrations)
- Plan relation loading strategy, transaction boundary, and ORM lifecycle constraints explicitly
- Identify security requirements (auth, validation, rate limiting)

## Step 3: Implement
- Create/modify files in this order:
  1. Database models + migrations
  2. Validation schemas (request/response)
  3. Service layer (business logic)
  4. API routes (thin, delegate to services)
  5. Tests (unit + integration)
- Use `stack/api-template.*` as reference
- Follow clean architecture: router -> service -> repository -> models

## Step 4: Verify
- Run `resources/checklist.md` items
- Run `../../_shared/core/common-checklist.md` items
- Ensure all tests pass
- Confirm OpenAPI docs are complete

## On Error
See `resources/error-playbook.md` for recovery steps.
