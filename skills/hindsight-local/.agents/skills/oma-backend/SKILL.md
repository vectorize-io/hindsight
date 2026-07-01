---
name: oma-backend
description: Backend specialist for APIs, databases, authentication with clean architecture (Repository/Service/Router pattern). Use for API, endpoint, REST, database, server, migration, and auth work.
---

# Backend Agent - API & Server Specialist

## Scheduling

### Goal
Implement or review backend APIs, authentication, database integration, server-side business logic, and migrations using the project's existing backend stack and clean architecture boundaries.

### Intent signature
- User asks for API, endpoint, REST, GraphQL, auth, server, migration, repository, service, router, or background job work.
- User needs backend code that coordinates validation, business logic, persistence, transactions, and backing services.

### When to use
- Building REST APIs or GraphQL endpoints
- Database design and migrations
- Authentication and authorization
- Server-side business logic
- Background jobs and queues

### When NOT to use
- Frontend UI -> use Frontend Agent
- Mobile-specific code -> use Mobile Agent

### Expected inputs
- Target feature, endpoint, migration, auth flow, or server behavior
- Existing backend stack files such as manifests, routes, services, models, and database config
- API contracts, schemas, validation rules, and persistence requirements
- Required verification commands or project conventions

### Expected outputs
- Backend code changes in router, service, repository, model, migration, or test files
- Validated inputs, safe queries, transaction boundaries, and error handling
- Verification results from the execution checklist

### Dependencies
- Project stack manifests and existing backend conventions
- `resources/execution-protocol.md`, `resources/checklist.md`, and `resources/orm-reference.md`
- Optional `stack/stack.yaml`, `stack/tech-stack.md`, snippets, and API templates
- Database, queue, cache, mail, auth, or external API resources configured through environment or secret managers

### Control-flow features
- Branches by detected stack, ORM/query pattern, auth requirement, migration impact, and transaction scope
- Reads and writes codebase files
- May touch local database migrations or generated code
- Must not hardcode secrets or share unsafe ORM lifecycle objects across concurrent work

## Structural Flow

### Entry
1. Detect the backend stack from project files first.
2. Identify affected router, service, repository, model, migration, and test boundaries.
3. Load stack-specific references only when needed.

### Scenes
1. **PREPARE**: Determine stack, architecture boundaries, and acceptance criteria.
2. **ACQUIRE**: Read existing routes, services, repositories, models, schemas, and config.
3. **ACT**: Implement backend changes with validation, business logic, persistence, and tests.
4. **VERIFY**: Run relevant lint, type, test, migration, and checklist commands.
5. **FINALIZE**: Report changed behavior, verification, and unresolved risks.

### Transitions
- If stack files exist, follow them before generic guidance.
- If ORM performance, relationship loading, transactions, or N+1 risk appears, use `resources/orm-reference.md`.
- If database schema impact is primary and API work is secondary, coordinate with `oma-db`.
- If auth server setup touches DB adapters or server libraries, keep it in backend scope.

### Failure and recovery
- If stack cannot be determined, ask the user or suggest running `/stack-set`.
- If verification fails, fix root cause before handoff.
- If required secrets or services are unavailable, document the blocker and keep code configurable.

### Exit
- Success: backend change is implemented, tested, and aligned with local architecture.
- Partial success: blocker, missing dependency, or verification gap is explicit.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Detect stack and conventions | `READ` | Manifests, stack files, existing code |
| Select implementation boundary | `SELECT` | Router/service/repository pattern |
| Validate inputs and schemas | `VALIDATE` | Stack validation library |
| Implement business logic | `WRITE` | Service layer code |
| Implement persistence | `WRITE` | Repository/model/migration code |
| Call external/backing services | `CALL_TOOL` | DB, queue, cache, auth, or API clients |
| Run verification | `CALL_TOOL` | Tests, typecheck, lint, migrations |
| Report result | `NOTIFY` | Final summary |

### Tools and instruments
- Project language/framework toolchain
- ORM or database client
- Test, lint, typecheck, and migration commands
- Stack-specific templates and snippets when present

### Canonical workflow path
```bash
rg --files
rg "route|router|service|repository|model|schema|migration" .
```

Then run the project's discovered verification commands, usually lint/typecheck/tests and migrations when schema changes are involved. Prefer `stack/stack.yaml` `verify:` commands when present.

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `CODEBASE` | Backend source, tests, schemas, migrations |
| `LOCAL_FS` | Stack references and generated artifacts |
| `PROCESS` | Test, lint, typecheck, migration commands |
| `CREDENTIALS` | Environment-managed DB URLs, API keys, secrets |
| `NETWORK` | External APIs or backing services when required |

### Preconditions
- Target behavior and affected backend boundary are identifiable.
- Project stack and verification commands can be inferred or are provided.
- Required credentials remain outside source code.

### Effects and side effects
- Mutates backend source files, tests, and possibly migrations.
- May change database schema, API behavior, auth behavior, or service contracts.
- May require generated clients or migration artifacts.

### Guardrails

1. **DRY (Don't Repeat Yourself)**: Business logic in `Service`, data access logic in `Repository`
2. **SOLID**:
   - **Single Responsibility**: Classes and functions should have one responsibility
   - **Dependency Inversion**: Use your framework's DI mechanism
3. **KISS**: Keep it simple and clear

### Architecture Pattern

```
Router (HTTP) → Service (Business Logic) → Repository (Data Access) → Models
```

### Repository Layer
- Encapsulate DB CRUD and query logic
- No business logic, return ORM entities

### Service Layer
- Business logic, Repository composition, external API calls
- Business decisions only here

### Router Layer
- Receive HTTP requests, input validation, call Service, return response
- No business logic, inject Service via DI

### Core Rules

1. **Clean architecture**: router → service → repository → models
2. **No business logic in route handlers**
3. **All inputs validated** with your stack's validation library
4. **Parameterized queries only** (never string interpolation)
5. **JWT + bcrypt for auth**; rate limit auth endpoints
6. **Async where supported**; type annotations on all signatures
7. **Custom exceptions** via centralized error module (not raw HTTP exceptions)
8. **Explicit ORM loading strategy**: do not rely on default relation loading when query shape matters
9. **Explicit transaction boundaries**: group one business operation into one request/service-scoped unit of work
10. **Safe ORM lifecycle**: do not share mutable ORM session/entity manager/client objects across concurrent work unless the ORM explicitly supports it
11. **Config from environment**: DB URLs, API keys, secrets, and feature flags come from env vars or secret managers; never hardcode in source
12. **Stateless services**: no in-memory session or user state between requests; use external stores (DB, Redis, cache) for shared state
13. **Backing services as resources**: DB, queue, cache, mail are swappable attached resources connected via config; Repository layer must not assume a specific instance

### Stack Detection

1. **Project files first**: Read existing code, package manifests (pyproject.toml, package.json, Cargo.toml, go.mod, pom.xml, etc.) to determine the tech stack
2. **stack/ second**: If `stack/` exists, use it as supplementary reference for coding conventions and snippet templates
3. **Neither exists**: Ask the user or suggest running `/stack-set`

### Stack-Specific Reference

- **Stack manifest (SSOT)**: `stack/stack.yaml`: structured declaration (`language`, `framework`, `orm`) and `verify:` contract consumed by `oma verify backend`. Schema: `variants/stack.schema.json`.
- Tech stack narrative: `stack/tech-stack.md`: human-readable reference only; `stack.yaml` wins on conflict.
- Code snippets (copy-paste ready): `stack/snippets.md`
- API template: `stack/api-template.*`

## References

Follow `resources/execution-protocol.md` step by step.
See `resources/examples.md` for input/output examples.
Use `resources/orm-reference.md` when the task involves ORM query performance, relationship loading, transactions, session/client lifecycle, or N+1 analysis.
Before submitting, run `resources/checklist.md`.
Vendor-specific execution protocols are injected automatically by `oma agent:spawn`.
Source files live under `../_shared/runtime/execution-protocols/{vendor}.md`.
- Execution steps: `resources/execution-protocol.md`
- Code examples: `resources/examples.md`
- Checklist: `resources/checklist.md`
- ORM reference: `resources/orm-reference.md`
- Error recovery: `resources/error-playbook.md`
- Context loading: `../_shared/core/context-loading.md`
- Reasoning templates: `../_shared/core/reasoning-templates.md`
- Clarification: `../_shared/core/clarification-protocol.md`
- Context budget: `../_shared/core/context-budget.md`
- Lessons learned: `../_shared/core/lessons-learned.md`
- Observability handoff: `../oma-observability/SKILL.md` §Integrations — propagators/baggage, span conventions, log correlation, PII redaction
