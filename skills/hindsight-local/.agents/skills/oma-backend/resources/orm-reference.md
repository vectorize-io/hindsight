# Backend Agent - ORM Reference

Use this guide when the task involves ORM query design, relation loading, transaction scoping, or session/client lifecycle decisions.

This document is intentionally synthesized from official ORM docs. The purpose is not to mirror each vendor page, but to extract the common operating rules that keep backend work correct and performant.

## ORM Rules

### 1. Relation loading must be chosen per use case

Backend Agent should therefore:
- choose loading strategy per endpoint or service method
- compare join-based eager loading, batched/select-in loading, and separate follow-up queries
- treat accidental lazy loads in loops as a performance bug

What this rule is derived from:
- Prisma documents N+1 mitigation through batched access patterns and join-based relation loading
- SQLAlchemy documents `joinedload`, `selectinload`, `lazyload`, and `raiseload` as explicit strategies
- TypeORM documents join-based loading and the lazy/eager trade-off
- Sequelize documents `include` as join-based eager loading and `separate: true` for `hasMany`
- Hibernate documents that select-based fetching is vulnerable to N+1

### 2. Transaction boundaries must follow business operations

Backend Agent should therefore:
- open one explicit transaction for one business operation
- keep read-modify-write flows inside the same transaction boundary
- avoid per-statement auto-commit designs
- never hold a database transaction open across user think time

What this rule is derived from:
- Hibernate explicitly calls session-per-operation and auto-commit-per-statement anti-patterns
- Sequelize promotes managed transactions around a callback-shaped unit of work
- SQLAlchemy models the session as transaction-oriented state

### 3. ORM lifecycle objects are not generic shared singletons

Backend Agent should therefore:
- reuse factory/client objects only when the ORM recommends reuse
- keep transaction-scoped mutable objects request-scoped or task-scoped
- never share stateful unit-of-work objects across concurrent tasks unless the vendor explicitly permits it

Operational reading:
- Prisma: reuse one `PrismaClient` in long-running apps; do not create a new client per query; do not disconnect after every request
- SQLAlchemy: `Session` and `AsyncSession` are mutable, transaction-scoped, and the documented model is session-per-thread / async-session-per-task
- Hibernate: prefer session-per-request over session-per-operation

### 4. Field projection is mandatory unless full entity hydration is required

Backend Agent should therefore:
- fetch only required columns, attributes, or nested fields
- prefer projections, DTO shaping, or raw results when domain objects are not needed
- treat over-fetching as a query design flaw, not a minor optimization item

What this rule is derived from:
- Prisma calls out over-fetching directly and supports nested `select`
- TypeORM recommends `select` and `getRawMany()`
- Sequelize documents `attributes` and `{ raw: true }`

### 5. Mapping defaults must never be trusted blindly

Lazy and eager defaults are convenience features, not query plans. Multiple vendors warn that both directions can fail differently:
- lazy defaults can explode into N+1
- eager defaults can create oversized joins, duplicate parent rows, or unnecessary payloads

Backend Agent should therefore:
- inspect relation defaults before changing repository or endpoint behavior
- override defaults at query time when access shape differs from mapping defaults
- prefer repository-level query shaping over relying on entity metadata alone

### 6. Query review must include access-path and row-shape checks

Backend Agent should therefore review:
- missing indexes on filters, joins, and foreign keys
- full table scans
- repeated identical queries that should be cached or batched
- row multiplication from `hasMany` / collection eager joins
- memory cost of hydrating full entities when raw results or projections are enough
