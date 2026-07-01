---
description: Backend API coding standards with clean architecture (Router/Service/Repository pattern)
globs:
alwaysApply: false
---

# Backend Coding Standards

## Core Rules

1. **Clean architecture**: router -> service -> repository -> models
2. **No business logic in route handlers**
3. **All inputs validated** with your stack's validation library
4. **Parameterized queries only** (never string interpolation)
5. **JWT + bcrypt for auth**; rate limit auth endpoints
6. **Async where supported**; type annotations on all signatures
7. **Custom exceptions** via centralized error module (not raw HTTP exceptions)
8. **Explicit ORM loading strategy**: do not rely on default relation loading when query shape matters
9. **Explicit transaction boundaries**: group one business operation into one request/service-scoped unit of work
10. **Safe ORM lifecycle**: do not share mutable ORM session/entity manager across concurrent work unless ORM explicitly supports it
11. **Config from environment, with graceful fallback**: DB URLs, API keys, secrets from env vars or secret managers, never hardcode. When integrating a third-party API (OpenAI, Anthropic, Stripe, etc.), write BOTH paths: (a) real call when `process.env.<KEY>` is present, (b) deterministic local fallback when absent. Mark the deferred branch with `// TODO(oma-deferred): integrate <vendor> when key is provisioned`. Shipping only the fallback (no env-conditional branch) leaves the spec unmet; shipping only the real call without fallback breaks demos when the key is missing.
12. **Stateless services**: no in-memory session or user state between requests — use external stores
13. **Backing services as resources**: DB, queue, cache are swappable resources connected via config

## Architecture

```
Router (HTTP) → Service (Business Logic) → Repository (Data Access) → Models
```

## Principles

- **DRY**: Business logic in Service, data access in Repository
- **Single Responsibility**: Classes and functions should have one responsibility
- **Dependency Inversion**: Use your framework's DI mechanism
- **KISS**: Keep it simple and clear
