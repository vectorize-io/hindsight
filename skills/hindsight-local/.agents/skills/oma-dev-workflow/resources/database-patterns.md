# Database & Local Infrastructure

Database migration workflows and local development infrastructure.

## Migration Tasks

```toml
# apps/api/mise.toml
[tasks.migrate]
description = "Run Alembic migrations"
run = "uv run alembic upgrade head"

[tasks.migrate:create]
description = "Create new migration"
run = "uv run alembic revision --autogenerate -m '{{arg(name)}}'"

[tasks.migrate:check]
description = "Check migration status"
run = "uv run alembic current"

[tasks.migrate:history]
description = "Show migration history"
run = "uv run alembic history --verbose"

[tasks.migrate:rollback]
description = "Rollback one migration"
run = "uv run alembic downgrade -1"

[tasks.migrate:reset]
description = "Reset database (all down then up)"
run = "uv run alembic downgrade base && uv run alembic upgrade head"
```

## Creating Migrations

```bash
# Create new migration
mise run //apps/api:migrate:create "add users table"

# Review generated migration
# Edit: apps/api/alembic/versions/xxx_add_users_table.py

# Apply migration
mise run //apps/api:migrate

# Check current version
mise run //apps/api:migrate:check

# View history
mise run //apps/api:migrate:history

# Rollback (if needed)
mise run //apps/api:migrate:rollback
```

## Root-Level Database Tasks

```toml
# Root mise.toml
[tasks.db:migrate]
description = "Run all database migrations"
depends = ["//apps/api:migrate"]

[tasks.db:reset]
description = "Reset all databases"
depends = ["//apps/api:migrate:reset"]
```

## Local Infrastructure (Docker)

```toml
# apps/api/mise.toml
[tasks.infra:up]
description = "Start local infrastructure (PostgreSQL, Redis, etc.)"
run = "docker compose -f docker-compose.infra.yml up -d"

[tasks.infra:down]
description = "Stop local infrastructure"
run = "docker compose -f docker-compose.infra.yml down"

[tasks.infra:logs]
description = "View infrastructure logs"
run = "docker compose -f docker-compose.infra.yml logs -f"

[tasks.infra:reset]
description = "Reset infrastructure (remove volumes)"
run = "docker compose -f docker-compose.infra.yml down -v"
```

## Root-Level Infrastructure Tasks

```toml
# Root mise.toml
[tasks.infra:up]
description = "Start all local infrastructure"
depends = ["//apps/api:infra:up"]

[tasks.infra:down]
description = "Stop all local infrastructure"
depends = ["//apps/api:infra:down"]
```
