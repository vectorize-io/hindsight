# Admin CLI

The `hindsight-admin` CLI provides administrative commands for managing your Hindsight deployment, including database migrations, backup, and restore operations.

## Installation

The admin CLI is included with the `hindsight-api` package:

```bash
pip install hindsight-api
# or
uv add hindsight-api
```

## Commands

### run-db-migration

Run database migrations to the latest version. This is useful when you want to run migrations separately from API startup (e.g., in CI/CD pipelines or before deploying a new version).

```bash
hindsight-admin run-db-migration [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--schema`, `-s` | Database schema to run migrations on | `public` |

**Examples:**

```bash
# Run migrations on the default public schema
hindsight-admin run-db-migration

# Run migrations on a specific tenant schema
hindsight-admin run-db-migration --schema tenant_acme
```

:::tip Disabling Auto-Migrations
To disable automatic migrations on API startup, set `HINDSIGHT_API_RUN_MIGRATIONS_ON_STARTUP=false`. This is useful when you want to run migrations as a separate step in your deployment pipeline.
:::

---

### backup

Create a backup of all Hindsight data to a zip file.

```bash
hindsight-admin backup OUTPUT [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `OUTPUT` | Output file path (will add `.zip` extension if not present) |

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--schema`, `-s` | Database schema to backup | `public` |

**Examples:**

```bash
# Backup to a file
hindsight-admin backup /backups/hindsight-2024-01-15.zip

# Backup a specific tenant schema
hindsight-admin backup /backups/tenant-acme.zip --schema tenant_acme
```

The backup includes:
- Memory banks and their configuration
- Documents and chunks
- Entities and their relationships
- Memory units (facts, experiences, observations)
- Entity cooccurrences and memory links

:::note Consistency
Backups are created within a database transaction with `REPEATABLE READ` isolation, ensuring a consistent snapshot across all tables.
:::

---

### restore

Restore data from a backup file. **Warning: This deletes all existing data in the target schema.**

```bash
hindsight-admin restore INPUT [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `INPUT` | Input backup file (.zip) |

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--schema`, `-s` | Database schema to restore to | `public` |
| `--yes`, `-y` | Skip confirmation prompt | `false` |

**Examples:**

```bash
# Restore with confirmation prompt
hindsight-admin restore /backups/hindsight-2024-01-15.zip

# Restore without confirmation (for scripts)
hindsight-admin restore /backups/hindsight-2024-01-15.zip --yes

# Restore to a specific tenant schema
hindsight-admin restore /backups/tenant-acme.zip --schema tenant_acme --yes
```

:::warning Data Loss
Restore will **delete all existing data** in the target schema before importing the backup. Always verify you have a recent backup before performing a restore.
:::

---

### decommission-worker

Release all tasks owned by a worker, resetting them from "processing" back to "pending" status so they can be picked up by other workers.

```bash
hindsight-admin decommission-worker WORKER_ID [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `WORKER_ID` | ID of the worker to decommission |

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--schema`, `-s` | Database schema | `public` |

**Examples:**

```bash
# Before scaling down - release tasks from workers being removed
hindsight-admin decommission-worker hindsight-worker-4
hindsight-admin decommission-worker hindsight-worker-3

# Release tasks from a crashed worker
hindsight-admin decommission-worker worker-2

# For a specific tenant schema
hindsight-admin decommission-worker worker-1 --schema tenant_acme
```

**When to Use:**

- **Scaling down**: Before removing worker replicas in Kubernetes
- **Graceful removal**: When taking a worker offline for maintenance
- **Crash recovery**: If a worker crashed while processing tasks
- **Stuck worker**: When a worker is unresponsive

:::tip Finding Worker IDs
Worker IDs default to the hostname. In Kubernetes StatefulSets, this is the pod name (e.g., `hindsight-worker-0`). You can also set a custom ID with `HINDSIGHT_API_WORKER_ID` or `--worker-id`.
:::

---

## Environment Variables

The admin CLI uses the same environment variables as the API service. The most important one is:

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_DATABASE_URL` | PostgreSQL connection string | `pg0` (embedded) |

**Example:**

```bash
# Use a specific database
export HINDSIGHT_API_DATABASE_URL=postgresql://user:pass@localhost:5432/hindsight
hindsight-admin backup /backups/mybackup.zip
```

