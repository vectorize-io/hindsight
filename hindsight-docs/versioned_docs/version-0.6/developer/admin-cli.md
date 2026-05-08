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

Run database migrations to the latest version. By default this migrates the base schema plus all tenant schemas discovered by the tenant extension. Use `--schema` for targeted migration of one schema. This is useful when you want to run migrations separately from API startup (e.g., in CI/CD pipelines or before deploying a new version).

```bash
hindsight-admin run-db-migration [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--schema`, `-s` | Database schema to run migrations on. If omitted, migrate the base schema plus all discovered tenant schemas. | All schemas |

**Examples:**

```bash
# Run migrations on the base schema plus all discovered tenant schemas
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


### decommission-workers

Release all currently-processing tasks from every worker, resetting them from "processing" back to "pending" status. Use this when one or more workers have crashed or been removed without graceful shutdown and you don't know which worker IDs to target.

```bash
hindsight-admin decommission-workers [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--schema`, `-s` | Database schema | `public` |
| `--yes`, `-y` | Skip confirmation prompt | `false` |

**Examples:**

```bash
# Release all processing tasks across all workers (with confirmation)
hindsight-admin decommission-workers

# Skip the confirmation prompt (useful in scripts)
hindsight-admin decommission-workers --yes

# Release tasks in a specific tenant schema
hindsight-admin decommission-workers --schema tenant_acme
```

**When to Use:**

- **Unknown dead workers**: Multiple workers crashed and you do not know their IDs
- **Fleet-wide recovery**: After an infrastructure event where many workers went down
- **"Just fix everything"**: A quick full-queue drain when per-worker cleanup is overkill

:::warning Disruptive
This releases **every** processing task regardless of worker, including tasks owned by healthy workers. Prefer `decommission-worker <WORKER_ID>` when you know which workers need cleanup.
:::

---

### worker-status

Show all currently-processing tasks grouped by worker, including operation type, bank, how long each task has been running, and when it was last updated. Useful for identifying orphaned tasks before decommissioning.

```bash
hindsight-admin worker-status [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--schema`, `-s` | Database schema | `public` |

**Examples:**

```bash
# Show all processing tasks across all workers
hindsight-admin worker-status

# Show processing tasks for a specific tenant schema
hindsight-admin worker-status --schema tenant_acme
```

**When to Use:**

- **Before decommissioning**: Inspect which workers have stale tasks and how long they have been stuck
- **Debugging throughput**: Diagnose why the queue is not draining (are tasks stuck in processing?)
- **Worker health check**: Spot workers whose `last_update_ago` keeps growing, indicating a dead or unresponsive worker

---

## Recovering stuck or zombie operations

A "zombie" operation is one stuck in `processing` indefinitely because the worker that claimed it is gone. The most common cause is an unstable `HINDSIGHT_API_WORKER_ID`: when it defaults to the container hostname, a Docker restart produces a new container ID, the new worker doesn't recognize the old worker's claims as its own, and those tasks are stranded.

**How to spot them:**

```bash
# List processing tasks grouped by worker — workers with a growing last_update_ago are dead
hindsight-admin worker-status

# Bank-level counters; pending_consolidation that never decreases is the usual symptom
curl -s http://localhost:8888/v1/default/banks/<bank_id>/stats
```

**How to recover:**

```bash
# You know which worker is dead (e.g. from worker-status):
hindsight-admin decommission-worker <old-worker-id>

# You don't know — release every processing task across the fleet:
hindsight-admin decommission-workers
```

Both commands reset `processing` rows back to `pending` so a live worker can claim them on the next poll.

**How to prevent it:**

Set `HINDSIGHT_API_WORKER_ID` to a stable value so worker identity survives restarts:

- **Docker**: pass `-e HINDSIGHT_API_WORKER_ID=hindsight-prod` (or per-replica names if running multiple containers)
- **Kubernetes (Helm)**: the chart's StatefulSet uses the pod name automatically — no extra config needed
- **Bare metal / pip**: pass `--worker-id <name>` or set the env var per process

See [Installation - Docker](./installation#docker) and [Configuration - Distributed Workers](./configuration#distributed-workers).

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
