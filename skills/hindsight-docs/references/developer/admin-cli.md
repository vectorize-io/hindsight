# Admin CLI

The `hindsight-admin` CLI provides administrative commands for managing your Hindsight deployment, including database migrations, backup, and restore operations.

## Installation

The admin CLI is included with the `hindsight-api` package — installing it puts the `hindsight-admin` executable on your `PATH`:

```bash
pip install hindsight-api
# or
uv add hindsight-api
```

## Running the CLI

`hindsight-admin` connects **directly to PostgreSQL** — it does not call the HTTP API. It reads the **same configuration as the API service** (environment variables, and a `.env` file in the current working directory), so it operates on whatever database `HINDSIGHT_API_DATABASE_URL` points to:

- **Default**: `pg0`, the embedded development database (must be run on the host that owns the pg0 data).
- **Production**: set `HINDSIGHT_API_DATABASE_URL=postgresql://user:pass@host:5432/hindsight`.

Because it talks to the database directly (binary `COPY`, `TRUNCATE`, etc.), the admin CLI is **PostgreSQL-only** (not supported on Oracle). Run it on the same host/container as your API deployment so it inherits the right configuration and has network access to the database:

```bash
# Bare metal / virtualenv (with the API's env or a .env in the working dir)
hindsight-admin worker-status

# Docker — exec into the API container
docker exec -it hindsight-api hindsight-admin backup /data/backup.zip

# Kubernetes — exec into an API pod
kubectl exec deploy/hindsight-api -- hindsight-admin run-db-migration
```

Use `--schema` to target a specific tenant schema (commands default to the configured base schema). See [Environment Variables](#environment-variables) below.

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
| `--embedding-dimension` | Expected embedding dimension to enforce after migrations. Omit to skip the post-migration dimension sync. | Skipped |
| `--skip-extension-reconcile` | Skip the post-migration vector / text-search index reconcile (it only does work when `HINDSIGHT_API_VECTOR_EXTENSION` / `HINDSIGHT_API_TEXT_SEARCH_EXTENSION` differs from a schema's existing indexes). Makes a no-change re-migration across many tenant schemas much faster; only use when the backend is unchanged. | Reconcile runs |

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
- Mental models and directives
- Webhooks and file storage
- Internal operational tables (async operations, audit log, graph-maintenance queue, and similar bookkeeping) so a restore reproduces a faithful full-database snapshot

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
| `--yes`, `-y` | Skip confirmation prompt | `false` |

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

### export-bank

Export an entire bank to a portable ZIP archive — documents, facts, observations, bank configuration, mental models, directives, and webhooks. Embeddings are **never** included; they are regenerated on import. This is the source half of a cross-instance migration (e.g. moving to a different embedding model, vector extension, or text-search backend). PostgreSQL only.

```bash
hindsight-admin export-bank --bank <BANK_ID> --output <FILE.zip> [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--bank`, `-b` | Bank id to export. | (required) |
| `--output`, `-o` | Path to write the `.zip` archive. | (required) |
| `--schema`, `-s` | Schema the bank lives in. | base schema |
| `--include-history` | Also export operational history (`audit_log`, `llm_requests`). | `false` |

**Examples:**

```bash
hindsight-admin export-bank --bank my-bank --output my-bank.zip

# include operational history
hindsight-admin export-bank --bank my-bank --output my-bank.zip --include-history
```

Read-only — safe to run against a live instance.

---

### import-bank

Restore a whole-bank archive (produced by `export-bank`) into **this** instance. Facts are re-embedded with this instance's configured embedding model and links/indexes are rebuilt; bank configuration, mental models, directives, and webhooks are restored exactly. No LLM fact-extraction runs, and because a migration restores state, it does **not** fire webhooks or re-run consolidation. PostgreSQL only.

```bash
hindsight-admin import-bank --archive <FILE.zip> [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--archive`, `-a` | Path to the `.zip` produced by `export-bank`. | (required) |
| `--schema`, `-s` | Target schema. | base schema |
| `--target-bank` | Override the bank id (defaults to the archive's source bank). | source bank |
| `--include-history` | Also restore history if present in the archive. | `false` |

**Examples:**

```bash
hindsight-admin import-bank --archive my-bank.zip
```

Run this against an instance configured with the **target** embedding model / vector extension / text-search backend — that's what re-embedding uses.

:::warning Target bank must not exist
Import restores a **whole bank** (config, facts, mental models, …) — it is **not a merge**. If a bank with the target id already exists, the command fails. Delete that bank first, or use `--target-bank` to restore under a fresh id.
:::

---

## Migrating a bank to a new instance

Changing a bank's **embedding model** (e.g. a 384-dim encoder → a 1024-dim one), **vector extension** (pgvector / vchord / pgvectorscale), or **text-search backend** can't be done in place on a populated bank — the stored vectors and indexes are tied to those settings. Because every embedding and index is a deterministic function of text already on disk, the supported path is to **move the bank to a fresh instance configured with the new settings and re-derive everything there — with no LLM re-extraction**.

`export-bank` / `import-bank` carry documents, facts, observations, bank config, mental models, directives, and webhooks — but never embeddings, which the target instance regenerates with its own model.

**Blue-green runbook:**

1. Stand up a **new instance** on a fresh database, configured with the new embedding model / vector extension / text-search backend.
2. Quiesce writes to the source bank (maintenance window) and run `hindsight-admin backup` for safety.
3. Export from the source, then import into the target:
   ```bash
   # on the source instance:
   hindsight-admin export-bank --bank my-bank --output my-bank.zip
   # on the target instance (configured with the new settings):
   hindsight-admin import-bank --archive my-bank.zip
   ```
4. Verify on the target: run representative recall queries and compare results.
5. Cut traffic over to the new instance. The old instance stays as an instant rollback until you're confident.

:::note Why a new instance, not in-place
The embedding model is server-level, and a bank's `memory_units.embedding` column has a single dimension shared across the schema, so a different-dimension or different-backend bank needs its own instance/database. The old vectors are never mutated, which makes rollback trivial.
:::

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
