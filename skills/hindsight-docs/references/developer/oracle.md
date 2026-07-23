# Oracle Database

Hindsight uses PostgreSQL as its default storage backend, but it also runs on
**Oracle Database 23ai** for organizations that standardize on Oracle
infrastructure. All memory operations — retain, recall, and reflect — work the
same way on Oracle; the backend is selected with a single environment variable.

This guide covers everything needed to run Hindsight against Oracle: the
prerequisites, the driver, a local quick start, provisioning a production
database, running migrations, and the handful of behavioural differences from
PostgreSQL.

:::info When to use Oracle
Oracle is the right choice when your organization already runs Oracle and needs
Hindsight to live inside that footprint. For everything else, the default
PostgreSQL backend is simpler to operate — see [Storage](./storage) for the
rationale. Oracle and PostgreSQL are configured independently; you pick one per
deployment.
:::

## Requirements

| Requirement | Details |
|-------------|---------|
| Oracle Database | **23ai** (23.4+). [Oracle Database Free 23ai](https://www.oracle.com/database/free/) works for development. |
| `VECTOR` type | Used for embeddings. Requires the schema to live in an **ASSM tablespace** (see below). |
| Oracle Text | Full-text search uses Oracle Text indexes. The schema user needs the `CTXAPP` role. |
| Driver | [`python-oracledb`](https://python-oracledb.readthedocs.io/) ≥ 2.5.0, running in **thin mode** — pure Python, no Oracle Instant Client required. |

:::warning The schema must use an ASSM tablespace
Oracle's `SYSTEM` tablespace uses *manual* segment space management (MSSM),
which **does not support `VECTOR` columns**. Create the Hindsight user in a
tablespace with **Automatic Segment Space Management (ASSM)** — otherwise
migrations fail when they create embedding columns. The provisioning SQL below
does this for you.
:::

## Install the driver

The Oracle driver is an optional extra — it is not bundled with the default
packages. Install it alongside Hindsight:

```bash
# With the packaged extra
pip install "hindsight-api-slim[oracle]"

# Or add the driver to an existing install (e.g. the full hindsight-api package)
pip install hindsight-api oracledb
```

If the driver is missing at startup, Hindsight fails with:
`python-oracledb is required for Oracle backend. Install it with: pip install oracledb`.

## Quick start (local Oracle)

The fastest way to try Hindsight on Oracle is the bundled helper script, which
starts a local **Oracle Database Free 23ai** container, provisions the test
user with the correct tablespace and grants, and prints a ready-to-use
connection URL:

```bash
# Start Oracle Free in Docker and bootstrap the hindsight_test user
./scripts/dev/start-oracle.sh

# ...prints:
#   export HINDSIGHT_API_DATABASE_BACKEND=oracle
#   export HINDSIGHT_API_DATABASE_URL='oracle+oracledb://hindsight_test:hindsight_test@localhost:1521/FREEPDB1'

# Stop and remove the container when done
./scripts/dev/stop-oracle.sh
```

A cold start takes 60–120s while the database initializes. If the first run
reports a provisioning error, the database was still starting up — just re-run
`./scripts/dev/start-oracle.sh` once the container is healthy (it is idempotent).

Once the script prints the connection URL, export the variables it shows, and
**also set the schema** to the Oracle user it created:

```bash
export HINDSIGHT_API_DATABASE_SCHEMA=HINDSIGHT_TEST
```

Then run migrations and start the API (see the steps below). Setting the schema
is required on Oracle — see [step 3](#3-configure-hindsight). This is the same
setup Hindsight's CI uses to test the Oracle backend.

## Production setup

### 1. Provision the schema user

Connect to your pluggable database as a privileged user (for example `SYSTEM`)
and create a dedicated tablespace and user for Hindsight. The tablespace **must**
use ASSM so `VECTOR` columns are supported:

```sql
-- ASSM tablespace (required for VECTOR columns). Size to your data volume.
CREATE BIGFILE TABLESPACE hindsight_ts
    DATAFILE 'hindsight_ts.dbf' SIZE 2G AUTOEXTEND ON NEXT 500M MAXSIZE UNLIMITED
    EXTENT MANAGEMENT LOCAL
    SEGMENT SPACE MANAGEMENT AUTO;

-- Dedicated schema user
CREATE USER hindsight IDENTIFIED BY "<strong-password>"
    DEFAULT TABLESPACE hindsight_ts
    TEMPORARY TABLESPACE temp
    QUOTA UNLIMITED ON hindsight_ts;

-- Object privileges Hindsight's migrations need
GRANT CONNECT, RESOURCE, CREATE TABLE, CREATE SEQUENCE, CREATE VIEW, CREATE PROCEDURE TO hindsight;

-- Oracle Text (full-text search indexes)
GRANT CTXAPP TO hindsight;
```

:::note Least privilege
`CONNECT` and `RESOURCE` cover the basics; the explicit `CREATE TABLE / SEQUENCE
/ VIEW / PROCEDURE` grants and `CTXAPP` are what the schema migrations require.
No `DBA` role is needed. On a managed service where `CREATE TABLESPACE` is not
available directly, provision the schema through the platform's admin tooling —
the requirements are unchanged: an **ASSM** default tablespace (needed for
`VECTOR` columns) plus the `CTXAPP` role.
:::

### 2. Build the connection URL

Hindsight uses SQLAlchemy-style URLs. The Oracle form is:

```
oracle+oracledb://USER:PASSWORD@HOST:PORT/SERVICE_NAME
```

| Part | Example | Notes |
|------|---------|-------|
| `USER` / `PASSWORD` | `hindsight` / `s3cret` | The schema user from step 1. URL-encode reserved characters (`@`, `/`, `:`) in the password. |
| `HOST:PORT` | `db.internal:1521` | The listener host and port (Oracle default is `1521`). |
| `SERVICE_NAME` | `FREEPDB1` | The **service name** of your pluggable database (not the SID). `FREEPDB1` for Oracle Free. |

Example:

```
oracle+oracledb://hindsight:s3cret@db.internal:1521/ORCLPDB1
```

:::warning Connection support: Easy Connect only
Hindsight builds the Oracle connection from the URL as a plain
`host:port/service_name` descriptor. **Wallet-based mTLS, TLS/TCPS, and TNS
aliases or full connect descriptors are not currently supported** by the
connection layer. In practice:

- **Oracle Autonomous Database** and other services that require a wallet /
  mTLS are not supported as-is — connect to a database reachable over a direct
  `host:port/service` listener.
- The driver does not negotiate TLS itself, so secure the connection at the
  network layer (private networking, VPN, or a TLS-terminating proxy).
:::

### 3. Configure Hindsight

Point Hindsight at Oracle with two environment variables:

```bash
export HINDSIGHT_API_DATABASE_BACKEND=oracle
export HINDSIGHT_API_DATABASE_URL='oracle+oracledb://hindsight:s3cret@db.internal:1521/ORCLPDB1'
export HINDSIGHT_API_DATABASE_SCHEMA=HINDSIGHT   # the Oracle user from step 1
```

`HINDSIGHT_API_DATABASE_BACKEND` defaults to `postgresql`; set it to `oracle` to
select the Oracle backend.

:::warning Set `DATABASE_SCHEMA` to your Oracle user
`HINDSIGHT_API_DATABASE_SCHEMA` defaults to `public` — a PostgreSQL concept. On
Oracle a schema **is a user**, there is no `public` schema, and leaving the
default makes migrations fail with `ORA-01435: user does not exist`. Set it to
the schema user you created in step 1, spelled exactly as Oracle stores it —
**uppercase** (e.g. `HINDSIGHT`) unless you created the user with a quoted
lower-case name.
:::

See [Configuration → Database](./configuration#database) for the full list of
database variables.

### 4. Run migrations

Hindsight runs the same schema migrations on Oracle as on PostgreSQL. By default
the API applies them automatically on startup
(`HINDSIGHT_API_RUN_MIGRATIONS_ON_STARTUP=true`). To run them explicitly — for
example in a controlled deploy step — use:

```bash
hindsight-admin run-db-migration
```

This routes through the dialect-aware migration runner and creates the Oracle
schema. (Unlike the admin CLI's data-movement commands, `run-db-migration`
is fully supported on Oracle — see [Limitations](#limitations-vs-postgresql).)

:::warning Migrate with your runtime embedding dimension
The embedding `VECTOR` columns are sized to the dimension of the configured
embeddings model. Run migrations with the **same embeddings provider/model you
will serve with** — otherwise the column dimension won't match the vectors the
API produces and retain fails with `ORA-51803: Vector dimension count must
match…` (for example, a schema built for a 384-dim local model rejects the
1536-dim vectors from OpenAI `text-embedding-3-small`). If you change the
embeddings model later, re-run migrations with `--embedding-dimension <N>` to
resize the columns.
:::

### 5. Start the API

```bash
hindsight-api
```

On startup Hindsight logs the resolved database (with credentials masked); it
should show your Oracle host and confirm the Oracle backend is active.

## Configuration reference

Oracle-relevant settings, all documented in full on the
[Configuration](./configuration) page:

| Variable | Purpose |
|----------|---------|
| `HINDSIGHT_API_DATABASE_BACKEND` | `postgresql` (default) or `oracle`. |
| `HINDSIGHT_API_DATABASE_URL` | `oracle+oracledb://…` connection URL. |
| `HINDSIGHT_API_DATABASE_SCHEMA` | Schema/user for the tables. On Oracle set this to your schema user (uppercase); the `public` default fails. |
| `HINDSIGHT_API_RUN_MIGRATIONS_ON_STARTUP` | Auto-apply migrations when the API boots (default `true`). |

## Limitations vs PostgreSQL

Memory operations behave identically on Oracle, but a few operational and
internal details differ:

- **Admin CLI data commands are PostgreSQL-only.** `hindsight-admin` backup,
  restore, bank export/import, and worker-status use asyncpg binary `COPY` and
  `TRUNCATE`, which are PostgreSQL-specific and not available on Oracle.
  Schema migrations (`run-db-migration`) *are* supported on Oracle.
- **No embedded database.** The `pg0` embedded PostgreSQL used for zero-config
  local development has no Oracle equivalent — Oracle always requires a running
  instance (use the [quick-start script](#quick-start-local-oracle) locally).
- **Consolidation reconciliation is skipped.** The similarity-based
  near-duplicate reconciliation pass in consolidation
  (`HINDSIGHT_API_CONSOLIDATION_DEDUP_THRESHOLD`) is a PostgreSQL-only path;
  consolidation still runs on Oracle, without that extra reconciliation step.
- **Entity resolution uses Oracle fuzzy matching.** Fuzzy entity lookup during
  retain uses Oracle's text matching rather than PostgreSQL's `pg_trgm` trigram
  matching. Behaviour is equivalent; the underlying mechanism differs.

## Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `python-oracledb is required for Oracle backend` | The driver isn't installed. Run `pip install oracledb` (or install the `[oracle]` extra). |
| `ORA-01435: user does not exist` on migration | `HINDSIGHT_API_DATABASE_SCHEMA` is unset (defaults to `public`) or misspelled. Set it to your Oracle schema user, uppercase (e.g. `HINDSIGHT`). |
| `ORA-51803: Vector dimension count must match` on retain | The schema was migrated with a different embedding dimension than the running embeddings model. Migrate with the same embeddings config, or re-run `run-db-migration --embedding-dimension <N>`. |
| Migration errors when creating embedding/`VECTOR` columns | The schema user's default tablespace is not ASSM (often the `SYSTEM` tablespace). Recreate the user in an ASSM tablespace as shown above. |
| Full-text search errors / missing Oracle Text index | The schema user is missing the `CTXAPP` role. Run `GRANT CTXAPP TO <user>;`. |
| `ORA-12514` / service not found | The URL uses a SID or wrong service name. Use the pluggable database **service name** (e.g. `FREEPDB1`), not the SID. |
| Login works manually but fails from Hindsight | A reserved character in the password isn't URL-encoded. Encode `@ / : ?` in the `DATABASE_URL`. |

## See also

- [Storage](./storage) — why PostgreSQL is the default, and how Oracle fits in
- [Configuration](./configuration#database) — all database environment variables
- [Installation](./installation) — packaging and deployment options
- [Admin CLI](./admin-cli) — administrative commands (PostgreSQL-only data operations)
