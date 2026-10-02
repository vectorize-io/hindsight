# Oracle Database

Hindsight uses PostgreSQL as its default storage backend, but it also runs on
**Oracle Database 23ai** and **Oracle AI Database 26ai** — including Oracle
Autonomous AI Database — for organizations that standardize on Oracle
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
| Oracle Database | **23ai** (23.4+) or **Oracle AI Database 26ai** (23.26). [Oracle Database Free](https://www.oracle.com/database/free/) works for development; Oracle Autonomous AI Database (including the Always Free tier) works for production — see [Autonomous AI Database](#autonomous-ai-database). |
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

#### Runtime user without DDL privileges

The schema user above owns the tables and runs the migrations. The API itself
does not need DDL: on Oracle AI Database 26ai, give it a separate user with
**schema privileges**, which cover the owner's current *and future* tables, so a
later migration does not need new grants:

```sql
CREATE USER hindsight_app IDENTIFIED BY "<strong-password>";
GRANT CREATE SESSION TO hindsight_app;
GRANT SELECT ANY TABLE, INSERT ANY TABLE, UPDATE ANY TABLE, DELETE ANY TABLE
    ON SCHEMA hindsight TO hindsight_app;
```

The runtime user needs no quota: rows are stored in the owner's schema. Point the
API at the runtime user and let migrations run as the owner:

```bash
export HINDSIGHT_API_DATABASE_URL='oracle+oracledb://hindsight_app:<password>@db.internal:1521/ORCLPDB1'
export HINDSIGHT_API_MIGRATION_DATABASE_URL='oracle+oracledb://hindsight:<password>@db.internal:1521/ORCLPDB1'
export HINDSIGHT_API_DATABASE_SCHEMA=HINDSIGHT   # the owner, not the runtime user
```

With `HINDSIGHT_API_MIGRATION_DATABASE_URL` set, the startup migrations and the
embedding-dimension check run as the owner; everything else runs as the runtime
user. On Oracle Database 23ai, which lacks the schema-level DML grants that 26ai
adds, grant the four object privileges on each table instead, and re-run the
grants after migrations that add tables.

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

#### Full connect descriptors and TNS aliases

The form above is Easy Connect, which can only express `host:port/service`. When you
need more — Autonomous Database over TCPS, `retry_count`, `ssl_server_dn_match`, or a
TNS alias from a `tnsnames.ora` — leave the host empty and pass the descriptor as a
`dsn` query parameter:

```
oracle+oracledb://USER:PASSWORD@/?dsn=DESCRIPTOR_OR_TNS_ALIAS
```

The descriptor has to be percent-encoded, like the password. For example, the
Autonomous Database descriptor

```
(description=(retry_count=20)(address=(protocol=tcps)(port=1522)(host=adb.example.com))(connect_data=(service_name=abc_low))(security=(ssl_server_dn_match=yes)))
```

becomes

```bash
export HINDSIGHT_API_DATABASE_URL="oracle+oracledb://hindsight:s3cret@/?dsn=$(python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.stdin.read().strip()))' <<< '(description=(retry_count=20)(address=(protocol=tcps)(port=1522)(host=adb.example.com))(connect_data=(service_name=abc_low))(security=(ssl_server_dn_match=yes)))')"
```

:::warning Wallet-based mTLS is still not supported
Hindsight passes no wallet directory or wallet password to the driver, so
connections that require a wallet (mTLS) do not work. TCPS with
`ssl_server_dn_match` does. Otherwise secure the connection at the network layer
(private networking, VPN, or a TLS-terminating proxy).
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

#### Embedding dimension

The baseline creates the embedding columns as `VECTOR(384, FLOAT32)`. On every
startup — and with `hindsight-admin run-db-migration --embedding-dimension <N>` —
Hindsight reconciles `memory_units.embedding` and `mental_models.embedding` with
the configured embeddings model, as it does on PostgreSQL:

| Column state | What happens |
|--------------|--------------|
| Same dimension as the model | Nothing. |
| Other dimension, table empty (a fresh install) | The column is replaced with `VECTOR(<N>, FLOAT32)` and its vector indexes are rebuilt with the same organization. Oracle cannot change a `VECTOR` dimension in place (`ALTER TABLE … MODIFY` fails with `ORA-51859` even on an empty table), so the column is renamed, re-added and the old one dropped; an interrupted run is finished by the next one, including the index rebuild (the dropped indexes' DDL is kept in a comment on the `embedding` column until they exist again). Workers booting together converge: one that loses a DDL race re-reads the catalog and carries on. |
| Other dimension, embeddings stored | Startup fails with an explicit error. Re-embed the data or configure a model with the stored dimension. |
| Flexible `VECTOR(*, *)` column (created by hand) | Accepted as long as every stored embedding has the model's dimension; never altered. |

So a deployment with a 1536-dimension model (OpenAI `text-embedding-3-small`,
Gemini `gemini-embedding-001` with 1536 output dimensions, …) needs no manual
DDL: the first startup sizes the empty tables.

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
| `HINDSIGHT_API_MIGRATION_DATABASE_URL` | URL of the schema owner, used for migrations and the embedding-dimension check when the API connects as a [runtime user](#runtime-user-without-ddl-privileges). |
| `HINDSIGHT_API_RUN_MIGRATIONS_ON_STARTUP` | Auto-apply migrations when the API boots (default `true`). |
| `HINDSIGHT_API_ORACLE_VECTOR_SEARCH` | `exact` (default) or `approx` — see [Vector search](#vector-search). |
| `HINDSIGHT_API_ORACLE_VECTOR_TARGET_ACCURACY` | Target accuracy (1–100) of `approx` semantic recall (default `95`). |
| `HINDSIGHT_API_DB_POOL_MAX_SIZE` | Upper bound of the connection pool (default `100`); lower it to fit the service's session limit — see [Autonomous AI Database](#autonomous-ai-database). |

## Autonomous AI Database

Hindsight runs on Oracle Autonomous AI Database, including the Always Free tier
(verified on Oracle AI Database 26ai 23.26.3). Oracle manages the instance; you
manage the schema, users, grants and the connection.

- **Connection.** Use a TLS connect descriptor (see [Full connect descriptors](#full-connect-descriptors-and-tns-aliases)):
  wallet-based mTLS is not supported, so set *Mutual TLS (mTLS) authentication*
  to *Not required* and restrict access with an access control list or a private
  endpoint. The `_low` or `_tp` service suits the API.
- **Sessions.** Always Free instances accept 30 concurrent sessions, shared with
  every other client of the database. Size `HINDSIGHT_API_DB_POOL_MAX_SIZE`
  (default `100`) well below that — for example `10` — so a load spike cannot
  exhaust the sessions other clients need.
- **Storage.** Always Free instances have 20 GB. `memory_units` is list-partitioned
  by bank, and every partition allocates its LOB segments up front, so each bank
  costs tens of MB even when nearly empty. The audit log
  (`HINDSIGHT_API_AUDIT_LOG_ENABLED`) stores full request bodies and grows fastest;
  enable it only while you need it.
- **Vector memory.** The vector pool is managed by the service: `vector_memory_size`
  reads `0` and cannot be set, and the pool grows when an HNSW index is created.
- **DDL latency.** Index creation and drops take seconds to minutes on a 1-ECPU
  instance (they wait for checkpoints); plan index changes outside peak hours.

## Vector search

Semantic recall ranks the bank's memories by `VECTOR_DISTANCE(…, COSINE)` against
the query embedding. The row-limiting clause decides whether that search is
exact or approximate:

- **`exact` (default)** — `FETCH EXACT FIRST n ROWS ONLY`. Every candidate row of
  the bank's partition is compared, so results are the true nearest neighbours.
  The `EXACT` keyword matters on Autonomous Database: there a bare `FETCH FIRST`
  is answered from a vector index whenever one exists, which with the baseline's
  global IVF index and the per-bank filter returned less than half of the true
  top-20 in our tests. Vector-ordered queries built elsewhere (temporal recall,
  link expansion) always use `EXACT`.
- **`approx`** — `HINDSIGHT_API_ORACLE_VECTOR_SEARCH=approx` switches semantic
  recall to `FETCH APPROX FIRST n ROWS ONLY WITH TARGET ACCURACY <n>` so Oracle
  may answer from the `memory_units.embedding` vector index. It only pays off for
  large banks: with the partition pruning on `bank_id`, exact search over a few
  thousand memories takes tens of milliseconds.

The baseline creates a global IVF index (`idx_mu_embedding_hnsw`, organization
`NEIGHBOR PARTITIONS`, despite its name). Oracle also offers local HNSW indexes on
partitioned tables (Oracle AI Database 26ai), which search only the bank's
partition:

```sql
DROP INDEX idx_mu_embedding_hnsw;
CREATE VECTOR INDEX idx_mu_embedding_hnsw ON memory_units (embedding)
    ORGANIZATION INMEMORY NEIGHBOR GRAPH DISTANCE COSINE
    WITH TARGET ACCURACY 95 LOCAL;
```

Whether the optimizer actually uses an index is cost-based; check with
`EXPLAIN PLAN` (an HNSW scan shows as `VECTOR INDEX HNSW SCAN`, an IVF scan as
access to its `VECTOR$…IVF_FLAT_CENTROIDS` tables) and measure recall against
`exact` before enabling `approx`.

## Full-text search

The BM25 arm uses an Oracle Text `CTXSYS.CONTEXT` index on `memory_units(text)`
with `SYNC (ON COMMIT)`. Every query term is wrapped in braces, so Oracle Text
reads it literally: reserved words (`NEAR`, `ABOUT`, …), operator characters and
`_` (Oracle Text's one-character wildcard) cannot change the expression, and
`snake_case` identifiers match the words the lexer indexed. Terms are
deduplicated, capped by `HINDSIGHT_API_BM25_MAX_QUERY_TERMS` like on
PostgreSQL, and combined with `ACCUM`, which ranks a memory higher the more query
terms it contains (`OR` scores it by its best single term, so a common word ranks
as high as the rare one the question is about).

## Hybrid search

Semantic, keyword, graph and temporal results are fused by Hindsight itself with
reciprocal rank fusion, then reranked by the configured reranker — the same
pipeline as on PostgreSQL, with every arm's scores in the recall trace. Oracle's
own hybrid search (hybrid vector indexes queried with `DBMS_HYBRID_VECTOR.SEARCH`)
is not used: a hybrid vector index computes its own embeddings with an
in-database ONNX model (or a provider called from the database), and queries it
with a `search_vector` must use that same model — a second vector space alongside
the embeddings Hindsight produces with its configured provider.

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
- **Approximate search is opt-in for the recall-time lookups.**
  `HINDSIGHT_API_ORACLE_VECTOR_SEARCH=approx` affects the semantic arm of
  recall; the other recall-time vector-ordered lookups (temporal recall, link
  expansion) always search exactly. Separately, semantic link construction
  during retain always searches approx — `HINDSIGHT_API_ORACLE_VECTOR_SEARCH`
  does not change it.

## Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `python-oracledb is required for Oracle backend` | The driver isn't installed. Run `pip install oracledb` (or install the `[oracle]` extra). |
| `ORA-01435: user does not exist` on migration | `HINDSIGHT_API_DATABASE_SCHEMA` is unset (defaults to `public`) or misspelled. Set it to your Oracle schema user, uppercase (e.g. `HINDSIGHT`). |
| Startup fails with `Cannot change embedding dimension from <X> to <N>` | The tables already hold embeddings of another dimension than the configured model. Re-embed the data (empty the tables and restart) or configure a model with `<X>` dimensions. See [Embedding dimension](#embedding-dimension). |
| `ORA-51803: Vector dimension count must match` on retain or recall | A flexible or hand-altered `VECTOR` column holds embeddings of several dimensions. Re-embed the stored rows with the configured model. |
| `ORA-01031: insufficient privileges` during startup migrations | The API connects as a runtime user without DDL rights. Set `HINDSIGHT_API_MIGRATION_DATABASE_URL` to the schema owner — see [Runtime user without DDL privileges](#runtime-user-without-ddl-privileges). |
| Semantic recall differs between Oracle Free and Autonomous Database | Fixed by `FETCH EXACT`: a bare `FETCH FIRST` is approximate on Autonomous Database when a vector index exists. Upgrade, or see [Vector search](#vector-search). |
| Migration errors when creating embedding/`VECTOR` columns | The schema user's default tablespace is not ASSM (often the `SYSTEM` tablespace). Recreate the user in an ASSM tablespace as shown above. |
| Full-text search errors / missing Oracle Text index | The schema user is missing the `CTXAPP` role. Run `GRANT CTXAPP TO <user>;`. |
| `ORA-12514` / service not found | The URL uses a SID or wrong service name. Use the pluggable database **service name** (e.g. `FREEPDB1`), not the SID. |
| Login works manually but fails from Hindsight | A reserved character in the password isn't URL-encoded. Encode `@ / : ?` in the `DATABASE_URL`. |

## See also

- [Storage](./storage) — why PostgreSQL is the default, and how Oracle fits in
- [Configuration](./configuration#database) — all database environment variables
- [Installation](./installation) — packaging and deployment options
- [Admin CLI](./admin-cli) — administrative commands (PostgreSQL-only data operations)
