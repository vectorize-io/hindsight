"""PostgreSQL-only admin utilities (backup, restore, migration, worker management).

Not supported on Oracle backends. Uses asyncpg.connect() directly, binary COPY,
TRUNCATE CASCADE, and REFRESH MATERIALIZED VIEW — all inherently PG-specific.
"""

import asyncio
import io
import json
import logging
import struct
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg
import typer

from ..config import DEFAULT_DATABASE_SCHEMA, HindsightConfig, load_dotenv_for_entrypoint
from ..engine.memories import get_memories
from ..engine.memory_engine import _current_schema
from ..engine.retain.bank_utils import _vector_index_clause
from ..engine.schema import fq_table_explicit as _fq_table
from ..engine.transfer import export_bank
from ..engine.vector_index_health import (
    BankIndexResult,
    drop_orphaned_bank_indexes,
    list_bank_ids,
    reconcile_bank_vector_indexes,
)
from ..extensions import TenantExtension, load_extension
from ..pg0 import parse_pg0_url, resolve_database_url

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(name="hindsight-admin", help="Hindsight administrative commands")

# Tables to backup/restore in foreign-key dependency order (parents first).
# Restore COPYs in this order and TRUNCATEs in reverse, so every child must
# appear after the tables it references.
#
# This must cover EVERY persistent PostgreSQL table in the schema — a missing
# entry silently drops that table's data on restore (and, worse, restore's
# `TRUNCATE banks CASCADE` wipes any FK-to-banks child like mental_models even
# when it was never backed up). test_admin_backup_restore.py asserts this list
# equals the live schema's tables, so adding a migration that creates a table
# without adding it here fails CI. Oracle-only tables (e.g. observation_sources)
# are intentionally absent — admin backup/restore is PostgreSQL-only.
BACKUP_TABLES = [
    "banks",
    "documents",
    "entities",
    "chunks",
    "memory_units",
    "invalidated_memory_units",
    "unit_entities",
    "entity_cooccurrences",
    "memory_links",
    "observation_history",
    "mental_models",
    "mental_model_history",
    "knowledge_pages",
    "directives",
    "async_operations",
    "webhooks",
    "file_storage",
    "audit_log",
    "llm_requests",
    "graph_maintenance_queue",
    "entity_maintenance_queue",
]

MANIFEST_VERSION = "2"


@dataclass(frozen=True)
class BackupColumn:
    """A PostgreSQL column shape required to decode a binary COPY stream."""

    name: str
    type_name: str


@dataclass(frozen=True)
class TableRestorePlan:
    """How one table's backed-up binary COPY stream is replayed onto the target.

    ``columns`` is the target column list handed to ``copy_to_table``, in stream
    order. When the target no longer has a backed-up column, its field is stripped
    from every tuple (``dropped_field_indices``) before the stream is replayed —
    binary COPY is positional, so the column list and the tuple fields must agree.
    """

    columns: list[str]
    dropped_field_indices: tuple[int, ...]
    source_field_count: int


# Header of a PostgreSQL binary COPY stream: an 11-byte signature, an int32 flags
# field, and an int32 header-extension length followed by that many bytes.
_COPY_BINARY_SIGNATURE = b"PGCOPY\n\xff\r\n\x00"
_COPY_BINARY_HEADER_LEN = len(_COPY_BINARY_SIGNATURE) + 8


def _strip_binary_copy_fields(data: bytes, plan: TableRestorePlan) -> bytes:
    """Drop `plan.dropped_field_indices` from every tuple of a binary COPY stream.

    Restore used to reject a backup whose columns the target no longer had — the
    preflight raised "target is missing backup columns …", which made any backup
    taken before a column-dropping migration unrestorable afterwards. Those columns
    are now ignored instead, but they cannot simply be left out of the
    ``copy_to_table`` column list: binary COPY carries no column identities, so each
    tuple's fields are matched to the column list purely by position and an unedited
    stream would desynchronise (or, worse, land values in the wrong columns). So the
    stream itself is rewritten here.

    Tuple format: int16 field count, then per field an int32 length (-1 for NULL)
    followed by that many bytes. An int16 of -1 is the end-of-data trailer.
    """
    if not plan.dropped_field_indices:
        return data
    if not data.startswith(_COPY_BINARY_SIGNATURE):
        raise ValueError("Backup stream is not in PostgreSQL binary COPY format")

    (extension_len,) = struct.unpack_from("!i", data, len(_COPY_BINARY_SIGNATURE) + 4)
    pos = _COPY_BINARY_HEADER_LEN + extension_len
    out = bytearray(data[:pos])

    dropped = set(plan.dropped_field_indices)
    kept_count = plan.source_field_count - len(dropped)
    while True:
        (field_count,) = struct.unpack_from("!h", data, pos)
        pos += 2
        if field_count == -1:  # end-of-data trailer
            out += struct.pack("!h", -1)
            break
        if field_count != plan.source_field_count:
            raise ValueError(
                f"Backup stream tuple has {field_count} fields, manifest declares {plan.source_field_count}"
            )
        out += struct.pack("!h", kept_count)
        for index in range(field_count):
            (length,) = struct.unpack_from("!i", data, pos)
            pos += 4
            payload = b"" if length == -1 else data[pos : pos + length]
            pos += max(length, 0)
            if index in dropped:
                continue
            out += struct.pack("!i", length)
            out += payload
    return bytes(out)


async def _table_columns(conn: asyncpg.Connection, schema: str, table: str) -> list[BackupColumn]:
    rows = await conn.fetch(
        """
        SELECT a.attname AS name, pg_catalog.format_type(a.atttypid, a.atttypmod) AS type_name
        FROM pg_catalog.pg_attribute AS a
        JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid
        JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
        WHERE n.nspname = $1 AND c.relname = $2 AND a.attnum > 0 AND NOT a.attisdropped
          AND a.attgenerated = ''
        ORDER BY a.attnum
        """,
        schema,
        table,
    )
    return [BackupColumn(name=row["name"], type_name=row["type_name"]) for row in rows]


async def _validate_restore_schema(
    conn: asyncpg.Connection, manifest: dict[str, Any], schema: str
) -> dict[str, TableRestorePlan]:
    """Validate every COPY stream against the target before destructive work starts.

    A column the target no longer has is **not** an error: a migration that drops a
    column would otherwise make every backup taken before it permanently
    unrestorable. Such columns are skipped (their fields are stripped from the
    stream by ``_strip_binary_copy_fields``) and reported, so the operator sees what
    was discarded instead of the restore failing outright.

    Type mismatches remain fatal. Type equality is an exact ``format_type`` string
    match. This is deliberately stricter than binary-COPY wire compatibility (e.g.
    ``varchar`` and ``text`` share a binary format yet compare unequal here): we
    would rather fail a genuinely-restorable backup with a clear, actionable error
    than silently risk a subtle binary mismatch. Restores blocked this way can be
    recovered by aligning the target schema.
    """
    plans: dict[str, TableRestorePlan] = {}
    errors: list[str] = []
    for table, table_manifest in manifest["tables"].items():
        source_columns = [BackupColumn(**column) for column in table_manifest["columns"]]
        target_by_name = {column.name: column for column in await _table_columns(conn, schema, table)}
        unknown = [
            (index, column.name) for index, column in enumerate(source_columns) if column.name not in target_by_name
        ]
        mismatched = [
            f"{column.name} ({column.type_name} in backup, {target_by_name[column.name].type_name} in target)"
            for column in source_columns
            if column.name in target_by_name and target_by_name[column.name].type_name != column.type_name
        ]
        if mismatched:
            errors.append(f"{table}: incompatible column types: {', '.join(mismatched)}")
        if unknown:
            typer.echo(
                f"  {table}: ignoring {len(unknown)} backup column(s) absent from the target schema: "
                f"{', '.join(name for _, name in unknown)}"
            )
        plans[table] = TableRestorePlan(
            columns=[column.name for column in source_columns if column.name in target_by_name],
            dropped_field_indices=tuple(index for index, _ in unknown),
            source_field_count=len(source_columns),
        )

    if errors:
        details = "; ".join(errors)
        raise ValueError(f"Backup schema is incompatible with target schema '{schema}': {details}")
    return plans


def _quote_identifier(identifier: str) -> str:
    """Quote a PostgreSQL identifier for catalog-derived dynamic SQL."""
    return '"' + identifier.replace('"', '""') + '"'


async def _sync_owned_sequences(conn: asyncpg.Connection, schema: str, tables: list[str]) -> None:
    """Advance non-cyclic identity sequences past restored column values."""
    sequences = await conn.fetch(
        """
        SELECT tbl.relname AS table_name,
               attr.attname AS column_name,
               seq_ns.nspname AS sequence_schema,
               seq.relname AS sequence_name,
               pg_seq.seqstart AS sequence_start,
               pg_seq.seqincrement AS sequence_increment,
               pg_seq.seqmin AS sequence_min,
               pg_seq.seqmax AS sequence_max
        FROM pg_catalog.pg_class AS tbl
        JOIN pg_catalog.pg_namespace AS tbl_ns ON tbl_ns.oid = tbl.relnamespace
        JOIN pg_catalog.pg_attribute AS attr
          ON attr.attrelid = tbl.oid AND attr.attnum > 0 AND NOT attr.attisdropped
        JOIN pg_catalog.pg_depend AS dep
          ON dep.refobjid = tbl.oid
         AND dep.refobjsubid = attr.attnum
         AND dep.classid = 'pg_catalog.pg_class'::regclass
         AND dep.refclassid = 'pg_catalog.pg_class'::regclass
         AND dep.deptype = 'i'
        JOIN pg_catalog.pg_class AS seq ON seq.oid = dep.objid AND seq.relkind = 'S'
        JOIN pg_catalog.pg_namespace AS seq_ns ON seq_ns.oid = seq.relnamespace
        JOIN pg_catalog.pg_sequence AS pg_seq ON pg_seq.seqrelid = seq.oid
        WHERE tbl_ns.nspname = $1
          AND tbl.relname = ANY($2::text[])
          AND attr.attidentity <> ''
          AND NOT pg_seq.seqcycle
        ORDER BY tbl.relname, attr.attnum
        """,
        schema,
        tables,
    )

    for sequence in sequences:
        aggregate = "MAX" if sequence["sequence_increment"] > 0 else "MIN"
        qualified_table = f"{_quote_identifier(schema)}.{_quote_identifier(sequence['table_name'])}"
        column = _quote_identifier(sequence["column_name"])
        restored_edge = await conn.fetchval(
            f"SELECT {aggregate}({column}) FROM {qualified_table} "
            f"WHERE {column} BETWEEN {int(sequence['sequence_min'])} AND {int(sequence['sequence_max'])}"
        )
        qualified_sequence = (
            f"{_quote_identifier(sequence['sequence_schema'])}.{_quote_identifier(sequence['sequence_name'])}"
        )
        restart_value = sequence["sequence_start"] if restored_edge is None else restored_edge
        await conn.execute(f"ALTER SEQUENCE {qualified_sequence} RESTART WITH {int(restart_value)}")
        if restored_edge is not None:
            # RESTART is transactional; consuming the restored edge gives the
            # sequence setval(edge, true) semantics without calculating a value
            # beyond a finite sequence's min/max bound.
            await conn.fetchval("SELECT nextval($1::regclass)", qualified_sequence)


def _effective_backup_tables() -> list[str]:
    """Core backup tables plus any bank-scoped tables a loaded extension declares.

    ``BACKUP_TABLES`` covers only the tables core owns. An extension that
    provisions its own bank-scoped tables (via ``TenantExtension``) declares
    them through ``extra_bank_tables()`` so they aren't dropped on restore.
    Extension tables are appended *after* the core set so restore's forward
    COPY inserts them after their FK parents (e.g. ``banks``) and the reversed
    TRUNCATE clears them before those parents.
    """
    tables = list(BACKUP_TABLES)
    tenant_extension = load_extension("TENANT", TenantExtension)
    if tenant_extension is not None:
        seen = set(tables)
        for spec in tenant_extension.extra_bank_tables():
            if spec.include_in_backup and spec.name not in seen:
                tables.append(spec.name)
                seen.add(spec.name)
    return tables


async def _admin_connect(db_url: str) -> asyncpg.Connection:
    """Open a raw asyncpg connection to an admin DB URL.

    ``resolve_database_url`` handles both plain ``postgres://`` (passthrough) and
    ``pg0://`` (boots the embedded server and returns its real libpq URL), so this
    is the only step needed to connect. JSON codecs are registered so ``jsonb``
    columns decode to Python objects (used by the export row dumps).
    """
    _pg0 = parse_pg0_url(db_url)
    is_pg0, instance_name = _pg0.is_pg0, _pg0.instance_name
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    conn = await asyncpg.connect(await resolve_database_url(db_url))
    for type_name in ("json", "jsonb"):
        await conn.set_type_codec(type_name, encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
    return conn


async def _backup(
    database_url: str,
    output_path: Path,
    schema: str = "public",
    backup_tables: list[str] | None = None,
) -> dict[str, Any]:
    """Backup all tables to a zip file using binary COPY protocol.

    ``backup_tables`` defaults to the core ``BACKUP_TABLES``; callers pass the
    extension-augmented list from ``_effective_backup_tables()``.
    """
    backup_tables = backup_tables if backup_tables is not None else BACKUP_TABLES
    conn = await asyncpg.connect(database_url)
    try:
        tables: dict[str, Any] = {}
        manifest: dict[str, Any] = {
            "version": MANIFEST_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "schema": schema,
            "tables": tables,
        }

        # Use a transaction with REPEATABLE READ isolation to get a consistent
        # snapshot across all tables. This prevents race conditions where
        # entity_cooccurrences could reference entities created after the
        # entities table was backed up.
        async with conn.transaction(isolation="repeatable_read"):
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, table in enumerate(backup_tables, 1):
                    typer.echo(f"  [{i}/{len(backup_tables)}] Backing up {table}...", nl=False)

                    buffer = io.BytesIO()

                    columns = await _table_columns(conn, schema, table)

                    # Pin the ordered columns into both the stream and manifest.
                    # PostgreSQL binary COPY does not encode column identities, so
                    # restore must validate this shape before truncating any data.
                    # asyncpg requires schema_name as separate parameter
                    await conn.copy_from_table(
                        table,
                        schema_name=schema,
                        columns=[column.name for column in columns],
                        output=buffer,
                        format="binary",
                    )

                    data = buffer.getvalue()
                    zf.writestr(f"{table}.bin", data)

                    # Get row count for manifest
                    qualified_table = _fq_table(table, schema)
                    row_count = await conn.fetchval(f"SELECT COUNT(*) FROM {qualified_table}")
                    tables[table] = {
                        "rows": row_count,
                        "size_bytes": len(data),
                        "columns": [{"name": column.name, "type_name": column.type_name} for column in columns],
                    }

                    typer.echo(f" {row_count} rows")

                zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        return manifest
    finally:
        await conn.close()


async def _restore(
    database_url: str,
    input_path: Path,
    schema: str = "public",
    backup_tables: list[str] | None = None,
) -> dict[str, Any]:
    """Restore all tables from a zip file using binary COPY protocol.

    ``backup_tables`` defaults to the core ``BACKUP_TABLES``; callers pass the
    extension-augmented list from ``_effective_backup_tables()``. Tables named
    here but absent from the archive are truncated then skipped for restore, so
    a stale extension registration never leaves pre-restore rows behind.
    """
    backup_tables = backup_tables if backup_tables is not None else BACKUP_TABLES
    conn = await asyncpg.connect(database_url)
    try:
        with zipfile.ZipFile(input_path, "r") as zf:
            # Read and validate manifest
            manifest: dict[str, Any] = json.loads(zf.read("manifest.json"))
            if manifest.get("version") != MANIFEST_VERSION:
                raise ValueError(f"Unsupported backup version: {manifest.get('version')}")

            # Complete the compatibility check before entering the transaction
            # that truncates tables. This turns historical schema drift into an
            # actionable error without risking the target's existing data.
            restore_plans = await _validate_restore_schema(conn, manifest, schema)

            # Use a transaction for atomic restore - either all tables are
            # restored or none are, preventing partial/inconsistent state.
            async with conn.transaction():
                typer.echo("  Clearing existing data...")
                # Truncate tables in reverse order (respects FK constraints)
                for table in reversed(backup_tables):
                    qualified_table = _fq_table(table, schema)
                    await conn.execute(f"TRUNCATE TABLE {qualified_table} CASCADE")

                # Restore tables in forward order
                for i, table in enumerate(backup_tables, 1):
                    filename = f"{table}.bin"
                    if filename not in zf.namelist():
                        typer.echo(f"  [{i}/{len(backup_tables)}] {table}: skipped (not in backup)")
                        continue

                    expected_rows = manifest["tables"].get(table, {}).get("rows", "?")
                    typer.echo(f"  [{i}/{len(backup_tables)}] Restoring {table}... {expected_rows} rows")

                    plan = restore_plans[table]
                    # Strips the fields of any column the target no longer has;
                    # a no-op when the schemas still line up.
                    buffer = io.BytesIO(_strip_binary_copy_fields(zf.read(filename), plan))
                    # asyncpg requires schema_name as separate parameter
                    await conn.copy_to_table(
                        table,
                        schema_name=schema,
                        columns=plan.columns,
                        source=buffer,
                        format="binary",
                    )

                # Refresh materialized view
                typer.echo("  Refreshing materialized views...")
                await conn.execute(f"REFRESH MATERIALIZED VIEW {_fq_table('memory_units_bm25', schema)}")

                typer.echo("  Synchronizing identity sequences...")
                await _sync_owned_sequences(conn, schema, backup_tables)

        return manifest
    finally:
        await conn.close()


async def _run_backup(db_url: str, output: Path, schema: str = "public") -> dict[str, Any]:
    """Resolve database URL and run backup."""
    _pg0 = parse_pg0_url(db_url)
    is_pg0, instance_name = _pg0.is_pg0, _pg0.instance_name
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    resolved_url = await resolve_database_url(db_url)
    return await _backup(resolved_url, output, schema, backup_tables=_effective_backup_tables())


async def _run_restore(db_url: str, input_file: Path, schema: str = "public") -> dict[str, Any]:
    """Resolve database URL and run restore."""
    _pg0 = parse_pg0_url(db_url)
    is_pg0, instance_name = _pg0.is_pg0, _pg0.instance_name
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    resolved_url = await resolve_database_url(db_url)
    return await _restore(resolved_url, input_file, schema, backup_tables=_effective_backup_tables())


@app.command()
def backup(
    output: Path = typer.Argument(..., help="Output file path (.zip)"),
    schema: str = typer.Option("public", "--schema", "-s", help="Database schema to backup"),
):
    """Backup the Hindsight database to a zip file."""
    config = HindsightConfig.from_env()

    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set HINDSIGHT_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    if output.suffix != ".zip":
        output = output.with_suffix(".zip")

    typer.echo(f"Backing up database (schema: {schema}) to {output}...")

    manifest = asyncio.run(_run_backup(config.database_url, output, schema))

    total_rows = sum(t["rows"] for t in manifest["tables"].values())
    typer.echo(f"Backed up {total_rows} rows across {len(manifest['tables'])} tables")
    typer.echo(f"Backup saved to {output}")


@app.command()
def restore(
    input_file: Path = typer.Argument(..., help="Input backup file (.zip)"),
    schema: str = typer.Option("public", "--schema", "-s", help="Database schema to restore to"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Restore the database from a backup file. WARNING: This deletes all existing data."""
    config = HindsightConfig.from_env()

    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set HINDSIGHT_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    if not input_file.exists():
        typer.echo(f"Error: File not found: {input_file}", err=True)
        raise typer.Exit(1)

    if not yes:
        typer.confirm(
            "This will DELETE all existing data and replace it with the backup. Continue?",
            abort=True,
        )

    typer.echo(f"Restoring database (schema: {schema}) from {input_file}...")

    manifest = asyncio.run(_run_restore(config.database_url, input_file, schema))

    total_rows = sum(t["rows"] for t in manifest["tables"].values())
    typer.echo(f"Restored {total_rows} rows across {len(manifest['tables'])} tables")
    typer.echo("Restore complete")


async def _run_migration(
    db_url: str,
    schema: str | None = None,
    base_schema: str = DEFAULT_DATABASE_SCHEMA,
    embedding_dimension: int | None = None,
    ensure_extensions: bool = True,
) -> list[str]:
    """Resolve database URL and run migrations for one schema or all discovered schemas."""
    from ..migrations import run_migrations_for_schemas

    _pg0 = parse_pg0_url(db_url)
    is_pg0, instance_name = _pg0.is_pg0, _pg0.instance_name
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    resolved_url = await resolve_database_url(db_url)

    config = HindsightConfig.from_env()
    tenant_extension = load_extension("TENANT", TenantExtension)
    if schema:
        schemas = [schema]
    else:
        schemas = [base_schema or DEFAULT_DATABASE_SCHEMA]
        if tenant_extension:
            tenants = await tenant_extension.list_tenants()
            schemas.extend(tenant.schema for tenant in tenants if tenant.schema)

        # Preserve order while removing duplicates.
        schemas = list(dict.fromkeys(schemas))

    # Migrate up to `migration_concurrency` schemas at once (each in its own
    # process); within a schema the work stays sequential. Run off the event
    # loop so the process pool's blocking joins don't stall it.
    await asyncio.to_thread(
        run_migrations_for_schemas,
        resolved_url,
        schemas,
        concurrency=config.migration_concurrency,
        migration_database_url=config.migration_database_url,
        embedding_dimension=embedding_dimension,
        vector_extension=config.vector_extension,
        text_search_extension=config.text_search_extension,
        pg_search_tokenizer=config.text_search_extension_pg_search_tokenizer,
        ensure_extensions=ensure_extensions,
    )

    # After core migrations, provision any extension-owned bank-scoped tables
    # per schema so extension schema evolves on the same lifecycle as core
    # schema (rather than via a lazy first-request path).
    if tenant_extension is not None:
        await _provision_extra_bank_tables(resolved_url, schemas, tenant_extension)

    return schemas


async def _provision_extra_bank_tables(
    resolved_url: str, schemas: list[str], tenant_extension: TenantExtension
) -> None:
    """Run the tenant extension's table provisioner for each migrated schema.

    Fires after core migrations complete so extension-owned bank tables are
    created/evolved on the same lifecycle as core schema. A failure aborts the
    migration command (and names the offending schema) rather than being
    swallowed — provisioning is idempotent, so the operator can fix and re-run.
    """
    for schema in schemas:
        conn = await asyncpg.connect(resolved_url)
        try:
            await tenant_extension.provision_bank_tables(conn, schema)
        except Exception as e:
            typer.echo(f"  Failed to provision extension tables for schema '{schema}': {e}", err=True)
            raise
        finally:
            await conn.close()


@app.command(name="run-db-migration")
def run_db_migration(
    schema: str | None = typer.Option(
        None,
        "--schema",
        "-s",
        help="Database schema to run migrations on. If omitted, migrate the base schema and all discovered tenant schemas.",
    ),
    embedding_dimension: int | None = typer.Option(
        None,
        "--embedding-dimension",
        help="Expected embedding dimension to enforce after migrations. Omit to skip dimension sync.",
    ),
    skip_extension_reconcile: bool = typer.Option(
        False,
        "--skip-extension-reconcile",
        help=(
            "Skip the post-migration vector / text-search index reconcile. This step only does "
            "work when the configured backend (HINDSIGHT_API_VECTOR_EXTENSION / "
            "HINDSIGHT_API_TEXT_SEARCH_EXTENSION) differs from a schema's existing indexes — a "
            "rare, operator-driven change. Skipping it makes a no-change re-migration over many "
            "tenant schemas much faster. Only use when you have NOT changed the backend; a "
            "backend change still needs a normal run to reshape the indexes."
        ),
    ),
):
    """Run database migrations to the latest version."""
    config = HindsightConfig.from_env()

    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set HINDSIGHT_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    if schema:
        typer.echo(f"Running database migrations for schema: {schema}...")
    else:
        typer.echo("Running database migrations for base schema and all discovered tenant schemas...")
    if skip_extension_reconcile:
        typer.echo("Skipping post-migration extension reconcile (--skip-extension-reconcile).")

    schemas = asyncio.run(
        _run_migration(
            config.database_url,
            schema=schema,
            base_schema=config.database_schema,
            embedding_dimension=embedding_dimension,
            ensure_extensions=not skip_extension_reconcile,
        )
    )

    typer.echo(f"Database migrations completed successfully for {len(schemas)} schema(s)")


async def _resolve_schemas(base_schema: str | None) -> list[str]:
    """Base schema plus every discovered tenant schema, de-duplicated in order."""
    schemas = [base_schema or DEFAULT_DATABASE_SCHEMA]
    tenant_extension = load_extension("TENANT", TenantExtension)
    if tenant_extension:
        tenants = await tenant_extension.list_tenants()
        schemas.extend(tenant.schema for tenant in tenants if tenant.schema)
    return list(dict.fromkeys(schemas))


async def _run_repair_bank(
    db_url: str,
    *,
    base_schema: str,
    schema: str | None,
    bank_id: str | None,
    dry_run: bool,
) -> list[BankIndexResult]:
    """Reconcile per-(bank, fact_type) vector index coverage over a raw connection.

    A single autocommit connection is used because ``CREATE INDEX CONCURRENTLY``
    cannot run inside a transaction block.

    Deliberately unbudgeted, unlike the background operation: this is an operator
    asking for convergence now, across as many banks as they named.
    """
    schemas = [schema] if schema else await _resolve_schemas(base_schema)
    index_clause = _vector_index_clause()
    # Guarded by the command, but assert so this helper is never called for a
    # backend without per-bank indexes.
    assert index_clause is not None

    conn = await _admin_connect(db_url)
    results: list[BankIndexResult] = []
    try:
        for target_schema in schemas:
            try:
                bank_ids = [bank_id] if bank_id else await list_bank_ids(conn, target_schema)
            except Exception as exc:  # noqa: BLE001 — one bad schema must not abort the sweep
                typer.echo(f"  schema '{target_schema}': skipped ({exc})", err=True)
                continue
            schema_results = [
                await reconcile_bank_vector_indexes(conn, target_schema, bid, index_clause, dry_run=dry_run)
                for bid in bank_ids
            ]
            results.extend(schema_results)
            # Only in --all mode: an index whose bank row is gone is unreachable
            # from every bank-scoped path, so this is the one place that can
            # collect it. Normally finds nothing — delete_bank drops a bank's
            # indexes while it still knows their names — but a deployment that
            # hit the #3485 wall could not run delete_bank at all.
            orphans = [] if bank_id else await drop_orphaned_bank_indexes(conn, target_schema, dry_run=dry_run)
            if orphans:
                typer.echo(
                    f"  schema '{target_schema}': {len(orphans)} orphaned index(es) "
                    f"{'to drop (dry-run)' if dry_run else 'dropped'} (no matching bank)"
                )
            typer.echo(
                f"  schema '{target_schema}': {len(bank_ids)} bank(s) scanned, "
                f"{sum(r.already_present for r in schema_results)} present, "
                f"{sum(r.created for r in schema_results)} created, "
                f"{sum(r.dropped for r in schema_results)} dropped, "
                f"{sum(r.skipped for r in schema_results)} to-create (dry-run), "
                f"{sum(r.would_drop for r in schema_results)} to-drop (dry-run), "
                f"{sum(r.failed for r in schema_results)} failed"
            )
        return results
    finally:
        await conn.close()


@app.command(name="repair-bank")
def repair_bank(
    bank_id: str | None = typer.Option(
        None,
        "--bank",
        "-b",
        help="Bank id to repair. Mutually exclusive with --all.",
    ),
    all_banks: bool = typer.Option(
        False,
        "--all",
        help="Repair every bank in the base schema and all discovered tenant schemas.",
    ),
    schema: str | None = typer.Option(
        None,
        "--schema",
        "-s",
        help="Limit to a single schema. Defaults to the base schema plus discovered tenant schemas.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report what would be repaired without creating or dropping any index.",
    ),
):
    """Reconcile per-(bank, fact_type) vector index coverage.

    With HINDSIGHT_API_VECTOR_INDEX_MIN_ROWS at its default of 0 every bank is
    owed all three indexes from the moment it exists, and this rebuilds whatever
    is missing or invalid — a bank whose creation lost its DDL to a deadlock, one
    restored around it, or one whose access method drifted after a backend
    switch. Nothing is dropped in that mode.

    With a threshold set, a (bank, fact_type) earns its index once it holds that
    many rows; below it the planner answers the same query exactly, and faster,
    from the (bank_id, fact_type) B-tree plus a top-N sort. This command then
    also drops what no longer qualifies. Either way it sheds indexes orphaned by
    a deleted bank, and all DDL is CONCURRENTLY, so it never blocks the live
    fleet.

    With a threshold set, writes keep this converged on their own — every write
    that could move a bank across it queues a vector_index_maintenance operation.
    Reach for the command when you want convergence without waiting for a write:
    after a restore or upgrade, after a backend switch, or to shed indexes in
    bulk on a deployment recovering from lock-table exhaustion (#3485).
    Idempotent and safe to re-run.
    """
    if bool(bank_id) == all_banks:
        typer.echo("Error: pass exactly one of --bank <id> or --all.", err=True)
        raise typer.Exit(2)

    config = HindsightConfig.from_env()
    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set HINDSIGHT_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    # Backend guard: backends with a single global vector index (AlloyDB ScaNN,
    # Oracle) have no per-bank indexes to repair.
    if _vector_index_clause() is None:
        typer.echo("Configured vector backend does not use per-bank vector indexes — nothing to repair.")
        return

    target = f"bank '{bank_id}'" if bank_id else "all banks"
    scope = f"schema '{schema}'" if schema else "base schema and all discovered tenant schemas"
    typer.echo(f"Repairing per-bank vector indexes for {target} across {scope}...")
    if dry_run:
        typer.echo("Dry run: no indexes will be created or dropped.")

    results = asyncio.run(
        _run_repair_bank(
            config.database_url,
            base_schema=config.database_schema,
            schema=schema,
            bank_id=bank_id,
            dry_run=dry_run,
        )
    )

    total_banks = len(results)
    total_present = sum(r.already_present for r in results)
    total_created = sum(r.created for r in results)
    total_dropped = sum(r.dropped for r in results)
    total_skipped = sum(r.skipped for r in results)
    total_would_drop = sum(r.would_drop for r in results)
    total_failed = sum(r.failed for r in results)
    typer.echo(
        f"Done: {len(results)} schema(s), {total_banks} bank(s) scanned, "
        f"{total_present} already present, {total_created} created, {total_dropped} dropped, "
        f"{total_skipped} to-create (dry-run), {total_would_drop} to-drop (dry-run), "
        f"{total_failed} failed"
    )
    if total_failed:
        failed_names = [name for r in results for name in r.failed_indexes]
        typer.echo(f"Failed indexes (dropped, retry with a re-run): {', '.join(failed_names)}", err=True)
        raise typer.Exit(1)


async def _run_export_bank(db_url: str, bank_id: str, output: Path, schema: str, include_history: bool) -> int:
    """Export a whole bank to a ZIP archive.

    The memories store is resolved from config here and handed to the export, because a bank whose
    memories live outside SQL cannot be read from this connection: the tables would be empty and the
    archive would come out well-formed and empty, which an operator only discovers at restore time.
    A Postgres deployment resolves to the SQL store, whose capability probe sends the export down
    exactly the path it always took.

    Resolving it in a short-lived CLI process is safe because the store connects lazily on first
    use rather than in `initialize()`.
    """
    conn = await _admin_connect(db_url)
    try:
        # export_bank resolves table names via fq_table (the _current_schema
        # contextvar); set it so the raw connection targets the right schema.
        _current_schema.set(schema)
        # _admin_connect registers JSON codecs, so row dumps already contain
        # decoded Python values (including JSON scalar strings).
        data = await export_bank(
            conn,
            bank_id,
            include_history=include_history,
            bank_rows_json_encoding="decoded",
            memories=get_memories(),
        )
    finally:
        await conn.close()

    output.write_bytes(data)
    return len(data)


@app.command(name="export-bank")
def export_bank_command(
    bank_id: str = typer.Option(..., "--bank", "-b", help="Bank id to export."),
    output: Path = typer.Option(..., "--output", "-o", help="Path to write the .zip archive."),
    schema: str | None = typer.Option(
        None,
        "--schema",
        "-s",
        help="Database schema the bank lives in. Defaults to the configured base schema.",
    ),
    include_history: bool = typer.Option(
        False,
        "--include-history",
        help="Also export operational history (audit_log, llm_requests). Off by default.",
    ),
):
    """Export an entire bank to a portable ZIP (no embeddings — regenerated on import).

    Carries documents, facts, observations, bank config, mental models, directives
    and webhooks so the bank can be imported into a new instance configured with a
    different embedding model / vector / text-search backend.
    """
    config = HindsightConfig.from_env()

    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set HINDSIGHT_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    target_schema = schema or config.database_schema or DEFAULT_DATABASE_SCHEMA
    typer.echo(f"Exporting bank '{bank_id}' from schema '{target_schema}'...")

    size = asyncio.run(_run_export_bank(config.database_url, bank_id, output, target_schema, include_history))

    typer.echo(f"Exported bank '{bank_id}' to {output} ({size} bytes)")


async def _run_import_bank(archive_path: Path, schema: str, target_bank_id: str | None, include_history: bool):
    """Boot a MemoryEngine (for the target's embedding model) and restore a bank archive."""
    # MemoryEngine is heavy (loads embeddings); import it lazily so other admin
    # commands don't pay for it. _current_schema is imported at module top.
    from ..engine.memory_engine import MemoryEngine
    from ..models import RequestContext

    archive_bytes = archive_path.read_bytes()
    # run_migrations=True so a fresh target instance is provisioned at this
    # instance's embedding dimension / vector / text-search backend before restore.
    engine = MemoryEngine(run_migrations=True)
    await engine.initialize()
    try:
        _current_schema.set(schema)
        context = RequestContext(internal=True, user_initiated=True)
        return await engine.import_bank_async(
            archive_bytes,
            context,
            target_bank_id=target_bank_id,
            include_history=include_history,
        )
    finally:
        await engine.close()


@app.command(name="import-bank")
def import_bank_command(
    archive: Path = typer.Option(..., "--archive", "-a", help="Path to the .zip produced by export-bank."),
    schema: str | None = typer.Option(
        None, "--schema", "-s", help="Target schema. Defaults to the configured base schema."
    ),
    target_bank: str | None = typer.Option(
        None, "--target-bank", help="Override the bank id (defaults to the archive's source bank)."
    ),
    include_history: bool = typer.Option(
        False, "--include-history", help="Also restore operational history if present in the archive."
    ),
):
    """Restore a whole bank from an export-bank archive into THIS instance.

    Re-embeds facts with this instance's configured embedding model and rebuilds
    links and indexes — the import half of a cross-instance migration. Run against
    an instance configured with the desired embedding / vector / text-search backend.
    The target bank must not already exist (import restores a whole bank, not a merge).
    """
    config = HindsightConfig.from_env()
    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set HINDSIGHT_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    target_schema = schema or config.database_schema or DEFAULT_DATABASE_SCHEMA
    typer.echo(f"Importing bank archive '{archive}' into schema '{target_schema}'...")

    result = asyncio.run(_run_import_bank(archive, target_schema, target_bank, include_history))

    typer.echo(
        f"Imported bank '{result.bank_id}': {result.documents_imported} doc(s), "
        f"{result.facts_imported} fact(s), {result.observations_imported} observation(s), "
        f"{result.mental_models_imported} mental model(s), "
        f"{result.mental_model_history_imported} mm-history row(s), "
        f"{result.knowledge_pages_imported} knowledge page(s), {result.directives_imported} directive(s), "
        f"{result.webhooks_imported} webhook(s), {result.history_rows_imported} history row(s)"
    )


async def _decommission_worker(db_url: str, worker_id: str, schema: str = "public") -> int:
    """Release all tasks owned by a worker, setting them back to pending status."""
    _pg0 = parse_pg0_url(db_url)
    is_pg0, instance_name = _pg0.is_pg0, _pg0.instance_name
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    resolved_url = await resolve_database_url(db_url)

    conn = await asyncpg.connect(resolved_url)
    try:
        table = _fq_table("async_operations", schema)
        result = await conn.fetch(
            f"""
            UPDATE {table}
            SET status = 'pending', worker_id = NULL, claimed_at = NULL, updated_at = now()
            WHERE worker_id = $1 AND status = 'processing'
            RETURNING operation_id
            """,
            worker_id,
        )
        return len(result)
    finally:
        await conn.close()


@app.command(name="decommission-worker")
def decommission_worker(
    worker_id: str = typer.Argument(..., help="Worker ID to decommission"),
    schema: str = typer.Option("public", "--schema", "-s", help="Database schema"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Release all tasks owned by a worker (sets status back to pending).

    Use this command when a worker has crashed or been removed without graceful shutdown.
    All tasks that were being processed by the worker will be released back to the queue
    so other workers can pick them up.
    """
    config = HindsightConfig.from_env()

    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set HINDSIGHT_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    if not yes:
        typer.confirm(
            f"This will release all tasks owned by worker '{worker_id}' back to pending. Continue?",
            abort=True,
        )

    typer.echo(f"Decommissioning worker '{worker_id}' (schema: {schema})...")

    count = asyncio.run(_decommission_worker(config.database_url, worker_id, schema))

    if count > 0:
        typer.echo(f"Released {count} task(s) from worker '{worker_id}'")
    else:
        typer.echo(f"No tasks found for worker '{worker_id}'")


async def _decommission_all_workers(db_url: str, schema: str = "public") -> list[dict[str, Any]]:
    """Release all processing tasks from all workers, setting them back to pending status."""
    _pg0 = parse_pg0_url(db_url)
    is_pg0, instance_name = _pg0.is_pg0, _pg0.instance_name
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    resolved_url = await resolve_database_url(db_url)

    conn = await asyncpg.connect(resolved_url)
    try:
        table = _fq_table("async_operations", schema)
        rows = await conn.fetch(
            f"""
            UPDATE {table}
            SET status = 'pending', worker_id = NULL, claimed_at = NULL, updated_at = now()
            WHERE status = 'processing'
            RETURNING operation_id, worker_id, operation_type
            """,
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


@app.command(name="decommission-workers")
def decommission_workers(
    schema: str = typer.Option("public", "--schema", "-s", help="Database schema"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Release all processing tasks from all workers (sets status back to pending).

    Use this command to recover from situations where one or more workers have crashed
    or been removed without graceful shutdown. All tasks currently in 'processing' status
    will be released back to the queue regardless of which worker owns them.
    """
    config = HindsightConfig.from_env()

    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set HINDSIGHT_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    if not yes:
        typer.confirm(
            "This will release ALL processing tasks from ALL workers back to pending. Continue?",
            abort=True,
        )

    typer.echo(f"Decommissioning all workers (schema: {schema})...")

    released = asyncio.run(_decommission_all_workers(config.database_url, schema))

    if released:
        # Group by worker_id for summary
        by_worker: dict[str, int] = {}
        for row in released:
            wid = row["worker_id"] or "unknown"
            by_worker[wid] = by_worker.get(wid, 0) + 1

        typer.echo(f"Released {len(released)} task(s):")
        for wid, count in by_worker.items():
            typer.echo(f"  {wid}: {count} task(s)")
    else:
        typer.echo("No processing tasks found")


async def _worker_status(db_url: str, schema: str = "public") -> list[dict[str, Any]]:
    """Get all processing tasks grouped by worker with their last update time."""
    _pg0 = parse_pg0_url(db_url)
    is_pg0, instance_name = _pg0.is_pg0, _pg0.instance_name
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    resolved_url = await resolve_database_url(db_url)

    conn = await asyncpg.connect(resolved_url)
    try:
        table = _fq_table("async_operations", schema)
        rows = await conn.fetch(
            f"""
            SELECT worker_id, operation_id, operation_type, bank_id,
                   claimed_at, updated_at,
                   now() - claimed_at AS running_for,
                   now() - updated_at AS last_update_ago
            FROM {table}
            WHERE status = 'processing'
            ORDER BY worker_id, claimed_at
            """,
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


@app.command(name="worker-status")
def worker_status(
    schema: str = typer.Option("public", "--schema", "-s", help="Database schema"),
):
    """Show all currently processing tasks grouped by worker.

    Displays each worker's active tasks with operation type, bank, how long
    the task has been running, and when it was last updated. Useful for
    identifying dead workers with orphaned tasks.
    """
    config = HindsightConfig.from_env()

    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set HINDSIGHT_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    rows = asyncio.run(_worker_status(config.database_url, schema))

    if not rows:
        typer.echo("No processing tasks found")
        return

    # Group by worker_id
    by_worker: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        wid = row["worker_id"] or "unknown"
        by_worker.setdefault(wid, []).append(row)

    typer.echo(f"Processing tasks across {len(by_worker)} worker(s):\n")
    for wid, tasks in by_worker.items():
        typer.echo(f"Worker: {wid} ({len(tasks)} task(s))")
        for task in tasks:
            op_id = str(task["operation_id"])[:8]
            running_for = task["running_for"]
            last_update = task["last_update_ago"]
            typer.echo(
                f"  {op_id}  {task['operation_type']:<20s} bank={task['bank_id']}"
                f"  running={running_for}  last_update={last_update} ago"
            )
        typer.echo("")


def main():
    load_dotenv_for_entrypoint()
    app()


if __name__ == "__main__":
    main()
