"""PostgreSQL-only admin utilities (backup, restore, migration, worker management).

Not supported on Oracle backends. Uses asyncpg.connect() directly, binary COPY,
and TRUNCATE CASCADE — all inherently PG-specific.
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
from ..db_url import is_oracle_url
from ..engine.memories import get_memories
from ..engine.memory_engine import _current_schema
from ..engine.retain.bank_utils import _vector_index_clause, bank_indexes_are_store_owned
from ..engine.schema import fq_table_explicit as _fq_table
from ..engine.storage import bank_storage_prefix, create_file_storage
from ..engine.transfer import TransferScope, export_bank
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
    # After "banks" for the same reason as "attachments" below: it FKs to it.
    "bank_aliases",
    # After "banks": attachments references it, so restore's forward COPY needs
    # the parent present, and the reversed TRUNCATE must clear the child first.
    "attachments",
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
    from ..engine.memories import get_memories
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
        store_owned_memories=get_memories().store_owned,
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


@dataclass
class RepairSweep:
    """What one ``repair-bank`` run did, and which schemas it could not do at all.

    A dataclass rather than a second return value: the bank list alone cannot
    distinguish "nothing needed doing" from "never got there", and the command must
    exit non-zero for the second.

    ``skipped_schemas`` means "not fully reconciled — re-run once the cause is
    cleared". A schema that failed partway appears in BOTH lists: the banks it did
    reconcile are real work that must still be reported (their failed-index names
    appear nowhere else), and the ones it did not still need the re-run. A schema
    whose banks were all repaired never appears here, even if the orphan sweep after
    them failed; that is reported on its own line and left out, because telling an
    operator to re-run a sweep that already did its work is how a report stops being
    believed.
    """

    banks: list[BankIndexResult]
    skipped_schemas: list[str]


async def _run_repair_bank(
    db_url: str,
    *,
    base_schema: str,
    schema: str | None,
    bank_id: str | None,
    dry_run: bool,
) -> RepairSweep:
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
    skipped_schemas: list[str] = []
    try:
        for target_schema in schemas:
            # The whole schema is inside the guard, not just list_bank_ids. Planning a
            # bank can now raise — a memories store that cannot say who owns a bank
            # must not be guessed at, or a transient blip would have the sweep rebuild
            # every index it failed on (#4615). That is worth failing on, but per
            # SCHEMA: an unreachable store used to take the whole command down with a
            # bare traceback from inside a list comprehension, so a deployment whose
            # admin process cannot reach the store repaired nothing at all, including
            # the ordinary SQL-owned banks the operator ran this for.
            # Appended as they complete, not built as a comprehension: a comprehension
            # binds nothing until it finishes, so a blip partway through discarded every
            # reconcile that had already run. See RepairSweep for what that means for
            # the two lists.
            bank_ids: list[str] = []
            schema_results: list[BankIndexResult] = []
            try:
                bank_ids = [bank_id] if bank_id else await list_bank_ids(conn, target_schema)
                for bid in bank_ids:
                    schema_results.append(
                        await reconcile_bank_vector_indexes(conn, target_schema, bid, index_clause, dry_run=dry_run)
                    )
            except Exception as exc:  # noqa: BLE001 — one bad schema must not abort the sweep
                results.extend(schema_results)
                done = f", {len(schema_results)} of {len(bank_ids)} bank(s) done" if schema_results else ""
                typer.echo(f"  schema '{target_schema}': skipped ({exc}){done}", err=True)
                skipped_schemas.append(target_schema)
                if isinstance(exc, asyncpg.PostgresConnectionError | asyncpg.InterfaceError):
                    # The sweep holds ONE connection for every schema, so this is not a
                    # property of the schema it happened on — every remaining schema
                    # fails the same way. Mark them unlooked-at and stop rather than
                    # print the same error once per tenant. Stop, not re-raise: raising
                    # reaches repair_bank's asyncio.run with no handler and discards
                    # every report, which is what the reporting split exists to avoid.
                    remaining = schemas[schemas.index(target_schema) + 1 :]
                    if remaining:
                        typer.echo(
                            f"  stopping: the shared connection is gone, so "
                            f"{len(remaining)} further schema(s) were not attempted.",
                            err=True,
                        )
                    skipped_schemas.extend(remaining)
                    break
                continue
            results.extend(schema_results)
            # Only in --all mode: an index whose bank row is gone is unreachable
            # from every bank-scoped path, so this is the one place that can collect
            # it. Normally finds nothing — delete_bank drops a bank's indexes while it
            # still knows their names — but a deployment that hit the #3485 wall could
            # not run delete_bank at all.
            #
            # Best-effort, and deliberately NOT a "skipped schema": the banks here were
            # reconciled, and reporting the schema as skipped would tell the operator
            # to re-run a sweep that already did its work. Per-index drop failures are
            # handled inside drop_orphaned_bank_indexes; this catches only the catalog
            # read behind it.
            orphans: list[str] = []
            if not bank_id:
                try:
                    orphans = await drop_orphaned_bank_indexes(conn, target_schema, dry_run=dry_run)
                except Exception as exc:  # noqa: BLE001 — the banks were repaired; orphan collection is a nicety
                    typer.echo(
                        f"  schema '{target_schema}': banks repaired, but orphan collection failed ({exc})", err=True
                    )
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
        return RepairSweep(banks=results, skipped_schemas=skipped_schemas)
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

    A bank whose memories a custom store owns is outside all of this, decided per
    bank rather than per deployment: it has no rows in memory_units, so nothing is
    built for it and nothing it already carries is taken away (#4615). Such a bank
    reports 0 present, 0 created either way; query pg_indexes to see what one still
    holds. If the store cannot say who owns a bank, that schema is reported skipped
    and this exits non-zero rather than guessing and rebuilding.

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

    sweep = asyncio.run(
        _run_repair_bank(
            config.database_url,
            base_schema=config.database_schema,
            schema=schema,
            bank_id=bank_id,
            dry_run=dry_run,
        )
    )
    results = sweep.banks

    total_banks = len(results)
    total_present = sum(r.already_present for r in results)
    total_created = sum(r.created for r in results)
    total_dropped = sum(r.dropped for r in results)
    total_skipped = sum(r.skipped for r in results)
    total_would_drop = sum(r.would_drop for r in results)
    total_failed = sum(r.failed for r in results)
    typer.echo(
        f"Done: {total_banks} bank(s) scanned, "
        f"{total_present} already present, {total_created} created, {total_dropped} dropped, "
        f"{total_skipped} to-create (dry-run), {total_would_drop} to-drop (dry-run), "
        f"{total_failed} failed"
    )
    # Both are reported before either exits. They are independent — a sweep can have
    # failed builds in one schema and be unable to look at another — and raising on
    # the first swallowed the index names, which appear nowhere else and are the whole
    # point of the failed line.
    if total_failed:
        failed_names = [name for r in results for name in r.failed_indexes]
        typer.echo(f"Failed indexes (dropped, retry with a re-run): {', '.join(failed_names)}", err=True)
    if sweep.skipped_schemas:
        # Some or all of a skipped schema's banks were not reconciled, so a run that
        # reported only successes would read as converged while whole tenants — or the
        # tail of one — still carry whatever they carried.
        typer.echo(
            f"Skipped {len(sweep.skipped_schemas)} schema(s): "
            f"{', '.join(sweep.skipped_schemas)}. Re-run once the cause is cleared.",
            err=True,
        )
    if total_failed or sweep.skipped_schemas:
        raise typer.Exit(1)


class RenameBankError(Exception):
    """A rename-bank precondition failed; nothing was changed."""


# FKs whose own columns include bank_id and that cannot be deferred as declared.
_RIGID_BANK_ID_FKS_SQL = """
    SELECT c.conrelid::regclass::text AS tbl, c.conname
    FROM pg_constraint c
    JOIN pg_namespace n ON n.oid = c.connamespace
    WHERE c.contype = 'f' AND n.nspname = $1 AND NOT c.condeferrable
      AND EXISTS (
          SELECT 1 FROM pg_attribute a
          WHERE a.attrelid = c.conrelid AND a.attnum = ANY (c.conkey) AND a.attname = 'bank_id'
      )
"""


async def _set_fks_deferrable(conn: asyncpg.Connection, fks: list[asyncpg.Record], clause: str) -> None:
    """Flip the given FKs' deferrability in one short transaction.

    ``ALTER CONSTRAINT`` only touches the catalog, but it still takes a table lock
    and queues behind any open transaction on the table — and everything else
    queues behind it. ``lock_timeout`` turns a long wait into a loud failure
    instead of a tenant-wide stall.
    """
    async with conn.transaction():
        await conn.execute("SET LOCAL lock_timeout = '5s'")
        for fk in fks:
            await conn.execute(f"ALTER TABLE {fk['tbl']} ALTER CONSTRAINT {_quote_identifier(fk['conname'])} {clause}")


async def _rename_bank(
    conn: asyncpg.Connection, schema: str, old_bank_id: str, new_bank_id: str, *, dry_run: bool
) -> dict[str, int]:
    """Move every row of ``old_bank_id`` to ``new_bank_id`` in one transaction.

    ``bank_id`` is the key every bank-scoped table carries, several composite FKs
    included (``documents(id, bank_id)``, ``mental_models(id, bank_id)``), so no
    update order satisfies an immediate FK check. The FKs are made DEFERRABLE
    just for the rename and put back afterwards, each flip in its own short
    transaction — doing it inside the rename would hold those table locks, and
    block every bank in the schema, for its whole duration. This is runtime DDL
    rather than a migration on purpose: rename is a rare admin operation, and the
    schema stays exactly as the migrations declare it.

    Only FKs this call flipped are put back. If that restore cannot happen (the
    process dies, or the restore times out on a lock) they stay DEFERRABLE
    INITIALLY IMMEDIATE, which checks every normal write exactly as before; a
    later rename will not restore them, since it only flips what it finds
    NOT DEFERRABLE, so the failure is reported with the constraint names.

    Returns the rows moved per table (tables the bank had no rows in are omitted).
    """
    rigid = await conn.fetch(_RIGID_BANK_ID_FKS_SQL, schema)
    try:
        await _set_fks_deferrable(conn, rigid, "DEFERRABLE INITIALLY IMMEDIATE")
    except asyncpg.exceptions.LockNotAvailableError as exc:
        raise RenameBankError(
            "could not lock the bank tables within 5s to prepare the rename; retry when fewer long transactions run"
        ) from exc
    try:
        return await _move_bank_rows(conn, schema, old_bank_id, new_bank_id, dry_run=dry_run)
    finally:
        # Never raise from here: it would replace the rename's own outcome (success
        # or its real error) with a failure that leaves every write checked as before.
        try:
            await _set_fks_deferrable(conn, rigid, "NOT DEFERRABLE")
        except Exception as exc:  # noqa: BLE001
            names = ", ".join(f"{fk['tbl']}.{fk['conname']}" for fk in rigid)
            typer.echo(
                f"Warning: could not restore NOT DEFERRABLE on {names} ({exc}). They still enforce every write "
                "as before; restore with ALTER TABLE ... ALTER CONSTRAINT ... NOT DEFERRABLE.",
                err=True,
            )


async def _move_bank_rows(
    conn: asyncpg.Connection, schema: str, old_bank_id: str, new_bank_id: str, *, dry_run: bool
) -> dict[str, int]:
    """The rename transaction proper; the bank_id FKs must already be DEFERRABLE.

    Deferring them lets the tables move in any order, and forcing them back to
    IMMEDIATE before the commit — or before the dry run's rollback — makes a
    missed table fail the whole rename instead of stranding rows. Tables are read
    from the catalog, so extension tables and tables added later are covered
    without a list to maintain.

    The ``FOR UPDATE`` on the bank row serialises the rename against writers:
    anything inserting a FK child of the bank takes a key-share lock on that row,
    so it either commits before the rename reads, or waits and then fails its FK
    check rather than writing under an id that no longer exists.
    """
    banks = _fq_table("banks", schema)
    tx = conn.transaction()
    await tx.start()
    try:
        if not await conn.fetchval(f"SELECT 1 FROM {banks} WHERE bank_id = $1 FOR UPDATE", old_bank_id):
            raise RenameBankError(f"bank '{old_bank_id}' does not exist in schema '{schema}'")
        if await conn.fetchval(f"SELECT 1 FROM {banks} WHERE bank_id = $1", new_bank_id):
            raise RenameBankError(f"bank '{new_bank_id}' already exists in schema '{schema}'")
        # An operation's task_payload names its bank, so one run after the rename
        # would work on (and could recreate) the old id.
        active = await conn.fetchval(
            f"SELECT count(*) FROM {_fq_table('async_operations', schema)} "
            "WHERE bank_id = $1 AND status IN ('pending', 'processing')",
            old_bank_id,
        )
        if active:
            raise RenameBankError(
                f"bank '{old_bank_id}' has {active} pending or processing operation(s); "
                "wait for them to finish or cancel them, then retry"
            )
        tables = await conn.fetch(
            """
            SELECT c.table_name
            FROM information_schema.columns c
            JOIN information_schema.tables t
              ON t.table_schema = c.table_schema AND t.table_name = c.table_name
            WHERE c.table_schema = $1 AND c.column_name = 'bank_id' AND t.table_type = 'BASE TABLE'
            ORDER BY c.table_name
            """,
            schema,
        )
        await conn.execute("SET CONSTRAINTS ALL DEFERRED")
        moved: dict[str, int] = {}
        for row in tables:
            status = await conn.execute(
                f"UPDATE {_fq_table(row['table_name'], schema)} SET bank_id = $1 WHERE bank_id = $2",
                new_bank_id,
                old_bank_id,
            )
            count = int(status.split()[-1])
            if count:
                moved[row["table_name"]] = count
        await conn.execute("SET CONSTRAINTS ALL IMMEDIATE")
    except BaseException:
        await tx.rollback()
        raise
    if dry_run:
        await tx.rollback()
    else:
        await tx.commit()
    return moved


async def _move_bank_files(conn: asyncpg.Connection, db_url: str, schema: str, old_id: str, new_id: str) -> int:
    """Re-key the bank's stored files, now that its rows carry the new id.

    A file's key spells the bank id (``bank_storage_prefix``), and no column move
    reaches inside a key: left alone, the bank's bytes stay under the old prefix,
    where neither the download route nor ``delete_bank``'s sweep — both of which
    look under the bank's *current* prefix — can reach them. The sweep missing
    them is the leak (#4502): on S3 or GCS the bank goes and its bytes stay, still
    billed.

    Copy first, repoint the row, and only then drop the old prefix, so an
    interrupted move leaves every row pointing at bytes that exist.

    ponytail: copies through this process, one object at a time. A server-side
    copy (S3 CopyObject) or, on the native backend, an UPDATE of file_storage
    would move bytes without reading them — worth it if renames get big or common.
    """
    old_prefix = bank_storage_prefix(old_id, schema)
    new_prefix = bank_storage_prefix(new_id, schema)
    # Only the native backend needs a pool; building one unconditionally keeps
    # this from branching on the storage type. Resolved like _admin_connect does,
    # so a pg0:// URL reaches the embedded server it already started.
    pool = await asyncpg.create_pool(await resolve_database_url(db_url), min_size=1, max_size=2)
    try:
        storage = create_file_storage(
            storage_type=HindsightConfig.from_env().file_storage_type,
            pool_getter=lambda: pool,
            schema=schema,
        )
        # starts_with, not LIKE: key segments are percent-encoded, so a prefix can
        # contain '%' and would read as a wildcard. Keys written before the tenant
        # layout sit outside the prefix and stay where they are: delete_bank sweeps
        # those from their rows, which the rename carries to the new id.
        rows = await conn.fetch(
            f"SELECT 'attachments' AS table_name, storage_key AS key FROM {_fq_table('attachments', schema)} "
            f"WHERE bank_id = $1 AND starts_with(storage_key, $2) "
            f"UNION ALL "
            f"SELECT 'documents', file_storage_key FROM {_fq_table('documents', schema)} "
            f"WHERE bank_id = $1 AND file_storage_key IS NOT NULL AND starts_with(file_storage_key, $2)",
            new_id,
            old_prefix,
        )
        columns = {"attachments": "storage_key", "documents": "file_storage_key"}
        moved = 0
        for row in rows:
            new_key = new_prefix + row["key"][len(old_prefix) :]
            try:
                data = await storage.retrieve(row["key"])
            except FileNotFoundError:
                # The row outlived its bytes (an earlier leak swept, a manual
                # cleanup). Leave it pointing where it does rather than at a key
                # with nothing behind it, and carry on: the rename is committed.
                typer.echo(f"Warning: {row['key']} has no stored bytes; its row keeps the old key.")
                continue
            await storage.store(file_data=data, key=new_key)
            column = columns[row["table_name"]]
            await conn.execute(
                f"UPDATE {_fq_table(row['table_name'], schema)} SET {column} = $1 WHERE bank_id = $2 AND {column} = $3",
                new_key,
                new_id,
                row["key"],
            )
            moved += 1
        # The originals, plus whatever else the bank left under the old prefix:
        # export and import archives, which no row names and which their
        # operation can produce again.
        try:
            await storage.delete_prefix(old_prefix)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"Warning: could not clear the old prefix {old_prefix} ({exc}); its files are now orphans.")
        return moved
    finally:
        await pool.close()


async def _run_rename_bank(
    db_url: str, schema: str, old_bank_id: str, new_bank_id: str, *, dry_run: bool
) -> dict[str, int]:
    """Rename the bank, then move its stored files and rebuild its vector indexes.

    Those indexes are partial on a ``bank_id`` literal, so after the rename they
    cover nothing; the reconcile sees the stale predicate and rebuilds them
    CONCURRENTLY, which is why it runs after the commit, outside the transaction.
    Until it finishes, recall on the bank runs without its index.
    """
    # Same unreachable-store case the reconcile below now handles, reported the same
    # way: this runs before anything is changed, so refusing is free and a traceback
    # would just look like a crash.
    try:
        old_is_store_owned = bank_indexes_are_store_owned(old_bank_id)
    except Exception as exc:  # noqa: BLE001 — cannot verify the precondition, so do not proceed
        raise RenameBankError(
            f"cannot tell whether bank '{old_bank_id}' keeps its memories outside SQL ({exc}); "
            f"nothing was changed. Re-run once the memories store is reachable."
        ) from exc
    if old_is_store_owned:
        raise RenameBankError(f"bank '{old_bank_id}' keeps its memories outside SQL; rename is not supported for it")
    conn = await _admin_connect(db_url)
    try:
        moved = await _rename_bank(conn, schema, old_bank_id, new_bank_id, dry_run=dry_run)
        if not dry_run:
            files = await _move_bank_files(conn, db_url, schema, old_bank_id, new_bank_id)
            typer.echo(f"Stored files: {files} re-keyed under the new bank id")
        index_clause = _vector_index_clause()
        if not dry_run and index_clause is not None:
            # The rename is already committed here, so a failure in the rebuild must
            # not surface as a bare traceback that buries that fact. Planning can now
            # raise (a memories store that cannot say who owns the new id is not
            # guessed at — see bank_indexes_are_store_owned), and the honest report is
            # "the rename worked, the index did not".
            try:
                result = await reconcile_bank_vector_indexes(conn, schema, new_bank_id, index_clause)
            except Exception as exc:  # noqa: BLE001 — the rename is committed; say so rather than traceback
                typer.echo(
                    f"Vector indexes: could not be rebuilt ({exc}). The rename itself succeeded. "
                    f"Re-run `hindsight-admin repair-bank --bank {new_bank_id}` once the cause is cleared; "
                    f"until then recall on this bank runs without its index.",
                    err=True,
                )
                return moved
            typer.echo(f"Vector indexes: {result.created} rebuilt, {result.dropped} dropped, {result.failed} failed")
            if result.failed:
                typer.echo(f"Re-run `hindsight-admin repair-bank --bank {new_bank_id}` to retry.", err=True)
        return moved
    finally:
        await conn.close()


@app.command(name="rename-bank")
def rename_bank(
    old_bank_id: str = typer.Option(..., "--from", help="Current bank id."),
    new_bank_id: str = typer.Option(..., "--to", help="New bank id. Must not exist yet."),
    schema: str | None = typer.Option(
        None,
        "--schema",
        "-s",
        help="Database schema the bank lives in. Defaults to the configured base schema.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run the whole rename, report what would move, then roll back.",
    ),
) -> None:
    """Rename a bank's id in place, keeping every memory, document and mental model.

    One transaction moves every row of the bank to the new id. Stop the bank's
    clients first: anything still calling the old id gets a 404 or, if it creates
    banks on demand, a new empty bank under the old id. Refused while the bank
    has pending or processing operations.
    """
    if not new_bank_id.strip() or new_bank_id == old_bank_id:
        typer.echo("Error: --to must be a non-empty id different from --from.", err=True)
        raise typer.Exit(2)

    config = HindsightConfig.from_env()
    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set HINDSIGHT_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    # The rename flips FK deferrability with PostgreSQL's ALTER CONSTRAINT and runs
    # over asyncpg; Oracle can only change deferrability by recreating the
    # constraint. Refuse before connecting rather than fail halfway.
    if is_oracle_url(config.database_url):
        typer.echo("Error: rename-bank is PostgreSQL-only; Oracle backends are not supported.", err=True)
        raise typer.Exit(1)

    target_schema = schema or config.database_schema or DEFAULT_DATABASE_SCHEMA
    typer.echo(f"Renaming bank '{old_bank_id}' -> '{new_bank_id}' in schema '{target_schema}'...")
    try:
        moved = asyncio.run(
            _run_rename_bank(config.database_url, target_schema, old_bank_id, new_bank_id, dry_run=dry_run)
        )
    except RenameBankError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    for table, count in sorted(moved.items()):
        typer.echo(f"  {table}: {count}")
    if dry_run:
        typer.echo(f"Dry run: {sum(moved.values())} row(s) would move; rolled back.")
    else:
        typer.echo(
            f"Renamed: {sum(moved.values())} row(s) moved. Point clients and API keys at '{new_bank_id}'; "
            "API servers may serve the old id from cache until bank_info_cache_ttl_seconds elapses."
        )


class _SingleConnectionPool:
    """Pool adapter over the admin CLI's one raw connection.

    File storage takes a pool because the API serves many requests from one; the
    CLI has a single connection and a single export running on it, so acquiring
    hands the same connection back. Only the native (PostgreSQL) backend touches
    this at all — an S3/GCS/Azure deployment never calls the pool.
    """

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def acquire(self) -> asyncpg.Connection:
        return self._conn

    async def release(self, conn: asyncpg.Connection) -> None:
        """No-op: the caller owns the connection's lifetime, and closes it."""


def _admin_file_storage(conn: asyncpg.Connection, schema: str) -> Any:
    """File storage for an export run from the CLI, on this instance's config.

    Attachment bytes live here rather than in a column, so a bank with
    attachments cannot be exported without it.
    """
    from ..engine.storage import create_file_storage

    config = HindsightConfig.from_env()
    return create_file_storage(
        storage_type=config.file_storage_type,
        pool_getter=lambda: _SingleConnectionPool(conn),
        schema=schema,
    )


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
            scope=TransferScope(data=True, bank_config=True, history=include_history),
            bank_rows_json_encoding="decoded",
            memories=get_memories(),
            file_storage=_admin_file_storage(conn, schema),
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
            scope=TransferScope(data=True, bank_config=True, history=include_history),
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
