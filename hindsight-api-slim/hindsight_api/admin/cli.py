"""PostgreSQL-only admin utilities (backup, restore, migration, worker management).

Not supported on Oracle backends. Uses asyncpg.connect() directly, binary COPY,
TRUNCATE CASCADE, and REFRESH MATERIALIZED VIEW — all inherently PG-specific.
"""

import asyncio
import io
import json
import logging
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg
import typer

from ..config import DEFAULT_DATABASE_SCHEMA, HindsightConfig
from ..engine.memory_engine import _current_schema
from ..engine.retain.bank_utils import _vector_index_clause
from ..engine.schema import fq_table_explicit as _fq_table
from ..engine.transfer import export_bank
from ..engine.vector_index_health import SchemaVectorIndexResult, repair_vector_indexes
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
    "directives",
    "async_operations",
    "webhooks",
    "file_storage",
    "audit_log",
    "llm_requests",
    "graph_maintenance_queue",
]

MANIFEST_VERSION = "2"


@dataclass(frozen=True)
class BackupColumn:
    """A PostgreSQL column shape required to decode a binary COPY stream."""

    name: str
    type_name: str


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
) -> dict[str, list[str]]:
    """Validate every COPY stream against the target before destructive work starts.

    Type equality is an exact ``format_type`` string match. This is deliberately
    stricter than binary-COPY wire compatibility (e.g. ``varchar`` and ``text``
    share a binary format yet compare unequal here): we would rather fail a
    genuinely-restorable backup with a clear, actionable error than silently risk
    a subtle binary mismatch. Restores blocked this way can be recovered by
    aligning the target schema.
    """
    restore_columns: dict[str, list[str]] = {}
    errors: list[str] = []
    for table, table_manifest in manifest["tables"].items():
        source_columns = [BackupColumn(**column) for column in table_manifest["columns"]]
        target_by_name = {column.name: column for column in await _table_columns(conn, schema, table)}
        missing = [column.name for column in source_columns if column.name not in target_by_name]
        mismatched = [
            f"{column.name} ({column.type_name} in backup, {target_by_name[column.name].type_name} in target)"
            for column in source_columns
            if column.name in target_by_name and target_by_name[column.name].type_name != column.type_name
        ]
        if missing:
            errors.append(f"{table}: target is missing backup columns {', '.join(missing)}")
        if mismatched:
            errors.append(f"{table}: incompatible column types: {', '.join(mismatched)}")
        restore_columns[table] = [column.name for column in source_columns]

    if errors:
        details = "; ".join(errors)
        raise ValueError(f"Backup schema is incompatible with target schema '{schema}': {details}")
    return restore_columns


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
            restore_columns = await _validate_restore_schema(conn, manifest, schema)

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

                    data = zf.read(filename)
                    buffer = io.BytesIO(data)
                    # asyncpg requires schema_name as separate parameter
                    await conn.copy_to_table(
                        table,
                        schema_name=schema,
                        columns=restore_columns[table],
                        source=buffer,
                        format="binary",
                    )

                # Refresh materialized view
                typer.echo("  Refreshing materialized views...")
                await conn.execute(f"REFRESH MATERIALIZED VIEW {_fq_table('memory_units_bm25', schema)}")

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
    if schema:
        schemas = [schema]
    else:
        tenant_extension = load_extension("TENANT", TenantExtension)

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

    return schemas


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
) -> list[SchemaVectorIndexResult]:
    """Reconcile per-(bank, fact_type) vector index coverage over a raw connection.

    A single autocommit connection is used because ``CREATE INDEX CONCURRENTLY``
    (used by ``repair_vector_indexes``) cannot run inside a transaction block.
    """
    schemas = [schema] if schema else await _resolve_schemas(base_schema)
    index_clause = _vector_index_clause()
    # Guarded by the command, but assert so this helper is never called for a
    # backend without per-bank indexes.
    assert index_clause is not None

    conn = await _admin_connect(db_url)
    try:
        results = await repair_vector_indexes(conn, schemas, index_clause, dry_run=dry_run, bank_id=bank_id)
        for result in results:
            typer.echo(
                f"  schema '{result.schema}': {result.banks_scanned} bank(s) scanned, "
                f"{result.already_present} present, {result.created} created, "
                f"{result.skipped} to-create (dry-run), {result.failed} failed"
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
    """Verify and repair a bank's per-(bank, fact_type) vector index coverage.

    Per-bank partial vector indexes are created when a bank is first created
    (instant on an empty bank). Banks that arrive populated — via logical
    restore, a cross-version upgrade, or a vector-extension switch — never hit
    that path, so their recall silently falls back to a global index +
    post-filter (slower, under-returning). This command detects missing OR
    invalid coverage (an INVALID leftover or an index whose access method
    drifted counts as missing) and rebuilds it with CREATE INDEX CONCURRENTLY,
    so it never blocks the live fleet. Idempotent and safe to re-run — the
    escape hatch after a restore, upgrade, or backend switch.
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

    total_banks = sum(r.banks_scanned for r in results)
    total_present = sum(r.already_present for r in results)
    total_created = sum(r.created for r in results)
    total_skipped = sum(r.skipped for r in results)
    total_failed = sum(r.failed for r in results)
    typer.echo(
        f"Done: {len(results)} schema(s), {total_banks} bank(s) scanned, "
        f"{total_present} already present, {total_created} created, "
        f"{total_skipped} to-create (dry-run), {total_failed} failed"
    )
    if total_failed:
        failed_names = [name for r in results for name in r.failed_indexes]
        typer.echo(f"Failed indexes (dropped, retry with a re-run): {', '.join(failed_names)}", err=True)
        raise typer.Exit(1)


async def _run_export_bank(db_url: str, bank_id: str, output: Path, schema: str, include_history: bool) -> int:
    """Export a whole bank to a ZIP archive."""
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
        f"{result.mental_model_history_imported} mm-history row(s), {result.directives_imported} directive(s), "
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
    app()


if __name__ == "__main__":
    main()
