"""PostgreSQL-only admin utilities (backup, restore, migration, worker management).

Not supported on Oracle backends. Uses asyncpg.connect() directly, binary COPY,
TRUNCATE CASCADE, and REFRESH MATERIALIZED VIEW — all inherently PG-specific.
"""

import asyncio
import io
import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg
import typer

from ..config import DEFAULT_DATABASE_SCHEMA, HindsightConfig
from ..engine.memory_engine import _current_schema
from ..engine.schema import fq_table_explicit as _fq_table
from ..engine.transfer import export_bank
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

MANIFEST_VERSION = "1"


async def _admin_connect(db_url: str) -> asyncpg.Connection:
    """Open a raw asyncpg connection to an admin DB URL.

    ``resolve_database_url`` handles both plain ``postgres://`` (passthrough) and
    ``pg0://`` (boots the embedded server and returns its real libpq URL), so this
    is the only step needed to connect. JSON codecs are registered so ``jsonb``
    columns decode to Python objects (used by the export row dumps).
    """
    is_pg0, instance_name, _ = parse_pg0_url(db_url)
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    conn = await asyncpg.connect(await resolve_database_url(db_url))
    for type_name in ("json", "jsonb"):
        await conn.set_type_codec(type_name, encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
    return conn


async def _backup(database_url: str, output_path: Path, schema: str = "public") -> dict[str, Any]:
    """Backup all tables to a zip file using binary COPY protocol."""
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
                for i, table in enumerate(BACKUP_TABLES, 1):
                    typer.echo(f"  [{i}/{len(BACKUP_TABLES)}] Backing up {table}...", nl=False)

                    buffer = io.BytesIO()

                    # Use binary COPY for exact type preservation
                    # asyncpg requires schema_name as separate parameter
                    await conn.copy_from_table(table, schema_name=schema, output=buffer, format="binary")

                    data = buffer.getvalue()
                    zf.writestr(f"{table}.bin", data)

                    # Get row count for manifest
                    qualified_table = _fq_table(table, schema)
                    row_count = await conn.fetchval(f"SELECT COUNT(*) FROM {qualified_table}")
                    tables[table] = {
                        "rows": row_count,
                        "size_bytes": len(data),
                    }

                    typer.echo(f" {row_count} rows")

                zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        return manifest
    finally:
        await conn.close()


async def _restore(database_url: str, input_path: Path, schema: str = "public") -> dict[str, Any]:
    """Restore all tables from a zip file using binary COPY protocol."""
    conn = await asyncpg.connect(database_url)
    try:
        with zipfile.ZipFile(input_path, "r") as zf:
            # Read and validate manifest
            manifest: dict[str, Any] = json.loads(zf.read("manifest.json"))
            if manifest.get("version") != MANIFEST_VERSION:
                raise ValueError(f"Unsupported backup version: {manifest.get('version')}")

            # Use a transaction for atomic restore - either all tables are
            # restored or none are, preventing partial/inconsistent state.
            async with conn.transaction():
                typer.echo("  Clearing existing data...")
                # Truncate tables in reverse order (respects FK constraints)
                for table in reversed(BACKUP_TABLES):
                    qualified_table = _fq_table(table, schema)
                    await conn.execute(f"TRUNCATE TABLE {qualified_table} CASCADE")

                # Restore tables in forward order
                for i, table in enumerate(BACKUP_TABLES, 1):
                    filename = f"{table}.bin"
                    if filename not in zf.namelist():
                        typer.echo(f"  [{i}/{len(BACKUP_TABLES)}] {table}: skipped (not in backup)")
                        continue

                    expected_rows = manifest["tables"].get(table, {}).get("rows", "?")
                    typer.echo(f"  [{i}/{len(BACKUP_TABLES)}] Restoring {table}... {expected_rows} rows")

                    data = zf.read(filename)
                    buffer = io.BytesIO(data)
                    # asyncpg requires schema_name as separate parameter
                    await conn.copy_to_table(table, schema_name=schema, source=buffer, format="binary")

                # Refresh materialized view
                typer.echo("  Refreshing materialized views...")
                await conn.execute(f"REFRESH MATERIALIZED VIEW {_fq_table('memory_units_bm25', schema)}")

        return manifest
    finally:
        await conn.close()


async def _run_backup(db_url: str, output: Path, schema: str = "public") -> dict[str, Any]:
    """Resolve database URL and run backup."""
    is_pg0, instance_name, _ = parse_pg0_url(db_url)
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    resolved_url = await resolve_database_url(db_url)
    return await _backup(resolved_url, output, schema)


async def _run_restore(db_url: str, input_file: Path, schema: str = "public") -> dict[str, Any]:
    """Resolve database URL and run restore."""
    is_pg0, instance_name, _ = parse_pg0_url(db_url)
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    resolved_url = await resolve_database_url(db_url)
    return await _restore(resolved_url, input_file, schema)


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
    typer.echo(f"Backed up {total_rows} rows across {len(BACKUP_TABLES)} tables")
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
    typer.echo(f"Restored {total_rows} rows across {len(BACKUP_TABLES)} tables")
    typer.echo("Restore complete")


async def _run_migration(
    db_url: str,
    schema: str | None = None,
    base_schema: str = DEFAULT_DATABASE_SCHEMA,
    embedding_dimension: int | None = None,
) -> list[str]:
    """Resolve database URL and run migrations for one schema or all discovered schemas."""
    from ..migrations import run_migrations_for_schemas

    is_pg0, instance_name, _ = parse_pg0_url(db_url)
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
        ensure_extensions=True,
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

    schemas = asyncio.run(
        _run_migration(
            config.database_url,
            schema=schema,
            base_schema=config.database_schema,
            embedding_dimension=embedding_dimension,
        )
    )

    typer.echo(f"Database migrations completed successfully for {len(schemas)} schema(s)")


async def _run_export_bank(db_url: str, bank_id: str, output: Path, schema: str, include_history: bool) -> int:
    """Export a whole bank to a ZIP archive."""
    conn = await _admin_connect(db_url)
    try:
        # export_bank resolves table names via fq_table (the _current_schema
        # contextvar); set it so the raw connection targets the right schema.
        _current_schema.set(schema)
        data = await export_bank(conn, bank_id, include_history=include_history)
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
    is_pg0, instance_name, _ = parse_pg0_url(db_url)
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
    is_pg0, instance_name, _ = parse_pg0_url(db_url)
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
    is_pg0, instance_name, _ = parse_pg0_url(db_url)
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
