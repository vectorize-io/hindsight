"""
Hindsight Admin CLI - backup and restore operations.
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

from ..config import HindsightConfig
from ..pg0 import parse_pg0_url, resolve_database_url


def _fq_table(table: str, schema: str) -> str:
    """Get fully-qualified table name with schema prefix."""
    return f"{schema}.{table}"


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(name="hindsight-admin", help="Hindsight administrative commands")

# Tables to backup/restore in dependency order
# Import must happen in this order due to foreign key constraints
BACKUP_TABLES = [
    "banks",
    "documents",
    "entities",
    "chunks",
    "memory_units",
    "unit_entities",
    "entity_cooccurrences",
    "memory_links",
]

MANIFEST_VERSION = "1"


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


async def _run_migration(db_url: str, schema: str = "public") -> None:
    """Resolve database URL and run migrations."""
    from ..migrations import run_migrations

    is_pg0, instance_name, _ = parse_pg0_url(db_url)
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    resolved_url = await resolve_database_url(db_url)
    run_migrations(resolved_url, schema=schema)


@app.command(name="run-db-migration")
def run_db_migration(
    schema: str = typer.Option("public", "--schema", "-s", help="Database schema to run migrations on"),
):
    """Run database migrations to the latest version."""
    config = HindsightConfig.from_env()

    if not config.database_url:
        typer.echo("Error: Database URL not configured.", err=True)
        typer.echo("Set HINDSIGHT_API_DATABASE_URL environment variable.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Running database migrations (schema: {schema})...")

    asyncio.run(_run_migration(config.database_url, schema))

    typer.echo("Database migrations completed successfully")


def main():
    app()


if __name__ == "__main__":
    main()
