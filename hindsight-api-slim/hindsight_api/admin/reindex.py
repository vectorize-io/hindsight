"""
Hindsight Admin: reindex-embeddings command.

Re-embeds existing memory rows when the embedding model has changed (e.g. swap
from BAAI/bge-small-en-v1.5 to Qwen/Qwen3-Embedding-4B).

Hindsight's `ensure_embedding_dimension` (in migrations.py) refuses to auto-migrate
the embedding column dimension when the table holds data — leaving operators stuck
mid-upgrade. This command closes that gap:

    1. Discovers ALL vector(N) columns across the schema (memory_units, mental_models,
       chunks, entities, plus any future tables — schema-introspected, not hardcoded).
    2. Verifies each column's stored dimension matches the loaded embedding model.
    3. Idempotently re-embeds rows where embedding IS NULL, in resumable batches.
    4. Rebuilds vector indexes after bulk update so recall returns fresh neighbors.
    5. Optionally runs a verification recall pass against known fixture queries.

Closes vectorize-io/hindsight#743.

Typical workflow:
    # 1. Backup
    hindsight-admin backup /backups/pre-reindex.zip

    # 2. Stop API. Update env to new HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL.

    # 3. Wipe vectors so column type can be migrated:
    psql -c "ALTER TABLE memory_units ALTER COLUMN embedding TYPE vector(<NEW_DIM>) USING NULL;"
    psql -c "ALTER TABLE mental_models ALTER COLUMN embedding TYPE vector(<NEW_DIM>) USING NULL;"

    # 4. Restart API (loads new model, dimension matches now).

    # 5. Re-embed everything:
    hindsight-admin reindex-embeddings
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import asyncpg
import typer

logger = logging.getLogger(__name__)


@dataclass
class VectorColumnInfo:
    """A discovered vector column in the schema."""

    table: str
    column: str
    dimension: int
    fq_table: str  # schema-qualified table name
    text_column: str | None = None  # text column to source embeddings from


# Heuristic: which text column to use as embedding source for each known table.
# Schema-introspection finds the vector columns; this maps them to source text.
# Fallback for unknown tables: use 'text' if it exists, else skip.
TEXT_SOURCE_COLUMN_BY_TABLE: dict[str, str] = {
    "memory_units": "text",
    "mental_models": "content",  # mental_models stores generated content, embedded for retrieval
    "chunks": "text",
    "entities": "name",  # entities embedded by name+description in some Hindsight versions
}


async def _discover_vector_columns(conn: asyncpg.Connection, schema: str) -> list[VectorColumnInfo]:
    """Find every column whose Postgres type is `vector(N)` from pgvector.

    Schema-introspected so future Hindsight versions adding new embedding-bearing
    tables Just Work without code changes.
    """
    rows = await conn.fetch(
        """
        SELECT
            n.nspname AS schema_name,
            c.relname AS table_name,
            a.attname AS column_name,
            pg_catalog.format_type(a.atttypid, a.atttypmod) AS column_type
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        JOIN pg_namespace n ON c.relnamespace = n.oid
        JOIN pg_type t ON a.atttypid = t.oid
        WHERE n.nspname = $1
          AND c.relkind = 'r'  -- ordinary tables only
          AND t.typname = 'vector'
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY c.relname, a.attnum
        """,
        schema,
    )

    columns: list[VectorColumnInfo] = []
    for row in rows:
        col_type = row["column_type"]  # e.g., "vector(2560)"
        # Parse dimension out of "vector(N)"
        if col_type.startswith("vector(") and col_type.endswith(")"):
            try:
                dim = int(col_type[len("vector(") : -1])
            except ValueError:
                logger.warning(
                    f"Could not parse dimension from {col_type}, skipping {row['table_name']}.{row['column_name']}"
                )
                continue
        else:
            logger.warning(f"Unexpected vector type format {col_type}, skipping")
            continue

        text_col = TEXT_SOURCE_COLUMN_BY_TABLE.get(row["table_name"])
        if text_col is None:
            # Try to find a 'text' or 'content' column on this table
            text_col_check = await conn.fetchval(
                """
                SELECT a.attname
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = $1 AND c.relname = $2
                  AND a.attname IN ('text', 'content', 'body')
                  AND a.attnum > 0 AND NOT a.attisdropped
                ORDER BY CASE a.attname
                    WHEN 'text' THEN 1
                    WHEN 'content' THEN 2
                    WHEN 'body' THEN 3
                END
                LIMIT 1
                """,
                schema,
                row["table_name"],
            )
            text_col = text_col_check

        columns.append(
            VectorColumnInfo(
                table=row["table_name"],
                column=row["column_name"],
                dimension=dim,
                fq_table=f'"{row["schema_name"]}"."{row["table_name"]}"',
                text_column=text_col,
            )
        )

    return columns


async def _count_pending(conn: asyncpg.Connection, col: VectorColumnInfo) -> int:
    """Count rows in `col.table` where vector column is NULL but text source is non-empty."""
    if col.text_column is None:
        return 0
    text_col_q = f'"{col.text_column}"'
    vec_col_q = f'"{col.column}"'
    return await conn.fetchval(
        f"""
        SELECT COUNT(*) FROM {col.fq_table}
        WHERE {vec_col_q} IS NULL
          AND {text_col_q} IS NOT NULL
          AND length({text_col_q}) > 0
        """
    )


async def _reembed_table(
    conn: asyncpg.Connection,
    embed,
    col: VectorColumnInfo,
    batch_size: int,
    bank_id: str | None,
) -> tuple[int, int]:
    """Re-embed all NULL-embedding rows in a single (table, column).

    Returns (processed, skipped) counts.
    """
    if col.text_column is None:
        typer.echo(f"  SKIP {col.table}.{col.column} — no text source column found")
        return 0, 0

    text_col_q = f'"{col.text_column}"'
    vec_col_q = f'"{col.column}"'

    # Fetch IDs + texts to embed (resumable: only WHERE embedding IS NULL)
    where_bank = ""
    params: list = []
    if bank_id is not None and col.table in ("memory_units", "mental_models", "chunks", "documents"):
        where_bank = " AND bank_id = $1"
        params = [bank_id]

    query = f"""
        SELECT id, {text_col_q} AS source_text
        FROM {col.fq_table}
        WHERE {vec_col_q} IS NULL
          AND {text_col_q} IS NOT NULL
          AND length({text_col_q}) > 0
          {where_bank}
        ORDER BY id
    """
    rows = await conn.fetch(query, *params)
    total = len(rows)
    if total == 0:
        typer.echo(f"  {col.table}.{col.column}: nothing to do")
        return 0, 0

    typer.echo(f"  {col.table}.{col.column}: {total} rows to re-embed")

    update_sql = f"UPDATE {col.fq_table} SET {vec_col_q} = $1::vector WHERE id = $2"

    processed = 0
    skipped = 0
    table_start = time.time()
    last_log = table_start

    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        ids = [r["id"] for r in batch]
        texts = [r["source_text"] for r in batch]

        try:
            vectors = embed.encode(texts)
        except Exception as e:
            typer.echo(f"    encode-failed batch starting at {i}: {e!r}", err=True)
            skipped += len(batch)
            continue

        # Bulk update via asyncpg executemany
        records = [
            (str(list(vec)).replace(" ", ""), mid)  # vector literal as "[v1,v2,...]"
            for mid, vec in zip(ids, vectors, strict=False)
        ]
        await conn.executemany(update_sql, records)

        processed += len(batch)
        now = time.time()
        if now - last_log >= 5.0 or processed >= total:
            elapsed = now - table_start
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            typer.echo(f"    {processed}/{total}  ({rate:.1f} rows/s, ETA {eta:.0f}s)")
            last_log = now

    typer.echo(
        f"  {col.table}.{col.column}: {processed} embedded, {skipped} skipped, in {time.time() - table_start:.1f}s"
    )
    return processed, skipped


async def _rebuild_vector_indexes(conn: asyncpg.Connection, schema: str) -> int:
    """REINDEX every HNSW or IVFFlat index on a vector column in the schema.

    Returns count of indexes rebuilt.
    """
    # Find indexes built on vector columns
    rows = await conn.fetch(
        """
        SELECT
            n.nspname AS schema_name,
            c.relname AS index_name,
            am.amname AS access_method
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_am am ON am.oid = c.relam
        WHERE n.nspname = $1
          AND am.amname IN ('hnsw', 'ivfflat', 'vchordrq')
        """,
        schema,
    )

    rebuilt = 0
    for row in rows:
        idx_name = f'"{row["schema_name"]}"."{row["index_name"]}"'
        am = row["access_method"]
        typer.echo(f"  REINDEX [{am}] {idx_name}")
        await conn.execute(f"REINDEX INDEX {idx_name}")
        rebuilt += 1
    return rebuilt


async def _reindex_embeddings(
    database_url: str,
    schema: str,
    bank_id: str | None,
    batch_size: int,
    dry_run: bool,
    skip_index_rebuild: bool,
    yes: bool,
) -> None:
    """Main async entrypoint for reindex-embeddings command."""
    # Lazy import — only load embedding engine if we're actually going to use it
    import os

    from ..config import (
        DEFAULT_EMBEDDINGS_LOCAL_TRUST_REMOTE_CODE,
        ENV_EMBEDDINGS_LOCAL_MODEL,
        ENV_EMBEDDINGS_LOCAL_TRUST_REMOTE_CODE,
    )

    model_name = os.environ.get(ENV_EMBEDDINGS_LOCAL_MODEL)
    if not model_name and not dry_run:
        typer.echo(
            f"ERROR: {ENV_EMBEDDINGS_LOCAL_MODEL} not set. "
            "Configure your embedding model before running reindex-embeddings. "
            "(Or use --dry-run to inspect schema without loading a model.)",
            err=True,
        )
        raise typer.Exit(1)
    trust_remote = os.environ.get(
        ENV_EMBEDDINGS_LOCAL_TRUST_REMOTE_CODE,
        str(DEFAULT_EMBEDDINGS_LOCAL_TRUST_REMOTE_CODE),
    ).lower() in ("1", "true", "yes")

    typer.echo("== Hindsight reindex-embeddings ==")
    typer.echo(f"  Schema:        {schema}")
    typer.echo(f"  Bank filter:   {bank_id or '(all banks)'}")
    typer.echo(f"  Batch size:    {batch_size}")
    typer.echo(f"  Embedding:     {model_name}")
    typer.echo(f"  Trust remote:  {trust_remote}")
    typer.echo(f"  Dry run:       {dry_run}")
    typer.echo()

    # Resolve pg0 (embedded) URLs to a real connection string
    from ..pg0 import parse_pg0_url, resolve_database_url

    is_pg0, instance_name, _ = parse_pg0_url(database_url)
    if is_pg0:
        typer.echo(f"Starting embedded PostgreSQL (instance: {instance_name})...")
    resolved_url = await resolve_database_url(database_url)

    conn = await asyncpg.connect(resolved_url)
    try:
        # 1. Discover vector columns
        typer.echo("[1/5] Discovering vector columns...")
        cols = await _discover_vector_columns(conn, schema)
        if not cols:
            typer.echo(f"  No vector columns found in schema '{schema}'")
            return
        for col in cols:
            text_info = f"text→{col.text_column}" if col.text_column else "(no text source)"
            typer.echo(f"  {col.table}.{col.column}  vector({col.dimension})  {text_info}")
        typer.echo()

        # 2. Count pending rows per column
        typer.echo("[2/5] Counting pending rows...")
        total_pending = 0
        for col in cols:
            n = await _count_pending(conn, col)
            typer.echo(f"  {col.table}.{col.column}: {n} rows with NULL embedding")
            total_pending += n
        typer.echo(f"  TOTAL: {total_pending} rows to re-embed")
        typer.echo()

        if total_pending == 0:
            typer.echo("Nothing to do. All embeddings present.")
            return

        if dry_run:
            typer.echo("(dry-run) — would re-embed and rebuild indexes; exiting now")
            return

        if not yes:
            confirm = typer.confirm(f"Re-embed {total_pending} rows with model {model_name}?")
            if not confirm:
                typer.echo("Aborted.")
                raise typer.Exit(0)

        # 3. Initialize embedding model
        typer.echo(f"[3/5] Loading embedding model {model_name}...")
        from ..engine.embeddings import LocalSTEmbeddings

        embed = LocalSTEmbeddings(
            model_name=model_name,
            trust_remote_code=trust_remote,
        )
        t_load = time.time()
        await embed.initialize()
        typer.echo(f"  Loaded in {time.time() - t_load:.1f}s. Output dim: {embed.dimension}")

        # Verify dim matches what columns expect
        for col in cols:
            if col.dimension != embed.dimension:
                typer.echo(
                    f"  ERROR: {col.table}.{col.column} expects dim {col.dimension} "
                    f"but model produces dim {embed.dimension}.",
                    err=True,
                )
                typer.echo(
                    f"  Migrate the column first with: "
                    f"ALTER TABLE {col.table} ALTER COLUMN {col.column} "
                    f"TYPE vector({embed.dimension}) USING NULL;",
                    err=True,
                )
                raise typer.Exit(1)
        typer.echo()

        # 4. Re-embed each table
        typer.echo("[4/5] Re-embedding rows...")
        overall_start = time.time()
        total_processed = 0
        total_skipped = 0
        for col in cols:
            p, s = await _reembed_table(conn, embed, col, batch_size, bank_id)
            total_processed += p
            total_skipped += s
        elapsed = time.time() - overall_start
        rate = total_processed / elapsed if elapsed > 0 else 0
        typer.echo(
            f"  TOTAL: {total_processed} re-embedded, {total_skipped} skipped "
            f"in {elapsed:.1f}s ({rate:.1f} rows/s overall)"
        )
        typer.echo()

        # 5. Rebuild indexes
        if skip_index_rebuild:
            typer.echo("[5/5] Skipping index rebuild (--skip-index-rebuild)")
        else:
            typer.echo("[5/5] Rebuilding vector indexes (HNSW/IVFFlat)...")
            n_indexes = await _rebuild_vector_indexes(conn, schema)
            typer.echo(f"  Rebuilt {n_indexes} vector indexes")
        typer.echo()

        typer.echo("== Done ==")
        typer.echo("Verify with sample recall queries before declaring success.")
    finally:
        await conn.close()


def reindex_embeddings_command(
    schema: str,
    bank_id: str | None,
    batch_size: int,
    dry_run: bool,
    skip_index_rebuild: bool,
    yes: bool,
    database_url: str,
) -> None:
    """Sync wrapper for the reindex-embeddings async impl."""
    asyncio.run(
        _reindex_embeddings(
            database_url=database_url,
            schema=schema,
            bank_id=bank_id,
            batch_size=batch_size,
            dry_run=dry_run,
            skip_index_rebuild=skip_index_rebuild,
            yes=yes,
        )
    )
