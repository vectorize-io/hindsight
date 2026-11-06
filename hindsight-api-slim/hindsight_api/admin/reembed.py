"""``hindsight-admin reembed``: move a schema's stored vectors to a new embedding model in place.

Postgres only. The servers keep running; the steps per schema are:

1. Add a nullable ``embedding_next`` column of the new dimension to ``memory_units`` and
   ``mental_models``. Adding a nullable column with no default is metadata only: no table copy.
2. Install a trigger that clears ``embedding_next`` whenever a row's text or vector changes, so a
   write that lands while the fill runs is simply re-embedded by a later pass.
3. Drop the old vector indexes. Every row is updated once; with an HNSW index still on the table each
   of those updates would pay an index insert. Recall keeps working on the old vectors, just
   without the index, until the swap.
4. Fill ``embedding_next`` in small keyset batches, each its own transaction, from the same text
   the write paths embed. Several batches are at the embedding model at once; the database work
   stays on one connection. A partial index over the rows still to do keeps every pass (and a
   resumed run) from rescanning the table.
5. Build the vector indexes on ``embedding_next`` (CONCURRENTLY), so the new column is served by an
   index the moment it becomes ``embedding``.
6. Swap under a short write lock (reads continue): embed the last stragglers, drop ``embedding``,
   rename ``embedding_next`` to ``embedding``. Both are catalog changes. The trigger becomes a guard
   that refuses vector writes from any connection not tagged with the new model (servers tag every
   connection with ``hindsight.embedding_model``); the worker defers those writes and HTTP answers 503.
7. Once every server runs the new model, ``reembed --finish`` removes the guard.

Every step is resumable: re-running the command picks up where it stopped.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from .._vector_index import (
    detect_vector_extension,
    index_type_keyword,
    index_using_clause,
    should_defer_index_creation,
    uses_per_bank_vector_indexes,
)
from ..config import get_config
from ..db_url import to_libpq_url
from ..engine.embeddings import EMBEDDING_MODEL_SETTING, REEMBED_GUARD_SQLSTATE, Embeddings, embedding_model_id
from ..engine.retain import embedding_processing
from ..engine.retain.types import ExtractedFact
from ..engine.vector_index_health import list_bank_ids, reconcile_bank_vector_indexes

logger = logging.getLogger(__name__)

TABLES = ("memory_units", "mental_models")
NEXT = "embedding_next"
# Columns whose change makes a row's freshly computed vector stale while the fill runs.
_SOURCE_COLUMNS = {"memory_units": ("text",), "mental_models": ("name", "content")}
# Each table's primary key, which the fill pages through and writes by. A mental model's id is only
# unique within its bank, so paging or matching on `id` alone could skip a row or write one bank's
# vector into another bank's model of the same id.
_KEYS = {"memory_units": ("id",), "mental_models": ("bank_id", "id")}
_TRIGGER = "hindsight_reembed"
_SWAP_LOCK_ATTEMPTS = 20


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _table(schema: str, table: str) -> str:
    return f"{_q(schema)}.{_q(table)}"


def _function(schema: str, table: str) -> str:
    return f"{_q(schema)}.{_q(f'{_TRIGGER}_{table}')}"


@dataclass
class SchemaReembed:
    """What one schema's run did."""

    schema: str
    status: str  # "swapped", "already swapped", "finished"
    rows_embedded: int = 0


async def _column_dimension(conn: asyncpg.Connection, schema: str, table: str, column: str) -> int | None:
    return await conn.fetchval(
        """
        SELECT a.atttypmod FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = $1 AND c.relname = $2 AND a.attname = $3 AND NOT a.attisdropped
        """,
        schema,
        table,
        column,
    )


async def _column_type(conn: asyncpg.Connection, schema: str, table: str, column: str) -> str:
    return await conn.fetchval(
        """
        SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = $1 AND c.relname = $2 AND a.attname = $3
        """,
        schema,
        table,
        column,
    )


async def _guard_installed(conn: asyncpg.Connection, schema: str) -> bool:
    return bool(
        await conn.fetchval(
            """
            SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE n.nspname = $1 AND p.proname = ANY($2::text[])
            """,
            schema,
            [f"{_TRIGGER}_{t}" for t in TABLES],
        )
    )


async def _install_function(conn: asyncpg.Connection, schema: str, table: str, body: str) -> None:
    """(Re)define the table's trigger function; the trigger itself is created once, by the fill."""
    await conn.execute(
        f"CREATE OR REPLACE FUNCTION {_function(schema, table)}() RETURNS trigger LANGUAGE plpgsql AS $hs$\n"
        f"BEGIN\n{body}\nRETURN NEW;\nEND\n$hs$"
    )


def _fill_body(table: str) -> str:
    changed = " OR ".join(f"NEW.{c} IS DISTINCT FROM OLD.{c}" for c in ("embedding", *_SOURCE_COLUMNS[table]))
    return f"IF TG_OP = 'UPDATE' AND ({changed}) THEN NEW.{NEXT} := NULL; END IF;"


def _guard_body(model_id: str) -> str:
    literal = "'" + model_id.replace("'", "''") + "'"
    message = "'hindsight re-embed: writes are shed until this server runs embedding model ' || " + literal
    return (
        f"IF current_setting('{EMBEDDING_MODEL_SETTING}', true) IS DISTINCT FROM {literal}\n"
        "   AND ((TG_OP = 'INSERT' AND NEW.embedding IS NOT NULL)\n"
        "        OR (TG_OP = 'UPDATE' AND NEW.embedding IS DISTINCT FROM OLD.embedding)) THEN\n"
        f"  RAISE EXCEPTION USING MESSAGE = {message}, ERRCODE = '{REEMBED_GUARD_SQLSTATE}';\n"
        "END IF;"
    )


async def _vector_index_names(conn: asyncpg.Connection, schema: str, table: str, column: str) -> list[str]:
    rows = await conn.fetch(
        """
        SELECT DISTINCT c.relname FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_class t ON t.oid = i.indrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(i.indkey)
        WHERE n.nspname = $1 AND t.relname = $2 AND a.attname = $3
        """,
        schema,
        table,
        column,
    )
    return [r["relname"] for r in rows]


async def _create_index(conn: asyncpg.Connection, schema: str, name: str, definition: str) -> None:
    """CREATE INDEX CONCURRENTLY, replacing an invalid leftover of an interrupted earlier run."""
    valid = await conn.fetchval(
        """
        SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = $1 AND c.relname = $2
        """,
        schema,
        name,
    )
    if valid:
        return
    if valid is False:
        await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_q(schema)}.{_q(name)}")
    await conn.execute(f"CREATE INDEX CONCURRENTLY {_q(name)} ON {definition}")


def _todo_predicate() -> str:
    return f"{NEXT} IS NULL AND embedding IS NOT NULL"


@dataclass
class _Todo:
    """A row still to embed."""

    key: list[Any]  # the table's _KEYS values
    version: str  # the row's xmin when read
    text: str  # what gets embedded


def fact_embedding_text(
    *,
    fact_type: str,
    text: str,
    entities: list[str],
    occurred_start: datetime | None,
    occurred_end: datetime | None,
    mentioned_at: datetime | None,
) -> str:
    """The text the write paths embed for a memory unit.

    Facts: their text plus dates and entity names, as retain/edit/import build it. Observations:
    their text alone, as consolidation embeds it.
    """
    if fact_type == "observation":
        return text
    fact = ExtractedFact(
        fact_text=text,
        fact_type=fact_type,
        entities=entities,
        occurred_start=occurred_start,
        occurred_end=occurred_end,
        mentioned_at=mentioned_at,
    )
    return embedding_processing.augment_texts_with_dates([fact], embedding_processing.format_readable_date)[0]


def mental_model_embedding_text(name: str | None, content: str | None) -> str:
    """The text the write paths embed for a mental model."""
    return f"{name or ''} {content or ''}"


async def _texts(conn: asyncpg.Connection, schema: str, table: str, after: list[Any] | None, limit: int) -> list[_Todo]:
    """The next batch of rows still to embed, keyset-paged on the table's primary key."""
    keys = _KEYS[table]
    key_cols = ", ".join(f"x.{k}" for k in keys)
    cursor = ""
    if after is not None:
        cursor = f"AND ({key_cols}) > ({', '.join(f'${i + 2}' for i in range(len(keys)))})"
    args: list[Any] = [limit, *(after or [])]
    where = f"WHERE {_todo_predicate()} {cursor} ORDER BY {key_cols} LIMIT $1"
    if table == "mental_models":
        rows = await conn.fetch(
            f"SELECT {key_cols}, x.xmin::text AS version, x.name, x.content FROM {_table(schema, table)} x {where}",
            *args,
        )
        return [
            _Todo([r[k] for k in keys], r["version"], mental_model_embedding_text(r["name"], r["content"]))
            for r in rows
        ]
    rows = await conn.fetch(
        f"""
        SELECT {key_cols}, x.xmin::text AS version, x.fact_type, x.text,
               x.occurred_start, x.occurred_end, x.mentioned_at,
               ARRAY(SELECT e.canonical_name FROM {_table(schema, "unit_entities")} ue
                     JOIN {_table(schema, "entities")} e ON e.id = ue.entity_id
                     WHERE ue.unit_id = x.id) AS entities
        FROM {_table(schema, table)} x {where}
        """,
        *args,
    )
    return [
        _Todo(
            [r[k] for k in keys],
            r["version"],
            fact_embedding_text(
                fact_type=r["fact_type"],
                text=r["text"],
                entities=list(r["entities"]),
                occurred_start=r["occurred_start"],
                occurred_end=r["occurred_end"],
                mentioned_at=r["mentioned_at"],
            ),
        )
        for r in rows
    ]


async def _fill_pass(
    conn: asyncpg.Connection,
    schema: str,
    table: str,
    embeddings: Embeddings,
    *,
    batch_size: int,
    concurrency: int,
    sleep_s: float,
) -> int:
    """One walk over the rows still to embed. Returns how many were read for embedding.

    Up to ``concurrency`` batches are at the embedding model at once: that call is the slow part
    (a remote model answers in hundreds of ms). Every read and write still goes through the one
    ``conn``, one at a time, so Postgres sees the same single connection however many batches are
    in flight; ``db`` serialises them because an asyncpg connection runs one query at a time.
    """
    keys = _KEYS[table]
    key_types = [await _column_type(conn, schema, table, k) for k in keys]
    n = len(keys)
    # unnest(keys..., versions, vectors) AS v(k0, k1, ..., version, e), matched on every key column.
    unnest_args = ", ".join(f"${i + 1}::{t}[]" for i, t in enumerate(key_types))
    match = " AND ".join(f"x.{k} = v.k{i}" for i, k in enumerate(keys))
    update = (
        f"UPDATE {_table(schema, table)} x SET {NEXT} = v.e::vector "
        f"FROM unnest({unnest_args}, ${n + 1}::text[], ${n + 2}::text[]) "
        f"AS v({', '.join(f'k{i}' for i in range(n))}, version, e) "
        f"WHERE {match} AND x.xmin::text = v.version"
    )
    db = asyncio.Lock()
    slots = asyncio.Semaphore(concurrency)

    async def embed_and_write(rows: list[_Todo]) -> None:
        try:
            vectors = await embedding_processing.generate_embeddings_batch(embeddings, [r.text for r in rows])
            # Only rows still at the version that was read: a row a server rewrote in between keeps
            # a NULL `embedding_next`, and a later pass embeds its new text.
            async with db:
                await conn.execute(
                    update,
                    *([r.key[i] for r in rows] for i in range(n)),
                    [r.version for r in rows],
                    ["[" + ",".join(str(float(x)) for x in v) + "]" for v in vectors],
                )
            if sleep_s:
                await asyncio.sleep(sleep_s)
        finally:
            slots.release()

    done = 0
    after = None
    # TaskGroup: one failed batch cancels the others and the error surfaces; a re-run resumes.
    async with asyncio.TaskGroup() as group:
        while True:
            await slots.acquire()
            async with db:
                rows = await _texts(conn, schema, table, after, batch_size)
            if not rows:
                slots.release()
                break
            group.create_task(embed_and_write(rows))
            done += len(rows)
            after = rows[-1].key
    return done


def _resolve_vector_extension(sync_url: str) -> str:
    engine = create_engine(to_libpq_url(sync_url), poolclass=NullPool)
    try:
        with engine.connect() as c:
            return detect_vector_extension(c, get_config().vector_extension)
    finally:
        engine.dispose()


async def _build_next_indexes(conn: asyncpg.Connection, schema: str, ext: str) -> None:
    """Vector indexes over ``embedding_next``; the rename at the swap carries them to ``embedding``."""
    clause = index_using_clause(ext).replace("(embedding ", f"({NEXT} ")
    for table in TABLES:
        if table == "memory_units" and uses_per_bank_vector_indexes(ext):
            continue
        rows = await conn.fetchval(f"SELECT count(*) FROM {_table(schema, table)} WHERE {NEXT} IS NOT NULL")
        if should_defer_index_creation(ext, rows):
            logger.warning("%s.%s: too few rows for a %s index yet; not built", schema, table, ext)
            continue
        name = f"idx_{table}_embedding_{index_type_keyword(ext)}"
        await _create_index(conn, schema, name, f"{_table(schema, table)} {clause}")
    if uses_per_bank_vector_indexes(ext):
        for bank_id in await list_bank_ids(conn, schema):
            await reconcile_bank_vector_indexes(conn, schema, bank_id, clause)


async def _swap(
    conn: asyncpg.Connection, schema: str, embeddings: Embeddings, *, batch_size: int, concurrency: int
) -> int:
    """Last stragglers, then ``embedding_next`` becomes ``embedding``, under a short write lock.

    SHARE ROW EXCLUSIVE stops writers and lets readers through. ``lock_timeout`` keeps a long
    transaction from parking every later query behind this one; the lock is retried instead.
    """
    tables = ", ".join(_table(schema, t) for t in TABLES)
    model_id = embedding_model_id(embeddings)
    for attempt in range(1, _SWAP_LOCK_ATTEMPTS + 1):
        try:
            async with conn.transaction():
                await conn.execute("SET LOCAL lock_timeout = '5s'")
                await conn.execute(f"LOCK TABLE {tables} IN SHARE ROW EXCLUSIVE MODE")
                done = 0
                for table in TABLES:
                    done += await _fill_pass(
                        conn, schema, table, embeddings, batch_size=batch_size, concurrency=concurrency, sleep_s=0
                    )
                for table in TABLES:
                    await conn.execute(f"ALTER TABLE {_table(schema, table)} DROP COLUMN embedding")
                    await conn.execute(f"ALTER TABLE {_table(schema, table)} RENAME COLUMN {NEXT} TO embedding")
                    await _install_function(conn, schema, table, _guard_body(model_id))
                return done
        except (asyncpg.LockNotAvailableError, asyncpg.DeadlockDetectedError) as exc:
            logger.warning("%s: swap lock not taken (attempt %d): %s", schema, attempt, exc)
            await asyncio.sleep(min(attempt, 10))
    raise RuntimeError(f"{schema}: could not take the swap lock after {_SWAP_LOCK_ATTEMPTS} attempts; re-run")


async def reembed_schema(
    conn: asyncpg.Connection,
    sync_url: str,
    schema: str,
    embeddings: Embeddings,
    *,
    batch_size: int = 256,
    concurrency: int = 4,
    sleep_s: float = 0.0,
) -> SchemaReembed:
    """Run (or resume) the re-embed of one schema. ``conn`` must be autocommit (no open transaction)."""
    dim = embeddings.dimension
    if await _column_dimension(conn, schema, "memory_units", NEXT) is None and await _guard_installed(conn, schema):
        return SchemaReembed(schema, "already swapped")

    ext = await asyncio.to_thread(_resolve_vector_extension, sync_url)
    if index_type_keyword(ext) == "hnsw" and dim > 2000:
        raise RuntimeError(f"pgvector HNSW indexes stop at 2000 dimensions; the new model has {dim}")

    for table in TABLES:
        if await _column_dimension(conn, schema, table, NEXT) not in (None, dim):
            # Left by an earlier run for a different model: its vectors are useless now.
            await conn.execute(f"ALTER TABLE {_table(schema, table)} DROP COLUMN {NEXT}")
        await conn.execute(f"ALTER TABLE {_table(schema, table)} ADD COLUMN IF NOT EXISTS {NEXT} vector({dim})")
        await _install_function(conn, schema, table, _fill_body(table))
        await conn.execute(f"DROP TRIGGER IF EXISTS {_TRIGGER} ON {_table(schema, table)}")
        await conn.execute(
            f"CREATE TRIGGER {_TRIGGER} BEFORE INSERT OR UPDATE ON {_table(schema, table)} "
            f"FOR EACH ROW EXECUTE FUNCTION {_function(schema, table)}()"
        )
        for name in await _vector_index_names(conn, schema, table, "embedding"):
            await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_q(schema)}.{_q(name)}")
        # Dropped with `embedding` at the swap, since its predicate names it.
        await _create_index(
            conn,
            schema,
            f"{_TRIGGER}_todo_{table}",
            f"{_table(schema, table)} ({', '.join(_KEYS[table])}) WHERE {_todo_predicate()}",
        )

    done = 0
    for table in TABLES:
        while True:
            # Writes that land during a pass clear their row again; keep walking until a pass
            # finds little left, then the swap mops up the rest under the lock.
            n = await _fill_pass(
                conn, schema, table, embeddings, batch_size=batch_size, concurrency=concurrency, sleep_s=sleep_s
            )
            done += n
            logger.info("%s.%s: embedded %d rows", schema, table, n)
            if n < batch_size:
                break

    await _build_next_indexes(conn, schema, ext)
    done += await _swap(conn, schema, embeddings, batch_size=batch_size, concurrency=concurrency)
    if uses_per_bank_vector_indexes(ext):
        # A bank created during the fill got its indexes on the old column, gone with it.
        for bank_id in await list_bank_ids(conn, schema):
            await reconcile_bank_vector_indexes(conn, schema, bank_id, index_using_clause(ext))
    await conn.execute(f"ANALYZE {', '.join(_table(schema, t) for t in TABLES)}")
    return SchemaReembed(schema, "swapped", done)


async def finish_schema(conn: asyncpg.Connection, schema: str) -> SchemaReembed:
    """Remove the write guard once every server runs the new model."""
    if await _column_dimension(conn, schema, "memory_units", NEXT) is not None:
        raise RuntimeError(f"{schema}: the re-embed has not swapped yet; run reembed without --finish first")
    for table in TABLES:
        await conn.execute(f"DROP TRIGGER IF EXISTS {_TRIGGER} ON {_table(schema, table)}")
        await conn.execute(f"DROP FUNCTION IF EXISTS {_function(schema, table)}()")
    return SchemaReembed(schema, "finished")
