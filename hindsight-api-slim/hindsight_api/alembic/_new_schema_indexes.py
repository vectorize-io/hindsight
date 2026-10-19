"""Build a brand-new schema's indexes plainly, not ``CONCURRENTLY``.

The migrations create and drop their indexes ``CONCURRENTLY`` so a deploy adding one never
blocks writes on a populated table. ``CONCURRENTLY`` pays for that by waiting, more than once,
for every transaction open anywhere in the DATABASE when it starts — not just on its table.

On a schema being created from nothing that wait buys nothing: its tables are empty and nothing
else is writing to them. It does cost: a single long transaction in any other schema (a bulk
UPDATE, a report, a long-running worker task) stalls each of the new schema's dozens of concurrent
index statements for up to its own duration, so provisioning one new tenant goes from seconds to
minutes, and every first request waiting on that tenant times out.

So when the migration run starts on a schema with no ``alembic_version`` yet, statements that
build or drop an index in THAT schema run without ``CONCURRENTLY``. A plain build on an empty
table is instant and locks only that table, which nothing else is using yet.

Anything else keeps ``CONCURRENTLY``: an existing schema (its tables may be populated and live),
the default schema, and any statement that does not name the new schema — so an index on a
shared table can never be built under a blocking lock by accident.
"""

import re
from typing import Any

from sqlalchemy import Connection, event, text
from sqlalchemy.engine import Engine, ExecutionContext

_CONCURRENT_INDEX_DDL = re.compile(
    r"^(\s*(?:CREATE\s+(?:UNIQUE\s+)?INDEX|DROP\s+INDEX))\s+CONCURRENTLY\b", re.IGNORECASE
)


def is_new_schema(connection: Connection, schema: str) -> bool:
    """Whether ``schema`` has never been migrated: it has no ``alembic_version`` table."""
    return (
        connection.execute(text("SELECT to_regclass(:t) IS NULL"), {"t": f'"{schema}".alembic_version'}).scalar()
        is True
    )


def plain_index_ddl(statement: str, schema: str) -> str:
    """``statement`` without ``CONCURRENTLY`` if it is index DDL naming ``schema``, else as is."""
    if f'"{schema}".' not in statement:
        return statement
    return _CONCURRENT_INDEX_DDL.sub(r"\1", statement, count=1)


def build_new_schema_indexes_plainly(engine: Engine, schema: str) -> None:
    """Rewrite the run's index DDL on ``schema`` to its plain form (see the module docstring)."""

    @event.listens_for(engine, "before_cursor_execute", retval=True)
    def _plain(
        conn: Connection,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: ExecutionContext | None,
        executemany: bool,
    ) -> tuple[str, Any]:
        # SQLAlchemy's retval=True contract for this event: (statement, parameters).
        return plain_index_ddl(statement, schema), parameters
