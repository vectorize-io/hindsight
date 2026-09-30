"""
Centralized schema-qualified table name helpers.

Single source of truth for producing ``"schema".table_name`` references
that respect both the active schema context and the database backend.
"""

from ..config import get_config

#: Tables the memories store owns. For a bank whose store is not Postgres these are
#: empty, so SQL against them from outside the store reads nothing, and writes rows
#: nothing will ever read (#4969). :func:`fq_table` and :func:`fq_table_explicit`
#: refuse to name them; only the Postgres store resolves them, through
#: :func:`fq_store_table` / :func:`fq_store_table_explicit`.
STORE_TABLES = frozenset(
    {
        "memory_units",
        "memory_links",
        "unit_entities",
        "documents",
        "chunks",
        "entities",
        "entity_cooccurrences",
        "invalidated_memory_units",
    }
)


class StoreTableAccessError(RuntimeError):
    """Code outside the memories store named a table the store owns."""


def _guard(table_name: str) -> None:
    if table_name in STORE_TABLES:
        raise StoreTableAccessError(
            f"{table_name!r} belongs to the memories store; go through get_memories() instead of SQL (#4969)"
        )


def _is_oracle() -> bool:
    """Return True when the configured database backend is Oracle."""
    return get_config().database_backend == "oracle"


def fq_table(table_name: str) -> str:
    """Get fully-qualified table name using the current schema context.

    On Oracle the schema is set at the session level (``ALTER SESSION SET
    CURRENT_SCHEMA``), so we return the bare table name.  On PostgreSQL
    we prefix with the schema from :func:`memory_engine.get_current_schema`.
    """
    _guard(table_name)
    return fq_store_table(table_name)


def fq_store_table(table_name: str) -> str:
    """:func:`fq_table` without the guard. Only the Postgres memories store may call it."""
    if _is_oracle():
        return table_name
    from .memory_engine import get_current_schema

    return f"{get_current_schema()}.{table_name}"


def fq_routine(name: str) -> str:
    """Schema-qualified name of a cross-tenant discovery routine.

    These routines are database-global — each enumerates ``pg_class`` across every
    schema and dispatches per schema — so exactly one copy exists, installed into
    the configured schema by ``b6d2f8a4c1e7``. Calling it through the configured
    schema rather than a hardcoded ``public.`` is what makes a deployment living
    in a dedicated non-``public`` schema work (#2638).

    Unlike :func:`fq_table` this ignores the per-request schema contextvar: the
    routines are deliberately cross-tenant, called from background loops that have
    no request context.
    """
    schema = get_config().database_schema or "public"
    return '"' + schema.replace('"', '""') + '".' + name


def fq_table_explicit(table: str, schema: str | None = None) -> str:
    """Get fully-qualified table name with an explicit schema override.

    Used by modules that don't rely on the context-variable schema
    (e.g. task_backend, worker poller) and instead pass the schema
    explicitly.
    """
    _guard(table)
    return fq_store_table_explicit(table, schema)


def fq_store_table_explicit(table: str, schema: str | None = None) -> str:
    """:func:`fq_table_explicit` without the guard. Only the Postgres memories store may call it."""
    if _is_oracle():
        return table
    if schema:
        return f'"{schema}".{table}'
    return table
