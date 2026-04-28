"""
Centralized schema-qualified table name helpers.

Single source of truth for producing ``"schema".table_name`` references
that respect both the active schema context and the database backend.
"""

from ..config import get_config


def _is_oracle() -> bool:
    """Return True when the configured database backend is Oracle."""
    return get_config().database_backend == "oracle"


def fq_table(table_name: str) -> str:
    """Get fully-qualified table name using the current schema context.

    On Oracle the schema is set at the session level (``ALTER SESSION SET
    CURRENT_SCHEMA``), so we return the bare table name.  On PostgreSQL
    we prefix with the schema from :func:`memory_engine.get_current_schema`.
    """
    if _is_oracle():
        return table_name
    from .memory_engine import get_current_schema

    return f"{get_current_schema()}.{table_name}"


def fq_table_explicit(table: str, schema: str | None = None) -> str:
    """Get fully-qualified table name with an explicit schema override.

    Used by modules that don't rely on the context-variable schema
    (e.g. task_backend, worker poller) and instead pass the schema
    explicitly.
    """
    if _is_oracle():
        return table
    if schema:
        return f'"{schema}".{table}'
    return table
