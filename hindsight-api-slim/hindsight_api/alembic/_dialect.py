"""Dialect dispatcher for Alembic migrations.

Each migration file declares a ``_pg_upgrade``/``_oracle_upgrade`` (and matching
downgrades) function and routes ``upgrade()``/``downgrade()`` through
``run_for_dialect``. The helper inspects the live connection's dialect name and
runs the matching function — or no-ops if the migration doesn't apply to the
current backend.

Use ``None`` (or omit the kwarg) when a migration intentionally has no effect
on a dialect; the helper treats it as a no-op.
"""

from __future__ import annotations

from collections.abc import Callable

from alembic import op

DialectFn = Callable[[], None]
_SUPPORTED = ("postgresql", "oracle")


def run_for_dialect(
    *,
    pg: DialectFn | None = None,
    oracle: DialectFn | None = None,
) -> None:
    """Dispatch to the function matching the current bind's dialect.

    Args:
        pg: Function to run when the active bind is PostgreSQL.
        oracle: Function to run when the active bind is Oracle.

    Unrecognized dialects raise; an explicit ``None`` for the active dialect
    is a no-op (the migration deliberately does nothing here).
    """
    name = op.get_bind().dialect.name
    if name not in _SUPPORTED:
        raise RuntimeError(f"Unsupported dialect for migration dispatch: {name!r}. Expected one of {_SUPPORTED}.")
    fn = {"postgresql": pg, "oracle": oracle}[name]
    if fn is not None:
        fn()
