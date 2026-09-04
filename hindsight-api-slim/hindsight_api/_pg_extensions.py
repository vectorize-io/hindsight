"""One rule for every PostgreSQL extension Hindsight installs: it lives in ``public``.

``CREATE EXTENSION`` installs into the *first* schema on the session
``search_path``. During migrations that path starts with the tenant schema
(see ``alembic/env.py``), so a bare ``CREATE EXTENSION pg_trgm`` in schema mode
lands the extension inside the tenant schema. The runtime connects with the
default ``"$user", public`` path and fully-qualifies its tables, so it never
sees that schema: operators and types the extension provides stop resolving,
and entity resolution fails with ``operator does not exist: text % text`` on
every retain, forever, with nothing surfaced to the caller (#4118).

Every extension therefore goes through :func:`create_extension`, which pins the
search path to ``public`` for the duration of the statement, and through
:func:`relocate_extension_to_public`, which repairs installations that a
previous version misplaced. Extensions that pin their own schema in their
control file (``relocatable = false``, e.g. ``vchord_bm25`` -> ``bm25_catalog``)
are left where PostgreSQL puts them — the search path does not override that,
and relocating them is not allowed.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy import text

logger = logging.getLogger(__name__)

#: Extension names are interpolated into DDL — PostgreSQL cannot bind an
#: identifier as a parameter — so every name is checked against this first.
_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

PUBLIC_SCHEMA = "public"

#: Every extension Hindsight creates, in any code path. ``ensure_extensions_in_public``
#: sweeps these to repair databases migrated by a version that misplaced them.
MANAGED_EXTENSIONS: tuple[str, ...] = (
    "vector",
    "vectorscale",
    "vchord",
    "alloydb_scann",
    "pg_trgm",
    "vchord_bm25",
    "pg_textsearch",
    "pg_search",
    "pgroonga",
)


class _Executable(Protocol):
    """The slice of SQLAlchemy's ``Connection`` these helpers need."""

    def execute(self, statement: Any, parameters: Any = None, *args: Any, **kwargs: Any) -> Any: ...


class _Transactional(_Executable, Protocol):
    """An ``_Executable`` whose transaction the caller may end (relocation only)."""

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


def _validate(name: str) -> str:
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid PostgreSQL extension name: {name!r}")
    return name


def create_extension(conn: _Executable, name: str, *, cascade: bool = False) -> None:
    """``CREATE EXTENSION IF NOT EXISTS`` with the install schema pinned to ``public``.

    The caller's ``search_path`` is restored afterwards. If the CREATE fails the
    restore is attempted but not allowed to mask the original error — callers
    that continue after a failure roll the transaction back, which restores the
    setting anyway.
    """
    _validate(name)
    previous = conn.execute(text("SELECT current_setting('search_path')")).scalar()
    conn.execute(text("SELECT set_config('search_path', :schema, false)"), {"schema": PUBLIC_SCHEMA})
    try:
        conn.execute(text(f"CREATE EXTENSION IF NOT EXISTS {name}{' CASCADE' if cascade else ''}"))
    finally:
        try:
            conn.execute(text("SELECT set_config('search_path', :previous, false)"), {"previous": previous or ""})
        except Exception:  # pragma: no cover - only reachable on an aborted transaction
            logger.debug("Could not restore search_path after creating extension %s", name)


@dataclass(frozen=True)
class InstalledExtension:
    """Where an installed extension lives, and whether PostgreSQL will move it."""

    schema: str
    relocatable: bool


def _installed_extension(conn: _Executable, name: str) -> InstalledExtension | None:
    """Look up an installed extension in the catalog, or None if it is absent."""
    row = conn.execute(
        text(
            "SELECT n.nspname, e.extrelocatable FROM pg_extension e "
            "JOIN pg_namespace n ON n.oid = e.extnamespace WHERE e.extname = :name"
        ),
        {"name": name},
    ).fetchone()
    return InstalledExtension(schema=row[0], relocatable=row[1]) if row else None


def extension_schema(conn: _Executable, name: str) -> str | None:
    """Return the schema the extension is installed in, or None if not installed."""
    _validate(name)
    installed = _installed_extension(conn, name)
    return installed.schema if installed else None


def relocate_extension_to_public(conn: _Transactional, name: str) -> bool:
    """Move a misplaced extension into ``public``; return True if it moved.

    A no-op when the extension is absent, already in ``public``, or not
    relocatable. Failures (typically: the migration role does not own the
    extension) are logged and swallowed — the caller is better off running with
    a misplaced extension than not running at all.
    """
    _validate(name)
    installed = _installed_extension(conn, name)
    if installed is None:
        return False
    schema = installed.schema
    if schema == PUBLIC_SCHEMA:
        return False
    if not installed.relocatable:
        # Its control file pins the schema (e.g. vchord_bm25 -> bm25_catalog);
        # the runtime adds those schemas to search_path instead.
        logger.debug("Extension %s is not relocatable; leaving it in schema '%s'", name, schema)
        return False

    logger.warning(
        "Extension %s is installed in schema '%s' instead of '%s'; relocating so the "
        "runtime can resolve its operators and types.",
        name,
        schema,
        PUBLIC_SCHEMA,
    )
    try:
        conn.execute(text(f'ALTER EXTENSION {name} SET SCHEMA "{PUBLIC_SCHEMA}"'))
        conn.commit()
    except Exception as exc:
        logger.warning(
            "Could not relocate extension %s from '%s' to '%s': %s. Continuing; "
            "queries that rely on it may fail until it is moved by an administrator.",
            name,
            schema,
            PUBLIC_SCHEMA,
            exc,
        )
        conn.rollback()
        return False
    logger.info("Extension %s relocated to the %s schema", name, PUBLIC_SCHEMA)
    return True


def ensure_extensions_in_public(conn: _Transactional, names: tuple[str, ...] = MANAGED_EXTENSIONS) -> None:
    """Repair extensions a previous version installed into a tenant schema.

    One catalog query, not one per extension: this runs before every schema
    migration, and a deployment sweeping tens of thousands of tenant schemas pays
    it once per schema.
    """
    misplaced = conn.execute(
        text(
            "SELECT e.extname FROM pg_extension e JOIN pg_namespace n ON n.oid = e.extnamespace "
            "WHERE e.extname = ANY(:names) AND n.nspname <> :public AND e.extrelocatable"
        ),
        {"names": list(names), "public": PUBLIC_SCHEMA},
    ).fetchall()
    for row in misplaced:
        relocate_extension_to_public(conn, row[0])
