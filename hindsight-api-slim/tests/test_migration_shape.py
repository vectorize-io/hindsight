"""Lint-style enforcement for the dialect-dispatched migration pattern.

Every file in ``alembic/versions/`` must route ``upgrade``/``downgrade`` through
``alembic._dialect.run_for_dialect`` so PG and Oracle stay in lockstep. This
test fails the build if a new migration is added without filling at least one
dialect slot — without it we'd silently re-introduce drift on Oracle the first
time someone copies an old PG migration as a template.
"""

from __future__ import annotations

import ast
import functools
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

VERSIONS_DIR = Path(__file__).resolve().parent.parent / "hindsight_api" / "alembic" / "versions"


def _migration_files() -> list[Path]:
    return sorted(p for p in VERSIONS_DIR.glob("*.py") if not p.name.startswith("__"))


@pytest.mark.parametrize("path", _migration_files(), ids=lambda p: p.name)
def test_migration_uses_dialect_dispatcher(path: Path) -> None:
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))

    imports_dispatcher = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "hindsight_api.alembic._dialect"
        and any(alias.name == "run_for_dialect" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert imports_dispatcher, (
        f"{path.name}: missing 'from hindsight_api.alembic._dialect import run_for_dialect'. "
        "All migrations must dispatch through run_for_dialect — see CLAUDE.md."
    )

    top_level_fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for required in ("upgrade", "downgrade"):
        assert required in top_level_fns, f"{path.name}: missing top-level def {required}()."
        assert _calls_run_for_dialect(top_level_fns[required]), (
            f"{path.name}: {required}() must call run_for_dialect(...)."
        )

    has_pg_slot = "_pg_upgrade" in top_level_fns
    has_oracle_slot = "_oracle_upgrade" in top_level_fns
    assert has_pg_slot or has_oracle_slot, (
        f"{path.name}: migration defines neither _pg_upgrade nor _oracle_upgrade — "
        "at least one dialect slot must be filled (set the other to None if intentional)."
    )


def _calls_run_for_dialect(fn: ast.FunctionDef) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_for_dialect":
            return True
    return False


def _execute_sql_strings(tree: ast.AST) -> list[str]:
    """All string SQL passed to ``op.execute(...)``, including f-strings.

    f-strings (``ast.JoinedStr``) are flattened by concatenating their constant
    parts — schema-prefix placeholders like ``{schema}memory_units`` collapse to
    ``memory_units``, which is exactly what the table-name regexes below need.
    """
    sqls: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and node.args):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "execute"):
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            sqls.append(arg.value)
        elif isinstance(arg, ast.JoinedStr):
            sqls.append(
                "".join(v.value for v in arg.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
            )
    return sqls


# ``\bmemory_units\b`` does not match inside ``invalidated_memory_units``: the
# preceding ``_`` is a word character, so there is no word boundary before the
# ``m``. The schema prefix (``"tenant".memory_units``) does produce a boundary.
_ALTER_MAIN_TABLE_RE = re.compile(r"ALTER\s+TABLE\s+\S*\bmemory_units\b", re.IGNORECASE)
_ALTER_SHADOW_TABLE_RE = re.compile(r"ALTER\s+TABLE\s+\S*invalidated_memory_units\b", re.IGNORECASE)
# Column adds only — ``ADD CONSTRAINT`` needs no shadow sync (the archive
# deliberately carries no constraints). PG uses ``ADD COLUMN``; Oracle uses an
# ``ADD ( col type, ... )`` block, whose first token is an identifier unless the
# block holds only constraints.
_ADD_COLUMN_RE = re.compile(r"ADD\s+COLUMN|ADD\s*\(\s*(?!CONSTRAINT\b)\w", re.IGNORECASE)


@dataclass(frozen=True)
class _MigrationIds:
    """Module-level ``revision`` / ``down_revision`` of one migration file.

    ``down_revisions`` normalizes the three legal forms (None / str / tuple —
    merge migrations declare multiple parents) into a tuple.
    """

    revision: str | None
    down_revisions: tuple[str, ...]


def _migration_ids(tree: ast.AST) -> _MigrationIds:
    revision: str | None = None
    down_revisions: tuple[str, ...] = ()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == "revision" and isinstance(value, ast.Constant) and isinstance(value.value, str):
                revision = value.value
            elif target.id == "down_revision":
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    down_revisions = (value.value,)
                elif isinstance(value, ast.Tuple):
                    down_revisions = tuple(
                        e.value for e in value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    )
    return _MigrationIds(revision=revision, down_revisions=down_revisions)


_SHADOW_TABLE_CREATED_IN = "c9a1b2d3e4f5"


@functools.cache
def _pre_shadow_revisions() -> frozenset[str]:
    """Revisions at or before the migration that created the shadow table.

    Computed by walking the ``down_revision`` graph backwards (BFS — merge
    migrations have multiple parents) from the shadow table's creation
    migration, so the exemption list never needs manual maintenance.
    """
    parents: dict[str, tuple[str, ...]] = {}
    for path in _migration_files():
        ids = _migration_ids(ast.parse(path.read_text(), filename=str(path)))
        if ids.revision is not None:
            parents[ids.revision] = ids.down_revisions
    ancestors: set[str] = set()
    stack = [_SHADOW_TABLE_CREATED_IN]
    while stack:
        cursor = stack.pop()
        if cursor in ancestors:
            continue
        ancestors.add(cursor)
        stack.extend(parents.get(cursor, ()))
    return frozenset(ancestors)


@pytest.mark.parametrize("path", _migration_files(), ids=lambda p: p.name)
def test_memory_units_column_adds_also_alter_shadow_table(path: Path) -> None:
    """Adding columns to ``memory_units`` must also alter ``invalidated_memory_units``.

    The curation archive was created with ``LIKE memory_units`` — a point-in-time
    snapshot that does not follow later ALTERs. The curation row-move builds its
    column list from ``memory_units`` (``_memory_unit_columns``) and INSERTs into
    the archive by name, so a column present on the main table but missing on the
    archive makes invalidation fail outright. Migrations that predate the shadow
    table are exempt (computed from the revision chain, not hardcoded).
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    revision = _migration_ids(tree).revision
    if isinstance(revision, str) and revision in _pre_shadow_revisions():
        pytest.skip("predates the shadow table")

    sqls = _execute_sql_strings(tree)
    adds_main_column = any(_ALTER_MAIN_TABLE_RE.search(s) and _ADD_COLUMN_RE.search(s) for s in sqls)
    if not adds_main_column:
        pytest.skip("does not add columns to memory_units")

    alters_shadow = any(_ALTER_SHADOW_TABLE_RE.search(s) for s in sqls)
    assert alters_shadow, (
        f"{path.name}: adds columns to memory_units but never alters invalidated_memory_units. "
        "The curation archive is a LIKE-snapshot that does not follow ALTERs; the row-move "
        "INSERT will fail (or silently drop data) unless both tables change in lockstep. "
        "Add the same columns to invalidated_memory_units in both dialect slots."
    )


def _is_manual_commit(node: ast.AST) -> bool:
    """True if ``node`` is ``op.execute("COMMIT")`` (any casing/whitespace)."""
    if not (isinstance(node, ast.Call) and node.args):
        return False
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "execute"):
        return False
    if not (isinstance(func.value, ast.Name) and func.value.id == "op"):
        return False
    first = node.args[0]
    return isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.strip().upper() == "COMMIT"


@pytest.mark.parametrize("path", _migration_files(), ids=lambda p: p.name)
def test_migration_uses_autocommit_block_not_manual_commit(path: Path) -> None:
    """Ban the ``op.execute("COMMIT")`` trick for escaping the migration transaction.

    ``CREATE/DROP INDEX CONCURRENTLY`` (and procedural ``COMMIT`` in ``DO`` blocks)
    must run outside Alembic's migration transaction. The manual-COMMIT trick
    happens to work on psycopg2 but breaks on psycopg/SQLAlchemy 2.1, where the
    next statement re-opens a transaction and PostgreSQL rejects CONCURRENTLY.
    Use ``with op.get_context().autocommit_block():`` instead.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = [node.lineno for node in ast.walk(tree) if _is_manual_commit(node)]
    assert not offenders, (
        f'{path.name}: op.execute("COMMIT") at line(s) {offenders}. '
        "Wrap CONCURRENTLY DDL in `with op.get_context().autocommit_block():` instead "
        "of manually committing — the COMMIT trick fails on psycopg/SQLAlchemy 2.1."
    )


def _executes_concurrently_ddl(tree: ast.AST) -> bool:
    """True if any string passed to ``op.execute(...)`` contains CONCURRENTLY."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and node.args):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "execute"):
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "CONCURRENTLY" in arg.value.upper():
            return True
    return False


def _uses_autocommit_block(tree: ast.AST) -> bool:
    return any(isinstance(node, ast.Attribute) and node.attr == "autocommit_block" for node in ast.walk(tree))


@pytest.mark.parametrize("path", _migration_files(), ids=lambda p: p.name)
def test_migration_concurrently_ddl_runs_in_autocommit_block(path: Path) -> None:
    """``CONCURRENTLY`` DDL must run inside an ``autocommit_block()``.

    PostgreSQL rejects ``CREATE/DROP INDEX CONCURRENTLY`` inside a transaction
    block, and Alembic wraps every migration in one. The only safe escape is
    ``with op.get_context().autocommit_block():``. This guards both the
    manual-COMMIT trick and a CONCURRENTLY statement with no escape at all.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    if not _executes_concurrently_ddl(tree):
        pytest.skip("no CONCURRENTLY DDL")
    assert _uses_autocommit_block(tree), (
        f"{path.name}: runs CONCURRENTLY DDL but never opens an autocommit_block(). "
        "Wrap it in `with op.get_context().autocommit_block():` — CONCURRENTLY cannot "
        "run inside Alembic's migration transaction."
    )
