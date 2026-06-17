"""Lint-style enforcement for the dialect-dispatched migration pattern.

Every file in ``alembic/versions/`` must route ``upgrade``/``downgrade`` through
``alembic._dialect.run_for_dialect`` so PG and Oracle stay in lockstep. This
test fails the build if a new migration is added without filling at least one
dialect slot — without it we'd silently re-introduce drift on Oracle the first
time someone copies an old PG migration as a template.
"""

from __future__ import annotations

import ast
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
