"""The pool skips asyncpg's release-time reset, and the invariants that allows.

asyncpg sends ``SELECT pg_advisory_unlock_all(); CLOSE ALL; UNLISTEN *;
RESET ALL;`` on every pool release — one Postgres round trip per acquire/release
cycle, and behind a transaction-mode pooler one server-side transaction too. On a
lookup-shaped endpoint that was 1 of ~5 round trips per request.

Skipping it is safe *for this codebase* rather than in general, and the second
half of this module is what keeps that true: it fails if anyone introduces the
session state the reset was cleaning up.

The one behaviour that genuinely changes: the pool's session setup can now only
ADD a GUC. A connection opened while the setup was sending one keeps it until the
connection is replaced, because nothing wipes it in between. That is invisible in
production, where the settings come from process-static config and therefore never
change within a pool's lifetime — but a test that rewrites config mid-process must
call ``pool.expire_connections()`` rather than expect the next acquire to have
forgotten (see tests/test_ann_iterative_scan.py).

Deterministic (no DB): asyncpg.create_pool is monkeypatched to capture kwargs,
and the invariant half reads the source tree.
"""

import ast
import inspect
import re
from pathlib import Path

import asyncpg
import pytest

from hindsight_api.engine.db import postgresql as pg_mod
from hindsight_api.engine.db.postgresql import PostgreSQLBackend, _NoResetConnection

_DSN = "postgresql://u:p@h:5432/db"


class _FakePool:
    def get_size(self):
        return 0

    def get_idle_size(self):
        return 0


@pytest.fixture
def captured_pool_kwargs(monkeypatch):
    captured: dict = {}

    async def fake_create_pool(dsn, **kwargs):
        captured["dsn"] = dsn
        captured.update(kwargs)
        return _FakePool()

    monkeypatch.setattr(pg_mod.asyncpg, "create_pool", fake_create_pool)
    return captured


class TestPoolWiring:
    @pytest.mark.asyncio
    async def test_the_pool_skips_the_reset_query(self, captured_pool_kwargs):
        backend = PostgreSQLBackend()
        await backend.initialize(_DSN)

        assert captured_pool_kwargs["connection_class"] is _NoResetConnection

    def test_empty_reset_query_is_what_asyncpg_skips_on(self):
        # asyncpg's Connection.reset() is `if reset_query: await self.execute(...)`,
        # so an empty string is the documented way to opt out. A subclass returning
        # None would raise there instead.
        assert _NoResetConnection.get_reset_query(None) == ""  # type: ignore[arg-type]

    def test_asyncpg_still_rolls_back_open_transactions(self):
        # The reset *query* is skipped; Connection._reset() is not, and that is
        # what rolls back a transaction a request left open. If a future asyncpg
        # moves the ROLLBACK into the reset query, this fails and the flag's
        # default has to be revisited.
        assert "ROLLBACK" in inspect.getsource(asyncpg.Connection._reset)


# --- Invariants that make skipping the reset safe -----------------------------
#
# Each pattern below is one clause of asyncpg's reset query. If the codebase
# starts producing state that clause was cleaning up, skipping the reset leaks it
# to the next acquirer of that pooled connection.

_API_ROOT = Path(__file__).resolve().parents[1] / "hindsight_api"

# migrations.py holds a grandfathered advisory lock (tracked for removal, see
# CLAUDE.md); it runs on its own connection, not a pooled one.
_ADVISORY_LOCK_ALLOWED = {_API_ROOT / "migrations.py"}

# The pool's own session GUCs (this is the state the reset used to wipe and the
# per-acquire setup used to re-send) plus extension bootstrap, which runs on a
# SQLAlchemy migration connection rather than the asyncpg pool.
_SESSION_SET_ALLOWED = {
    _API_ROOT / "engine" / "db" / "postgresql.py",
    _API_ROOT / "engine" / "memory_engine.py",
    _API_ROOT / "_pg_extensions.py",
}

_LISTEN = re.compile(r"(?<!UN)\bLISTEN\b|\bNOTIFY\b", re.IGNORECASE)
_ADVISORY = re.compile(r"\bpg_(try_)?advisory_", re.IGNORECASE)
# A statement that *starts* with SET is the session-scoped `SET x = ...` command,
# which survives into the next acquire. `SET LOCAL` is transaction-scoped and
# always fine, and an `UPDATE ... SET col = ...` fragment starts with UPDATE.
_SESSION_SET = re.compile(r"^SET\s+(?!LOCAL\b)[a-z_][a-z0-9_.]*\s*(=|TO\b)", re.IGNORECASE)
# set_config(name, value, is_local); is_local=false is a session-scoped write.
_SESSION_SET_CONFIG = re.compile(r"set_config\([^)]*,\s*false\s*\)", re.IGNORECASE)


def _sql_literals(path: Path) -> list[str]:
    """Every string literal in a module except docstrings.

    f-strings are reassembled into one string (interpolations become a
    placeholder) rather than yielding each constant chunk separately: a query
    written as f"UPDATE {fq_table('banks')} SET config = ..." would otherwise
    look like a bare ``SET`` statement.

    Docstrings are excluded because this file's own prose, and the comments that
    explain the reset query, quote the very SQL being banned.
    """
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            doc = node.body[0] if node.body else None
            if isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant) and isinstance(doc.value.value, str):
                docstrings.add(id(doc.value))

    out: list[str] = []

    class _Collect(ast.NodeVisitor):
        def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                else:
                    parts.append("{}")
            out.append("".join(parts))
            # Do not recurse: the constant chunks were just consumed, and any
            # nested f-string inside an interpolation is a formatting detail.

        def visit_Constant(self, node: ast.Constant) -> None:
            if isinstance(node.value, str) and id(node) not in docstrings:
                out.append(node.value)

    _Collect().visit(tree)
    return out


def _api_modules() -> list[Path]:
    return [p for p in _API_ROOT.rglob("*.py") if "alembic" not in p.parts]


def _offenders(pattern: re.Pattern, allowed: set[Path], *, per_statement: bool = False) -> list[str]:
    """Literals matching ``pattern``, as ``path: snippet`` lines.

    ``per_statement`` splits each literal on ``;`` and anchors the match at the
    start of a statement — the only way to tell the session-scoped ``SET x = 1``
    command from the ``SET`` clause of an ``UPDATE``.
    """
    hits = []
    for path in _api_modules():
        if path in allowed:
            continue
        for literal in _sql_literals(path):
            fragments = [f.strip() for f in literal.split(";")] if per_statement else [literal]
            for fragment in fragments:
                if pattern.search(fragment):
                    hits.append(f"{path.relative_to(_API_ROOT.parent)}: {fragment.strip()[:120]}")
    return hits


class TestResetInvariants:
    def test_no_listen_or_notify(self):
        # `UNLISTEN *` in the reset query only matters if something LISTENs.
        assert _offenders(_LISTEN, set()) == []

    def test_no_advisory_locks_outside_migrations(self):
        # `pg_advisory_unlock_all()` in the reset query only matters if a pooled
        # connection can hold one. Advisory locks are banned project-wide anyway.
        assert _offenders(_ADVISORY, _ADVISORY_LOCK_ALLOWED) == []

    def test_no_session_scoped_set_outside_the_pool_setup(self):
        # `RESET ALL` in the reset query only matters for session-scoped writes.
        # Everything else must use SET LOCAL, which the transaction ends anyway.
        assert _offenders(_SESSION_SET, _SESSION_SET_ALLOWED, per_statement=True) == []
        assert _offenders(_SESSION_SET_CONFIG, _SESSION_SET_ALLOWED) == []
