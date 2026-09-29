"""Every migration that installs a maintenance routine must honour ownership.

The cross-schema discovery routines are installed by ``CREATE OR REPLACE``, so
the last migration to touch one wins. An installation that has replaced a
routine with its own implementation loses it the next time any migration
reinstalls it, silently, and only finds out from the resulting load.

``execute_unless_owned`` is the guard. This test is what keeps it applied: a new
migration that reaches for ``op.execute`` on one of these routines fails here
rather than in someone's production database months later.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from hindsight_api.alembic._owned import (
    ENV_EXTERNALLY_OWNED_ROUTINES,
    execute_unless_owned,
    externally_owned,
)

# The routines a deployment is allowed to own. Adding one here without also
# guarding its install sites makes this test fail, which is the intent.
OWNABLE = (
    "mental_models_with_cron",
    "banks_needing_consolidation",
    "schemas_with_expired_rows",
    "schemas_with_expired_operations",
)

VERSIONS = Path(__file__).resolve().parents[1] / "hindsight_api" / "alembic" / "versions"

_TOUCHES = re.compile(
    r"(CREATE OR REPLACE FUNCTION|DROP FUNCTION)[^;]*?\b(" + "|".join(OWNABLE) + r")\b",
    re.IGNORECASE | re.DOTALL,
)


def _unguarded_calls(path: Path) -> list[str]:
    """Names installed via a bare ``op.execute`` in this migration."""
    tree = ast.parse(path.read_text())
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_op_execute = (
            isinstance(func, ast.Attribute)
            and func.attr == "execute"
            and isinstance(func.value, ast.Name)
            and func.value.id == "op"
        )
        if not is_op_execute or not node.args:
            continue
        arg = node.args[0]
        sql = arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None
        if sql is None:
            # An f-string or a name: fall back to the raw segment so a formatted
            # CREATE OR REPLACE cannot slip through by not being a literal.
            sql = ast.get_source_segment(path.read_text(), arg) or ""
        m = _TOUCHES.search(sql)
        if m:
            bad.append(m.group(2))
    return bad


def _migration_files() -> list[Path]:
    return sorted(p for p in VERSIONS.glob("*.py") if p.name != "__init__.py")


def test_every_migration_that_installs_an_ownable_routine_uses_the_guard():
    offenders: dict[str, list[str]] = {}
    for path in _migration_files():
        names = _unguarded_calls(path)
        if names:
            offenders[path.name] = sorted(set(names))
    assert not offenders, (
        "These migrations install or drop an ownable maintenance routine with a bare "
        "op.execute, which silently overwrites a deployment's own implementation. "
        "Use execute_unless_owned(<name>, <sql>) instead:\n  "
        + "\n  ".join(f"{f}: {', '.join(n)}" for f, n in offenders.items())
    )


def test_the_guard_is_actually_applied_somewhere():
    """Guards against the above passing because the regex stopped matching."""
    guarded = sum(p.read_text().count("execute_unless_owned(") for p in _migration_files())
    assert guarded >= len(OWNABLE), (
        f"expected at least {len(OWNABLE)} guarded call sites, found {guarded} — "
        "if the routines were renamed, update OWNABLE"
    )


class TestOwnership:
    def test_unset_means_nothing_is_owned(self, monkeypatch):
        monkeypatch.delenv(ENV_EXTERNALLY_OWNED_ROUTINES, raising=False)
        for name in OWNABLE:
            assert externally_owned(name) is False

    def test_empty_means_nothing_is_owned(self, monkeypatch):
        monkeypatch.setenv(ENV_EXTERNALLY_OWNED_ROUTINES, "")
        assert externally_owned("mental_models_with_cron") is False

    @pytest.mark.parametrize(
        "raw",
        [
            "mental_models_with_cron",
            " mental_models_with_cron ",
            "banks_needing_consolidation,mental_models_with_cron",
            "banks_needing_consolidation, mental_models_with_cron",
            "mental_models_with_cron,,",
        ],
    )
    def test_parsing_tolerates_spacing_and_empties(self, monkeypatch, raw):
        monkeypatch.setenv(ENV_EXTERNALLY_OWNED_ROUTINES, raw)
        assert externally_owned("mental_models_with_cron") is True

    def test_only_the_named_routine_is_owned(self, monkeypatch):
        monkeypatch.setenv(ENV_EXTERNALLY_OWNED_ROUTINES, "mental_models_with_cron")
        assert externally_owned("mental_models_with_cron") is True
        assert externally_owned("banks_needing_consolidation") is False

    def test_execute_unless_owned_skips_when_owned(self, monkeypatch):
        monkeypatch.setenv(ENV_EXTERNALLY_OWNED_ROUTINES, "mental_models_with_cron")
        ran: list[str] = []
        monkeypatch.setattr(
            "hindsight_api.alembic._owned.op",
            type("_Op", (), {"execute": staticmethod(lambda sql: ran.append(sql))})(),
        )
        execute_unless_owned("mental_models_with_cron", "SELECT 1")
        assert ran == []

    def test_execute_unless_owned_runs_when_not_owned(self, monkeypatch):
        monkeypatch.delenv(ENV_EXTERNALLY_OWNED_ROUTINES, raising=False)
        ran: list[str] = []
        monkeypatch.setattr(
            "hindsight_api.alembic._owned.op",
            type("_Op", (), {"execute": staticmethod(lambda sql: ran.append(sql))})(),
        )
        execute_unless_owned("mental_models_with_cron", "SELECT 1")
        assert ran == ["SELECT 1"]

    def test_env_is_read_at_call_time_not_import_time(self, monkeypatch):
        """Migrations run long after import; a cached value would miss the setting."""
        monkeypatch.delenv(ENV_EXTERNALLY_OWNED_ROUTINES, raising=False)
        assert externally_owned("mental_models_with_cron") is False
        monkeypatch.setenv(ENV_EXTERNALLY_OWNED_ROUTINES, "mental_models_with_cron")
        assert externally_owned("mental_models_with_cron") is True
