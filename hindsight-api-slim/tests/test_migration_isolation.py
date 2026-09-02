"""HINDSIGHT_API_MIGRATION_ISOLATION decides where migrations run.

Alembic drives PostgreSQL through SQLAlchemy's sync engine, i.e. psycopg2, which has
no free-threaded build: importing it on a free-threaded interpreter re-enables the GIL
for the life of the process. A server that migrates on startup would spend the rest of
its life single-threaded, having done the damage before serving a request. "auto"
therefore isolates exactly there, and the explicit values let a deployment decide.
"""

import sysconfig
from unittest.mock import patch

import pytest

from hindsight_api import migrations
from hindsight_api.config import _parse_migration_isolation


def _isolates(mode: str, monkeypatch) -> bool:
    monkeypatch.setenv("HINDSIGHT_API_MIGRATION_ISOLATION", mode)
    monkeypatch.delenv(migrations._CHILD_MARKER, raising=False)
    # get_config caches, so read the decision through a config carrying this mode.
    with patch.object(migrations, "get_config") as get_config:
        get_config.return_value.migration_isolation = mode
        return migrations._should_isolate_migrations()


def test_always_isolates(monkeypatch):
    assert _isolates("always", monkeypatch) is True


def test_never_isolates(monkeypatch):
    assert _isolates("never", monkeypatch) is False


def test_auto_follows_the_interpreter(monkeypatch):
    """auto isolates only where psycopg2 would cost the process its free-threading."""
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    assert _isolates("auto", monkeypatch) is free_threaded


def test_child_never_recurses(monkeypatch):
    """The subprocess must run the migration, not spawn another one."""
    monkeypatch.setenv(migrations._CHILD_MARKER, "1")
    with patch.object(migrations, "get_config") as get_config:
        get_config.return_value.migration_isolation = "always"
        assert migrations._should_isolate_migrations() is False


@pytest.mark.parametrize("bad", ["", "yes", "subprocess", "Auto ", "1"])
def test_rejects_unknown_values(monkeypatch, bad):
    """Silently defaulting would run migrations in the wrong process, invisibly."""
    monkeypatch.setenv("HINDSIGHT_API_MIGRATION_ISOLATION", bad)
    if bad.strip().lower() in ("auto", "always", "never"):
        pytest.skip("valid after normalisation")
    with pytest.raises(ValueError, match="HINDSIGHT_API_MIGRATION_ISOLATION"):
        _parse_migration_isolation()


def test_defaults_to_auto(monkeypatch):
    monkeypatch.delenv("HINDSIGHT_API_MIGRATION_ISOLATION", raising=False)
    assert _parse_migration_isolation() == "auto"


def test_case_and_whitespace_are_tolerated(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_MIGRATION_ISOLATION", "  ALWAYS  ")
    assert _parse_migration_isolation() == "always"
