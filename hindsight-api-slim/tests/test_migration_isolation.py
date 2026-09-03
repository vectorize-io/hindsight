"""HINDSIGHT_API_MIGRATION_ISOLATION decides where migrations run.

Alembic drives PostgreSQL through SQLAlchemy's sync engine, i.e. psycopg2, which has
no free-threaded build: importing it on a free-threaded interpreter re-enables the GIL
for the life of the process. A server that migrates on startup would spend the rest of
its life single-threaded, having done the damage before serving a request. "auto"
therefore isolates exactly there, and "true"/"false" let a deployment decide.
"""

import io
import json
import os
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


def test_true_isolates(monkeypatch):
    assert _isolates("true", monkeypatch) is True


def test_false_does_not_isolate(monkeypatch):
    assert _isolates("false", monkeypatch) is False


def test_auto_follows_the_interpreter(monkeypatch):
    """auto isolates only where psycopg2 would cost the process its free-threading."""
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    assert _isolates("auto", monkeypatch) is free_threaded


def test_child_never_recurses(monkeypatch):
    """The subprocess must run the migration, not spawn another one."""
    monkeypatch.setenv(migrations._CHILD_MARKER, "1")
    with patch.object(migrations, "get_config") as get_config:
        get_config.return_value.migration_isolation = "true"
        assert migrations._should_isolate_migrations() is False


@pytest.mark.parametrize("bad", ["", "yes", "no", "1", "0", "always", "never", "subprocess"])
def test_rejects_unknown_values(monkeypatch, bad):
    """Silently defaulting would run migrations in the wrong process, invisibly.

    "1"/"0"/"yes"/"no" are rejected on purpose: the flag is not a general bool parser,
    and the three spellings it does take are the ones documented.
    """
    monkeypatch.setenv("HINDSIGHT_API_MIGRATION_ISOLATION", bad)
    with pytest.raises(ValueError, match="HINDSIGHT_API_MIGRATION_ISOLATION"):
        _parse_migration_isolation()


def test_defaults_to_auto(monkeypatch):
    monkeypatch.delenv("HINDSIGHT_API_MIGRATION_ISOLATION", raising=False)
    assert _parse_migration_isolation() == "auto"


def test_case_and_whitespace_are_tolerated(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_MIGRATION_ISOLATION", "  TRUE  ")
    assert _parse_migration_isolation() == "true"


def test_payload_travels_on_stdin_not_argv():
    """A whole-fleet sweep names every tenant schema; argv would hit ARG_MAX.

    ``run_migrations_for_schemas`` is called with all schemas at once, and the
    entrypoint is documented at 20k of them — hundreds of KB of JSON, past ARG_MAX on
    macOS. It also must not be captured: an hour-long sweep would show nothing until
    it finished.
    """
    schemas = [f"tenant_{i:05d}" for i in range(20_000)]
    with patch.object(migrations.subprocess, "run") as run:
        run.return_value.returncode = 0
        migrations._run_in_migration_child("run_migrations_for_schemas", {"schemas": schemas})

    (argv,), kwargs = run.call_args
    assert argv[1:] == ["-m", "hindsight_api.migrations"], "the payload must not be an argument"
    assert json.loads(kwargs["input"])["kwargs"]["schemas"] == schemas
    assert "capture_output" not in kwargs and "stdout" not in kwargs, "child output must stream"
    assert kwargs["env"][migrations._CHILD_MARKER] == "1"


def test_child_failure_is_raised(monkeypatch):
    with patch.object(migrations.subprocess, "run") as run:
        run.return_value.returncode = 3
        with pytest.raises(RuntimeError, match="exit 3"):
            migrations._run_in_migration_child("run_migrations", {"database_url": "postgresql://x"})


def test_main_reads_stdin_and_dispatches(monkeypatch):
    """The child entrypoint reconstructs the exact call the parent asked for."""
    kwargs = {"database_url": "postgresql://x", "schema": "tenant_a"}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"target": "run_migrations", "kwargs": kwargs})))
    with patch.object(migrations, "run_migrations") as run_migrations_:
        migrations._main()
    run_migrations_.assert_called_once_with(**kwargs)
    assert os.environ[migrations._CHILD_MARKER] == "1"


def test_main_rejects_an_unknown_target(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"target": "drop_everything", "kwargs": {}})))
    with pytest.raises(SystemExit, match="drop_everything"):
        migrations._main()
