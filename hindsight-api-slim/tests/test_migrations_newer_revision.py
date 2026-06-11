"""Newer-bank rolling-deployment handling in _run_migrations_internal.

command.upgrade() does not raise ResolutionError directly: alembic's
ScriptDirectory._catch_revision_errors wraps it in CommandError. The
rolling-deployment skip must therefore handle the wrapped form too
(github issue #2114).
"""

import logging

import pytest
from alembic.script.revision import ResolutionError
from alembic.util.exc import CommandError

from hindsight_api import migrations

DB_URL = "postgresql://user:pass@localhost/db"


def _raise_wrapped_resolution_error(_cfg, _revision):
    # Mirrors alembic's _catch_revision_errors wrapping.
    try:
        raise ResolutionError("No such revision or branch 'c1d2e3f4a5b6'", "c1d2e3f4a5b6")
    except ResolutionError as err:
        raise CommandError("Can't locate revision identified by 'c1d2e3f4a5b6'") from err


def test_wrapped_resolution_error_skips_migrations(monkeypatch, caplog):
    monkeypatch.setattr(migrations.command, "upgrade", _raise_wrapped_resolution_error)
    with caplog.at_level(logging.WARNING):
        migrations._run_migrations_internal(DB_URL, "/tmp/alembic")
    assert "newer migration revision" in caplog.text


def test_bare_resolution_error_still_skips_migrations(monkeypatch, caplog):
    def raise_resolution_error(_cfg, _revision):
        raise ResolutionError("No such revision or branch 'c1d2e3f4a5b6'", "c1d2e3f4a5b6")

    monkeypatch.setattr(migrations.command, "upgrade", raise_resolution_error)
    with caplog.at_level(logging.WARNING):
        migrations._run_migrations_internal(DB_URL, "/tmp/alembic")
    assert "newer migration revision" in caplog.text


def test_unrelated_command_error_propagates(monkeypatch):
    def raise_command_error(_cfg, _revision):
        raise CommandError("Path doesn't exist: '/tmp/alembic'")

    monkeypatch.setattr(migrations.command, "upgrade", raise_command_error)
    with pytest.raises(CommandError):
        migrations._run_migrations_internal(DB_URL, "/tmp/alembic")
