"""HINDSIGHT_EMBED_DB_PORT controls the embedded PostgreSQL port.

Loads ``pg0.py`` in isolation so the test exercises only the constructor's port
resolution without importing the full ``hindsight_api`` package or starting a
real PostgreSQL.
"""

import importlib.util
from pathlib import Path

_PG0 = Path(__file__).resolve().parents[1] / "hindsight_api" / "pg0.py"
_spec = importlib.util.spec_from_file_location("hindsight_api_pg0_under_test", _PG0)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EmbeddedPostgres = _mod.EmbeddedPostgres


def test_env_sets_port(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_EMBED_DB_PORT", "5544")
    assert EmbeddedPostgres().port == 5544


def test_explicit_port_overrides_env(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_EMBED_DB_PORT", "5544")
    assert EmbeddedPostgres(port=5999).port == 5999


def test_invalid_env_is_ignored(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_EMBED_DB_PORT", "not-a-port")
    assert EmbeddedPostgres().port is None


def test_unset_env_defaults_to_none(monkeypatch):
    monkeypatch.delenv("HINDSIGHT_EMBED_DB_PORT", raising=False)
    assert EmbeddedPostgres().port is None
