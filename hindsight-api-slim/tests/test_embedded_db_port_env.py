"""HINDSIGHT_EMBED_DB_PORT controls the embedded PostgreSQL port.

Loads ``pg0.py`` in isolation so the test exercises only the constructor's port
resolution without importing the full ``hindsight_api`` package or starting a
real PostgreSQL.
"""

import importlib.util
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("bad_port", ["0", "-1", "70000", "99999"])
def test_out_of_range_env_is_ignored(monkeypatch, bad_port):
    monkeypatch.setenv("HINDSIGHT_EMBED_DB_PORT", bad_port)
    assert EmbeddedPostgres().port is None


@pytest.mark.parametrize("edge_port", ["1", "65535"])
def test_valid_edge_ports_accepted(monkeypatch, edge_port):
    monkeypatch.setenv("HINDSIGHT_EMBED_DB_PORT", edge_port)
    assert EmbeddedPostgres().port == int(edge_port)
