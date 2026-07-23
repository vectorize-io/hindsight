"""Tests for uvx daemon interpreter compatibility."""

from unittest.mock import patch

from lib import daemon


def test_uvx_defaults_to_python_313(monkeypatch):
    monkeypatch.delenv("UV_PYTHON", raising=False)
    with patch("lib.daemon.subprocess.run") as run:
        daemon._run_embed({}, ["status"])
    assert run.call_args.kwargs["env"]["UV_PYTHON"] == "3.13"


def test_uvx_preserves_explicit_python_override(monkeypatch):
    with patch("lib.daemon.subprocess.run") as run:
        daemon._run_embed({}, ["status"], env={"UV_PYTHON": "3.12"})
    assert run.call_args.kwargs["env"]["UV_PYTHON"] == "3.12"


def test_uvx_replaces_blank_python_override(monkeypatch):
    with patch("lib.daemon.subprocess.run") as run:
        daemon._run_embed({}, ["status"], env={"UV_PYTHON": "  "})
    assert run.call_args.kwargs["env"]["UV_PYTHON"] == "3.13"


def test_development_embed_does_not_pin_python(monkeypatch):
    monkeypatch.delenv("UV_PYTHON", raising=False)
    with patch("lib.daemon.subprocess.run") as run:
        daemon._run_embed({"embedPackagePath": "/tmp/hindsight-embed"}, ["status"])
    assert "UV_PYTHON" not in run.call_args.kwargs["env"]


def test_background_prestart_passes_python_313_to_uvx(monkeypatch):
    monkeypatch.delenv("UV_PYTHON", raising=False)
    with (
        patch("lib.daemon._check_health", return_value=False),
        patch("lib.daemon._is_embed_available", return_value=True),
        patch("lib.daemon.detect_llm_config", return_value={}),
        patch("lib.daemon.get_llm_env_vars", return_value={}),
        patch("lib.daemon.subprocess.Popen") as popen,
    ):
        daemon.prestart_daemon_background({})
    assert popen.call_args.kwargs["env"]["UV_PYTHON"] == "3.13"
