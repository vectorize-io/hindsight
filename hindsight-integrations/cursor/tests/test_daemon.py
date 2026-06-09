"""Tests for the cloud-default + useLocalDaemon resolution in lib.daemon.

Pinned by the V2 audit (2026-06-02) that flagged Goal-5 (Default to Cloud)
as failing because an empty ``hindsightApiUrl`` previously fell through to
the local daemon path rather than the hosted backend.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lib.config import DEFAULT_HINDSIGHT_API_URL
from lib.daemon import get_api_url


class TestGetApiUrlResolution:
    def test_explicit_url_wins(self):
        config = {"hindsightApiUrl": "http://my-server:9000", "apiPort": 9077}
        assert get_api_url(config) == "http://my-server:9000"

    def test_empty_url_no_local_server_no_useLocalDaemon_falls_back_to_cloud(self):
        """Goal-5 regression pin: no URL + no daemon opt-in → hosted backend."""
        config = {"hindsightApiUrl": "", "apiPort": 65535, "useLocalDaemon": False}
        with patch("lib.daemon._check_health", return_value=False):
            assert get_api_url(config) == DEFAULT_HINDSIGHT_API_URL

    def test_empty_url_with_running_local_server_uses_local(self):
        """Backward-compat: a developer who already started a daemon keeps it."""
        config = {"hindsightApiUrl": "", "apiPort": 9077, "useLocalDaemon": False}
        with patch("lib.daemon._check_health", return_value=True):
            assert get_api_url(config) == "http://127.0.0.1:9077"

    def test_useLocalDaemon_true_without_allow_daemon_start_still_falls_back(self):
        """Recall path (allow_daemon_start=False) never starts a daemon on its own."""
        config = {"hindsightApiUrl": "", "apiPort": 65535, "useLocalDaemon": True}
        with patch("lib.daemon._check_health", return_value=False):
            assert get_api_url(config, allow_daemon_start=False) == DEFAULT_HINDSIGHT_API_URL

    def test_useLocalDaemon_true_with_daemon_start_failure_falls_back_to_cloud(self):
        """If the opt-in daemon start fails, fall back to the hosted backend
        rather than hard-erroring — the plugin should keep working."""
        config = {"hindsightApiUrl": "", "apiPort": 65535, "useLocalDaemon": True}
        with (
            patch("lib.daemon._check_health", return_value=False),
            patch(
                "lib.daemon._ensure_daemon_running",
                side_effect=RuntimeError("hindsight-embed not found"),
            ),
        ):
            assert get_api_url(config, allow_daemon_start=True) == DEFAULT_HINDSIGHT_API_URL

    def test_env_var_HINDSIGHT_USE_LOCAL_DAEMON_takes_effect(self, monkeypatch):
        """HINDSIGHT_USE_LOCAL_DAEMON env var is in the load_config override map."""
        from lib.config import load_config

        monkeypatch.setenv("CURSOR_PLUGIN_ROOT", "/nonexistent")
        monkeypatch.setenv("HINDSIGHT_USE_LOCAL_DAEMON", "true")
        cfg = load_config()
        assert cfg["useLocalDaemon"] is True


class TestDocumentIdIsUniquePerRetain:
    """Pin the V2 audit fix: full-session retains used to share document_id=
    session_id and silently dropped earlier turns. Each retain now creates
    a distinct document so multi-turn sessions accumulate."""

    def test_retain_module_constructs_timestamp_suffixed_document_id(self):
        # Light-touch source pin — the document_id derivation is a single
        # f-string that mixes session_id with a millisecond wall clock.
        # We just confirm the source still has the expected shape; the
        # behavioural test is the live E2E retain roundtrip.
        src = open("scripts/retain.py").read()
        assert 'document_id = f"{session_id}-{int(time.time() * 1000)}"' in src
        assert "document_id = session_id" not in src
