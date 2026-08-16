"""Tests for daemonize() — subprocess.Popen re-exec instead of os.fork()."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest


def test_daemon_pid_receipt_tracks_current_process(monkeypatch, tmp_path):
    import json

    from hindsight_api.daemon import remove_daemon_pid_receipt, write_daemon_pid_receipt

    pid_path = tmp_path / "daemon.pid"
    monkeypatch.setenv("HINDSIGHT_EMBED_DAEMON_PID_FILE", str(pid_path))
    monkeypatch.setattr("hindsight_api.daemon._process_birth_marker", lambda _pid: "birth-a")

    assert write_daemon_pid_receipt() == pid_path
    assert json.loads(pid_path.read_text()) == {
        "version": 1,
        "pid": os.getpid(),
        "birth_marker": "birth-a",
    }

    remove_daemon_pid_receipt(pid_path)
    assert not pid_path.exists()


def test_daemon_pid_receipt_does_not_remove_replacement(monkeypatch, tmp_path):
    from hindsight_api.daemon import remove_daemon_pid_receipt

    pid_path = tmp_path / "daemon.pid"
    monkeypatch.setattr("hindsight_api.daemon._process_birth_marker", lambda _pid: "birth-a")
    pid_path.write_text('{"version":1,"pid":999,"birth_marker":"birth-b"}')

    remove_daemon_pid_receipt(pid_path)
    assert pid_path.exists()


def test_daemonize_parent_reexecs_via_popen(monkeypatch, tmp_path):
    """Parent path: daemonize() must spawn a child via subprocess.Popen and exit."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("_HINDSIGHT_DAEMON_CHILD", raising=False)
    monkeypatch.setattr(sys, "argv", ["hindsight-api", "--daemon", "--port", "9999"])

    log_path = tmp_path / "daemon.log"
    monkeypatch.setattr("hindsight_api.daemon.DAEMON_LOG_PATH", log_path)

    captured: dict = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        proc = MagicMock()
        proc.pid = 99999
        return proc

    with (
        patch("hindsight_api.daemon.subprocess.Popen", side_effect=fake_popen),
        pytest.raises(SystemExit) as exc_info,
    ):
        from hindsight_api.daemon import daemonize

        daemonize()

    assert exc_info.value.code == 0

    # Verify child command does NOT contain --daemon
    assert "--daemon" not in captured["cmd"]
    # Verify it uses the module entry point
    assert "-m" in captured["cmd"]
    assert "hindsight_api.main" in captured["cmd"]
    # Verify remaining args are preserved
    assert "--port" in captured["cmd"]
    assert "9999" in captured["cmd"]

    # Verify env has the daemon child marker
    env = captured["kwargs"]["env"]
    assert env["_HINDSIGHT_DAEMON_CHILD"] == "1"

    # Verify detach kwargs
    kwargs = captured["kwargs"]
    assert kwargs.get("start_new_session") is True


def test_daemonize_child_does_not_reexec(monkeypatch, tmp_path):
    """Child path: when _HINDSIGHT_DAEMON_CHILD=1, daemonize() does NOT call
    Popen — it only redirects stdio."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("_HINDSIGHT_DAEMON_CHILD", "1")

    log_path = tmp_path / "daemon.log"
    monkeypatch.setattr("hindsight_api.daemon.DAEMON_LOG_PATH", log_path)

    with (
        patch("hindsight_api.daemon.subprocess.Popen") as mock_popen,
        patch("hindsight_api.daemon._redirect_stdio_to_log") as mock_redirect,
    ):
        from hindsight_api.daemon import daemonize

        daemonize()
        mock_popen.assert_not_called()
        mock_redirect.assert_called_once()


def test_daemonize_windows_noop(monkeypatch, tmp_path):
    """On Windows, daemonize() just creates the log directory."""
    monkeypatch.setattr(sys, "platform", "win32")

    log_path = tmp_path / "subdir" / "daemon.log"
    monkeypatch.setattr("hindsight_api.daemon.DAEMON_LOG_PATH", log_path)

    with patch("hindsight_api.daemon.subprocess.Popen") as mock_popen:
        from hindsight_api.daemon import daemonize

        daemonize()
        mock_popen.assert_not_called()

    assert log_path.parent.exists()
