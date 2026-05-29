"""Tests for daemon mode host/port resolution (resolve_daemon_host_port).

Verifies that --daemon honors explicit --host / HINDSIGHT_API_HOST overrides
instead of unconditionally pinning to 127.0.0.1.

See: https://github.com/vectorize-io/hindsight/issues/1402
"""

from hindsight_api.daemon import DEFAULT_DAEMON_PORT
from hindsight_api.main import _parse_cli_args, resolve_daemon_host_port

# Default config values matching production defaults
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8888


class _Config:
    host = DEFAULT_HOST
    port = DEFAULT_PORT
    log_level = "info"


class TestResolveDaemonHostPort:
    """Test resolve_daemon_host_port under various override scenarios."""

    def test_defaults_to_localhost_when_no_override(self, monkeypatch):
        """With no explicit host setting, daemon should bind to 127.0.0.1."""
        monkeypatch.delenv("HINDSIGHT_API_HOST", raising=False)
        resolved = resolve_daemon_host_port(
            args_host=DEFAULT_HOST,
            args_port=DEFAULT_PORT,
            explicit_host=False,
            explicit_port=False,
        )
        assert resolved.host == "127.0.0.1"
        assert resolved.port == DEFAULT_DAEMON_PORT

    def test_honors_cli_host_flag(self, monkeypatch):
        """--host 0.0.0.0 --daemon should bind to 0.0.0.0, not 127.0.0.1."""
        monkeypatch.delenv("HINDSIGHT_API_HOST", raising=False)
        resolved = resolve_daemon_host_port(
            args_host="0.0.0.0",
            args_port=DEFAULT_PORT,
            explicit_host=True,
            explicit_port=False,
        )
        assert resolved.host == "0.0.0.0"

    def test_honors_cli_host_flag_matching_config_default(self, monkeypatch):
        """An explicit --host should be honored even when it matches config."""
        monkeypatch.delenv("HINDSIGHT_API_HOST", raising=False)
        resolved = resolve_daemon_host_port(
            args_host=DEFAULT_HOST,
            args_port=DEFAULT_PORT,
            explicit_host=True,
            explicit_port=False,
        )
        assert resolved.host == DEFAULT_HOST

    def test_honors_env_var_host(self, monkeypatch):
        """HINDSIGHT_API_HOST=0.0.0.0 --daemon should bind to 0.0.0.0."""
        monkeypatch.setenv("HINDSIGHT_API_HOST", "0.0.0.0")
        # When env var is set, config.host already reflects it, so
        # args_host matches the config default, but the env var presence is the signal.
        resolved = resolve_daemon_host_port(
            args_host="0.0.0.0",
            args_port=DEFAULT_PORT,
            explicit_host=False,
            explicit_port=False,
        )
        assert resolved.host == "0.0.0.0"

    def test_honors_custom_host_via_env(self, monkeypatch):
        """HINDSIGHT_API_HOST=10.0.0.5 should be respected in daemon mode."""
        monkeypatch.setenv("HINDSIGHT_API_HOST", "10.0.0.5")
        resolved = resolve_daemon_host_port(
            args_host="10.0.0.5",
            args_port=DEFAULT_PORT,
            explicit_host=False,
            explicit_port=False,
        )
        assert resolved.host == "10.0.0.5"

    def test_cli_flag_overrides_env_var(self, monkeypatch):
        """--host flag should take precedence over env var."""
        monkeypatch.setenv("HINDSIGHT_API_HOST", "10.0.0.5")
        resolved = resolve_daemon_host_port(
            args_host="192.168.1.1",
            args_port=DEFAULT_PORT,
            explicit_host=True,
            explicit_port=False,
        )
        assert resolved.host == "192.168.1.1"

    def test_custom_port_preserved(self, monkeypatch):
        """--port 9999 --daemon should keep 9999, not switch to daemon default."""
        monkeypatch.delenv("HINDSIGHT_API_HOST", raising=False)
        resolved = resolve_daemon_host_port(
            args_host=DEFAULT_HOST,
            args_port=9999,
            explicit_host=False,
            explicit_port=True,
        )
        assert resolved.port == 9999

    def test_explicit_port_matching_config_is_preserved(self, monkeypatch):
        """--port should be honored even when it matches HINDSIGHT_API_PORT."""
        monkeypatch.delenv("HINDSIGHT_API_HOST", raising=False)
        resolved = resolve_daemon_host_port(
            args_host=DEFAULT_HOST,
            args_port=9555,
            explicit_host=False,
            explicit_port=True,
        )
        assert resolved.port == 9555

    def test_default_port_becomes_daemon_port(self, monkeypatch):
        """No --port flag in daemon mode should use DEFAULT_DAEMON_PORT."""
        monkeypatch.delenv("HINDSIGHT_API_HOST", raising=False)
        resolved = resolve_daemon_host_port(
            args_host=DEFAULT_HOST,
            args_port=DEFAULT_PORT,
            explicit_host=False,
            explicit_port=False,
        )
        assert resolved.port == DEFAULT_DAEMON_PORT

    def test_env_port_without_cli_still_uses_daemon_default(self, monkeypatch):
        """HINDSIGHT_API_PORT alone does not override daemon mode's default."""
        monkeypatch.delenv("HINDSIGHT_API_HOST", raising=False)
        resolved = resolve_daemon_host_port(
            args_host=DEFAULT_HOST,
            args_port=9555,
            explicit_host=False,
            explicit_port=False,
        )
        assert resolved.port == DEFAULT_DAEMON_PORT

    def test_argparse_marks_abbreviated_port_as_explicit(self, monkeypatch):
        """Argparse-accepted --po should be treated like explicit --port."""
        monkeypatch.delenv("HINDSIGHT_API_HOST", raising=False)
        config = _Config()
        config.port = 9555
        parsed = _parse_cli_args(["--daemon", "--po", "9555"], config)
        assert parsed.args.port == 9555
        assert parsed.explicit_port is True

        resolved = resolve_daemon_host_port(
            args_host=parsed.args.host,
            args_port=parsed.args.port,
            explicit_host=parsed.explicit_host,
            explicit_port=parsed.explicit_port,
        )
        assert resolved.port == 9555

    def test_argparse_marks_abbreviated_host_as_explicit(self, monkeypatch):
        """Argparse-accepted --ho should be treated like explicit --host."""
        monkeypatch.delenv("HINDSIGHT_API_HOST", raising=False)
        config = _Config()
        parsed = _parse_cli_args(["--daemon", "--ho", DEFAULT_HOST], config)
        assert parsed.args.host == DEFAULT_HOST
        assert parsed.explicit_host is True

        resolved = resolve_daemon_host_port(
            args_host=parsed.args.host,
            args_port=parsed.args.port,
            explicit_host=parsed.explicit_host,
            explicit_port=parsed.explicit_port,
        )
        assert resolved.host == DEFAULT_HOST
