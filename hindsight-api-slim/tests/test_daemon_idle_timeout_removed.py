"""The daemon no longer auto-exits when idle.

Idleness was measured from the last request *start*, so a retain/reflect call
that ran longer than the timeout was SIGTERM'd mid-flight.  The whole mechanism
was removed; ``--idle-timeout`` stays parseable (older launchers still pass it)
but is inert.

See: https://github.com/vectorize-io/hindsight/issues/3903
"""

import hindsight_api.daemon as daemon_module
import hindsight_api.main as main_module
from hindsight_api.main import _parse_cli_args


class _Config:
    host = "0.0.0.0"
    port = 8888
    log_level = "info"


def test_idle_timeout_flag_is_still_accepted():
    """hindsight-embed and the coding-agent integrations pass this flag."""
    parsed = _parse_cli_args(["--daemon", "--idle-timeout", "300"], _Config())
    assert parsed.args.idle_timeout == 300


def test_idle_timeout_defaults_to_zero():
    assert _parse_cli_args(["--daemon"], _Config()).args.idle_timeout == 0


def test_no_idle_timeout_machinery_remains():
    """A non-zero value must not resurrect a shutdown path anywhere."""
    assert not hasattr(daemon_module, "IdleTimeoutMiddleware")
    assert not hasattr(daemon_module, "DEFAULT_IDLE_TIMEOUT")
    assert "idle_middleware" not in main_module.main.__code__.co_varnames
