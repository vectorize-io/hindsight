"""Tests for uvx daemon interpreter compatibility and Devin-specific isolation.

The first group is ported verbatim from the Claude Code plugin — `daemon.py`'s
uvx handling is shared logic and must not drift. The second group pins the two
values that deliberately *do* differ, so the two plugins can each auto-manage a
local daemon on one machine without fighting over a port or a profile.
"""

import os
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from lib import daemon
from lib.state import read_state, write_state

# A successful `daemon stop`. _run_embed() returns a CompletedProcess and
# stop_daemon() reads its returncode, so a bare MagicMock would compare unequal
# to 0 and quietly send every one of these tests down the failed-stop repair
# path — including a real health probe against port 9078.
_STOP_OK = SimpleNamespace(returncode=0, stdout="stopped", stderr="")
_STOP_FAILED = SimpleNamespace(returncode=1, stdout="", stderr="could not stop daemon")


def test_uvx_defaults_to_python_313(monkeypatch):
    monkeypatch.delenv("UV_PYTHON", raising=False)
    with patch("lib.daemon.subprocess.run") as run:
        daemon._run_embed({}, ["status"])
    assert run.call_args.kwargs["env"]["UV_PYTHON"] == "3.13"


def test_uvx_preserves_explicit_python_override():
    with patch("lib.daemon.subprocess.run") as run:
        daemon._run_embed({}, ["status"], env={"UV_PYTHON": "3.12"})
    assert run.call_args.kwargs["env"]["UV_PYTHON"] == "3.12"


def test_uvx_replaces_blank_python_override():
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


class TestDevinCliDaemonIsolation:
    """This plugin must not collide with a co-installed Claude Code plugin.

    Both plugins can auto-manage their own `hindsight-embed` daemon. If they
    shared a profile name or a default port, whichever started second would
    either reconfigure the other's profile or trip `_clear_port()` and SIGTERM
    a live daemon out from under the other agent. Sharing memory is supported,
    but via a shared `hindsightApiUrl` — not by accident.
    """

    def test_profile_name_is_devin_cli(self):
        assert daemon.PROFILE_NAME == "devin-cli", (
            "profile must not be 'claude-code' — a shared profile lets one plugin's config overwrite the other's"
        )

    def test_default_port_is_9078_not_claude_codes_9077(self):
        """No apiPort configured → 9078, so a local CC daemon on 9077 is untouched."""
        with patch("lib.daemon._check_health", return_value=True) as health:
            url = daemon.get_api_url({})
        assert url == "http://127.0.0.1:9078"
        assert "9078" in health.call_args[0][0]

    def test_prestart_uses_the_same_default_port(self):
        """get_api_url and prestart must agree, or prestart warms the wrong port."""
        with patch("lib.daemon._check_health", return_value=True) as health:
            daemon.prestart_daemon_background({})
        assert health.call_args[0][0] == "http://127.0.0.1:9078"

    def test_explicit_api_port_overrides_the_default(self):
        with patch("lib.daemon._check_health", return_value=True):
            url = daemon.get_api_url({"apiPort": 9077})
        assert url == "http://127.0.0.1:9077"

    def test_external_api_url_skips_the_daemon_entirely(self):
        """Mode 1: no health check, no daemon start — just use the URL."""
        with patch("lib.daemon._check_health") as health:
            url = daemon.get_api_url({"hindsightApiUrl": "https://hs.example.com"})
        assert url == "https://hs.example.com"
        health.assert_not_called()

    def test_prestart_is_a_noop_in_external_api_mode(self):
        with patch("lib.daemon._check_health") as health:
            daemon.prestart_daemon_background({"hindsightApiUrl": "https://hs.example.com"})
        health.assert_not_called()

    def test_recall_path_refuses_to_start_a_cold_daemon(self):
        """allow_daemon_start=False must raise, not block the hook on a 30s start."""
        import pytest

        with patch("lib.daemon._check_health", return_value=False):
            with pytest.raises(RuntimeError, match="9078"):
                daemon.get_api_url({}, allow_daemon_start=False)


class TestSecretsStayOutOfArgv:
    """`profile create --env KEY=VALUE` puts its arguments in the process listing.

    hindsight-embed's daemon manager copies its own os.environ when it spawns the
    API server, and both call sites already pass the key in the subprocess env,
    so the key reaches the daemon without ever appearing on a command line.
    """

    def test_api_key_is_not_passed_as_a_command_line_argument(self):
        args = daemon._profile_env_args(
            {
                "HINDSIGHT_API_LLM_PROVIDER": "openai",
                "HINDSIGHT_API_LLM_API_KEY": "sk-secret",
            }
        )

        assert "sk-secret" not in " ".join(args)
        assert "--env" in args and "HINDSIGHT_API_LLM_PROVIDER=openai" in args

    def test_empty_values_are_dropped(self):
        assert daemon._profile_env_args({"HINDSIGHT_API_LLM_MODEL": ""}) == []


class TestEmbedAvailabilityChecksTheRealCommand:
    def test_standalone_hindsight_embed_without_uvx_is_not_available(self):
        """`_get_embed_command()` always uses uvx when embedPackagePath is unset.

        Accepting a bare `hindsight-embed` on PATH made the preflight pass and
        then startup fail on a missing uvx.
        """
        with patch("lib.daemon.shutil.which", side_effect=lambda name: None if name == "uvx" else "/usr/bin/" + name):
            assert daemon._is_embed_available({}) is False

    def test_embed_package_path_requires_uv(self, tmp_path):
        with patch("lib.daemon.shutil.which", return_value=None):
            assert daemon._is_embed_available({"embedPackagePath": str(tmp_path)}) is False

    def test_embed_package_path_with_uv_present_is_available(self, tmp_path):
        with patch("lib.daemon.shutil.which", return_value="/usr/bin/uv"):
            assert daemon._is_embed_available({"embedPackagePath": str(tmp_path)}) is True


class TestPrestartClaimsOwnershipSoSessionEndCanStop:
    """A pre-started daemon that no one owns outlives every session.

    `stop_daemon` only acts on a `started_by_plugin` marker. The synchronous
    start path writes it once health confirms readiness; the background
    pre-start returns immediately and used to write nothing, so SessionEnd
    skipped the stop and the daemon leaked.
    """

    def _prestart(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
        with (
            patch("lib.daemon._check_health", return_value=False),
            patch("lib.daemon._is_embed_available", return_value=True),
            patch("lib.daemon.detect_llm_config", return_value={"provider": "openai"}),
            patch("lib.daemon.get_llm_env_vars", return_value={}),
            patch("lib.daemon.subprocess.Popen") as popen,
        ):
            daemon.prestart_daemon_background({})
        return popen

    def test_background_prestart_records_plugin_ownership(self, tmp_path, monkeypatch):
        popen = self._prestart(tmp_path, monkeypatch)

        popen.assert_called_once()
        state = read_state(daemon.DAEMON_STATE_FILE)
        assert state is not None, "no state written — SessionEnd would skip the stop"
        assert state["started_by_plugin"] is True
        assert state["port"] == 9078

    def test_session_end_then_stops_the_daemon_it_started(self, tmp_path, monkeypatch):
        self._prestart(tmp_path, monkeypatch)

        with patch("lib.daemon._run_embed", return_value=_STOP_OK) as run_embed:
            daemon.stop_daemon({})

        run_embed.assert_called_once()
        assert run_embed.call_args[0][1] == ["daemon", "--profile", daemon.PROFILE_NAME, "stop"]

    def test_no_ownership_is_claimed_when_a_daemon_is_already_running(self, tmp_path, monkeypatch):
        """The health check returns early — that daemon is someone else's."""
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))

        with patch("lib.daemon._check_health", return_value=True):
            daemon.prestart_daemon_background({})

        assert read_state(daemon.DAEMON_STATE_FILE) is None


class TestConcurrentSessionsShareOneDaemon:
    """One daemon serves every concurrent Devin CLI session.

    Its lifecycle used to hang off a single global marker, so the first session
    to end stopped the daemon out from under the others — and they had no way
    to notice until their next hook failed.
    """

    def _prestart(self, tmp_path, monkeypatch, session_id):
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
        with (
            patch("lib.daemon._check_health", return_value=False),
            patch("lib.daemon._is_embed_available", return_value=True),
            patch("lib.daemon.detect_llm_config", return_value={"provider": "openai"}),
            patch("lib.daemon.get_llm_env_vars", return_value={}),
            patch("lib.daemon.subprocess.Popen"),
        ):
            daemon.prestart_daemon_background({}, session_id=session_id)

    def test_the_first_session_to_end_does_not_stop_a_shared_daemon(self, tmp_path, monkeypatch):
        self._prestart(tmp_path, monkeypatch, "sess-a")
        daemon.register_session("sess-b")

        with patch("lib.daemon._run_embed", return_value=_STOP_OK) as run_embed:
            daemon.stop_daemon({}, session_id="sess-a")

        run_embed.assert_not_called()
        assert daemon._registered_sessions(read_state(daemon.DAEMON_STATE_FILE, {})) == ["sess-b"]

    def test_the_last_session_to_end_stops_it(self, tmp_path, monkeypatch):
        self._prestart(tmp_path, monkeypatch, "sess-a")
        daemon.register_session("sess-b")

        with patch("lib.daemon._run_embed", return_value=_STOP_OK):
            daemon.stop_daemon({}, session_id="sess-a")
        with patch("lib.daemon._run_embed", return_value=_STOP_OK) as run_embed:
            daemon.stop_daemon({}, session_id="sess-b")

        run_embed.assert_called_once()
        assert run_embed.call_args[0][1] == ["daemon", "--profile", daemon.PROFILE_NAME, "stop"]

    def test_a_solo_session_still_stops_its_own_daemon(self, tmp_path, monkeypatch):
        self._prestart(tmp_path, monkeypatch, "sess-a")

        with patch("lib.daemon._run_embed", return_value=_STOP_OK) as run_embed:
            daemon.stop_daemon({}, session_id="sess-a")

        run_embed.assert_called_once()

    def test_ownership_is_released_before_the_stop_command_runs(self, tmp_path, monkeypatch):
        """A session arriving mid-stop must not attach to a daemon being killed.

        `daemon stop` runs outside the interprocess lock on purpose — it is
        allowed 10s and SessionStart only 5s, so holding the lock across it
        would trade this race for a guaranteed hook timeout. What closes the
        race instead is that *ownership* is given up in the same locked write
        that decides to stop, so by the time the stop command runs there is no
        marker left for a new session to be counted against.

        The late session's id is still recorded. Ownership is what gates the
        stop decision, not the presence of an id — and if the stop fails, that
        id is the only record that a live session is using the daemon being
        handed back. See test_a_session_that_arrives_mid_stop_survives_a_failed_stop.
        """
        self._prestart(tmp_path, monkeypatch, "sess-a")
        seen = {}

        def _register_midway(*_args, **_kwargs):
            daemon.register_session("sess-late")
            seen["state"] = read_state(daemon.DAEMON_STATE_FILE, {})
            return _STOP_OK

        with patch("lib.daemon._run_embed", side_effect=_register_midway):
            daemon.stop_daemon({}, session_id="sess-a")

        assert not seen["state"].get("started_by_plugin"), "a session registered against a daemon already being stopped"
        assert seen["state"].get("sessions") == ["sess-late"], "the late session left no record of itself"

    def test_a_session_that_is_still_running_keeps_its_daemon_owned(self, tmp_path, monkeypatch):
        """Control: releasing ownership is only correct on the stop path."""
        self._prestart(tmp_path, monkeypatch, "sess-a")
        daemon.register_session("sess-b")

        with patch("lib.daemon._run_embed", return_value=_STOP_OK):
            daemon.stop_daemon({}, session_id="sess-a")

        state = read_state(daemon.DAEMON_STATE_FILE, {})
        assert state.get("started_by_plugin") is True
        assert state.get("port") == 9078

    def test_a_failed_stop_hands_ownership_back(self, tmp_path, monkeypatch):
        """Otherwise a daemon that refused to die belongs to nobody, forever.

        Ownership is released before the stop runs, so if the stop then fails
        the daemon keeps running with no `started_by_plugin` marker —
        register_session() declines, and no later SessionEnd ever retries.
        """
        self._prestart(tmp_path, monkeypatch, "sess-a")
        failed = SimpleNamespace(returncode=1, stdout="", stderr="no such daemon")

        with (
            patch("lib.daemon._run_embed", return_value=failed),
            patch("lib.daemon._check_health", return_value=True),
        ):
            daemon.stop_daemon({}, session_id="sess-a")

        state = read_state(daemon.DAEMON_STATE_FILE, {})
        assert state.get("started_by_plugin") is True
        assert state.get("port") == 9078
        # The sessions that were registered have all ended; what comes back is
        # the marker, not a stale roster.
        assert daemon._registered_sessions(state) == []

        # And the restored marker is usable: a new session can register against
        # it, so its SessionEnd retries the stop.
        daemon.register_session("sess-b")
        assert daemon._registered_sessions(read_state(daemon.DAEMON_STATE_FILE, {})) == ["sess-b"]

    def test_a_nonzero_exit_counts_as_a_failed_stop(self, tmp_path, monkeypatch):
        """_run_embed() does not pass check=True, so a failure is a returncode.

        Treating any returned CompletedProcess as success meant the common
        failure mode — the daemon reporting it could not stop — was read as a
        clean shutdown.
        """
        self._prestart(tmp_path, monkeypatch, "sess-a")
        failed = SimpleNamespace(returncode=1, stdout="", stderr="boom")

        with (
            patch("lib.daemon._run_embed", return_value=failed),
            patch("lib.daemon._check_health", return_value=True) as health,
        ):
            daemon.stop_daemon({}, session_id="sess-a")

        health.assert_called_once()
        # Well under _check_health()'s 10s default: SessionEnd only gets 10s in
        # total and `daemon stop` may already have spent it.
        assert health.call_args.kwargs["timeout"] == 2

    def test_a_daemon_that_did_die_is_not_reclaimed(self, tmp_path, monkeypatch):
        """A nonzero exit does not prove the daemon survived.

        Restoring the marker unconditionally would leave one pointing at a port
        with nothing behind it, and the next SessionEnd would try to stop a
        daemon that is already gone.
        """
        self._prestart(tmp_path, monkeypatch, "sess-a")
        failed = SimpleNamespace(returncode=1, stdout="", stderr="boom")

        with (
            patch("lib.daemon._run_embed", return_value=failed),
            patch("lib.daemon._check_health", return_value=False),
        ):
            daemon.stop_daemon({}, session_id="sess-a")

        assert read_state(daemon.DAEMON_STATE_FILE, {}) == {}

    def test_a_reclaim_does_not_clobber_a_newer_daemon(self, tmp_path, monkeypatch):
        """A fresh daemon may have claimed the port while the stop was running.

        That state belongs to a live session; overwriting it with the dead
        daemon's would drop that session from the registry and strand it.
        """
        self._prestart(tmp_path, monkeypatch, "sess-a")
        newer = {"port": 9078, "started_by_plugin": True, "sessions": ["sess-new"]}

        def _restart_midway(*_args, **_kwargs):
            write_state(daemon.DAEMON_STATE_FILE, newer)
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")

        with (
            patch("lib.daemon._run_embed", side_effect=_restart_midway),
            patch("lib.daemon._check_health", return_value=True),
        ):
            daemon.stop_daemon({}, session_id="sess-a")

        assert read_state(daemon.DAEMON_STATE_FILE, {}) == newer

    def test_registering_confers_no_ownership_when_the_daemon_is_not_ours(self, tmp_path, monkeypatch):
        """An externally-started daemon must not become plugin-owned by registration.

        The id is recorded — see register_session()'s docstring for why that has
        to happen even with no marker present — but recording it must not be
        mistaken for owning the daemon, because ownership is what stop_daemon()
        reads to decide it may kill the process.
        """
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))

        daemon.register_session("sess-a")

        state = read_state(daemon.DAEMON_STATE_FILE, {})
        assert not state.get("started_by_plugin"), "an external daemon was claimed by a mere registration"

        # And the stop path agrees: no marker, no stop.
        with patch("lib.daemon._run_embed") as run:
            daemon.stop_daemon({}, session_id="sess-a")
        run.assert_not_called()

    def test_state_from_an_older_plugin_version_still_stops(self, tmp_path, monkeypatch):
        """No `sessions` key means nobody registered — keep the old behaviour."""
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
        write_state(daemon.DAEMON_STATE_FILE, {"port": 9078, "started_by_plugin": True})

        with patch("lib.daemon._run_embed", return_value=_STOP_OK) as run_embed:
            daemon.stop_daemon({}, session_id="sess-a")

        run_embed.assert_called_once()


class TestMalformedDaemonStateDoesNotBreakSessionEnd:
    @pytest.mark.parametrize("bad_state", [None, [], "a string"])
    def test_non_dict_daemon_state_is_treated_as_no_daemon(self, tmp_path, monkeypatch, bad_state):
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
        write_state(daemon.DAEMON_STATE_FILE, bad_state)

        with patch("lib.daemon._run_embed", return_value=_STOP_OK) as run_embed:
            daemon.stop_daemon({}, session_id="sess-a")

        run_embed.assert_not_called()


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))

# Each child widens the read-modify-write window from microseconds to 300ms by
# slowing the write, so a lost update is certain without the lock rather than
# dependent on process-scheduling luck. The sleep sits inside the locked region,
# so with the lock the children simply serialise.
_CHILD = """
import os, sys, time
sys.path.insert(0, {scripts!r})
from lib import state
_real_write = state.write_state
def _slow_write(name, data):
    time.sleep(0.3)
    return _real_write(name, data)
state.write_state = _slow_write
from lib.daemon import {fn}
{setup}
while not os.path.exists({go!r}):
    time.sleep(0.01)
{call}
"""


def _race(tmp_path, session_ids, fn, call_tmpl, setup=""):
    """Run `fn` concurrently in real subprocesses, released by a shared go-file."""
    go = str(tmp_path / "go")
    env = {**os.environ, "HINDSIGHT_DEVIN_CLI_DATA_DIR": str(tmp_path)}
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _CHILD.format(
                    scripts=SCRIPTS_DIR,
                    go=go,
                    fn=fn,
                    setup=setup,
                    call=call_tmpl.format(sid=repr(sid)),
                ),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for sid in session_ids
    ]
    time.sleep(0.5)  # let every child reach the barrier
    open(go, "w").close()
    for p in procs:
        _, err = p.communicate(timeout=30)
        assert p.returncode == 0, err.decode()


class TestConcurrentRegistrationIsNotLost:
    """register_session() does a read-modify-write on shared daemon state.

    Unlocked, two SessionStart hooks both read the same list, each append only
    their own id, and the later write drops the earlier session — whose
    SessionEnd then sees an empty registry and stops the daemon while that
    session is still working.

    Driven through real subprocesses because the guarantee is an interprocess
    one (flock); threads in one interpreter would not prove it.
    """

    def test_no_registration_is_lost_under_concurrency(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
        write_state(
            daemon.DAEMON_STATE_FILE,
            {"port": 9078, "started_by_plugin": True, "sessions": []},
        )
        ids = [f"sess-{i}" for i in range(4)]

        _race(tmp_path, ids, "register_session", "register_session({sid})")

        registered = daemon._registered_sessions(read_state(daemon.DAEMON_STATE_FILE, {}))
        assert sorted(registered) == sorted(ids)

    def test_no_deregistration_is_lost_under_concurrency(self, tmp_path, monkeypatch):
        """Two SessionEnds racing must not each drop the other and both stop."""
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
        ids = [f"sess-{i}" for i in range(4)]
        write_state(
            daemon.DAEMON_STATE_FILE,
            {"port": 9078, "started_by_plugin": True, "sessions": [*ids, "sess-survivor"]},
        )

        _race(tmp_path, ids, "stop_daemon", "stop_daemon({{}}, session_id={sid})")

        registered = daemon._registered_sessions(read_state(daemon.DAEMON_STATE_FILE, {}))
        assert registered == ["sess-survivor"], "a concurrent SessionEnd dropped the wrong id"


# Neutralise everything prestart does besides claiming ownership: the health
# probe (so each child believes it must start a daemon), the availability and
# LLM checks (so it gets that far), and the launch itself.
_NO_DAEMON = """
from lib import daemon
daemon._check_health = lambda *a, **k: False
daemon._is_embed_available = lambda *a, **k: True
daemon.detect_llm_config = lambda *a, **k: {}
daemon.get_llm_env_vars = lambda *a, **k: {}
daemon.subprocess.Popen = lambda *a, **k: None
"""


class TestConcurrentPrestartIsNotLost:
    """SessionStart's background pre-start claims the daemon — without erasing it.

    Every child here sees no daemon on the port, because that is the real
    situation: pre-start returns before the daemon listens, so the next
    session's health check races it and also decides to start one. Writing the
    ownership marker straight over the state at that moment leaves one id in a
    registry that should hold four, and the three erased sessions are then
    stopped out from under while they are still working.
    """

    def test_no_prestart_registration_is_lost_under_concurrency(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
        ids = [f"sess-{i}" for i in range(4)]

        _race(
            tmp_path,
            ids,
            "prestart_daemon_background",
            "prestart_daemon_background({{}}, session_id={sid})",
            setup=_NO_DAEMON,
        )

        state_after = read_state(daemon.DAEMON_STATE_FILE, {})
        assert state_after.get("started_by_plugin") is True
        assert sorted(daemon._registered_sessions(state_after)) == sorted(ids)

    def test_waiting_for_a_cold_daemon_does_not_erase_the_registry(self, tmp_path, monkeypatch):
        """The synchronous path has no session id of its own — and must keep the others'.

        A hook that finds no daemon waits for one and then records ownership.
        It ran with no `sessions` key at all, so a single retain hook firing
        before the daemon listened wiped the id SessionStart had just seeded.
        """
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
        write_state(
            daemon.DAEMON_STATE_FILE,
            {"port": 9078, "started_by_plugin": True, "sessions": ["sess-live"]},
        )

        daemon._claim_daemon_ownership(9078)

        state_after = read_state(daemon.DAEMON_STATE_FILE, {})
        assert daemon._registered_sessions(state_after) == ["sess-live"]

    def test_a_marker_for_another_port_is_not_inherited(self, tmp_path, monkeypatch):
        """Sessions registered against a different daemon must not carry over."""
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
        write_state(
            daemon.DAEMON_STATE_FILE,
            {"port": 9077, "started_by_plugin": True, "sessions": ["sess-other-daemon"]},
        )

        daemon._claim_daemon_ownership(9078, session_id="sess-new")

        state_after = read_state(daemon.DAEMON_STATE_FILE, {})
        assert state_after["port"] == 9078
        assert daemon._registered_sessions(state_after) == ["sess-new"]


class TestReclaimStaysInsideTheHookBudget:
    """The reclaim probe and the stop before it share one SessionEnd timeout.

    `daemon stop` is allowed the whole budget. A probe with a timeout of its own
    is therefore additive, and a stop that ran long could take the hook past the
    point where Devin CLI kills it — losing the deregistration the hook exists
    to perform.
    """

    def test_the_budget_constant_tracks_the_installed_sessionend_timeout(self):
        import install

        installed = install.build_hooks()["SessionEnd"][0]["hooks"][0]["timeout"]
        assert daemon._SESSION_END_BUDGET_SECONDS == installed, (
            "the budget the stop and reclaim probe divide up must be the timeout "
            "install.py actually writes for SessionEnd"
        )

    def test_no_probe_once_the_budget_is_spent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
        probes = []
        monkeypatch.setattr(daemon, "_check_health", lambda *a, **k: probes.append(a) or True)

        daemon._reclaim_after_failed_stop({}, {"port": 9078}, probe_timeout=0)

        assert probes == [], "probed the daemon with no hook budget left"
        assert read_state(daemon.DAEMON_STATE_FILE, {}) == {}, "reclaimed without confirming the daemon is alive"

    def test_a_slow_stop_shrinks_the_probe(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
        write_state(
            daemon.DAEMON_STATE_FILE,
            {"port": 9078, "started_by_plugin": True, "sessions": ["s1"]},
        )
        monkeypatch.setattr(
            daemon,
            "_run_embed",
            lambda *a, **k: subprocess.CompletedProcess([], 1, "", "could not stop"),
        )
        clock = iter([0.0, 9.0])  # the stop burned 9 of the 10 seconds
        monkeypatch.setattr(daemon.time, "monotonic", lambda: next(clock))
        seen = {}
        monkeypatch.setattr(daemon, "_reclaim_after_failed_stop", lambda *a, **k: seen.update(k))

        daemon.stop_daemon({}, session_id="s1")

        assert seen["probe_timeout"] == 1.0

    def test_a_fast_stop_still_caps_the_probe(self, tmp_path, monkeypatch):
        """The common failure is a quick nonzero exit, which must keep the full probe."""
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
        write_state(
            daemon.DAEMON_STATE_FILE,
            {"port": 9078, "started_by_plugin": True, "sessions": ["s1"]},
        )
        monkeypatch.setattr(
            daemon,
            "_run_embed",
            lambda *a, **k: subprocess.CompletedProcess([], 1, "", "could not stop"),
        )
        seen = {}
        monkeypatch.setattr(daemon, "_reclaim_after_failed_stop", lambda *a, **k: seen.update(k))

        daemon.stop_daemon({}, session_id="s1")

        assert seen["probe_timeout"] == daemon._RECLAIM_PROBE_SECONDS


class TestSessionsArrivingDuringAStopAreNotLost:
    """SessionStart can land in the window where the daemon is being torn down.

    stop_daemon() clears the ownership marker in the same locked write that
    decides to stop, then runs `daemon stop` outside the lock. A SessionStart
    that health-checks the still-running daemon in that gap reaches
    register_session() with nothing in the state file.

    Dropping that registration is only safe while the stop actually succeeds —
    the session then finds nothing listening and starts its own daemon. When the
    stop *fails*, _reclaim_after_failed_stop() hands ownership back, and a
    registry that forgot the gap arrival is a registry that under-counts a live
    session: the next SessionEnd sees no other sessions and kills the daemon out
    from under it, which is the exact failure the registry was added to prevent.
    """

    def _prestart(self, tmp_path, monkeypatch, session_id):
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
        with (
            patch("lib.daemon._check_health", return_value=False),
            patch("lib.daemon._is_embed_available", return_value=True),
            patch("lib.daemon.detect_llm_config", return_value={"provider": "openai"}),
            patch("lib.daemon.get_llm_env_vars", return_value={}),
            patch("lib.daemon.subprocess.Popen"),
        ):
            daemon.prestart_daemon_background({}, session_id=session_id)

    def _stop_registering_midway(self, late_id, result):
        def _side_effect(*_args, **_kwargs):
            daemon.register_session(late_id)
            return result

        return _side_effect

    def test_a_failed_stop_hands_ownership_back_with_the_gap_session_registered(self, tmp_path, monkeypatch):
        self._prestart(tmp_path, monkeypatch, "sess-a")

        with (
            patch("lib.daemon._run_embed", side_effect=self._stop_registering_midway("sess-late", _STOP_FAILED)),
            patch("lib.daemon._check_health", return_value=True),
        ):
            daemon.stop_daemon({}, session_id="sess-a")

        state = read_state(daemon.DAEMON_STATE_FILE, {})
        assert state.get("started_by_plugin"), "ownership was not handed back after a failed stop"
        assert state.get("sessions") == ["sess-late"], "the session that arrived during the stop was erased"

    def test_the_gap_session_then_keeps_the_reclaimed_daemon_alive(self, tmp_path, monkeypatch):
        """The registry is only worth restoring if it still gates the next stop."""
        self._prestart(tmp_path, monkeypatch, "sess-a")

        with (
            patch("lib.daemon._run_embed", side_effect=self._stop_registering_midway("sess-late", _STOP_FAILED)),
            patch("lib.daemon._check_health", return_value=True),
        ):
            daemon.stop_daemon({}, session_id="sess-a")

        # An unrelated session ending must not take the daemon with it while
        # sess-late is still working.
        with patch("lib.daemon._run_embed", return_value=_STOP_OK) as run_embed:
            daemon.stop_daemon({}, session_id="sess-other")

        run_embed.assert_not_called()

    def test_a_gap_session_is_carried_onto_the_next_daemon_it_uses(self, tmp_path, monkeypatch):
        """After a *successful* stop the gap session restarts the daemon it needs.

        Its id is already in the state file, and the fresh ownership marker must
        adopt it rather than write over it — otherwise the session is invisible
        again the moment its daemon comes back.
        """
        self._prestart(tmp_path, monkeypatch, "sess-a")

        with patch("lib.daemon._run_embed", side_effect=self._stop_registering_midway("sess-late", _STOP_OK)):
            daemon.stop_daemon({}, session_id="sess-a")

        self._prestart(tmp_path, monkeypatch, "sess-new")

        state = read_state(daemon.DAEMON_STATE_FILE, {})
        assert state.get("sessions") == ["sess-late", "sess-new"]

    def test_a_marker_for_another_port_does_not_donate_its_sessions(self, tmp_path, monkeypatch):
        """Carrying ids forward is scoped to the port they were recorded against."""
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
        write_state(daemon.DAEMON_STATE_FILE, {"port": 9999, "sessions": ["ghost"]})

        self._prestart(tmp_path, monkeypatch, "sess-new")

        state = read_state(daemon.DAEMON_STATE_FILE, {})
        assert state["sessions"] == ["sess-new"], "sessions from a different daemon's marker were adopted"
