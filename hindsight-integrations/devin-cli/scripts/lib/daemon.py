"""Hindsight-embed daemon lifecycle management.

Port of the Claude Code plugin's daemon.py, adapted for Python subprocess
calls from Devin CLI's ephemeral hook processes.

Manages three connection modes (same as the Claude Code plugin):
  1. External API — user provides hindsightApiUrl (skip daemon entirely)
  2. Existing local server — user already has hindsight running
  3. Auto-managed daemon — plugin starts/stops hindsight-embed

Uses its own `devin-cli` hindsight-embed profile and a different default port
(9078 vs. the Claude Code plugin's 9077) so both integrations can run their
own local daemons side by side without colliding. To share one memory bank
across both Claude Code and Devin CLI, point both at the same external server
via `hindsightApiUrl` instead of running two local daemons.
"""

import os
import platform
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .client import USER_AGENT
from .llm import detect_llm_config, get_llm_env_vars
from .state import locked_read_modify_write

DAEMON_STATE_FILE = "daemon.json"
DAEMON_LOCK_FILE = "daemon.lock"
PROFILE_NAME = "devin-cli"

# SessionEnd's timeout in the hook config install.py writes. The daemon stop and
# the reclaim probe that may follow it share this one budget, so it lives here
# rather than as a literal at each call site.
_SESSION_END_BUDGET_SECONDS = 10
_RECLAIM_PROBE_SECONDS = 2


def _get_embed_command(config: dict) -> list:
    """Get the command to run hindsight-embed."""
    embed_path = config.get("embedPackagePath")
    if embed_path:
        return ["uv", "run", "--directory", embed_path, "hindsight-embed"]

    version = config.get("embedVersion", "latest")
    package = f"hindsight-embed@{version}" if version else "hindsight-embed@latest"
    return ["uvx", package]


def _set_uvx_python_compat(cmd: list, env: dict) -> None:
    """Use a Python version compatible with uvx-managed Hindsight packages."""
    if cmd and cmd[0] == "uvx" and not env.get("UV_PYTHON", "").strip():
        env["UV_PYTHON"] = "3.13"


# Never passed on the command line — see _profile_env_args().
_SECRET_ENV_VARS = {"HINDSIGHT_API_LLM_API_KEY"}


def _profile_env_args(env: dict) -> list:
    """Build `--env KEY=VALUE` args for `hindsight-embed profile create`.

    Secrets are deliberately omitted. Command-line arguments are world-readable
    in process listings, so a key passed this way is visible to every other user
    on the machine for as long as the command runs. It does not need to be here:
    hindsight-embed's daemon manager copies its own os.environ when it spawns the
    API server, and both callers already pass the key in the subprocess env.
    """
    return [
        arg
        for name, value in env.items()
        if value and name not in _SECRET_ENV_VARS
        for arg in ("--env", f"{name}={value}")
    ]


def _run_embed(config: dict, args: list, env: dict = None, timeout: int = 10) -> subprocess.CompletedProcess:
    """Run a hindsight-embed command and return the result."""
    cmd = _get_embed_command(config) + args
    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    _set_uvx_python_compat(cmd, run_env)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=run_env,
    )


def _is_embed_available(config: dict) -> bool:
    """Check that the executable `_get_embed_command()` will actually invoke exists.

    Accepting *either* uvx or hindsight-embed is not enough. With
    `embedPackagePath` set the command is `uv run --directory ...`, so a missing
    `uv` still fails; without it the command is always `uvx`, so a machine
    carrying only a standalone `hindsight-embed` would pass this check and then
    fail at startup. Check the command we are about to run instead.
    """
    if shutil.which(_get_embed_command(config)[0]) is None:
        return False

    embed_path = config.get("embedPackagePath")
    if embed_path:
        return os.path.isdir(embed_path)
    return True


def _check_health(base_url: str, timeout: int = 10) -> bool:
    """Quick health check against a Hindsight server.

    Default timeout is 10s: under load an alive-but-busy daemon mid
    fact-extraction may not answer /health within a couple of seconds. A
    too-short timeout yields a false negative and can trigger a restart/kill
    loop via _ensure_daemon_running().
    """
    try:
        url = f"{base_url.rstrip('/')}/health"
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_api_url(config: dict, debug_fn=None, allow_daemon_start: bool = False) -> str:
    """Determine the API URL, optionally starting daemon if needed.

    Connection mode priority:
      1. External API (hindsightApiUrl configured)
      2. Existing local server (check port health)
      3. Auto-managed daemon (only if allow_daemon_start=True)

    allow_daemon_start=False for recall (bounded hook timeout); True for
    retain (async, more headroom for a cold daemon start).
    """
    external_url = config.get("hindsightApiUrl")
    if external_url:
        if debug_fn:
            debug_fn(f"Using external API: {external_url}")
        return external_url

    port = config.get("apiPort", 9078)
    base_url = f"http://127.0.0.1:{port}"

    if _check_health(base_url):
        if debug_fn:
            debug_fn(f"Existing server healthy on port {port}")
        return base_url

    if not allow_daemon_start:
        raise RuntimeError(
            f"No Hindsight server on port {port}. Set hindsightApiUrl for external "
            f"API, start hindsight-embed manually, or wait for the retain hook to "
            f"auto-start the daemon."
        )

    if debug_fn:
        debug_fn(f"No server on port {port}, attempting daemon start")

    try:
        _ensure_daemon_running(config, port, debug_fn)
    except Exception as e:
        if debug_fn:
            debug_fn(f"Daemon start failed: {e}")
        raise RuntimeError(
            "No Hindsight server available. Set hindsightApiUrl for external API, "
            "or ensure hindsight-embed is installed for local daemon mode."
        ) from e

    return base_url


def _ensure_daemon_running(config: dict, port: int, debug_fn=None):
    """Start the hindsight-embed daemon if not already running."""
    if not _is_embed_available(config):
        raise RuntimeError(
            "hindsight-embed not found (uvx not on PATH). "
            "Install with: pip install hindsight-embed, or set hindsightApiUrl."
        )

    base_url = f"http://127.0.0.1:{port}"

    try:
        llm_config = detect_llm_config(config)
    except RuntimeError as e:
        raise RuntimeError(f"Cannot start daemon: {e}") from e

    llm_env = get_llm_env_vars(llm_config)

    daemon_env = dict(llm_env)
    idle_timeout = config.get("daemonIdleTimeout", 300)
    daemon_env["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] = str(idle_timeout)

    if platform.system() == "Darwin":
        daemon_env["HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU"] = "1"
        daemon_env["HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU"] = "1"

    if debug_fn:
        debug_fn(f'Configuring "{PROFILE_NAME}" profile...')

    profile_args = [
        "profile",
        "create",
        PROFILE_NAME,
        "--merge",
        "--port",
        str(port),
    ]
    profile_args.extend(_profile_env_args(daemon_env))

    try:
        result = _run_embed(config, profile_args, daemon_env, timeout=10)
        if result.returncode != 0:
            if debug_fn:
                debug_fn(f"Profile create stderr: {result.stderr.strip()}")
            raise RuntimeError(f"Profile create failed (exit {result.returncode}): {result.stderr}")
        if debug_fn:
            debug_fn("Profile configured")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Profile create timed out")
    except FileNotFoundError:
        raise RuntimeError(
            "hindsight-embed not found. Install with: pip install hindsight-embed "
            "or set hindsightApiUrl for external API mode."
        )

    if debug_fn:
        debug_fn("Starting daemon...")

    try:
        result = _run_embed(
            config,
            ["daemon", "--profile", PROFILE_NAME, "start"],
            daemon_env,
            timeout=30,
        )
        if debug_fn:
            debug_fn(f"Daemon start exit={result.returncode} stdout={result.stdout.strip()}")
        if result.returncode != 0 and "already running" not in result.stderr.lower():
            raise RuntimeError(f"Daemon start failed (exit {result.returncode}): {result.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Daemon start timed out")

    if debug_fn:
        debug_fn("Waiting for daemon to be ready...")

    for attempt in range(30):
        if _check_health(base_url):
            if debug_fn:
                debug_fn(f"Daemon ready after {attempt + 1} attempts")
            _claim_daemon_ownership(port, debug_fn=debug_fn)
            return
        time.sleep(1)

    raise RuntimeError("Daemon failed to become ready within 30 seconds")


def _claim_daemon_ownership(port: int, session_id: str = "", debug_fn=None) -> bool:
    """Record that the plugin owns the daemon on `port`, keeping the registry.

    Both start paths end here, and both used to write the marker straight over
    whatever was there. That drops the session registry stop_daemon() decides
    on:

    - Two SessionStart hooks reach the background path together whenever the
      second one's health check races the first one's start — the common case,
      since the daemon takes seconds to listen. A bare write leaves only the
      second session's id.
    - The synchronous path wrote no `sessions` key at all, so a hook that had
      to wait for the daemon erased the id SessionStart had just seeded.

    Either way a session ends up unregistered, and when some *other* session
    ends it reads a registry that no longer mentions the first and stops the
    daemon out from under it — exactly the failure the registry was added to
    prevent. So: merge, under the same lock register_session() uses.

    Returns False if ownership could not be recorded.
    """

    def _claim(state):
        prior_port = state.get("port")
        # A marker naming a *different* port belongs to a different daemon; its
        # session list must not be carried over onto this one. A state naming no
        # port is the other case, and it must be carried: stop_daemon() clears
        # the whole marker before running the stop, and register_session() keeps
        # recording into what is left, so those ids are live sessions that
        # health-checked this very port during the gap. Dropping them here is
        # how one goes missing from the registry and has its daemon stopped
        # under it.
        carried = _registered_sessions(state) if prior_port is None or prior_port == port else []
        if not state.get("started_by_plugin") or prior_port != port:
            state = {
                "port": port,
                "started_by_plugin": True,
                "started_at": time.time(),
                "pid": os.getpid(),
            }
        sessions = carried
        if session_id and session_id not in sessions:
            sessions = sessions + [session_id]
        state["sessions"] = sessions
        return state, None

    try:
        locked_read_modify_write(DAEMON_STATE_FILE, DAEMON_LOCK_FILE, _claim)
    except OSError as e:
        # No lock means no safe way to record ownership. Leaving the daemon
        # unowned means nothing stops it and its own idle timeout reaps it —
        # strictly better than the failure on the other side, which is stopping
        # a daemon a live session is still talking to.
        if debug_fn:
            debug_fn(f"Could not record daemon ownership: {e}")
        return False
    return True


def prestart_daemon_background(config: dict, session_id: str = "", debug_fn=None):
    """Fire off daemon startup in the background — non-blocking.

    Called from SessionStart so the daemon is warm by the time the first
    recall or retain hook fires. Returns immediately.

    `session_id` seeds the daemon's session registry so SessionEnd knows
    whether anyone else is still using it — see register_session().
    """
    if config.get("hindsightApiUrl"):
        return

    port = config.get("apiPort", 9078)
    if _check_health(f"http://127.0.0.1:{port}"):
        if debug_fn:
            debug_fn(f"Daemon already running on port {port}, skipping pre-start")
        return

    if not _is_embed_available(config):
        if debug_fn:
            debug_fn("hindsight-embed not available, skipping pre-start")
        return

    try:
        llm_config = detect_llm_config(config)
    except RuntimeError as e:
        if debug_fn:
            debug_fn(f"No LLM configured, skipping daemon pre-start: {e}")
        return

    llm_env = get_llm_env_vars(llm_config)
    embed_cmd = _get_embed_command(config)
    daemon_env = dict(os.environ)
    daemon_env.update(llm_env)
    _set_uvx_python_compat(embed_cmd, daemon_env)
    idle_timeout = config.get("daemonIdleTimeout", 300)
    daemon_env["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] = str(idle_timeout)
    if platform.system() == "Darwin":
        daemon_env["HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU"] = "1"
        daemon_env["HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU"] = "1"

    profile_args = ["profile", "create", PROFILE_NAME, "--merge", "--port", str(port)]
    profile_args.extend(_profile_env_args(llm_env))

    profile_str = shlex.join(embed_cmd + profile_args)
    daemon_str = shlex.join(embed_cmd + ["daemon", "--profile", PROFILE_NAME, "start"])

    subprocess.Popen(
        f"{profile_str} && {daemon_str}",
        shell=True,
        env=daemon_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Claim ownership now rather than on confirmed readiness, which is what the
    # synchronous path can do because it waits. Without this marker SessionEnd
    # reads no `started_by_plugin` and skips the stop, so every pre-started
    # daemon outlives its session. Claiming early is safe: the health check
    # above already returned if anything was listening on this port, so this
    # can only ever mark a daemon the plugin itself launched — and stop_daemon
    # targets the plugin's own profile, not a raw pid.
    #
    if not _claim_daemon_ownership(port, session_id=session_id, debug_fn=debug_fn):
        return

    if debug_fn:
        debug_fn(f"Daemon pre-start initiated in background (port {port})")


def _registered_sessions(state: dict) -> list:
    """Session ids currently using the plugin-managed daemon.

    Absent on state files written by earlier versions of this plugin, which is
    read as "nobody is registered" so those installs keep the previous
    stop-on-first-SessionEnd behaviour rather than leaking a daemon forever.
    """
    sessions = state.get("sessions")
    return [s for s in sessions if isinstance(s, str)] if isinstance(sessions, list) else []


def register_session(session_id: str, debug_fn=None) -> None:
    """Record that `session_id` is using the plugin-managed daemon.

    The daemon is shared by every concurrent Devin CLI session, but its
    lifecycle used to be tracked by one global marker — so the first session to
    end stopped the daemon out from under the others, and they had no way to
    notice. stop_daemon() now only stops once the last registered session is
    gone.

    Held under the same interprocess lock the turn counter uses. Two
    SessionStart hooks racing here would otherwise both read the same list,
    each append only its own id, and the later write would drop the earlier
    session — whose SessionEnd then sees an empty registry and stops the daemon
    while that session is still working.
    """
    if not session_id:
        return

    def _add(state):
        # Recorded whether or not an ownership marker is present. stop_daemon()
        # clears the marker in the same locked write that decides to stop, and
        # the stop itself then runs outside the lock — so a SessionStart that
        # health-checks the still-running daemon in that gap lands here on an
        # empty state. Declining to record it left that live session invisible,
        # and if the stop then failed, _reclaim_after_failed_stop() handed
        # ownership back with an empty registry: a later SessionEnd read no
        # other sessions and stopped the daemon out from under one that was
        # still working. Recording unconditionally only ever *delays* a stop,
        # and the id is carried onto a marker by _claim_daemon_ownership() only
        # when that marker names the same port.
        sessions = _registered_sessions(state)
        if session_id not in sessions:
            state["sessions"] = sessions + [session_id]
        return state, None

    try:
        locked_read_modify_write(DAEMON_STATE_FILE, DAEMON_LOCK_FILE, _add)
    except OSError as e:
        # Losing a registration costs a premature daemon stop, not a wrong
        # answer, and the next hook restarts it. Not worth failing SessionStart.
        if debug_fn:
            debug_fn(f"Could not register session with daemon state: {e}")


@dataclass
class _StopDecision:
    """What stop_daemon() should do, decided in one locked read of the state.

    `released` carries the ownership state given up on the "stop" path, so it
    can be handed back if the stop turns out not to have stopped anything.
    """

    action: str
    remaining: list
    released: dict


def _reclaim_after_failed_stop(
    config: dict,
    released: dict,
    debug_fn=None,
    probe_timeout: float = _RECLAIM_PROBE_SECONDS,
) -> None:
    """Take ownership back when `daemon stop` did not actually stop anything.

    Ownership is given up *before* the stop runs, which is what keeps a new
    session from attaching to a daemon already condemned. The cost is that a
    stop which fails would otherwise leave a daemon nobody owns:
    register_session() declines without `started_by_plugin`, so no later
    SessionEnd would ever retry, and the daemon would live until its idle
    timeout reaped it — or forever, if that timeout is disabled.

    Health is re-checked first because a nonzero exit does not mean the daemon
    survived; reclaiming one that did die would leave a marker pointing at a
    port with nothing behind it.
    """
    port = released.get("port")
    if isinstance(port, bool) or not isinstance(port, int):
        port = config.get("apiPort", 9078)

    # Deliberately far below _check_health()'s 10s default, and further capped
    # by the caller to whatever is left of SessionEnd's 10s budget after the
    # stop — which is allowed that entire budget on its own, so "2s" as a fixed
    # value could take the hook to 12s and get it killed. A false negative here
    # costs only the repair marker, which leaves exactly the behaviour this
    # function exists to improve on; a slow probe costs the whole hook.
    if probe_timeout <= 0:
        if debug_fn:
            debug_fn("No hook budget left to probe the daemon; leaving ownership released")
        return
    if not _check_health(f"http://127.0.0.1:{port}", timeout=probe_timeout):
        return

    def _restore(state):
        # Not if someone re-claimed the port meanwhile: that state belongs to a
        # daemon a newer session started, and its session list must not be
        # overwritten by a dead one's.
        if state.get("started_by_plugin"):
            return state, False
        # Every session registered *before* the stop has ended, but sessions
        # that started during it are recorded on the cleared state and are
        # still live — the daemon they health-checked is the one that just
        # refused to die. Handing ownership back with a hardcoded empty list
        # erased them, and the next SessionEnd then stopped the daemon under
        # them. Restoring the marker is what lets those sessions' own
        # SessionEnd retry the stop.
        return {**released, "sessions": _registered_sessions(state)}, True

    try:
        restored = locked_read_modify_write(DAEMON_STATE_FILE, DAEMON_LOCK_FILE, _restore)
    except OSError as e:
        if debug_fn:
            debug_fn(f"Could not restore daemon ownership after a failed stop: {e}")
        return

    if debug_fn and restored:
        debug_fn("Daemon still running after a failed stop; ownership restored for a later retry")


def stop_daemon(config: dict, session_id: str = "", debug_fn=None):
    """Stop the daemon, once the last session using it has ended.

    Deregistration runs under the same lock as register_session(), so two
    SessionEnd hooks racing cannot each drop the other's id and both conclude
    they were last.

    The `daemon stop` itself runs *outside* the lock, deliberately. It is
    allowed 10s and SessionStart — which calls register_session() through the
    same lock — is only allowed 5s by the installed hook config, so holding the
    lock across the stop would trade this race for a guaranteed hook timeout.
    What closes the race instead is that ownership is given up in the same
    locked write that decides to stop: a session arriving in the gap reads no
    `started_by_plugin`, declines to register, and so is never counted against
    a daemon that is already being torn down. If the stop then fails, that
    ownership is handed back — see _reclaim_after_failed_stop().
    """

    def _deregister(state):
        # Dict default is applied by locked_read_modify_write, so a daemon.json
        # holding `null` or `[]` arrives here as {} rather than reaching .get().
        if not state.get("started_by_plugin"):
            return state, _StopDecision("skip", [], {})
        others = [s for s in _registered_sessions(state) if s != session_id]
        if others:
            state["sessions"] = others
            return state, _StopDecision("keep", others, {})
        # Clearing the whole state, not just `sessions`, is what closes the
        # race: it is the same write that decides to stop, so there is no
        # window in which a new session can register against a daemon this
        # call has already committed to killing.
        return {}, _StopDecision("stop", [], dict(state))

    try:
        decision = locked_read_modify_write(DAEMON_STATE_FILE, DAEMON_LOCK_FILE, _deregister)
    except OSError as e:
        if debug_fn:
            debug_fn(f"Could not read daemon state, skipping stop: {e}")
        return

    if decision.action == "skip":
        if debug_fn:
            debug_fn("Daemon not started by plugin, skipping stop")
        return

    if decision.action == "keep":
        # A session that crashes never deregisters, so this can hold a daemon
        # open past its last real user — bounded by the daemon's own idle
        # timeout (HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT), which reaps it anyway.
        if debug_fn:
            debug_fn(f"{len(decision.remaining)} session(s) still using the daemon, skipping stop")
        return

    if debug_fn:
        debug_fn("Stopping daemon...")

    stopped = False
    stop_started = time.monotonic()
    try:
        result = _run_embed(
            config,
            ["daemon", "--profile", PROFILE_NAME, "stop"],
            timeout=_SESSION_END_BUDGET_SECONDS,
        )
        # _run_embed() does not pass check=True, so a stop that failed arrives
        # as a nonzero returncode rather than an exception. Reading only stdout
        # here would report that failure as a success.
        stopped = result.returncode == 0
        if debug_fn:
            debug_fn(f"Daemon stop ({result.returncode}): {(result.stdout or result.stderr).strip()}")
    except Exception as e:
        # A session that health-checked just before this stop keeps a handle on
        # a daemon that is about to die. It self-heals: the recall and retain
        # paths start a daemon when none is listening.
        if debug_fn:
            debug_fn(f"Daemon stop error: {e}")

    if not stopped:
        # Whatever is left of the hook's budget, capped at the probe's own
        # timeout. The stop above is allowed the whole budget, so a stop that
        # burned it leaves nothing to probe with.
        remaining = _SESSION_END_BUDGET_SECONDS - (time.monotonic() - stop_started)
        _reclaim_after_failed_stop(
            config,
            decision.released,
            debug_fn,
            probe_timeout=min(_RECLAIM_PROBE_SECONDS, remaining),
        )
