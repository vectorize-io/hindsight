"""Hindsight-embed daemon lifecycle management.

Port of: HindsightServer in @vectorize-io/hindsight-all, adapted for Python
subprocess calls from ephemeral hook processes.

Manages three connection modes (same as Openclaw):
  1. External API — user provides hindsightApiUrl (skip daemon entirely)
  2. Existing local server — user already has hindsight running
  3. Auto-managed daemon — plugin starts/stops hindsight-embed

In Claude Code's ephemeral model, daemon state is tracked via files in
$CLAUDE_PLUGIN_DATA/state/. The daemon itself is a background OS process
managed by hindsight-embed's built-in daemon command.
"""

import os
import platform
import subprocess
import time
import urllib.error
import urllib.request

from .client import USER_AGENT
from .llm import detect_llm_config, get_llm_env_vars
from .state import read_state, write_state

DAEMON_STATE_FILE = "daemon.json"
PROFILE_NAME = "claude-code"


def _get_embed_command(config: dict) -> list:
    """Get the command to run hindsight-embed.

    Port of: getEmbedCommand() in @vectorize-io/hindsight-all
    """
    embed_path = config.get("embedPackagePath")
    if embed_path:
        return ["uv", "run", "--directory", embed_path, "hindsight-embed"]

    version = config.get("embedVersion", "latest")
    package = f"hindsight-embed@{version}" if version else "hindsight-embed@latest"
    return ["uvx", package]


def _run_embed(config: dict, args: list, env: dict = None, timeout: int = 10) -> subprocess.CompletedProcess:
    """Run a hindsight-embed command and return the result."""
    cmd = _get_embed_command(config) + args
    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=run_env,
    )


def _is_embed_available(config: dict) -> bool:
    """Quick check if hindsight-embed is available on PATH.

    Avoids the slow uvx download path when the tool isn't installed.
    """
    import shutil

    embed_path = config.get("embedPackagePath")
    if embed_path:
        return os.path.isdir(embed_path)
    # Check for uvx or hindsight-embed on PATH
    return shutil.which("uvx") is not None or shutil.which("hindsight-embed") is not None


def _check_health(base_url: str, timeout: int = 2) -> bool:
    """Quick health check against a Hindsight server."""
    try:
        url = f"{base_url.rstrip('/')}/health"
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_api_url(config: dict, debug_fn=None, allow_daemon_start: bool = False) -> str:
    """Determine the API URL, optionally starting daemon if needed.

    Returns the API URL to use for recall/retain, handling all three modes.

    Connection mode priority:
      1. External API (hindsightApiUrl configured)
      2. Existing local server (check port health)
      3. Auto-managed daemon (only if allow_daemon_start=True)

    The allow_daemon_start flag prevents the recall hook (10s timeout) from
    blocking on a 30s daemon startup. Only the retain hook (async, 15s) should
    attempt daemon start.
    """
    # Mode 1: External API
    external_url = config.get("hindsightApiUrl")
    if external_url:
        if debug_fn:
            debug_fn(f"Using external API: {external_url}")
        return external_url

    # Mode 2 & 3: Local server
    port = config.get("apiPort", 9077)
    base_url = f"http://127.0.0.1:{port}"

    # Check if something is already running on this port
    if _check_health(base_url):
        if debug_fn:
            debug_fn(f"Existing server healthy on port {port}")
        return base_url

    # Mode 3: Auto-start daemon (only when allowed)
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
    """Start the hindsight-embed daemon if not already running.

    Port of: HindsightServer.start() in @vectorize-io/hindsight-all
    """
    # Fast-fail if hindsight-embed toolchain is not available
    if not _is_embed_available(config):
        raise RuntimeError(
            "hindsight-embed not found (uvx not on PATH). "
            "Install with: pip install hindsight-embed, or set hindsightApiUrl."
        )

    base_url = f"http://127.0.0.1:{port}"

    # Detect LLM config (needed for daemon's fact extraction)
    try:
        llm_config = detect_llm_config(config)
    except RuntimeError as e:
        raise RuntimeError(f"Cannot start daemon: {e}") from e

    llm_env = get_llm_env_vars(llm_config)

    # Build daemon environment
    daemon_env = dict(llm_env)
    idle_timeout = config.get("daemonIdleTimeout", 300)
    daemon_env["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] = str(idle_timeout)

    # On macOS, force CPU for embeddings/reranker (mirrors Openclaw)
    if platform.system() == "Darwin":
        daemon_env["HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU"] = "1"
        daemon_env["HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU"] = "1"

    # Step 1: Configure profile
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
    for env_name, env_val in daemon_env.items():
        if env_val:
            profile_args.extend(["--env", f"{env_name}={env_val}"])

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

    # Step 2: Start daemon
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

    # Step 3: Wait for ready (poll health endpoint)
    if debug_fn:
        debug_fn("Waiting for daemon to be ready...")

    for attempt in range(30):
        if _check_health(base_url):
            if debug_fn:
                debug_fn(f"Daemon ready after {attempt + 1} attempts")
            # Track that we started this daemon
            write_state(
                DAEMON_STATE_FILE,
                {
                    "port": port,
                    "started_by_plugin": True,
                    "started_at": time.time(),
                    "pid": os.getpid(),
                },
            )
            return
        time.sleep(1)

    raise RuntimeError("Daemon failed to become ready within 30 seconds")


def prestart_daemon_background(config: dict, debug_fn=None):
    """Fire off daemon startup in the background — non-blocking.

    Called from SessionStart hook to warm up the daemon before the first
    recall or retain hook fires. Returns immediately; the daemon starts
    asynchronously as a detached OS process.
    """
    if config.get("hindsightApiUrl"):
        return  # External API mode — no local daemon needed

    port = config.get("apiPort", 9077)
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
    daemon_env = dict(os.environ)
    daemon_env.update(llm_env)
    idle_timeout = config.get("daemonIdleTimeout", 300)
    daemon_env["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] = str(idle_timeout)
    if platform.system() == "Darwin":
        daemon_env["HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU"] = "1"
        daemon_env["HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU"] = "1"

    embed_cmd = _get_embed_command(config)

    profile_args = ["profile", "create", PROFILE_NAME, "--merge", "--port", str(port)]
    for env_name, env_val in llm_env.items():
        if env_val:
            profile_args.extend(["--env", f"{env_name}={env_val}"])

    import shlex
    profile_str = shlex.join(embed_cmd + profile_args)
    daemon_str = shlex.join(embed_cmd + ["daemon", "--profile", PROFILE_NAME, "start"])

    import subprocess as _sp
    _sp.Popen(
        f"{profile_str} && {daemon_str}",
        shell=True,
        env=daemon_env,
        stdout=_sp.DEVNULL,
        stderr=_sp.DEVNULL,
        start_new_session=True,
    )
    if debug_fn:
        debug_fn(f"Daemon pre-start initiated in background (port {port})")


def stop_daemon(config: dict, debug_fn=None):
    """Stop the daemon if it was started by this plugin.

    Called during cleanup. Only stops if we started it (tracked in state).
    """
    state = read_state(DAEMON_STATE_FILE)
    if not state or not state.get("started_by_plugin"):
        if debug_fn:
            debug_fn("Daemon not started by plugin, skipping stop")
        return

    if debug_fn:
        debug_fn("Stopping daemon...")

    try:
        result = _run_embed(
            config,
            ["daemon", "--profile", PROFILE_NAME, "stop"],
            timeout=10,
        )
        if debug_fn:
            debug_fn(f"Daemon stop: {result.stdout.strip()}")
    except Exception as e:
        if debug_fn:
            debug_fn(f"Daemon stop error: {e}")

    # Clear state
    write_state(DAEMON_STATE_FILE, {})
