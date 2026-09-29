"""Local-embedded Hindsight runtime: import probe, install hint, the per-profile env
file the standalone ``hindsight-embed`` daemon consumes, and the health-grace export."""

from __future__ import annotations

import contextlib
import importlib
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.secret_scope import UnscopedSecretError, get_secret

from .settings import _DEFAULT_IDLE_TIMEOUT, _daemon_llm_provider, _parse_int_setting

logger = logging.getLogger(__name__.rpartition(".")[0])

# Read by hindsight_embed.daemon_embed_manager AT IMPORT TIME: how long to wait
# for a slow /health before killing the daemon as stale. Busy hosts exceed the
# upstream 2s check and get needlessly restarted, so it's plugin config.
# Env var the embedded daemon manager reads (at import time, as a module-level constant) to size the grace
# window it waits for a slow /health before declaring a daemon stale and killing it. We surface it as plugin
# config so users can raise it without hand-setting an env var, consistent with "config.json, not raw env
# vars". See #13125.
_PORT_HEALTH_GRACE_ENV = "HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT"

# Stale embedded-daemon connection markers (client recreated, operation retried once).
_RETRIABLE_CONNECTION_MARKERS = (
    "cannot connect to host",
    # Connection-establishment / DNS failure message patterns. These surface when the exception TYPE is
    # generic (RuntimeError/Exception from a local shim, MCP bridge, subprocess wrapper, or an SDK that
    # re-raises without chaining) so the _TRANSPORT_ERROR_TYPES check never fires, and the error carries no
    # HTTP status. Without message-level matching they fall through to FailoverReason.unknown, which misses
    # the transport eager-fallback path in the retry loop (unknown retries the same dead endpoint for the
    # full budget before fallback). Ported from anomalyco/opencode#40707, which hit the same bug shape:
    # serialized midstream errors matched by type only. Deliberately EXCLUDES mid-stream disconnect strings
    # ("connection reset by peer", "peer closed connection", "unexpected eof", "socket hang up") — those
    # belong to _SERVER_DISCONNECT_PATTERNS, whose classification step runs later and routes large sessions
    # to context-overflow compression. A connection that was never established cannot be a server-side
    # overflow rejection, so these are safe to classify as plain retryable transport.
    "connection refused",
    "connect call failed",
    "clientconnectorerror",
)


def _export_port_health_grace_timeout(config: dict[str, Any]) -> None:
    """Export the daemon health grace timeout BEFORE ``daemon_embed_manager`` is
    imported. Only when configured; ``setdefault`` so an explicit env override wins."""
    raw = config.get("port_health_grace_timeout")
    if raw is None or raw == "":
        return
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return logger.warning("Invalid Hindsight port_health_grace_timeout %r; ignoring.", raw)
    if seconds < 0:
        return logger.warning("Negative Hindsight port_health_grace_timeout %r; ignoring.", raw)
    os.environ.setdefault(_PORT_HEALTH_GRACE_ENV, repr(seconds))


@dataclass(frozen=True)
class LocalRuntimeStatus:
    """Whether local_embedded can run in this process, and why not when it cannot."""

    available: bool
    reason: str | None = None


def _check_local_runtime() -> LocalRuntimeStatus:
    """Whether what local_embedded needs IN THIS PROCESS imports cleanly: an HTTP client and
    the daemon manager. Nothing else belongs here.

    The server itself (``hindsight-api``, its embedding/reranking stack, pg0) runs as a separate
    process that ``hindsight_embed`` starts — from an installed ``hindsight-api`` binary, else a
    ``uvx hindsight-api`` fallback in its own environment. So we no longer probe for the
    top-level ``hindsight`` module or ``sentence_transformers``: requiring those in-process is
    what forced ``hindsight-all`` (the whole hindsight-api-slim tree) into Hermes' single pinned
    venv, where it cannot resolve — api-slim needs protobuf>=7.35.1 against mem0ai's and modal's
    protobuf<7.0, and otel-semconv>=0.65b0 against mistralai's <0.61. Both packages probed here
    are declared in this plugin's pyproject and conflict with nothing.
    """
    try:
        for module in ("hindsight_client", "hindsight_embed.daemon_embed_manager"):
            importlib.import_module(module)
        return LocalRuntimeStatus(available=True)
    except Exception as exc:
        return LocalRuntimeStatus(available=False, reason=str(exc))


def _local_runtime_hint(reason: str | None) -> str:
    """Guidance when what local_embedded needs in-process is missing.

    Both packages are declared in this plugin's ``pyproject.toml``, so a miss means the
    environment was rebuilt without them (a pm generation that dropped the plugin member, a
    stripped venv), not that the user has to install a server by hand. Reinstalling the plugin
    is the fix. NousResearch/hermes-agent#7718, #123784.
    """
    text = (reason or "").lower()
    if "no module named" in text and any(m in text for m in ("hindsight_client", "hindsight_embed")):
        return (
            " The plugin's own packages (hindsight-client, hindsight-embed) are missing from "
            "this environment: run 'hermes plugins install hindsight' (or 'hermes memory setup') "
            "to reinstall them. The Hindsight server itself is NOT needed here — it runs as a "
            "separate process."
        )
    return ""


# Interpreter-selection variables of this process: meaningful only for the interpreter that set
# them. hindsight-embed drops them from the daemon child itself since the release after 0.10.2;
# on 0.10.1/0.10.2 we have to keep them away from it ourselves (see _start_daemon_in_clean_child).
_PARENT_INTERPRETER_ENV = frozenset({"PYTHONPATH", "PYTHONHOME", "PYTHONSAFEPATH", "VIRTUAL_ENV"})

# Ask the installed hindsight-embed to start the daemon, reading the config off stdin so an LLM
# API key never appears in the process list.
_DAEMON_START_SNIPPET = (
    "import json, sys\n"
    "from hindsight_embed import get_embed_manager\n"
    "sys.exit(0 if get_embed_manager().ensure_running(json.load(sys.stdin), sys.argv[1]) else 1)\n"
)

# Generous: the first start on a machine with no hindsight-api binary downloads the server through
# uvx before the manager's own 180s health deadline even begins.
_DAEMON_START_TIMEOUT = 900


def _embed_scrubs_parent_env() -> bool:
    """Whether the installed hindsight-embed keeps our PYTHONPATH out of the daemon child itself.

    Unreadable for any reason reads as "no", so the safe path (our own clean child) is the default.
    """
    try:
        from hindsight_embed import daemon_embed_manager

        return hasattr(daemon_embed_manager, "_strip_parent_interpreter_env")
    except Exception:
        return False


def _start_daemon_in_clean_child(config: dict[str, str], profile: str) -> bool:
    """Start the daemon from a short-lived child that never had our interpreter's PYTHONPATH.

    Workaround for hindsight-embed <= 0.10.2, which copies ``os.environ`` into the daemon process.
    Hermes' package-manager install exports ``PYTHONPATH=<repo>:<its 3.14 generation>``
    (pm/environments.py), and the daemon usually runs through ``uvx hindsight-api`` on whatever
    Python uv picks. When those minor versions differ the server imports Hermes' pydantic and dies
    on ``ModuleNotFoundError: No module named 'pydantic_core._pydantic_core'``; when they happen to
    match it works, which is why this fails on some machines and not others.

    Deliberately NOT done by scrubbing ``os.environ`` around ``ensure_running``: that hole would be
    process-wide for the whole spawn *and* health wait — minutes while uvx downloads the server on
    a first run — and Hermes' own children rely on that PYTHONPATH. A child process confines it.
    """
    env = {key: value for key, value in os.environ.items() if key not in _PARENT_INTERPRETER_ENV}
    try:
        result = subprocess.run(
            [sys.executable, "-c", _DAEMON_START_SNIPPET, profile],
            input=json.dumps(config),
            env=env,
            capture_output=True,
            text=True,
            timeout=_DAEMON_START_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Could not start the Hindsight daemon helper for profile %r: %s", profile, exc)
        return False
    if result.returncode != 0:
        logger.warning(
            "Hindsight daemon helper failed for profile %r: %s",
            profile,
            (result.stderr or result.stdout or "").strip()[-500:],
        )
    return result.returncode == 0


def _start_daemon(config: dict[str, str], profile: str) -> str:
    """Start (or reuse) the out-of-process daemon for *profile* and return its base URL.

    This is what ``hindsight.HindsightEmbedded`` does internally — it is a composition of
    ``hindsight_client.Hindsight`` + ``hindsight_embed.get_embed_manager()`` and nothing more
    (see hindsight-all/hindsight/embedded.py). Calling the manager directly keeps the identical
    daemon, profile, profile ``.env`` and pg0 database while dropping the ``hindsight-all``
    dependency that cannot be installed alongside Hermes.

    *config* is an environment mapping, not a structured record — it is handed straight to the
    daemon manager as the subprocess's env, so a dict of ``HINDSIGHT_*`` names is the interface.
    It carries only explicitly-set keys: an omitted key is resolved by the
    daemon from the profile's ``.env``, then the parent environment, then its own default, and
    sending a placeholder instead would overwrite the profile's real value (#3253).
    """
    from hindsight_embed import get_embed_manager

    manager = get_embed_manager()
    if _embed_scrubs_parent_env():
        started = manager.ensure_running(config, profile)
    else:
        started = manager.is_running(profile) or _start_daemon_in_clean_child(config, profile)
    if not started:
        raise RuntimeError(f"Failed to start the Hindsight daemon for profile {profile!r}")
    return manager.get_url(profile)


def _stop_daemon(profile: str) -> bool:
    """Stop the daemon for *profile* (used when the profile env drifted and must be re-read).
    The database lives outside the process, so a stop loses no data."""
    try:
        from hindsight_embed import get_embed_manager

        return bool(get_embed_manager().stop(profile))
    except Exception as exc:
        logger.warning("Could not stop the Hindsight daemon for profile %r: %s", profile, exc)
        return False


def _daemon_is_running(profile: str) -> bool:
    """Whether the daemon for *profile* is up (used for status reporting, never to gate a call:
    the retry path in ``_run_hindsight_operation`` recreates the client, which restarts it)."""
    try:
        from hindsight_embed import get_embed_manager

        return bool(get_embed_manager().is_running(profile))
    except Exception:
        return False


def _load_simple_env(path) -> dict[str, str]:
    """Parse a KEY=VALUE env file (comments/blank lines ignored). utf-8-sig: also used
    on the Hermes .env during post_setup, where a Notepad BOM would stick to the first key."""
    if not path.exists():
        return {}
    pairs = (
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        if line and not line.startswith("#") and "=" in line
    )
    return {key.strip(): value.strip() for key, value in pairs}


def _embedded_profile_env_path(config: dict[str, Any]) -> Path:
    profile = str(config.get("profile", "hermes") or "hermes")
    return Path.home() / ".hindsight" / "profiles" / f"{profile}.env"


def _on_disk_llm_api_key(config: dict[str, Any]) -> str:
    """The key currently persisted in the profile env file ("" when absent)."""
    with contextlib.suppress(Exception):
        return _load_simple_env(_embedded_profile_env_path(config)).get("HINDSIGHT_API_LLM_API_KEY", "") or ""
    return ""


def _embedded_llm_api_key(config: dict[str, Any]) -> str:
    """Resolve the LLM API key: explicit config first, then the profile secret
    scope, then the on-disk profile env as a last resort.

    The disk fallback is the durability core: the background daemon-start
    worker usually runs with no secret scope, and without it the client would
    be built keyless — its ``ensure_running(config)`` merge would then
    overwrite the file's good key with emptiness inside the upstream manager
    (``_register_profile`` → ``create_profile`` rewrite). Falling back to the
    persisted key keeps the in-process client, the file compare, and the
    daemon subprocess all keyed from the same surviving copy.
    """
    if config.get("llmApiKey") or config.get("llm_api_key"):
        return config.get("llmApiKey") or config.get("llm_api_key")
    # NOTE: the vault item is named HINDSIGHT_API_LLM_API_KEY (matching the
    # daemon's env var), not HINDSIGHT_LLM_API_KEY (the setup-wizard name).
    # Accept both so vault-fed scopes resolve regardless of which name the
    # secret source carries.
    try:
        scoped = get_secret("HINDSIGHT_API_LLM_API_KEY", "") or get_secret("HINDSIGHT_LLM_API_KEY", "")
    except UnscopedSecretError:
        # Multiplexed gateway with no profile scope on this thread: never let
        # a missing scope read os.environ (another profile's key may live
        # there). Fall through to the on-disk copy below.
        scoped = ""
    if scoped:
        return scoped
    return _on_disk_llm_api_key(config)


def _may_rewrite_profile_env(config: dict[str, Any]) -> bool:
    """Whether rewriting the profile env file is safe right now.

    False exactly when the build has no key (no scope, no config key) but the
    file holds one: a rewrite would clobber live credentials with emptiness.
    All other mismatches (model/provider/base-url/idle-timeout drift, missing
    file, key rotation to a new non-empty value) remain writable.
    """
    if _build_embedded_profile_env(config).get("HINDSIGHT_API_LLM_API_KEY"):
        return True
    return not _on_disk_llm_api_key(config)


def _build_embedded_profile_env(config: dict[str, Any], *, llm_api_key: str | None = None) -> dict[str, str]:
    """Build the profile-scoped env that standalone hindsight-embed consumes."""
    if llm_api_key is None:
        llm_api_key = _embedded_llm_api_key(config)
    env_values = {
        "HINDSIGHT_API_LLM_PROVIDER": str(_daemon_llm_provider(config.get("llm_provider", ""))),
        "HINDSIGHT_API_LLM_API_KEY": str(llm_api_key or ""),
        "HINDSIGHT_API_LLM_MODEL": str(config.get("llm_model", "")),
        "HINDSIGHT_API_LOG_LEVEL": "info",
    }
    # Base URL is per-profile like the key beside it (the scoped key must not go to the default's host);
    # on the scopeless daemon worker a miss is a miss, never os.environ (same rule as the key above).
    base_url = config.get("llm_base_url")
    if not base_url:
        try:
            base_url = get_secret("HINDSIGHT_API_LLM_BASE_URL", "") or ""
        except UnscopedSecretError:
            base_url = ""
    if base_url:
        env_values["HINDSIGHT_API_LLM_BASE_URL"] = str(base_url)
    if (idle_timeout := config.get("idle_timeout")) is None:
        idle_timeout = os.environ.get("HINDSIGHT_IDLE_TIMEOUT")
    if idle_timeout is not None and idle_timeout != "":
        env_values["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] = str(_parse_int_setting(idle_timeout, _DEFAULT_IDLE_TIMEOUT))
    return env_values


def _secure_write_profile_env(profile_env: Path, content: str) -> None:
    """Create/overwrite *profile_env* owner-only (0600); a pre-existing file is
    tightened BEFORE the plaintext LLM API key is written."""
    if profile_env.exists():
        with contextlib.suppress(OSError):
            os.chmod(profile_env, 0o600)
    fd = os.open(str(profile_env), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)


def _validate_profile_env_permissions(profile_env: Path) -> None:
    """Post-write check: owner-only on POSIX (Windows ACLs aren't mode bits; skipped)."""
    if os.name != "posix":
        return
    import stat

    if stat.S_IMODE(profile_env.stat().st_mode) != 0o600:
        with contextlib.suppress(OSError):
            os.chmod(profile_env, 0o600)
        if stat.S_IMODE(profile_env.stat().st_mode) != 0o600:
            raise PermissionError(f"Embedded Hindsight profile environment is not owner-only: {profile_env}")


def _materialize_embedded_profile_env(config: dict[str, Any], *, llm_api_key: str | None = None) -> Path:
    """Write the profile env file; never leave a plaintext key in a file whose
    permissions could not be verified."""
    profile_env = _embedded_profile_env_path(config)
    profile_env.parent.mkdir(parents=True, exist_ok=True)
    env_values = _build_embedded_profile_env(config, llm_api_key=llm_api_key)
    content = "".join(f"{key}={value}\n" for key, value in env_values.items())
    try:
        _secure_write_profile_env(profile_env, content)
        _validate_profile_env_permissions(profile_env)
    except BaseException:
        with contextlib.suppress(OSError):
            profile_env.unlink()
        raise
    return profile_env
