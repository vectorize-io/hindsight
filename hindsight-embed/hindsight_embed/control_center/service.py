"""Business logic for the control center.

A thin layer over ``ProfileManager`` (config read/write) and ``daemon_client``
(lifecycle) that returns typed results. The HTTP server in ``server.py`` only
does routing, token enforcement, and JSON (de)serialization on top of this — so
this module is what the tests exercise directly.
"""

from dataclasses import asdict, dataclass

import httpx

from .. import daemon_client
from ..profile_manager import ENV_API_PORT, ENV_CP_PORT, ProfileManager

# The "simple alias" keys that ProfileManager.load_profile_config injects for
# backward compatibility. They must never be written back into a profile's
# .env, or they'd pollute it with lowercase duplicates.
_ALIAS_KEYS = frozenset(
    {
        "llm_api_key",
        "llm_provider",
        "llm_model",
        "llm_base_url",
        "log_level",
        "idle_timeout",
    }
)

# Env var names the wizard owns. Writing config replaces these; everything else
# in the profile (idle timeout, bank id, custom keys) is preserved untouched.
_ENV_PROVIDER = "HINDSIGHT_API_LLM_PROVIDER"
_ENV_API_KEY = "HINDSIGHT_API_LLM_API_KEY"
_ENV_MODEL = "HINDSIGHT_API_LLM_MODEL"
_ENV_BASE_URL = "HINDSIGHT_API_LLM_BASE_URL"
# Per-profile component versions (default = the embed's own version when unset).
_ENV_API_VERSION = "HINDSIGHT_EMBED_API_VERSION"
_ENV_CP_VERSION = "HINDSIGHT_EMBED_CP_VERSION"

# Sentinel the UI sends back when the api-key field is left untouched, so we
# keep the stored key instead of overwriting it with the masked placeholder.
API_KEY_UNCHANGED = "__unchanged__"


@dataclass(frozen=True)
class ProfileSummary:
    """One row in the control center's profile list."""

    name: str  # "" for the default profile
    display_name: str  # "default" for the default profile
    port: int
    is_active: bool
    daemon_running: bool
    provider: str | None
    model: str | None
    has_api_key: bool


@dataclass(frozen=True)
class ProfileConfigView:
    """The editable LLM configuration of a single profile (key masked)."""

    name: str
    display_name: str
    provider: str | None
    model: str | None
    base_url: str | None
    api_key_masked: str | None  # None when no key is set
    has_api_key: bool
    api_port: int  # effective API/daemon port
    ui_port: int  # effective control-plane port
    ui_port_is_default: bool  # True when UI port is derived (API + offset), not pinned
    api_version: str | None  # pinned hindsight-api version (None = use embed default)
    cp_version: str | None  # pinned control-plane version (None = use embed default)


@dataclass(frozen=True)
class DaemonActionResult:
    """Outcome of a lifecycle action (start/stop/restart) or a status query."""

    ok: bool
    running: bool
    url: str | None
    message: str


@dataclass(frozen=True)
class EnvFileView:
    """The raw contents of a profile's .env file, for direct editing."""

    name: str
    display_name: str
    path: str
    content: str
    exists: bool


@dataclass(frozen=True)
class ProfilePathsView:
    """On-disk references for a profile (config, logs, database, URLs)."""

    name: str
    display_name: str
    port: int
    config_path: str
    log_path: str
    lock_path: str
    database_url: str
    database_path: str | None  # local pg0 instance dir, when using pg0://
    daemon_url: str
    ui_url: str


@dataclass(frozen=True)
class LogTailView:
    """The tail of a profile's daemon log."""

    path: str
    exists: bool
    content: str


@dataclass(frozen=True)
class UiStatusView:
    """Status of the control plane (Next.js) UI for a profile."""

    running: bool
    url: str


@dataclass(frozen=True)
class ProfileDeleteResult:
    """Outcome of deleting a profile."""

    ok: bool
    message: str


@dataclass(frozen=True)
class HealthView:
    """Live health of a profile's API daemon and control-plane UI."""

    api_ok: bool
    api_detail: str  # e.g. "healthy · db connected", "unreachable", "HTTP 503"
    ui_ok: bool


def normalize_profile(name: str | None) -> str:
    """Map the URL/display name to the internal profile id ("" = default)."""
    if name is None or name in ("default", ""):
        return ""
    return name


def _display_name(name: str) -> str:
    return name or "default"


def _browser_url(url: str) -> str:
    """Render daemon/UI URLs with the localhost hostname for the browser.

    0.0.0.0 isn't routable, and we present localhost everywhere for consistency.
    """
    return url.replace("://0.0.0.0:", "://localhost:").replace("://127.0.0.1:", "://localhost:")


def _mask_key(key: str | None) -> str | None:
    if not key:
        return None
    if len(key) <= 8:
        return "****"
    return f"{key[:3]}…{key[-4:]}"


def _read_raw_env(name: str) -> dict[str, str]:
    """Parse a profile's .env into real env-var keys only (no simple aliases)."""
    pm = ProfileManager()
    path = pm.resolve_profile_paths(name).config
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in _ALIAS_KEYS:
            continue
        env[key] = value.strip()
    return env


def _write_raw_env(name: str, env: dict[str, str]) -> None:
    """Persist env vars to a profile's .env (default or named), 0600."""
    pm = ProfileManager()
    if name:
        # Named profile: create_profile writes the file and updates metadata
        # (port allocation/last_used) in one shot.
        pm.create_profile(name, env)
        path = pm.resolve_profile_paths(name).config
    else:
        path = pm.resolve_profile_paths("").config
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{k}={v}\n" for k, v in env.items()))
    # The .env holds the API key — keep it owner-only.
    path.chmod(0o600)


def list_profiles() -> list[ProfileSummary]:
    """List all profiles with live daemon status and their LLM provider/model."""
    pm = ProfileManager()
    summaries: list[ProfileSummary] = []
    for info in pm.list_profiles():
        config = pm.load_profile_config(info.name)
        summaries.append(
            ProfileSummary(
                name=info.name,
                display_name=_display_name(info.name),
                port=info.port,
                is_active=info.is_active,
                daemon_running=info.daemon_running,
                provider=config.get("llm_provider"),
                model=config.get("llm_model"),
                has_api_key=bool(config.get("llm_api_key")),
            )
        )
    return summaries


def get_profile_config(name: str) -> ProfileConfigView:
    """Return the editable LLM config of a profile, with the API key masked."""
    name = normalize_profile(name)
    pm = ProfileManager()
    config = pm.load_profile_config(name)
    api_key = config.get("llm_api_key")
    paths = pm.resolve_profile_paths(name)
    raw = _read_raw_env(name)
    return ProfileConfigView(
        name=name,
        display_name=_display_name(name),
        provider=config.get("llm_provider"),
        model=config.get("llm_model"),
        base_url=config.get("llm_base_url"),
        api_key_masked=_mask_key(api_key),
        has_api_key=bool(api_key),
        api_port=paths.port,
        ui_port=paths.ui_port,
        ui_port_is_default=ENV_CP_PORT not in raw,
        api_version=raw.get(_ENV_API_VERSION),
        cp_version=raw.get(_ENV_CP_VERSION),
    )


def save_llm_config(
    name: str,
    provider: str,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
    api_port: str | None = None,
    ui_port: str | None = None,
    api_version: str | None = None,
    cp_version: str | None = None,
) -> ProfileConfigView:
    """Write the wizard's LLM settings into a profile's .env.

    Existing unrelated keys (idle timeout, custom vars) are preserved. Passing
    ``API_KEY_UNCHANGED`` (or None) for ``api_key`` keeps the stored key; passing
    empty string clears it. An empty ``model`` removes the override. ``base_url``
    is no longer in the wizard: None preserves an existing override, "" clears it.
    ``api_port`` pins HINDSIGHT_API_PORT; an empty ``ui_port`` removes the override
    so the control-plane port follows API + offset. Empty version fields remove
    the per-profile version pin (use the embed default).
    """
    name = normalize_profile(name)
    if not provider:
        raise ValueError("provider is required")

    env = _read_raw_env(name)
    env[_ENV_PROVIDER] = provider

    if api_key == API_KEY_UNCHANGED or api_key is None:
        pass  # keep whatever is stored
    elif api_key == "":
        env.pop(_ENV_API_KEY, None)
    else:
        env[_ENV_API_KEY] = api_key

    if model:
        env[_ENV_MODEL] = model
    else:
        env.pop(_ENV_MODEL, None)

    # base_url is no longer in the wizard; only touch it when explicitly passed
    # (None preserves an existing override; "" clears it). Editable via raw .env.
    if base_url:
        env[_ENV_BASE_URL] = base_url
    elif base_url == "":
        env.pop(_ENV_BASE_URL, None)

    # Ports live in the .env. API port is pinned; an empty UI port means "derive
    # from API + offset" (remove the override). No validation by design.
    if api_port:
        env[ENV_API_PORT] = api_port
    if ui_port:
        env[ENV_CP_PORT] = ui_port
    else:
        env.pop(ENV_CP_PORT, None)

    # Component versions: pin if set, else remove the override (use embed default).
    if api_version:
        env[_ENV_API_VERSION] = api_version
    else:
        env.pop(_ENV_API_VERSION, None)
    if cp_version:
        env[_ENV_CP_VERSION] = cp_version
    else:
        env.pop(_ENV_CP_VERSION, None)

    _write_raw_env(name, env)
    return get_profile_config(name)


def _daemon_config(name: str) -> dict[str, str]:
    """Build the config dict daemon_client.ensure_daemon_running expects."""
    return ProfileManager().load_profile_config(name)


def _http_get(url: str, timeout: float = 2.0) -> httpx.Response | None:
    """GET helper that returns the response or None on failure (patchable in tests)."""
    try:
        with httpx.Client(timeout=timeout) as client:
            return client.get(url)
    except Exception:
        return None


def health(name: str) -> HealthView:
    """Probe the profile's API daemon (/health) and control-plane UI right now.

    This is polled regularly by the control center for the selected profile.
    """
    name = normalize_profile(name)
    paths = ProfileManager().resolve_profile_paths(name)
    api_ok = False
    resp = _http_get(f"http://127.0.0.1:{paths.port}/health")
    if resp is None:
        api_detail = "unreachable"
    elif resp.status_code != 200:
        api_detail = f"HTTP {resp.status_code}"
    else:
        try:
            data = resp.json()
            api_ok = data.get("status") == "healthy"
            api_detail = f"{data.get('status', 'ok')} · db {data.get('database', '?')}"
        except Exception:
            api_ok, api_detail = True, "ok"
    return HealthView(api_ok=api_ok, api_detail=api_detail, ui_ok=daemon_client.is_ui_running(name))


def daemon_status(name: str) -> DaemonActionResult:
    """Report whether the profile's daemon is running, with its URL."""
    name = normalize_profile(name)
    running = daemon_client.is_daemon_running(name)
    url = _browser_url(daemon_client.get_daemon_url(name)) if running else None
    return DaemonActionResult(
        ok=True,
        running=running,
        url=url,
        message="Daemon is running." if running else "Daemon is not running.",
    )


def start_daemon(name: str) -> DaemonActionResult:
    """Start the profile's daemon (no-op if already healthy)."""
    name = normalize_profile(name)
    ok = daemon_client.ensure_daemon_running(_daemon_config(name), name)
    running = daemon_client.is_daemon_running(name)
    return DaemonActionResult(
        ok=ok,
        running=running,
        url=_browser_url(daemon_client.get_daemon_url(name)) if running else None,
        message="Daemon started." if ok else "Failed to start daemon — check the logs.",
    )


def stop_daemon(name: str) -> DaemonActionResult:
    """Stop the profile's daemon."""
    name = normalize_profile(name)
    if not daemon_client.is_daemon_running(name):
        return DaemonActionResult(ok=True, running=False, url=None, message="Daemon was not running.")
    ok = daemon_client.stop_daemon(name)
    return DaemonActionResult(
        ok=ok,
        running=daemon_client.is_daemon_running(name),
        url=None,
        message="Daemon stopped." if ok else "Failed to stop daemon.",
    )


def restart_daemon(name: str) -> DaemonActionResult:
    """Restart the profile's daemon to pick up new config."""
    name = normalize_profile(name)
    if daemon_client.is_daemon_running(name):
        if not daemon_client.stop_daemon(name):
            return DaemonActionResult(ok=False, running=True, url=None, message="Failed to stop daemon for restart.")
    return start_daemon(name)


# --------------------------------------------------------------------------
# raw .env editing
# --------------------------------------------------------------------------
def read_env_file(name: str) -> EnvFileView:
    """Return the raw text of a profile's .env for direct editing.

    Unlike get_profile_config, this exposes the file verbatim (API key
    included) — it's a power-user editor for a local, single-user tool.
    """
    name = normalize_profile(name)
    path = ProfileManager().resolve_profile_paths(name).config
    content = path.read_text() if path.exists() else ""
    return EnvFileView(
        name=name,
        display_name=_display_name(name),
        path=str(path),
        content=content,
        exists=path.exists(),
    )


def write_env_file(name: str, content: str) -> EnvFileView:
    """Overwrite a profile's .env with raw text (chmod 0600)."""
    name = normalize_profile(name)
    path = ProfileManager().resolve_profile_paths(name).config
    path.parent.mkdir(parents=True, exist_ok=True)
    # Normalize to a trailing newline so the file stays POSIX-clean.
    if content and not content.endswith("\n"):
        content += "\n"
    path.write_text(content)
    path.chmod(0o600)
    return read_env_file(name)


# --------------------------------------------------------------------------
# file references + logs
# --------------------------------------------------------------------------
def get_profile_paths(name: str) -> ProfilePathsView:
    """Resolve the on-disk references and URLs for a profile."""
    import os
    from pathlib import Path

    from .. import get_embed_manager

    name = normalize_profile(name)
    paths = ProfileManager().resolve_profile_paths(name)

    # Honor the same DB override the daemon uses, else the per-profile pg0.
    database_url = os.getenv("HINDSIGHT_EMBED_API_DATABASE_URL") or get_embed_manager().get_database_url(name)
    database_path: str | None = None
    if database_url.startswith("pg0://"):
        instance = database_url.removeprefix("pg0://")
        database_path = str(Path.home() / ".pg0" / "instances" / instance)

    return ProfilePathsView(
        name=name,
        display_name=_display_name(name),
        port=paths.port,
        config_path=str(paths.config),
        log_path=str(paths.log),
        lock_path=str(paths.lock),
        database_url=database_url,
        database_path=database_path,
        daemon_url=_browser_url(daemon_client.get_daemon_url(name)),
        ui_url=_browser_url(daemon_client.get_ui_url(name)),
    )


def tail_log(name: str, lines: int = 200, source: str = "daemon") -> LogTailView:
    """Return the last ``lines`` lines of the profile's daemon or control-plane log.

    ``source`` is "daemon" (the API daemon log) or "ui" (the control-plane log,
    so you can see why the UI failed to start).
    """
    name = normalize_profile(name)
    paths = ProfileManager().resolve_profile_paths(name)
    path = paths.ui_log if source == "ui" else paths.log
    if not path.exists():
        return LogTailView(path=str(path), exists=False, content="")
    # Small logs — reading then slicing is simpler than seeking and is fine for
    # the interactive tail in the UI.
    tail = path.read_text(errors="replace").splitlines()[-max(lines, 1) :]
    return LogTailView(path=str(path), exists=True, content="\n".join(tail))


# --------------------------------------------------------------------------
# control plane (Next.js UI)
# --------------------------------------------------------------------------
def ui_status(name: str) -> UiStatusView:
    """Whether the control plane UI is running for this profile, and its URL."""
    name = normalize_profile(name)
    return UiStatusView(running=daemon_client.is_ui_running(name), url=_browser_url(daemon_client.get_ui_url(name)))


def start_ui(name: str) -> UiStatusView:
    """Start the control plane UI (ensuring the daemon is up first)."""
    name = normalize_profile(name)
    # The control plane talks to the daemon API, so it needs the daemon running.
    daemon_client.ensure_daemon_running(_daemon_config(name), name)
    daemon_client.start_ui(name)
    return ui_status(name)


def stop_ui(name: str) -> UiStatusView:
    """Stop the control plane UI."""
    name = normalize_profile(name)
    daemon_client.stop_ui(name)
    return ui_status(name)


def restart_ui(name: str) -> UiStatusView:
    """Restart the control plane UI (e.g. to pick up a new UI port).

    Always stops first — after a UI-port change the UI is on the *old* port, so
    an is-running check on the configured port would skip the stop and orphan it.
    stop_ui kills both the configured and the recorded (actual) port.
    """
    name = normalize_profile(name)
    daemon_client.stop_ui(name)
    return start_ui(name)


def delete_profile(name: str) -> ProfileDeleteResult:
    """Stop the profile's daemon (if running) and delete the profile.

    The default profile cannot be deleted (it has no metadata entry and is the
    fallback) — ProfileManager raises, which we surface as a failure.
    """
    name = normalize_profile(name)
    if not name:
        return ProfileDeleteResult(ok=False, message="The default profile cannot be deleted.")
    # Free the port / processes before removing the config it points at.
    if daemon_client.is_daemon_running(name):
        daemon_client.stop_daemon(name)
    if daemon_client.is_ui_running(name):
        daemon_client.stop_ui(name)
    try:
        ProfileManager().delete_profile(name)
    except ValueError as exc:
        return ProfileDeleteResult(ok=False, message=str(exc))
    return ProfileDeleteResult(ok=True, message=f"Deleted profile '{name}'.")


def to_json(obj: object) -> dict:
    """Serialize a control-center dataclass to a JSON-ready dict."""
    return asdict(obj)  # type: ignore[call-overload]
