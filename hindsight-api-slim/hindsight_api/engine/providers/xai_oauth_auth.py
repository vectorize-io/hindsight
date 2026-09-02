"""
xAI OAuth credential manager for the ``xai-oauth`` subscription provider.

Serves a consumer SuperGrok subscription — flat-rate, no API key — by holding a
user-consented OAuth grant obtained from xAI's own OIDC issuer at
``https://auth.x.ai`` through the RFC 8628 device-code flow, and refreshing it
with a plain ``grant_type=refresh_token`` POST. For API-key access to the same
``api.x.ai`` endpoint, use ``provider: openai`` with a base URL; this module is
the subscription lane's credential half.

Relationship to the other subscription providers
------------------------------------------------
``codex_auth.py`` and ``nous_auth.py`` read a credential file that a vendor CLI
created and keep it fresh. This module follows their store conventions — one
JSON file, ``0600``, temp-file + ``os.replace`` rewrite, a per-store
``fcntl.flock`` advisory lock — but owns the login itself, because it must never
read or write the Grok CLI's ``~/.grok/auth.json``. The login is therefore an
explicit, interactive entrypoint (``python -m
hindsight_api.engine.providers.xai_oauth_auth login``) and nothing on the
request path can reach it.

Store
-----
``~/.hindsight/xai_oauth.json`` by default (the directory ``daemon.py`` and
``llamacpp_llm.py`` already use), overridable with
``HINDSIGHT_API_XAI_OAUTH_TOKEN_PATH``. It holds the access token, the refresh
token, ``expires_at``, ``obtained_at``, the granted scope, and the discovered
``token_endpoint``.

xAI issues a **new refresh token with every refresh** and retires the old one,
so the store is not a read-only credential drop: whichever process refreshes
must be able to write back, and every process serving this provider must read
the same file rather than its own copy of one login's output.

Multi-instance safety
---------------------
Refresh state lives in the store, not on the manager: several provider
instances (one per configured lane) share one credential. Each refresh re-reads
the store after taking the lock and skips when a sibling's refresh already
landed, and the anti-spin minimum gap is measured from the store's
``obtained_at`` rather than an in-memory clock.

Logging
-------
Token values, refresh tokens, user codes and Authorization headers are never
passed to the logger at any level; log lines carry byte counts, expiry
timestamps, scope names and status codes only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CLIENT_ID",
    "DEFAULT_MIN_REFRESH_GAP_SECONDS",
    "DEFAULT_REFRESH_SKEW_SECONDS",
    "DEFAULT_REFRESH_TIMEOUT_SECONDS",
    "DEFAULT_SCOPE",
    "DEVICE_CODE_GRANT_TYPE",
    "ENV_CLIENT_ID",
    "ENV_REFRESH_SKEW_SECONDS",
    "ENV_REFRESH_TIMEOUT_SECONDS",
    "ENV_SCOPE",
    "ENV_TOKEN_PATH",
    "LOGIN_COMMAND",
    "StoredCredential",
    "XaiOAuthDiscoveryError",
    "XaiOAuthError",
    "XaiOAuthLoginRequiredError",
    "XaiOAuthManager",
    "XaiOAuthRefreshError",
    "default_token_path",
    "device_code_login",
    "discover_endpoints",
    "read_credential",
    "request_device_code",
    "poll_device_token",
    "write_credential",
]


# ---------------------------------------------------------------------------
# Environment surface
# ---------------------------------------------------------------------------

#: Relocate the token store. The read and the write both follow it, so this is
#: a full relocation rather than a read-side override.
ENV_TOKEN_PATH = "HINDSIGHT_API_XAI_OAUTH_TOKEN_PATH"

#: Override the OAuth client id used for both login and refresh.
ENV_CLIENT_ID = "HINDSIGHT_API_XAI_OAUTH_CLIENT_ID"

#: Override the scope string requested at login.
ENV_SCOPE = "HINDSIGHT_API_XAI_OAUTH_SCOPE"

#: Per-HTTP-call timeout for discovery, device-code and refresh requests.
ENV_REFRESH_TIMEOUT_SECONDS = "HINDSIGHT_API_XAI_OAUTH_REFRESH_TIMEOUT_SECONDS"

#: How long before ``expires_at`` a token counts as due for refresh.
ENV_REFRESH_SKEW_SECONDS = "HINDSIGHT_API_XAI_OAUTH_REFRESH_SKEW_SECONDS"


# ---------------------------------------------------------------------------
# Vendor constants
# ---------------------------------------------------------------------------

XAI_OAUTH_ISSUER = "https://auth.x.ai"
XAI_OAUTH_DISCOVERY_URL = f"{XAI_OAUTH_ISSUER}/.well-known/openid-configuration"

#: Fallback device-authorization endpoint, used only when the discovery
#: document omits ``device_authorization_endpoint``.
XAI_OAUTH_DEVICE_CODE_URL = f"{XAI_OAUTH_ISSUER}/oauth2/device/code"

#: Public OAuth client id published in xAI's Apache-2.0 Grok CLI sources
#: (``crates/codegen/xai-grok-shell/src/auth/config.rs`` in
#: github.com/xai-org/grok-build). Not a secret: a device-code public client has
#: no client secret by construction.
DEFAULT_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"

#: Scope string the vendor's own device-code login requests.
DEFAULT_SCOPE = "openid profile email offline_access grok-cli:access api:access"

DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"

#: Refresh this many seconds before ``expires_at``. 60s matches
#: ``codex_auth.py``'s skew; deployments that touch the provider rarely (a cron
#: or gateway shape) can widen it with ``ENV_REFRESH_SKEW_SECONDS``.
DEFAULT_REFRESH_SKEW_SECONDS = 60.0

#: Hard per-request timeout for every call this module makes.
DEFAULT_REFRESH_TIMEOUT_SECONDS = 20.0

#: Anti-spin floor: a credential obtained less than this many seconds ago is
#: not refreshed again. Measured from the store's ``obtained_at``.
DEFAULT_MIN_REFRESH_GAP_SECONDS = 30.0

#: Wait this long for the cross-process store lock before giving up.
AUTH_LOCK_TIMEOUT_SECONDS = 20.0

#: OAuth token-endpoint statuses that mean the grant itself is dead.
#:
#: RFC 6749 section 5.2 answers a spent or revoked ``refresh_token`` with 400
#: ``invalid_grant``, and 401 covers client authentication. A 403 is not a
#: token-endpoint error shape at all — it is what an edge or policy layer in
#: front of the issuer returns — so it stays off this set: quarantining on one
#: would trade a transient upstream refusal for a mandatory interactive
#: re-login, which is the same reasoning ``xai_oauth_llm`` applies to a 403
#: from the API side.
TERMINAL_REFRESH_STATUSES = frozenset({400, 401})

LOGIN_COMMAND = "python -m hindsight_api.engine.providers.xai_oauth_auth login"

_STORE_LOCKS_GUARD = threading.Lock()
_STORE_LOCKS: dict[Path, threading.Lock] = {}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class XaiOAuthError(RuntimeError):
    """Base class for every credential-side failure in this module."""


class XaiOAuthLoginRequiredError(XaiOAuthError):
    """Raised when only an interactive login can restore service.

    Carries the exact command to run. Nothing on the request path ever starts
    the device-code flow itself, so this is the terminus of the unattended
    path rather than a prompt.
    """


class XaiOAuthDiscoveryError(XaiOAuthError):
    """Raised when OIDC discovery fails or returns an endpoint off the xAI origin."""


class XaiOAuthRefreshError(XaiOAuthError):
    """Raised when a refresh fails in a way a later attempt might survive."""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoredCredential:
    """One credential as it exists on disk.

    ``expires_at`` is ``None`` when the store never recorded one. Such a
    credential is treated as due for refresh, never as valid.
    """

    access_token: str
    refresh_token: str
    expires_at: float | None
    obtained_at: float
    scope: str
    token_endpoint: str

    def seconds_left(self, now: float | None = None) -> float:
        """Seconds until expiry, or ``0.0`` when the store recorded no expiry."""
        if self.expires_at is None:
            return 0.0
        return self.expires_at - (time.time() if now is None else now)


def default_token_path() -> Path:
    """Return the token-store path, honoring ``ENV_TOKEN_PATH``.

    Resolved on each call rather than cached at import, so the environment is
    read at the point of use.
    """
    configured = os.environ.get(ENV_TOKEN_PATH, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".hindsight" / "xai_oauth.json"


def _path_scoped_lock(token_path: Path) -> threading.Lock:
    """Return the process-wide lock for one store path."""
    key = token_path.expanduser().resolve(strict=False)
    with _STORE_LOCKS_GUARD:
        lock = _STORE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _STORE_LOCKS[key] = lock
        return lock


@contextlib.contextmanager
def token_store_lock(token_path: Path, timeout_seconds: float = AUTH_LOCK_TIMEOUT_SECONDS) -> Iterator[None]:
    """Hold the advisory lock for one token store.

    An in-process lock keyed by the resolved path serialises this process's own
    managers; a ``fcntl.flock`` on ``<store>.lock`` extends that across
    processes. Where ``fcntl`` is unavailable (Windows) only the in-process lock
    applies — the same degradation ``codex_auth.py`` and ``nous_auth.py`` take.
    """
    with _path_scoped_lock(token_path):
        if fcntl is None:  # pragma: no cover - Windows
            logger.debug("fcntl unavailable; xai-oauth refresh proceeds without a cross-process lock.")
            yield
            return

        lock_path = token_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+") as lock_file:
            deadline = time.monotonic() + max(1.0, timeout_seconds)
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Timed out waiting for the xai-oauth token store lock") from None
                    time.sleep(0.05)
            try:
                yield
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_credential(token_path: Path) -> StoredCredential:
    """Load the credential from ``token_path``.

    Raises
    ------
    XaiOAuthLoginRequiredError:
        When the file is missing, unreadable, malformed, or carries no
        ``refresh_token`` — every shape from which only a login recovers.
    """
    if not token_path.exists():
        raise XaiOAuthLoginRequiredError(
            f"No xai-oauth credential found at {token_path}. "
            f"Run the xai-oauth login on the host that owns the token store: {LOGIN_COMMAND}"
        )

    try:
        with open(token_path) as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise XaiOAuthLoginRequiredError(
            f"The xai-oauth credential at {token_path} is unreadable ({type(exc).__name__}). "
            f"Run the xai-oauth login on the host that owns the token store: {LOGIN_COMMAND}"
        ) from exc

    tokens = data.get("tokens") if isinstance(data, dict) else None
    tokens = tokens if isinstance(tokens, dict) else {}
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")

    if not isinstance(refresh_token, str) or not refresh_token:
        raise XaiOAuthLoginRequiredError(
            f"The xai-oauth credential at {token_path} has no refresh_token. "
            f"Run the xai-oauth login on the host that owns the token store: {LOGIN_COMMAND}"
        )

    return StoredCredential(
        access_token=access_token if isinstance(access_token, str) else "",
        refresh_token=refresh_token,
        expires_at=_coerce_float(data.get("expires_at")),
        obtained_at=_coerce_float(data.get("obtained_at")) or 0.0,
        scope=str(data.get("scope") or ""),
        token_endpoint=str(data.get("token_endpoint") or ""),
    )


def write_credential(token_path: Path, credential: StoredCredential) -> None:
    """Persist ``credential`` to ``token_path`` as one atomic replacement.

    The payload is written to a sibling temp file, fsynced, chmod-ed ``0600``
    where the platform supports it, and moved into place with ``os.replace`` —
    so a reader never observes a partially written store.
    """
    payload = {
        "auth_mode": "xai-oauth-device-code",
        "tokens": {
            "access_token": credential.access_token,
            "refresh_token": credential.refresh_token,
        },
        "expires_at": credential.expires_at,
        "obtained_at": credential.obtained_at,
        "scope": credential.scope,
        "token_endpoint": credential.token_endpoint,
    }

    parent = token_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".xai_oauth.", suffix=".json.tmp", dir=str(parent))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        with contextlib.suppress(OSError):
            os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, token_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
    with contextlib.suppress(OSError):
        os.chmod(token_path, 0o600)


def quarantine_credential(token_path: Path, *, code: str, reason: str) -> None:
    """Strip the dead tokens from the store and record why.

    A grant the token endpoint has rejected terminally cannot be revived by
    retrying, so the tokens are removed and the next process fails fast on the
    login remediation instead of re-attempting a refresh that cannot succeed.
    Only the error code and reason are recorded — never a token value. Must be
    called while holding :func:`token_store_lock`.
    """
    try:
        with open(token_path) as handle:
            data = json.load(handle)
        current: dict[str, Any] = data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        current = {}

    current["tokens"] = {}
    current["expires_at"] = None
    current["last_auth_error"] = {
        "code": code,
        "reason": reason,
        "at": time.time(),
    }

    parent = token_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".xai_oauth.", suffix=".json.tmp", dir=str(parent))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(current, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        with contextlib.suppress(OSError):
            os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, token_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def validate_oauth_endpoint(url: str, *, field: str) -> str:
    """Return ``url`` if it is HTTPS on the xAI origin, else raise.

    The discovery result is cached in the store and every later refresh posts
    the refresh token to the cached ``token_endpoint``, so a substituted
    endpoint would turn one bad discovery response into a standing credential
    leak. Pinning scheme and host to ``x.ai`` (or a ``*.x.ai`` subdomain) is
    what stops the substitution outliving the request that carried it.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise XaiOAuthDiscoveryError(f"xAI OIDC discovery returned a non-HTTPS {field}: {url!r}")
    host = (parsed.hostname or "").lower()
    if not host:
        raise XaiOAuthDiscoveryError(f"xAI OIDC discovery {field} has no hostname: {url!r}")
    if host != "x.ai" and not host.endswith(".x.ai"):
        raise XaiOAuthDiscoveryError(
            f"xAI OIDC discovery {field} host {host!r} is not on the xAI origin (expected x.ai or a *.x.ai subdomain)"
        )
    return url


def discover_endpoints(client: httpx.Client) -> dict[str, str]:
    """Fetch the OIDC discovery document and return the endpoints we use.

    Returns ``token_endpoint`` and ``device_authorization_endpoint``. Both are
    validated against the xAI origin. When the document omits
    ``device_authorization_endpoint``, :data:`XAI_OAUTH_DEVICE_CODE_URL` is used
    instead — RFC 8628 registers the metadata key but does not require issuers
    to publish it.
    """
    try:
        response = client.get(XAI_OAUTH_DISCOVERY_URL, headers={"Accept": "application/json"})
    except httpx.RequestError as exc:
        raise XaiOAuthDiscoveryError(f"xAI OIDC discovery request failed: {type(exc).__name__}") from exc

    if response.status_code != 200:
        raise XaiOAuthDiscoveryError(f"xAI OIDC discovery returned HTTP {response.status_code}")

    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise XaiOAuthDiscoveryError("xAI OIDC discovery returned a non-JSON body") from exc

    if not isinstance(payload, dict):
        raise XaiOAuthDiscoveryError("xAI OIDC discovery body was not a JSON object")

    token_endpoint = str(payload.get("token_endpoint") or "").strip()
    if not token_endpoint:
        raise XaiOAuthDiscoveryError("xAI OIDC discovery body has no token_endpoint")
    device_endpoint = str(payload.get("device_authorization_endpoint") or "").strip() or XAI_OAUTH_DEVICE_CODE_URL

    return {
        "token_endpoint": validate_oauth_endpoint(token_endpoint, field="token_endpoint"),
        "device_authorization_endpoint": validate_oauth_endpoint(
            device_endpoint, field="device_authorization_endpoint"
        ),
    }


# ---------------------------------------------------------------------------
# Device-code login (interactive only)
# ---------------------------------------------------------------------------


def request_device_code(client: httpx.Client, *, device_endpoint: str, client_id: str, scope: str) -> dict[str, Any]:
    """Start the device-code flow and return the authorization response.

    Raises :class:`XaiOAuthError` when the endpoint answers non-200 or omits a
    field RFC 8628 section 3.2 makes required.
    """
    response = client.post(
        device_endpoint,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        data={"client_id": client_id, "scope": scope},
    )
    if response.status_code != 200:
        raise XaiOAuthError(f"xAI device-code request failed with HTTP {response.status_code}")

    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise XaiOAuthError("xAI device-code response was not JSON") from exc
    if not isinstance(payload, dict):
        raise XaiOAuthError("xAI device-code response was not a JSON object")

    missing = [key for key in ("device_code", "user_code", "verification_uri", "expires_in") if key not in payload]
    if missing:
        raise XaiOAuthError(f"xAI device-code response is missing fields: {', '.join(missing)}")
    return payload


def poll_device_token(
    client: httpx.Client,
    *,
    token_endpoint: str,
    device_code: str,
    client_id: str,
    expires_in: int,
    interval: int,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Poll the token endpoint until the user approves, or the code dies.

    Follows RFC 8628 section 3.5: waits ``interval`` seconds between polls,
    widens that interval by 5 seconds on ``slow_down``, keeps polling on
    ``authorization_pending``, and raises on ``expired_token`` or any other
    error code. Stops once ``expires_in`` seconds have elapsed.

    ``sleeper`` and ``monotonic`` are injected so the timing rules can be
    asserted without wall-clock waits.
    """
    deadline = monotonic() + max(1, int(expires_in))
    current_interval = max(1, int(interval))

    while monotonic() < deadline:
        response = client.post(
            token_endpoint,
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            data={
                "grant_type": DEVICE_CODE_GRANT_TYPE,
                "client_id": client_id,
                "device_code": device_code,
            },
        )
        if response.status_code == 200:
            payload = response.json()
            if not isinstance(payload, dict) or not payload.get("access_token"):
                raise XaiOAuthError("xAI device-code token response carried no access_token")
            if not payload.get("refresh_token"):
                raise XaiOAuthError("xAI device-code token response carried no refresh_token")
            return payload

        try:
            error_payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise XaiOAuthError(
                f"xAI device-code polling returned a non-JSON HTTP {response.status_code} body"
            ) from exc

        error_code = str(error_payload.get("error") or "") if isinstance(error_payload, dict) else ""
        if error_code == "authorization_pending":
            sleeper(current_interval)
            continue
        if error_code == "slow_down":
            current_interval += 5
            sleeper(current_interval)
            continue
        if error_code == "expired_token":
            raise XaiOAuthError("The xAI device code expired before it was approved. Start the login again.")
        raise XaiOAuthError(f"xAI device-code polling failed with error {error_code or 'unknown'}")

    raise XaiOAuthError("Timed out waiting for xAI device-code approval.")


def device_code_login(
    *,
    token_path: Path | None = None,
    client_id: str | None = None,
    scope: str | None = None,
    timeout_seconds: float | None = None,
    writer: Callable[[str], None] = print,
) -> Path:
    """Run the interactive device-code login and write the credential store.

    Prints the verification URL and user code for the operator to approve in a
    browser, then polls until the grant lands. The user code is written through
    ``writer`` (stdout by default) and is never passed to the logger.

    Returns the store path that was written. This function is only reachable
    from the module's ``login`` entrypoint — no request path calls it.
    """
    path = token_path or default_token_path()
    resolved_client_id = client_id or os.environ.get(ENV_CLIENT_ID, "").strip() or DEFAULT_CLIENT_ID
    resolved_scope = scope or os.environ.get(ENV_SCOPE, "").strip() or DEFAULT_SCOPE
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else _env_float(ENV_REFRESH_TIMEOUT_SECONDS, DEFAULT_REFRESH_TIMEOUT_SECONDS)
    )

    with httpx.Client(timeout=httpx.Timeout(max(20.0, timeout)), headers={"Accept": "application/json"}) as client:
        endpoints = discover_endpoints(client)
        device = request_device_code(
            client,
            device_endpoint=endpoints["device_authorization_endpoint"],
            client_id=resolved_client_id,
            scope=resolved_scope,
        )
        verification = str(device.get("verification_uri_complete") or device["verification_uri"])
        writer("")
        writer("To authorize Hindsight against your SuperGrok subscription:")
        writer(f"  1. Open: {verification}")
        writer(f"  2. If prompted, enter code: {device['user_code']}")
        writer("Waiting for approval...")

        payload = poll_device_token(
            client,
            token_endpoint=endpoints["token_endpoint"],
            device_code=str(device["device_code"]),
            client_id=resolved_client_id,
            expires_in=int(device["expires_in"]),
            interval=int(device.get("interval") or 5),
        )

    now = time.time()
    expires_in = _coerce_float(payload.get("expires_in"))
    credential = StoredCredential(
        access_token=str(payload["access_token"]),
        refresh_token=str(payload["refresh_token"]),
        expires_at=(now + expires_in) if expires_in is not None else None,
        obtained_at=now,
        scope=str(payload.get("scope") or resolved_scope),
        token_endpoint=endpoints["token_endpoint"],
    )
    with token_store_lock(path):
        write_credential(path, credential)
    logger.info(
        "xai-oauth credential stored at %s (scope=%s, expires_at=%s)",
        path,
        credential.scope,
        credential.expires_at,
    )
    writer(f"Stored the xai-oauth credential at {path}")
    return path


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s; using %s", name, default)
        return default


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class XaiOAuthManager:
    """Keeps one xAI OAuth credential usable for unattended requests.

    Proactively refreshes a token inside the skew window and reactively
    refreshes once when the API rejects one. Never starts the device-code flow:
    a credential that only a login can restore raises
    :class:`XaiOAuthLoginRequiredError` instead.
    """

    def __init__(
        self,
        token_path: Path | None = None,
        *,
        client_id: str | None = None,
        refresh_skew_seconds: float | None = None,
        refresh_timeout_seconds: float | None = None,
        min_refresh_gap_seconds: float = DEFAULT_MIN_REFRESH_GAP_SECONDS,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._token_path = token_path or default_token_path()
        self._client_id = client_id or os.environ.get(ENV_CLIENT_ID, "").strip() or DEFAULT_CLIENT_ID
        self._skew_seconds = (
            refresh_skew_seconds
            if refresh_skew_seconds is not None
            else _env_float(ENV_REFRESH_SKEW_SECONDS, DEFAULT_REFRESH_SKEW_SECONDS)
        )
        self._timeout_seconds = (
            refresh_timeout_seconds
            if refresh_timeout_seconds is not None
            else _env_float(ENV_REFRESH_TIMEOUT_SECONDS, DEFAULT_REFRESH_TIMEOUT_SECONDS)
        )
        self._min_refresh_gap_seconds = min_refresh_gap_seconds
        self._owns_client = http_client is None
        self._http_client = http_client or httpx.Client(timeout=httpx.Timeout(self._timeout_seconds))

    @property
    def token_path(self) -> Path:
        return self._token_path

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_access_token(self, min_ttl_seconds: float | None = None) -> str:
        """Return an access token good for at least ``min_ttl_seconds``.

        ``min_ttl_seconds`` defaults to the configured skew. A credential whose
        store recorded no expiry counts as having none left, so it is refreshed
        rather than used on trust.
        """
        required_ttl = self._skew_seconds if min_ttl_seconds is None else max(min_ttl_seconds, 0.0)
        credential = read_credential(self._token_path)
        if credential.access_token and credential.seconds_left() > required_ttl:
            return credential.access_token
        return self.refresh(reason="proactive (token inside the refresh window)", required_ttl=required_ttl)

    def refresh(
        self,
        *,
        reason: str = "",
        required_ttl: float | None = None,
        rejected_token: str | None = None,
    ) -> str:
        """Refresh the stored credential and return the new access token.

        Takes the store lock, then re-reads the store: when a sibling manager's
        refresh already produced a token that satisfies ``required_ttl`` (or,
        for a reactive refresh, any token other than ``rejected_token``), that
        token is returned and no request is sent. This is what keeps N provider
        instances sharing one credential down to one refresh.

        Raises
        ------
        XaiOAuthLoginRequiredError:
            When the token endpoint terminally rejects the grant, or the store
            holds nothing to refresh.
        XaiOAuthRefreshError:
            On a transient failure, and when the anti-spin minimum gap has not
            elapsed since the stored credential was obtained.
        """
        ttl = self._skew_seconds if required_ttl is None else required_ttl

        with token_store_lock(self._token_path, timeout_seconds=max(AUTH_LOCK_TIMEOUT_SECONDS, self._timeout_seconds)):
            credential = read_credential(self._token_path)

            if rejected_token is None:
                if credential.access_token and credential.seconds_left() > ttl:
                    logger.debug("xai-oauth refresh skipped: the store already holds a token outside the window")
                    return credential.access_token
            elif credential.access_token and credential.access_token != rejected_token:
                logger.debug("xai-oauth refresh skipped: the store already holds a token newer than the rejected one")
                return credential.access_token

            gap = time.time() - credential.obtained_at
            if gap < self._min_refresh_gap_seconds:
                raise XaiOAuthRefreshError(
                    f"xai-oauth refused to refresh again {gap:.1f}s after the stored credential was obtained "
                    f"(minimum gap {self._min_refresh_gap_seconds:.0f}s). The upstream is rejecting a token that "
                    "was just issued."
                )

            return self._refresh_locked(credential, reason=reason)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _token_endpoint(self, credential: StoredCredential) -> str:
        if credential.token_endpoint:
            return validate_oauth_endpoint(credential.token_endpoint, field="token_endpoint")
        return discover_endpoints(self._http_client)["token_endpoint"]

    def _refresh_locked(self, credential: StoredCredential, *, reason: str) -> str:
        """Exchange the refresh token and persist the result. Lock must be held.

        A terminal status (see :data:`TERMINAL_REFRESH_STATUSES`) is retried
        exactly once — a single network-level oddity should not cost the
        operator a re-login — and then, unless the store has meanwhile moved to
        a different grant, quarantines the store and raises the login
        remediation.
        """
        endpoint = self._token_endpoint(credential)
        log_reason = f" ({reason})" if reason else ""
        logger.info("Refreshing the xai-oauth access token%s", log_reason)

        last_status: int | None = None
        for attempt in (1, 2):
            try:
                response = self._http_client.post(
                    endpoint,
                    headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
                    data={
                        "grant_type": "refresh_token",
                        "client_id": self._client_id,
                        "refresh_token": credential.refresh_token,
                    },
                    timeout=self._timeout_seconds,
                )
            except httpx.RequestError as exc:
                raise XaiOAuthRefreshError(f"xai-oauth refresh network error: {type(exc).__name__}") from exc

            if response.status_code == 200:
                return self._persist_refreshed(credential, response)

            last_status = response.status_code
            if response.status_code not in TERMINAL_REFRESH_STATUSES:
                raise XaiOAuthRefreshError(f"xai-oauth refresh failed with HTTP {response.status_code}")

            logger.warning(
                "xai-oauth refresh rejected with HTTP %d (attempt %d/2)",
                response.status_code,
                attempt,
            )

        # xAI rotates the refresh token on every successful refresh, so a
        # rejection can simply mean some other writer spent this one first and
        # the store already holds its replacement. Quarantining then would
        # destroy a live grant and force an interactive login for nothing. The
        # in-process lock plus the recheck in refresh() rule that out among
        # this host's managers, but not against a store shared with a host
        # whose filesystem does not honour flock — so re-read before wiping.
        superseded = self._superseded_access_token(credential)
        if superseded is not None:
            logger.info("xai-oauth refresh rejection ignored: the store already holds a different grant")
            return superseded

        quarantine_credential(
            self._token_path,
            code=f"refresh_rejected_{last_status}",
            reason="the token endpoint rejected the refresh_token twice",
        )
        raise XaiOAuthLoginRequiredError(
            f"The xAI token endpoint rejected the stored grant (HTTP {last_status}). "
            f"Run the xai-oauth login on the host that owns the token store: {LOGIN_COMMAND}"
        )

    def _superseded_access_token(self, rejected: StoredCredential) -> str | None:
        """Return a usable access token when the store moved on under us.

        ``None`` when the store still holds the grant that was just rejected,
        or holds nothing readable — both cases the caller must quarantine.
        """
        try:
            current = read_credential(self._token_path)
        except XaiOAuthError:
            return None
        if current.refresh_token == rejected.refresh_token or not current.access_token:
            return None
        return current.access_token

    def _persist_refreshed(self, credential: StoredCredential, response: httpx.Response) -> str:
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise XaiOAuthRefreshError("xai-oauth refresh returned a non-JSON body") from exc
        if not isinstance(payload, dict):
            raise XaiOAuthRefreshError("xai-oauth refresh body was not a JSON object")

        access_token = str(payload.get("access_token") or "")
        if not access_token:
            raise XaiOAuthRefreshError("xai-oauth refresh returned no access_token")

        now = time.time()
        expires_in = _coerce_float(payload.get("expires_in"))
        refreshed = replace(
            credential,
            access_token=access_token,
            refresh_token=str(payload.get("refresh_token") or credential.refresh_token),
            expires_at=(now + expires_in) if expires_in is not None else None,
            obtained_at=now,
            scope=str(payload.get("scope") or credential.scope),
        )
        write_credential(self._token_path, refreshed)
        logger.info(
            "xai-oauth access token refreshed (%d bytes, expires_at=%s, scope=%s)",
            len(access_token),
            refreshed.expires_at,
            refreshed.scope,
        )
        return access_token

    def close(self) -> None:
        """Close the HTTP client when this manager created it."""
        if self._owns_client:
            self._http_client.close()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Command-line entrypoint: ``python -m ...xai_oauth_auth login``."""
    parser = argparse.ArgumentParser(prog="xai_oauth_auth", description="Manage the xai-oauth credential store.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    login = subparsers.add_parser("login", help="Run the interactive xAI device-code login.")
    login.add_argument("--token-path", default=None, help="Write the credential here instead of the default store.")

    args = parser.parse_args(argv)
    if args.command == "login":
        try:
            device_code_login(token_path=Path(args.token_path) if args.token_path else None)
        except XaiOAuthError as exc:
            print(f"xai-oauth login failed: {exc}", file=sys.stderr)
            return 1
        return 0
    return 1  # pragma: no cover - argparse rejects unknown commands first


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    raise SystemExit(main())
