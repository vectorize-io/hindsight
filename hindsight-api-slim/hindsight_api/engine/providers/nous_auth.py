"""
Native Nous Portal OAuth authentication manager.

The Nous Portal inference endpoint (https://inference-api.nousresearch.com/v1)
speaks the OpenAI-compatible wire format but authenticates with a short-lived,
inference-scoped JWT rather than a static API key. Hermes obtains that JWT once
via an interactive browser login (``hermes portal``) and persists the resulting
OAuth state — ``access_token`` + ``refresh_token`` — under ``providers.nous`` in
``~/.hermes/auth.json``.

This manager reads that file *directly* and refreshes the access token itself,
exactly mirroring ``codex_auth.py`` (read ``~/.codex/auth.json`` + native
refresh). It deliberately does **not** import the Hermes ``hermes_cli`` package:
that package is the interactive CLI, not a library Hindsight can depend on. The
refresh request shape is mirrored from Hermes' own resolver
(``POST {portal}/api/oauth/token`` with an ``x-nous-refresh-token`` header and a
``grant_type=refresh_token`` form body), so server-side changes affect both
clients identically. The inference bearer is the access token itself — in
Hermes' state the ``agent_key`` field is literally ``= access_token``.

Single-use refresh tokens
-------------------------
Nous refresh tokens are single-use with server-side reuse-detection: if two
processes refresh with the same ``refresh_token``, or a rotated token is not
persisted back, the Portal revokes the whole session as a theft signal. Because
Hindsight shares ``~/.hermes/auth.json`` with a possibly-running Hermes agent,
every refresh here is performed while holding the **same cross-process advisory
lock Hermes uses** (``~/.hermes/auth.lock`` via ``fcntl.flock``) and re-reads the
latest ``refresh_token`` from disk under that lock before exchanging it. That is
the protocol Hermes follows too, so the two coordinate safely through the file.

Usage
-----
    mgr = NousAuthManager.from_file()
    token = mgr.ensure_fresh_token()        # proactive; refreshes if near expiry
    ...                                       # use token as Bearer
    mgr.refresh_tokens(force=True)           # reactive, on a 401
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — mirrored from Hermes' canonical Nous resolver
# (hermes_cli/auth.py: DEFAULT_NOUS_* and _refresh_access_token). Endpoints and
# client id are overridable via the same env vars Hermes honours, so a staging
# Portal or a future change can be pointed at without a code change.
# ---------------------------------------------------------------------------

_NOUS_PORTAL_BASE_URL = (
    os.environ.get("HERMES_PORTAL_BASE_URL")
    or os.environ.get("NOUS_PORTAL_BASE_URL")
    or "https://portal.nousresearch.com"
)
_NOUS_INFERENCE_BASE_URL = os.environ.get("NOUS_INFERENCE_BASE_URL") or "https://inference-api.nousresearch.com/v1"
_NOUS_CLIENT_ID = "hermes-cli"

# Proactively refresh this many seconds before the JWT ``exp`` claim — matches
# the 120s skew Hermes' own runtime resolver uses for Nous.
_NOUS_TOKEN_REFRESH_SKEW_SECONDS = 120

# OAuth error codes the Portal returns when the refresh_token itself is no
# longer usable. These are terminal — retrying will not succeed; the user must
# re-run ``hermes portal``.
_NOUS_TERMINAL_REFRESH_ERROR_CODES = frozenset(
    {"invalid_grant", "invalid_token", "refresh_token_reused", "refresh_token_expired"}
)

_AUTH_LOCK_TIMEOUT_SECONDS = 20.0


def _default_auth_file() -> Path:
    return Path.home() / ".hermes" / "auth.json"


class NousNotLoggedInError(RuntimeError):
    """Raised when ``~/.hermes/auth.json`` has no usable Nous OAuth state.

    Remediation: run ``hermes portal`` to log in to Nous Portal.
    """


class NousRefreshExpiredError(RuntimeError):
    """Raised when the Nous refresh_token itself is permanently invalid.

    The user must re-run ``hermes portal`` to obtain new credentials. Callers
    should surface a clear remediation message and stop retrying.
    """


@contextlib.contextmanager
def _hermes_auth_lock(auth_file: Path, timeout_seconds: float = _AUTH_LOCK_TIMEOUT_SECONDS) -> Iterator[None]:
    """Cross-process advisory lock on the Hermes auth store.

    Uses ``<auth_file>.lock`` (i.e. ``~/.hermes/auth.lock``) with
    ``fcntl.flock(LOCK_EX)`` — the exact same lock file and primitive Hermes'
    ``_auth_store_lock`` takes — so a refresh here is mutually exclusive with a
    concurrently-running Hermes agent. Degrades to a no-op (with a debug log)
    where ``fcntl`` is unavailable (Windows); the single-process in-memory lock
    still serialises this process's own refreshes.
    """
    if fcntl is None:  # pragma: no cover - Windows
        logger.debug("fcntl unavailable; Nous refresh proceeds without a cross-process lock.")
        yield
        return

    lock_path = auth_file.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock_file:
        deadline = time.monotonic() + max(1.0, timeout_seconds)
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for the Hermes auth store lock") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class NousAuthManager:
    """Sync Nous Portal OAuth credential manager.

    Holds the access_token + refresh_token in memory and handles
    proactive/reactive refresh. A ``threading.Lock`` gives single-flight
    semantics within the process; the cross-process ``fcntl`` lock guards
    against a concurrent Hermes agent (see module docstring).
    """

    def __init__(
        self,
        access_token: str,
        refresh_token: str | None,
        auth_file: Path,
        *,
        portal_base_url: str = _NOUS_PORTAL_BASE_URL,
        inference_base_url: str = _NOUS_INFERENCE_BASE_URL,
        client_id: str = _NOUS_CLIENT_ID,
    ) -> None:
        self.access_token = access_token
        self.refresh_token = refresh_token
        self._auth_file = auth_file
        self._portal_base_url = portal_base_url.rstrip("/")
        self._inference_base_url = inference_base_url.rstrip("/")
        self._client_id = client_id
        self._lock = threading.Lock()
        self._http_client = httpx.Client(timeout=30.0)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, auth_file: Path | None = None) -> "NousAuthManager":
        """Build a manager from ``providers.nous`` in the Hermes auth store.

        Raises
        ------
        NousNotLoggedInError:
            If the file is missing, unreadable, or has no Nous OAuth state with
            an ``access_token``.
        """
        if auth_file is None:
            auth_file = _default_auth_file()

        if not auth_file.exists():
            raise NousNotLoggedInError(
                f"Hermes auth file not found: {auth_file}. Run 'hermes portal' to log in to Nous Portal."
            )

        try:
            with open(auth_file) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise NousNotLoggedInError(f"Could not read Hermes auth file {auth_file}: {type(e).__name__}") from e

        state = cls._nous_state(data)
        if not state:
            raise NousNotLoggedInError(
                "Hermes is not logged into Nous Portal (no providers.nous OAuth state). Run 'hermes portal'."
            )

        access_token = state.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise NousNotLoggedInError("Nous OAuth state has no access_token. Re-authenticate with 'hermes portal'.")

        return cls(
            access_token=access_token,
            refresh_token=state.get("refresh_token"),
            auth_file=auth_file,
            portal_base_url=cls._optional_url(state.get("portal_base_url")) or _NOUS_PORTAL_BASE_URL,
            inference_base_url=cls._optional_url(state.get("inference_base_url")) or _NOUS_INFERENCE_BASE_URL,
            client_id=str(state.get("client_id") or _NOUS_CLIENT_ID),
        )

    @staticmethod
    def _nous_state(data: dict[str, Any]) -> dict[str, Any]:
        """Pull the ``providers.nous`` state dict out of a loaded auth store."""
        providers = data.get("providers")
        if not isinstance(providers, dict):
            return {}
        state = providers.get("nous")
        return state if isinstance(state, dict) else {}

    @staticmethod
    def _optional_url(value: Any) -> str | None:
        return value.rstrip("/") if isinstance(value, str) and value.strip() else None

    @property
    def base_url(self) -> str:
        return self._inference_base_url

    # ------------------------------------------------------------------
    # Token state
    # ------------------------------------------------------------------

    @staticmethod
    def load_refresh_token_from_file(auth_file: Path) -> str | None:
        """Read ``providers.nous.refresh_token`` from ``auth_file``.

        Returns ``None`` when the file is unreadable or omits the field. Does
        not raise — the caller degrades to using the in-memory token.
        """
        try:
            with open(auth_file) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        return NousAuthManager._nous_state(data).get("refresh_token")

    @staticmethod
    def _decode_jwt_exp_unixtime(token: str) -> int | None:
        """Return the JWT ``exp`` claim as a unix timestamp, or None on failure.

        The signature is not verified — the server is the source of truth on
        acceptance. This only schedules proactive refresh.
        """
        try:
            parts = token.split(".")
            if len(parts) < 2:
                return None
            payload_b64 = parts[1]
            padding = "=" * (-len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64 + padding).decode("utf-8"))
            exp = payload.get("exp")
            return int(exp) if exp is not None else None
        except (ValueError, TypeError, json.JSONDecodeError, binascii.Error):
            return None

    def _token_is_stale(self, skew_seconds: int = _NOUS_TOKEN_REFRESH_SKEW_SECONDS) -> bool:
        """True when the cached access_token is past expiry (with skew).

        Returns False when expiry cannot be determined — we'd rather use a
        possibly-expired token and recover via the reactive 401 path than
        refresh aggressively on every request when ``exp`` is unparseable.
        """
        exp = self._decode_jwt_exp_unixtime(self.access_token)
        if exp is None:
            return False
        return exp <= int(time.time()) + skew_seconds

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_state_atomic(self, updated: dict[str, Any]) -> None:
        """Patch ``providers.nous`` in ``_auth_file`` and write atomically.

        Re-reads the on-disk store first so fields written by Hermes (other
        providers, the credential pool, rotated tokens) are never clobbered,
        then patches only the Nous OAuth fields and ``os.replace``s into place
        (atomic on POSIX within the same filesystem). Must be called while
        holding :func:`_hermes_auth_lock`.
        """
        try:
            with open(self._auth_file) as f:
                loaded = json.load(f)
            current: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            current = {}

        providers = current.get("providers")
        if not isinstance(providers, dict):
            providers = {}
            current["providers"] = providers
        state = providers.get("nous")
        if not isinstance(state, dict):
            state = {}
            providers["nous"] = state

        state.update(updated)
        # The inference bearer is the access token itself; keep agent_key in
        # sync so Hermes' own resolver/status sees the rotation too.
        state["agent_key"] = updated.get("access_token", state.get("access_token"))
        current["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        parent = self._auth_file.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".auth.", suffix=".json.tmp", dir=str(parent))
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(current, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            with contextlib.suppress(OSError):
                os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, self._auth_file)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_oauth_error_code(response: httpx.Response) -> str | None:
        """Pull the OAuth error code out of a 4xx refresh response, if present."""
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(body, dict):
            return None
        err = body.get("error")
        if isinstance(err, str):
            return err
        if isinstance(err, dict) and isinstance(err.get("code"), str):
            return err["code"]
        code = body.get("error_code")
        return code if isinstance(code, str) else None

    def refresh_tokens(self, reason: str = "", *, force: bool = False) -> None:
        """Single-flight Nous OAuth token refresh.

        Serialised through ``self._lock`` (in-process single-flight) and
        :func:`_hermes_auth_lock` (cross-process, vs a running Hermes agent).
        The latest ``refresh_token`` is re-read from disk under the lock before
        the exchange — single-use tokens make using a stale in-memory RT a
        session-revoking mistake.

        Raises
        ------
        NousRefreshExpiredError:
            On a terminal refresh error (expired/reused/invalid grant).
        RuntimeError:
            For other refresh failures (network, 5xx, missing refresh_token).
        """
        token_before_lock = self.access_token
        with self._lock:
            if force:
                if self.access_token != token_before_lock:
                    return  # another caller already refreshed while we waited
            elif not self._token_is_stale():
                return

            with _hermes_auth_lock(self._auth_file):
                # Re-read the freshest refresh_token persisted by whoever rotated
                # last (this process or Hermes). Using a stale RT is exactly what
                # trips the Portal's single-use reuse-detection.
                disk_rt = self.load_refresh_token_from_file(self._auth_file)
                if disk_rt:
                    self.refresh_token = disk_rt

                if not self.refresh_token:
                    raise RuntimeError(
                        "Nous access_token is expired but no refresh_token is available. "
                        "Run 'hermes portal' to re-authenticate."
                    )

                log_reason = f" ({reason})" if reason else ""
                logger.info(f"Refreshing Nous Portal access_token{log_reason}")

                try:
                    response = self._http_client.post(
                        f"{self._portal_base_url}/api/oauth/token",
                        headers={"x-nous-refresh-token": self.refresh_token},
                        data={"grant_type": "refresh_token", "client_id": self._client_id},
                        timeout=30.0,
                    )
                except httpx.RequestError as e:
                    raise RuntimeError(f"Nous OAuth refresh network error: {type(e).__name__}") from e

                if response.status_code != 200:
                    code = self._extract_oauth_error_code(response)
                    if code in _NOUS_TERMINAL_REFRESH_ERROR_CODES or response.status_code in (400, 401):
                        raise NousRefreshExpiredError(
                            f"Nous refresh_token is no longer valid (status={response.status_code}, "
                            f"error={code or 'none'}). Run 'hermes portal' to re-authenticate."
                        )
                    raise RuntimeError(f"Nous OAuth refresh failed with HTTP {response.status_code}")

                try:
                    body = response.json()
                except (json.JSONDecodeError, ValueError) as e:
                    raise RuntimeError(f"Nous OAuth refresh returned non-JSON body: {e}") from e

                new_access = body.get("access_token")
                if not new_access:
                    raise RuntimeError("Nous OAuth refresh returned no access_token")
                new_refresh = body.get("refresh_token") or self.refresh_token

                # Update in-memory state first so waiters see fresh credentials
                # even if the disk write fails.
                self.access_token = new_access
                self.refresh_token = new_refresh

                persisted: dict[str, Any] = {"access_token": new_access, "refresh_token": new_refresh}
                expires_in = body.get("expires_in")
                if isinstance(expires_in, (int, float)):
                    persisted["expires_at"] = datetime.fromtimestamp(
                        time.time() + float(expires_in), tz=timezone.utc
                    ).isoformat()
                try:
                    self._persist_state_atomic(persisted)
                except OSError as e:
                    logger.warning(
                        f"Nous refresh succeeded but persisting auth.json failed: {type(e).__name__}. "
                        "In-memory credentials are current; the on-disk rotated token was not saved."
                    )
                logger.info("Nous Portal access_token refreshed successfully")

    def ensure_fresh_token(self) -> str:
        """Refresh proactively if near/at expiry, then return the bearer token.

        Cheap when fresh (a JWT exp decode + comparison).
        """
        if self._token_is_stale():
            self.refresh_tokens(reason="proactive (token near expiry)")
        return self.access_token

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http_client.close()
