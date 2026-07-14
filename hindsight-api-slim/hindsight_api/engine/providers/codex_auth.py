"""
Shared Codex OAuth authentication manager.

Extracted from ``CodexLLM`` so that both ``CodexLLM`` and
``CodexOAuthEmbeddings`` can share JWT-expiry detection, single-flight
token refresh, and atomic file persistence without duplicating the logic.

Usage
-----
Create a manager from the auth file::

    mgr = CodexAuthManager.from_file()

Then call ``ensure_fresh_token()`` before each outbound request and
``refresh_tokens(reason=..., force=...)`` on a reactive 401.
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
# Module-level constants (shared with codex_llm.py via re-export there)
# ---------------------------------------------------------------------------

# OAuth refresh endpoint and client id, mirrored from the canonical
# ``@openai/codex`` CLI (codex-rs/login/src/auth/manager.rs on
# github.com/openai/codex). The endpoint is overridable via env var so that
# future Codex changes or staging environments can be pointed at without a
# code change — same env var name the upstream CLI uses.
_CODEX_REFRESH_TOKEN_URL = os.environ.get("CODEX_REFRESH_TOKEN_URL_OVERRIDE", "https://auth.openai.com/oauth/token")
_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

# Proactively refresh this many seconds before the JWT ``exp`` claim. The
# upstream Codex CLI uses no skew (it refreshes at ``exp <= now``); the
# extra window reduces races where a request leaves the client with a token
# that the server has already declared expired by the time it arrives.
_CODEX_TOKEN_REFRESH_SKEW_SECONDS = 60

# OAuth error codes that the refresh endpoint returns when the refresh_token
# itself is no longer usable. These are terminal — retrying refresh will not
# succeed; the user must re-run ``codex auth login``.
_CODEX_TERMINAL_REFRESH_ERROR_CODES = frozenset(
    {"refresh_token_expired", "refresh_token_reused", "refresh_token_invalidated"}
)
_CODEX_AUTH_LOCK_TIMEOUT_SECONDS = 20.0
_CODEX_AUTH_LOCKS_GUARD = threading.Lock()
_CODEX_AUTH_LOCKS: dict[Path, threading.Lock] = {}


def default_codex_auth_file() -> Path:
    """Return the path to Codex's ``auth.json``.

    Honors the ``CODEX_HOME`` environment variable — the same variable the
    canonical ``@openai/codex`` CLI uses to relocate its config/credentials
    directory — and falls back to ``~/.codex`` when it is unset or empty.

    Resolved lazily on each call (rather than cached at import time) so that
    the environment is read at the point of use.
    """
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home) / "auth.json"
    return Path.home() / ".codex" / "auth.json"


def _path_scoped_lock(auth_file: Path) -> threading.Lock:
    key = auth_file.expanduser().resolve(strict=False)
    with _CODEX_AUTH_LOCKS_GUARD:
        lock = _CODEX_AUTH_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _CODEX_AUTH_LOCKS[key] = lock
        return lock


@contextlib.contextmanager
def _codex_auth_lock(auth_file: Path, timeout_seconds: float = _CODEX_AUTH_LOCK_TIMEOUT_SECONDS):
    """Cross-process advisory lock for one Codex auth store."""
    with _path_scoped_lock(auth_file):
        if fcntl is None:  # pragma: no cover - Windows
            logger.debug("fcntl unavailable; Codex refresh proceeds without a cross-process lock.")
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
                        raise TimeoutError("Timed out waiting for the Codex auth store lock") from None
                    time.sleep(0.05)
            try:
                yield
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class CodexRefreshExpiredError(RuntimeError):
    """Raised when the Codex refresh_token itself is no longer valid.

    The user must re-run ``codex auth login`` to obtain new credentials.
    Callers should surface a clear remediation message and stop retrying.
    """


class CodexAuthManager:
    """Sync Codex OAuth credential manager.

    Holds the access_token, refresh_token, and account_id in memory and
    handles proactive/reactive refresh using a ``threading.Lock`` for
    single-flight semantics (safe to use from multiple threads or via
    ``asyncio.to_thread``).

    Parameters
    ----------
    access_token:
        The current bearer token.
    account_id:
        The OpenAI account ID embedded in the Codex request headers.
    refresh_token:
        The OAuth refresh token. May be ``None`` when the auth file omits it;
        the provider still works as a one-shot loader in that case.
    auth_file:
        Path to the Codex ``auth.json``. Used for re-reading the refresh token
        on demand and for atomic persistence of rotated credentials.
    """

    def __init__(
        self,
        access_token: str,
        account_id: str,
        refresh_token: str | None,
        auth_file: Path,
    ) -> None:
        self.access_token = access_token
        self.account_id = account_id
        self.refresh_token = refresh_token
        self._auth_file = auth_file
        self._lock = threading.Lock()
        self._http_client = httpx.Client(timeout=30.0)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, auth_file: Path | None = None) -> "CodexAuthManager":
        """Build a manager by reading credentials from ``auth_file``.

        Parameters
        ----------
        auth_file:
            Defaults to ``$CODEX_HOME/auth.json`` (or ``~/.codex/auth.json``
            when ``CODEX_HOME`` is unset).

        Raises
        ------
        FileNotFoundError:
            If the auth file does not exist.
        ValueError:
            If the auth file is missing ``access_token`` or has an unexpected
            ``auth_mode``.
        """
        if auth_file is None:
            auth_file = default_codex_auth_file()

        if not auth_file.exists():
            raise FileNotFoundError(f"Codex auth file not found: {auth_file}. Run 'codex auth login' to authenticate.")

        with open(auth_file) as f:
            data = json.load(f)

        auth_mode = data.get("auth_mode")
        if auth_mode != "chatgpt":
            raise ValueError(f"Expected Codex auth_mode='chatgpt', got: {auth_mode}")

        tokens = data.get("tokens") or {}
        access_token = tokens.get("access_token")
        if not access_token:
            raise ValueError("No access_token found in Codex auth file. Run 'codex auth login' again.")

        account_id = tokens.get("account_id") or ""
        refresh_token = tokens.get("refresh_token")

        return cls(
            access_token=access_token,
            account_id=account_id,
            refresh_token=refresh_token,
            auth_file=auth_file,
        )

    # ------------------------------------------------------------------
    # Token state helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_refresh_token_from_file(auth_file: Path) -> str | None:
        """Read ``tokens.refresh_token`` from ``auth_file``.

        Returns ``None`` when the file is unreadable or omits the field.
        Does not raise — the provider degrades to one-shot mode.
        """
        try:
            with open(auth_file) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                f"Codex auth file unreadable when loading refresh_token: {type(e).__name__}. "
                "Token refresh will not be available; the access_token in memory will be used until it expires."
            )
            return None
        return data.get("tokens", {}).get("refresh_token")

    @staticmethod
    def _load_tokens_from_file(auth_file: Path) -> dict[str, Any] | None:
        try:
            with open(auth_file) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        tokens = data.get("tokens")
        return tokens if isinstance(tokens, dict) else None

    def _adopt_tokens(self, tokens: dict[str, Any]) -> bool:
        """Adopt a newer on-disk Codex token set if present."""
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        account_id = tokens.get("account_id")

        changed = False
        if isinstance(access_token, str) and access_token and access_token != self.access_token:
            self.access_token = access_token
            changed = True
        if isinstance(refresh_token, str) and refresh_token and refresh_token != self.refresh_token:
            self.refresh_token = refresh_token
            changed = True
        if isinstance(account_id, str) and account_id and account_id != self.account_id:
            self.account_id = account_id
            changed = True
        return changed

    @staticmethod
    def _decode_jwt_exp_unixtime(token: str) -> int | None:
        """Return the JWT ``exp`` claim as a unix timestamp, or None on parse failure.

        We do not verify the signature — the server is the source of truth
        on whether the token is actually accepted. This is only used to
        schedule proactive refresh.
        """
        try:
            parts = token.split(".")
            if len(parts) < 2:
                return None
            payload_b64 = parts[1]
            padding = "=" * (-len(payload_b64) % 4)
            payload_bytes = base64.urlsafe_b64decode(payload_b64 + padding)
            payload = json.loads(payload_bytes.decode("utf-8"))
            exp = payload.get("exp")
            return int(exp) if exp is not None else None
        except (ValueError, TypeError, json.JSONDecodeError, binascii.Error):
            return None

    def _token_is_stale(self, skew_seconds: int = _CODEX_TOKEN_REFRESH_SKEW_SECONDS) -> bool:
        """True when the cached access_token is past expiry (with skew).

        Returns False when expiry cannot be determined — we'd rather use a
        possibly-expired token and recover via the reactive 401 path than
        refresh aggressively on every request when ``exp`` parsing fails.
        """
        exp = self._decode_jwt_exp_unixtime(self.access_token)
        if exp is None:
            return False
        return exp <= int(time.time()) + skew_seconds

    def _token_is_fresh_with_known_expiry(self, skew_seconds: int = _CODEX_TOKEN_REFRESH_SKEW_SECONDS) -> bool:
        """True only when the cached token has a known expiry outside the skew window."""
        exp = self._decode_jwt_exp_unixtime(self.access_token)
        return exp is not None and exp > int(time.time()) + skew_seconds

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_auth_atomic(self, updated_tokens: dict[str, Any]) -> None:
        """Write rotated tokens back to ``_auth_file`` atomically.

        Re-reads the on-disk file first to avoid clobbering fields written
        by another process, patches ``tokens.*`` and ``last_refresh``, then
        writes to a sibling tempfile and calls ``os.replace`` (atomic on
        POSIX and Windows within the same filesystem).
        """
        current: dict[str, Any]
        try:
            with open(self._auth_file) as f:
                loaded = json.load(f)
            current = loaded if isinstance(loaded, dict) else {"auth_mode": "chatgpt", "tokens": {}}
        except (OSError, json.JSONDecodeError):
            current = {"auth_mode": "chatgpt", "tokens": {}}

        existing_tokens = current.get("tokens")
        tokens: dict[str, Any] = existing_tokens if isinstance(existing_tokens, dict) else {}
        for key in ("access_token", "refresh_token", "id_token", "account_id"):
            if key in updated_tokens and updated_tokens[key] is not None:
                tokens[key] = updated_tokens[key]
        current["tokens"] = tokens
        current["last_refresh"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        parent = self._auth_file.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".auth.", suffix=".json.tmp", dir=str(parent))
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(current, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.chmod(tmp_path, 0o600)
            except OSError:
                pass
            os.replace(tmp_path, self._auth_file)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Error extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_oauth_error_code(response: httpx.Response) -> str | None:
        """Pull the OAuth error code out of a 4xx response body, if present.

        The refresh endpoint returns shapes like
        ``{"error": "...", "error_code": "..."}`` or
        ``{"error": {"code": "..."}}``.
        """
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(body, dict):
            return None
        err = body.get("error")
        if isinstance(err, dict):
            code = err.get("code")
            if isinstance(code, str):
                return code
        code = body.get("error_code")
        if isinstance(code, str):
            return code
        if isinstance(err, str):
            return err
        return None

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_tokens(self, reason: str = "", *, force: bool = False) -> None:
        """Synchronous single-flight OAuth token refresh.

        Serialized through ``self._lock`` so concurrent threads produce one
        network request. The first caller refreshes; the rest wake up and
        skip if the token is no longer stale (proactive) or if the token
        has already changed (reactive / force).

        Parameters
        ----------
        reason:
            Free-form string included in log lines for diagnostics.
        force:
            When True, refresh even if the JWT exp claim looks fresh.
            Used by the reactive 401 path.

        Raises
        ------
        CodexRefreshExpiredError:
            When the server returns a terminal error code or any 401.
        RuntimeError:
            For other refresh failures (network, 5xx, etc.).
        """
        token_before_lock = self.access_token
        with self._lock:
            if force:
                if self.access_token != token_before_lock:
                    return
            else:
                if not self._token_is_stale():
                    return

            with _codex_auth_lock(self._auth_file):
                disk_tokens = self._load_tokens_from_file(self._auth_file)
                if disk_tokens and self._adopt_tokens(disk_tokens):
                    if force or self._token_is_fresh_with_known_expiry():
                        return

                if not self.refresh_token:
                    raise RuntimeError(
                        "Codex access_token is expired but no refresh_token is available. "
                        "Run 'codex auth login' to re-authenticate."
                    )

                log_reason = f" ({reason})" if reason else ""
                logger.info(f"Refreshing Codex OAuth access_token{log_reason}")

                request_access_token = self.access_token
                request_refresh_token = self.refresh_token
                request_body = {
                    "client_id": _CODEX_CLIENT_ID,
                    "grant_type": "refresh_token",
                    "refresh_token": request_refresh_token,
                }
                try:
                    response = self._http_client.post(
                        _CODEX_REFRESH_TOKEN_URL,
                        json=request_body,
                        headers={"Content-Type": "application/json"},
                        timeout=30.0,
                    )
                except httpx.RequestError as e:
                    raise RuntimeError(f"Codex OAuth refresh network error: {type(e).__name__}") from e

                if response.status_code == 401:
                    error_code = self._extract_oauth_error_code(response)
                    disk_tokens = self._load_tokens_from_file(self._auth_file)
                    if disk_tokens and (
                        disk_tokens.get("access_token") != request_access_token
                        or disk_tokens.get("refresh_token") != request_refresh_token
                    ):
                        self._adopt_tokens(disk_tokens)
                        return
                    if error_code in _CODEX_TERMINAL_REFRESH_ERROR_CODES:
                        raise CodexRefreshExpiredError(
                            f"Codex refresh_token is permanently invalid (error.code={error_code}). "
                            "Run 'codex auth login' to re-authenticate."
                        )
                    raise CodexRefreshExpiredError(
                        f"Codex OAuth refresh returned 401 with unrecognized error code "
                        f"({error_code or 'none'}). Run 'codex auth login' to re-authenticate."
                    )

                if response.status_code >= 400:
                    raise RuntimeError(f"Codex OAuth refresh failed with HTTP {response.status_code}")

                try:
                    body = response.json()
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Codex OAuth refresh returned non-JSON body: {e}") from e

                new_access = body.get("access_token")
                if not new_access:
                    raise RuntimeError("Codex OAuth refresh returned no access_token")

                new_refresh = body.get("refresh_token") or self.refresh_token
                new_id_token = body.get("id_token")

                # Update in-memory state first so waiters see fresh credentials
                # immediately, even if disk write fails.
                self.access_token = new_access
                self.refresh_token = new_refresh

                persisted: dict[str, Any] = {
                    "access_token": new_access,
                    "refresh_token": new_refresh,
                }
                if new_id_token:
                    persisted["id_token"] = new_id_token

                try:
                    self._persist_auth_atomic(persisted)
                except OSError as e:
                    logger.warning(
                        f"Codex OAuth refresh succeeded but persisting auth.json failed: {type(e).__name__}. "
                        "In-memory credentials are up to date; on-disk file is stale."
                    )

                logger.info("Codex OAuth access_token refreshed successfully")

    def ensure_fresh_token(self) -> None:
        """Proactively refresh the access_token if it is near or past expiry.

        Cheap when the token is fresh (just decodes the JWT exp claim and
        returns).
        """
        if self._token_is_stale():
            self.refresh_tokens(reason="proactive (token near expiry)")

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http_client.close()
