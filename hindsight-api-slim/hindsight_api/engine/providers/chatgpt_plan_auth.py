"""
ChatGPT plan credentials for the ``chatgpt-plan`` provider.

Implements OpenAI's "Sign in with ChatGPT" flow for open-source apps: a public
client registered dynamically on first sign-in (``client_id=dynamic_agent_client``
plus ``agent_name_hint``), an authorization-code + PKCE grant over a
``127.0.0.1`` loopback callback, and rotating refresh tokens. The resulting
access token is a bearer credential for the public ``api.openai.com/v1``
Responses API, billed to the user's ChatGPT plan. This is the documented path;
``openai-codex`` instead reuses the Codex CLI's client and its private backend.

Store
-----
One directory per ChatGPT account registration (``~/.hindsight/chatgpt-plan`` by
default, ``HINDSIGHT_API_LLM_CHATGPT_PLAN_HOME`` to relocate) holding:

* ``credentials.json`` — the credential record in the format OpenAI documents
  (email, subject, issued ``client_id``, tokens, scopes, ``expires_in``,
  ``saved_at``). Written atomically with mode ``0600``.
* ``host_id`` — this host's ``ext_agent_host_id``. Kept in its own file so that
  importing a credential file made on another machine never replaces it.

Refresh tokens rotate, so every refresh happens under :func:`oauth_store_lock`
and re-reads the file first: a sibling process may already have rotated it.

Interactive commands (``login``, ``status``, ``logout``, ``import``) run as
``python -m hindsight_api.engine.providers.chatgpt_plan_auth <command>``. The
server never starts a login.

Token values are never logged or printed.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import contextlib
import hashlib
import json
import logging
import os
import secrets
import sys
import tempfile
import time
import uuid
import webbrowser
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import aiohttp
import jwt
from aiohttp import web

from ...config import get_config
from ..aiohttp_session import LoopLocalSession, per_phase_timeout
from .oauth_store_lock import oauth_store_lock

logger = logging.getLogger(__name__)

ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = f"{ISSUER}/api/accounts/authorize"
TOKEN_URL = f"{ISSUER}/api/accounts/oauth/token"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"
RESOURCE = "https://api.openai.com/v1"

#: The scope that authorizes inference on the user's ChatGPT plan. A valid ID token
#: without it means the user declined plan usage.
PLAN_SCOPE = "chatgpt.tokens.use.direct"
REQUESTED_SCOPE = f"openid profile email offline_access resource.invoke {PLAN_SCOPE}"

#: First-registration entrypoint. Never saved, never used for token exchange.
DYNAMIC_CLIENT_ID = "dynamic_agent_client"
AGENT_NAME = "Hindsight"

DEFAULT_CALLBACK_PORT = 1455
CALLBACK_PATH = "/auth/callback"
CREDENTIAL_FILE = "credentials.json"
HOST_ID_FILE = "host_id"
LOGIN_COMMAND = "python -m hindsight_api.engine.providers.chatgpt_plan_auth login"

#: Refresh errors after which only a new login helps.
TERMINAL_REFRESH_ERROR_CODES = frozenset(
    {
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
        "invalid_client",
    }
)

# Access tokens live one hour; refresh this long before expiry.
_REFRESH_SKEW_SECONDS = 300
_LOCK_TIMEOUT_SECONDS = 30.0
_HTTP_TIMEOUT_SECONDS = 30.0
_LOGIN_TIMEOUT_SECONDS = 600.0
_ID_TOKEN_LEEWAY_SECONDS = 60
_ID_TOKEN_ALGORITHMS = frozenset({"RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384"})


class ChatGPTPlanAuthError(RuntimeError):
    """A credential operation failed; may be transient (network, 5xx)."""


class ChatGPTPlanLoginRequiredError(ChatGPTPlanAuthError):
    """Only a new interactive login can restore service."""


def default_home() -> Path:
    """Credential directory: ``HINDSIGHT_API_LLM_CHATGPT_PLAN_HOME`` or ``~/.hindsight/chatgpt-plan``."""
    configured = get_config().llm_chatgpt_plan_home
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".hindsight" / "chatgpt-plan"


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _write_private(path: Path, text: str) -> None:
    """Atomically replace ``path`` with ``text``, mode ``0600``."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        with contextlib.suppress(OSError):
            os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def load_or_create_host_id(home: Path) -> str:
    """Return this host's ``ext_agent_host_id``, creating and persisting it once."""
    path = home / HOST_ID_FILE
    with contextlib.suppress(FileNotFoundError):
        existing = path.read_text().strip()
        if existing:
            return existing
    host_id = f"urn:uuid:{uuid.uuid4()}"
    _write_private(path, host_id + "\n")
    return host_id


def read_credential(path: Path) -> dict[str, Any]:
    """Load the credential record; raise :class:`ChatGPTPlanLoginRequiredError` if absent."""
    try:
        with open(path) as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise ChatGPTPlanLoginRequiredError(
            f"No ChatGPT plan credentials at {path}. Run `{LOGIN_COMMAND}` on a machine with a browser "
            "(see the chatgpt-plan setup docs for headless servers)."
        ) from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ChatGPTPlanLoginRequiredError(
            f"ChatGPT plan credentials at {path} are unreadable ({type(exc).__name__}). Run `{LOGIN_COMMAND}`."
        ) from exc
    if not isinstance(data, dict):
        raise ChatGPTPlanLoginRequiredError(f"ChatGPT plan credentials at {path} are malformed. Run `{LOGIN_COMMAND}`.")
    return data


def write_credential(path: Path, credential: dict[str, Any]) -> None:
    _write_private(path, json.dumps(credential, indent=2) + "\n")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jwt_exp(token: str) -> float | None:
    """Unverified ``exp`` claim, for scheduling refresh only."""
    try:
        payload_b64 = token.split(".")[1]
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
        exp = payload.get("exp")
        return float(exp) if isinstance(exp, (int, float)) else None
    except (IndexError, ValueError, binascii.Error, AttributeError):
        return None


def expires_at(credential: dict[str, Any]) -> float | None:
    """Access-token expiry as unix seconds: the JWT ``exp``, else ``saved_at + expires_in``."""
    token = credential.get("access_token")
    if isinstance(token, str) and token:
        exp = _jwt_exp(token)
        if exp is not None:
            return exp
    expires_in = credential.get("expires_in")
    saved_at = credential.get("saved_at")
    if isinstance(expires_in, (int, float)) and isinstance(saved_at, str):
        try:
            return datetime.fromisoformat(saved_at.replace("Z", "+00:00")).timestamp() + float(expires_in)
        except ValueError:
            return None
    return None


def needs_refresh(credential: dict[str, Any], now: float | None = None) -> bool:
    """True when there is no access token or it expires within the skew window."""
    if not credential.get("access_token"):
        return True
    exp = expires_at(credential)
    if exp is None:
        # Unknown expiry: use the token and let a 401 trigger the refresh.
        return False
    now = time.time() if now is None else now
    earliest = credential.get("earliest_refresh_at")
    if isinstance(earliest, (int, float)) and now < earliest and now < exp:
        return False
    return exp - now <= _REFRESH_SKEW_SECONDS


def has_plan_scope(credential: dict[str, Any]) -> bool:
    scopes = credential.get("scopes")
    return isinstance(scopes, list) and PLAN_SCOPE in scopes


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


async def _post_form(session: aiohttp.ClientSession, url: str, form: dict[str, str]) -> tuple[int, Any]:
    """POST a form; return ``(status, parsed JSON body or None)``."""
    try:
        async with session.post(url, data=form, headers={"Accept": "application/json"}) as response:
            status = response.status
            text = await response.text(errors="replace")
    except (aiohttp.ClientError, TimeoutError) as exc:
        raise ChatGPTPlanAuthError(f"Request to {url} failed: {type(exc).__name__}") from exc
    try:
        return status, json.loads(text) if text else None
    except json.JSONDecodeError:
        return status, None


def oauth_error_code(body: Any) -> str | None:
    """Pull an OAuth error code out of ``{"error": "..."}`` / ``{"error": {"code": ...}}`` shapes."""
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if isinstance(err, dict):
        code = err.get("code")
        return code if isinstance(code, str) else None
    if isinstance(body.get("error_code"), str):
        return body["error_code"]
    return err if isinstance(err, str) else None


def _same_origin_as_issuer(url: str, field: str) -> str:
    """Pin discovered endpoints to the issuer's origin so a bad discovery reply cannot leak a token."""
    issuer = urlparse(ISSUER)
    parsed = urlparse(url)
    if (parsed.scheme, parsed.netloc) != (issuer.scheme, issuer.netloc):
        raise ChatGPTPlanAuthError(f"OpenAI discovery returned {field} outside {ISSUER}: {url!r}")
    return url


async def discover(session: aiohttp.ClientSession) -> dict[str, str]:
    """Return ``jwks_uri`` and ``revocation_endpoint`` from OpenAI's OIDC discovery document."""
    try:
        async with session.get(DISCOVERY_URL, headers={"Accept": "application/json"}) as response:
            status = response.status
            payload = await response.json(content_type=None)
    except (aiohttp.ClientError, TimeoutError, json.JSONDecodeError) as exc:
        raise ChatGPTPlanAuthError(f"OpenAI OIDC discovery failed: {type(exc).__name__}") from exc
    if status != 200 or not isinstance(payload, dict):
        raise ChatGPTPlanAuthError(f"OpenAI OIDC discovery returned HTTP {status}")
    result: dict[str, str] = {}
    for field in ("jwks_uri", "revocation_endpoint"):
        value = payload.get(field)
        if isinstance(value, str) and value:
            result[field] = _same_origin_as_issuer(value, field)
    return result


async def fetch_jwks(session: aiohttp.ClientSession, jwks_uri: str) -> dict[str, Any]:
    try:
        async with session.get(jwks_uri, headers={"Accept": "application/json"}) as response:
            status = response.status
            payload = await response.json(content_type=None)
    except (aiohttp.ClientError, TimeoutError, json.JSONDecodeError) as exc:
        raise ChatGPTPlanAuthError(f"Fetching OpenAI JWKS failed: {type(exc).__name__}") from exc
    if status != 200 or not isinstance(payload, dict):
        raise ChatGPTPlanAuthError(f"Fetching OpenAI JWKS returned HTTP {status}")
    return payload


# ---------------------------------------------------------------------------
# Login pieces
# ---------------------------------------------------------------------------


def new_pkce() -> tuple[str, str]:
    """Return ``(verifier, S256 challenge)``."""
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode()
    return verifier, challenge


def redirect_uri_for(port: int) -> str:
    # 127.0.0.1, never localhost: only the port may vary between sign-ins.
    return f"http://127.0.0.1:{port}{CALLBACK_PATH}"


def build_authorize_url(
    *,
    client_id: str,
    host_id: str,
    redirect_uri: str,
    state: str,
    nonce: str,
    code_challenge: str,
    id_token_hint: str | None = None,
    login_hint: str | None = None,
) -> str:
    """Authorization URL. ``agent_name_hint`` goes only on a first registration."""
    params = {
        "client_id": client_id,
        "ext_agent_host_id": host_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": REQUESTED_SCOPE,
        "resource": RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge_method": "S256",
        "code_challenge": code_challenge,
    }
    if client_id == DYNAMIC_CLIENT_ID:
        params["agent_name_hint"] = AGENT_NAME
    if id_token_hint:
        params["id_token_hint"] = id_token_hint
    if login_hint:
        params["login_hint"] = login_hint
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def parse_callback(query: dict[str, str], *, expected_state: str, pending_client_id: str) -> tuple[str, str]:
    """Validate the loopback callback; return ``(code, issued client_id)``."""
    if not secrets.compare_digest(query.get("state", ""), expected_state):
        raise ChatGPTPlanAuthError("Sign-in callback state did not match; aborting.")
    error = query.get("error")
    if error:
        if error == "access_denied":
            raise ChatGPTPlanAuthError("Sign-in was declined in the browser (access_denied).")
        raise ChatGPTPlanAuthError(f"Sign-in failed: {error}")
    code = query.get("code")
    if not code:
        raise ChatGPTPlanAuthError("Sign-in callback carried no authorization code.")
    returned_client_id = query.get("client_id")
    if pending_client_id == DYNAMIC_CLIENT_ID:
        if not returned_client_id or returned_client_id == DYNAMIC_CLIENT_ID:
            raise ChatGPTPlanAuthError("Registration incomplete: the callback carried no issued client_id.")
        return code, returned_client_id
    if returned_client_id and returned_client_id != pending_client_id:
        raise ChatGPTPlanAuthError("Sign-in callback returned a different client_id than this registration; rejected.")
    return code, pending_client_id


async def exchange_code(
    session: aiohttp.ClientSession, *, client_id: str, code: str, code_verifier: str, redirect_uri: str
) -> dict[str, Any]:
    status, body = await _post_form(
        session,
        TOKEN_URL,
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": redirect_uri,
            "resource": RESOURCE,
        },
    )
    if status != 200 or not isinstance(body, dict) or not body.get("access_token"):
        raise ChatGPTPlanAuthError(f"Code exchange failed: HTTP {status} ({oauth_error_code(body) or 'no error code'})")
    return body


def validate_id_token(id_token: str, *, jwks: dict[str, Any], client_id: str, nonce: str) -> dict[str, Any]:
    """Verify signature, ``iss``, ``aud``, ``exp`` and ``nonce``; return the claims."""
    try:
        header = jwt.get_unverified_header(id_token)
        keys = jwt.PyJWKSet.from_dict(jwks).keys
    except jwt.PyJWTError as exc:
        raise ChatGPTPlanAuthError(f"ID token or JWKS unreadable: {exc}") from exc
    alg = header.get("alg")
    if alg not in _ID_TOKEN_ALGORITHMS:
        raise ChatGPTPlanAuthError(f"ID token uses an unexpected algorithm: {alg!r}")
    kid = header.get("kid")
    candidates = [k for k in keys if k.key_id == kid] if kid else keys
    if len(candidates) != 1:
        raise ChatGPTPlanAuthError("No unique JWKS key matches the ID token.")
    try:
        claims = jwt.decode(
            id_token,
            key=candidates[0].key,
            algorithms=[alg],
            audience=client_id,
            issuer=ISSUER,
            leeway=_ID_TOKEN_LEEWAY_SECONDS,
            options={"require": ["exp", "iss", "aud", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise ChatGPTPlanAuthError(f"ID token validation failed: {exc}") from exc
    if not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
        raise ChatGPTPlanAuthError("ID token nonce did not match this sign-in attempt.")
    return claims


def build_credential(
    token_response: dict[str, Any], *, claims: dict[str, Any], client_id: str, host_id: str
) -> dict[str, Any]:
    """Credential record in the shape OpenAI documents for local storage."""
    scope = token_response.get("scope")
    return {
        "email": claims.get("email"),
        "issuer": ISSUER,
        "subject": claims["sub"],
        "client_id": client_id,
        "ext_agent_host_id": host_id,
        "id_token": token_response.get("id_token"),
        "access_token": token_response["access_token"],
        "refresh_token": token_response.get("refresh_token"),
        "token_type": token_response.get("token_type", "Bearer"),
        "expires_in": token_response.get("expires_in"),
        "earliest_refresh_at": token_response.get("earliest_refresh_at"),
        "scopes": sorted(scope.split()) if isinstance(scope, str) else [],
        "saved_at": _utc_now_iso(),
    }


async def _start_callback_listener(port: int, timeout: float) -> Callable[[], Awaitable[dict[str, str]]]:
    """Start the loopback listener on 127.0.0.1; return a waiter for the callback query."""
    loop = asyncio.get_running_loop()
    received: asyncio.Future[dict[str, str]] = loop.create_future()

    async def handle(request: web.Request) -> web.Response:
        if not received.done():
            received.set_result(dict(request.query))
        return web.Response(
            text="Hindsight received the sign-in response. You can close this tab and return to the terminal.",
            content_type="text/plain",
        )

    app = web.Application()
    app.router.add_get(CALLBACK_PATH, handle)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", port).start()
    except OSError as exc:
        await runner.cleanup()
        raise ChatGPTPlanAuthError(f"Cannot listen on 127.0.0.1:{port} ({exc.strerror}); try --port.") from exc

    async def wait() -> dict[str, str]:
        try:
            return await asyncio.wait_for(received, timeout=timeout)
        finally:
            await runner.cleanup()

    return wait


async def login(
    home: Path,
    *,
    port: int = DEFAULT_CALLBACK_PORT,
    open_browser: Callable[[str], Any] | None = webbrowser.open,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the interactive sign-in and write the credential file.

    ``open_browser=None`` prints the URL instead (``--no-browser``); in that case
    ``id_token_hint`` is left out so no token lands in terminal scrollback.
    Re-running against an existing record reuses its issued ``client_id`` and
    requires the same ChatGPT account.
    """
    path = home / CREDENTIAL_FILE
    host_id = load_or_create_host_id(home)
    existing: dict[str, Any] | None = None
    with contextlib.suppress(ChatGPTPlanLoginRequiredError):
        existing = read_credential(path)

    client_id = (existing or {}).get("client_id") or DYNAMIC_CLIENT_ID
    verifier, challenge = new_pkce()
    state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    redirect_uri = redirect_uri_for(port)
    url = build_authorize_url(
        client_id=client_id,
        host_id=host_id,
        redirect_uri=redirect_uri,
        state=state,
        nonce=nonce,
        code_challenge=challenge,
        id_token_hint=(existing or {}).get("id_token") if open_browser is not None else None,
        login_hint=(existing or {}).get("email"),
    )

    wait = await _start_callback_listener(port, _LOGIN_TIMEOUT_SECONDS)
    if open_browser is None or not open_browser(url):
        out(f"Open this URL in a browser on this machine to continue:\n\n{url}\n")
    else:
        out("Opened your browser to sign in with ChatGPT. Waiting for the callback...")
    try:
        query = await wait()
    except TimeoutError:
        raise ChatGPTPlanAuthError("Timed out waiting for the browser sign-in.") from None

    code, issued_client_id = parse_callback(query, expected_state=state, pending_client_id=client_id)

    async with aiohttp.ClientSession(timeout=per_phase_timeout(_HTTP_TIMEOUT_SECONDS)) as session:
        tokens = await exchange_code(
            session, client_id=issued_client_id, code=code, code_verifier=verifier, redirect_uri=redirect_uri
        )
        id_token = tokens.get("id_token")
        if not isinstance(id_token, str) or not id_token:
            raise ChatGPTPlanAuthError("Token response carried no ID token.")
        endpoints = await discover(session)
        if "jwks_uri" not in endpoints:
            raise ChatGPTPlanAuthError("OpenAI discovery document has no jwks_uri.")
        jwks = await fetch_jwks(session, endpoints["jwks_uri"])

    claims = validate_id_token(id_token, jwks=jwks, client_id=issued_client_id, nonce=nonce)
    if existing and existing.get("subject") and existing["subject"] != claims.get("sub"):
        raise ChatGPTPlanAuthError(
            f"Signed in as a different ChatGPT account than the one stored in {path}. "
            "Use a separate HINDSIGHT_API_LLM_CHATGPT_PLAN_HOME per account."
        )

    credential = build_credential(tokens, claims=claims, client_id=issued_client_id, host_id=host_id)
    if not credential.get("refresh_token"):
        raise ChatGPTPlanAuthError("Token response carried no refresh token (offline_access not granted).")
    async with oauth_store_lock(path, timeout_seconds=_LOCK_TIMEOUT_SECONDS, label="chatgpt-plan credential store"):
        write_credential(path, credential)
    return credential


# ---------------------------------------------------------------------------
# Runtime manager
# ---------------------------------------------------------------------------


class ChatGPTPlanAuth:
    """Supplies a fresh access token and serializes rotating refreshes across processes."""

    def __init__(self, path: Path, credential: dict[str, Any]) -> None:
        self.path = path
        self._credential = credential
        self._http = LoopLocalSession(timeout=per_phase_timeout(_HTTP_TIMEOUT_SECONDS))

    @classmethod
    def from_home(cls, home: Path) -> ChatGPTPlanAuth:
        path = home / CREDENTIAL_FILE
        credential = read_credential(path)
        if not credential.get("client_id") or not credential.get("refresh_token"):
            raise ChatGPTPlanLoginRequiredError(
                f"ChatGPT plan credentials at {path} have no active session. Run `{LOGIN_COMMAND}`."
            )
        if not has_plan_scope(credential):
            raise ChatGPTPlanLoginRequiredError(
                f"ChatGPT plan usage is not enabled for this sign-in ({PLAN_SCOPE} was not granted). "
                f"Run `{LOGIN_COMMAND}` and allow Hindsight to use your ChatGPT plan."
            )
        return cls(path, credential)

    async def get_access_token(self) -> str:
        if needs_refresh(self._credential):
            await self.refresh()
        return self._credential["access_token"]

    async def refresh(self, rejected_token: str | None = None) -> None:
        """Rotate the token set unless another holder of the lock already did.

        ``rejected_token`` is the access token the API just refused (reactive
        refresh); a fresh on-disk token equal to it is not reused.
        """
        async with oauth_store_lock(
            self.path, timeout_seconds=_LOCK_TIMEOUT_SECONDS, label="chatgpt-plan credential store"
        ):
            current = read_credential(self.path)
            self._credential = current
            token = current.get("access_token")
            if token and token != rejected_token and not needs_refresh(current):
                return
            refresh_token = current.get("refresh_token")
            client_id = current.get("client_id")
            if not refresh_token or not client_id:
                raise ChatGPTPlanLoginRequiredError(
                    f"ChatGPT plan credentials at {self.path} have no refresh token. Run `{LOGIN_COMMAND}`."
                )

            logger.info("Refreshing ChatGPT plan access token")
            status, body = await _post_form(
                self._http.get(),
                TOKEN_URL,
                {
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "refresh_token": refresh_token,
                    "resource": RESOURCE,
                },
            )
            if status != 200 or not isinstance(body, dict) or not body.get("access_token"):
                code = oauth_error_code(body)
                if code in TERMINAL_REFRESH_ERROR_CODES:
                    cleared = {k: v for k, v in current.items() if k not in ("access_token", "refresh_token")}
                    cleared["last_auth_error"] = {"code": code, "at": _utc_now_iso()}
                    write_credential(self.path, cleared)
                    self._credential = cleared
                    raise ChatGPTPlanLoginRequiredError(
                        f"ChatGPT plan refresh token is no longer usable ({code}). Run `{LOGIN_COMMAND}`."
                    )
                raise ChatGPTPlanAuthError(f"ChatGPT plan token refresh failed: HTTP {status} ({code or 'no code'})")

            scope = body.get("scope")
            refreshed = {
                **current,
                "access_token": body["access_token"],
                "refresh_token": body.get("refresh_token") or refresh_token,
                "token_type": body.get("token_type", current.get("token_type", "Bearer")),
                "expires_in": body.get("expires_in"),
                "earliest_refresh_at": body.get("earliest_refresh_at"),
                "scopes": sorted(scope.split()) if isinstance(scope, str) else current.get("scopes", []),
                "saved_at": _utc_now_iso(),
            }
            if body.get("id_token"):
                refreshed["id_token"] = body["id_token"]
            refreshed.pop("last_auth_error", None)
            write_credential(self.path, refreshed)
            self._credential = refreshed
            logger.info("ChatGPT plan access token refreshed")
            if not has_plan_scope(refreshed):
                raise ChatGPTPlanLoginRequiredError(
                    f"ChatGPT plan usage is no longer granted ({PLAN_SCOPE} missing). Run `{LOGIN_COMMAND}`."
                )

    async def close(self) -> None:
        await self._http.close()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def logout(home: Path, out: Callable[[str], None] = print) -> bool:
    """Revoke the refresh token, then clear tokens locally. Returns whether revocation was confirmed.

    The issued client id, account identity and host id are kept for a later sign-in.
    """
    path = home / CREDENTIAL_FILE
    async with oauth_store_lock(path, timeout_seconds=_LOCK_TIMEOUT_SECONDS, label="chatgpt-plan credential store"):
        credential = read_credential(path)
        refresh_token = credential.get("refresh_token")
        confirmed = False
        if refresh_token and credential.get("client_id"):
            async with aiohttp.ClientSession(timeout=per_phase_timeout(_HTTP_TIMEOUT_SECONDS)) as session:
                for attempt in range(3):
                    try:
                        endpoint = (await discover(session)).get("revocation_endpoint")
                        if not endpoint:
                            break
                        status, _ = await _post_form(
                            session,
                            endpoint,
                            {
                                "token": refresh_token,
                                "token_type_hint": "refresh_token",
                                "client_id": credential["client_id"],
                            },
                        )
                    except ChatGPTPlanAuthError:
                        status = 0
                    if status == 200:
                        confirmed = True
                        break
                    if status and status < 500:
                        break
                    await asyncio.sleep(2**attempt)
        cleared = {
            k: v
            for k, v in credential.items()
            if k not in ("access_token", "refresh_token", "id_token", "expires_in", "earliest_refresh_at", "scopes")
        }
        write_credential(path, cleared)
    if confirmed:
        out("Signed out; the session was revoked.")
    else:
        out(
            "Signed out locally, but remote revocation was not confirmed. "
            "You can disconnect Hindsight in ChatGPT Settings."
        )
    return confirmed


def import_credential(home: Path, source: Path, out: Callable[[str], None] = print) -> None:
    """Install a credential file made on another machine, keeping this host's own host id."""
    imported = read_credential(source)
    if not imported.get("client_id") or not imported.get("refresh_token"):
        raise ChatGPTPlanAuthError(f"{source} is not an active chatgpt-plan credential file.")
    host_id = load_or_create_host_id(home)
    imported["ext_agent_host_id"] = host_id
    write_credential(home / CREDENTIAL_FILE, imported)
    out(
        f"Imported credentials for {imported.get('email') or imported.get('subject')} into {home / CREDENTIAL_FILE} "
        f"(host id {host_id}). This host now owns refreshes: delete the source copy and do not use or log out "
        "of it elsewhere, since the refresh token rotates."
    )


def status_lines(home: Path) -> list[str]:
    path = home / CREDENTIAL_FILE
    credential = read_credential(path)
    exp = expires_at(credential)
    host_id_path = home / HOST_ID_FILE
    host_id = host_id_path.read_text().strip() if host_id_path.exists() else "(none)"
    lines = [
        f"Credential file: {path}",
        f"Account: {credential.get('email') or '(unknown)'} (subject {credential.get('subject') or '?'})",
        f"Client id: {credential.get('client_id') or '(none)'}",
        f"Host id: {host_id}",
        f"Scopes: {' '.join(credential.get('scopes') or []) or '(none)'}",
        f"Plan usage enabled: {'yes' if has_plan_scope(credential) else 'no'}",
        f"Signed in: {'yes' if credential.get('refresh_token') else 'no'}",
        "Access token expires: "
        + (datetime.fromtimestamp(exp, timezone.utc).isoformat() if exp is not None else "(unknown)"),
    ]
    if credential.get("last_auth_error"):
        lines.append(f"Last auth error: {credential['last_auth_error'].get('code')}")
    lines.append("Usage and limits: https://chatgpt.com/settings/usage")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chatgpt_plan_auth", description="Manage the chatgpt-plan (Sign in with ChatGPT) credentials."
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help="Credential directory (default: HINDSIGHT_API_LLM_CHATGPT_PLAN_HOME or ~/.hindsight/chatgpt-plan)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    login_parser = sub.add_parser("login", help="Sign in with ChatGPT in a browser on this machine.")
    login_parser.add_argument("--port", type=int, default=DEFAULT_CALLBACK_PORT, help="Loopback callback port.")
    login_parser.add_argument("--no-browser", action="store_true", help="Print the sign-in URL instead of opening it.")
    sub.add_parser("status", help="Show the stored account, client, scopes and expiry (no token values).")
    sub.add_parser("logout", help="Revoke the session and clear local tokens.")
    import_parser = sub.add_parser("import", help="Install a credential file made on another machine.")
    import_parser.add_argument("source", type=Path)
    args = parser.parse_args(argv)
    home: Path = args.home.expanduser() if args.home else default_home()

    try:
        if args.command == "login":
            credential = asyncio.run(
                login(home, port=args.port, open_browser=None if args.no_browser else webbrowser.open)
            )
            print(f"Signed in as {credential.get('email') or credential['subject']}; saved {home / CREDENTIAL_FILE}.")
            if not has_plan_scope(credential):
                print(
                    f"ChatGPT plan usage was not granted ({PLAN_SCOPE} missing); the chatgpt-plan provider "
                    "will refuse to run. Sign in again and allow plan usage.",
                    file=sys.stderr,
                )
                return 1
        elif args.command == "status":
            print("\n".join(status_lines(home)))
        elif args.command == "logout":
            asyncio.run(logout(home))
        elif args.command == "import":
            if args.source.expanduser().resolve() == (home / CREDENTIAL_FILE).resolve():
                print("Source is already the local credential file.", file=sys.stderr)
                return 1
            import_credential(home, args.source.expanduser())
    except ChatGPTPlanAuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    sys.exit(main())
