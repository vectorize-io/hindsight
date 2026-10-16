"""
ChatGPT plan credentials for the ``chatgpt`` provider.

Implements OpenAI's "Sign in with ChatGPT" flow for open-source apps: a public
client registered dynamically on first sign-in (``client_id=dynamic_agent_client``
plus ``agent_name_hint``), an authorization-code + PKCE grant over a
``127.0.0.1`` loopback callback, and rotating refresh tokens. The resulting
access token is a bearer credential for the public ``api.openai.com/v1``
Responses API, billed to the user's ChatGPT plan. This is the documented path;
``openai-codex`` instead reuses the Codex CLI's client and its private backend.

Store
-----
One directory per ChatGPT account registration (``~/.hindsight/chatgpt`` by
default, ``HINDSIGHT_API_LLM_CHATGPT_HOME`` to relocate) holding:

* ``credentials.json`` — the credential record in the format OpenAI documents
  (email, subject, issued ``client_id``, tokens, scopes, ``expires_in``,
  ``saved_at``). Written atomically with mode ``0600``.
* ``host_id`` — this host's ``ext_agent_host_id``. Kept in its own file so that
  importing a credential file made on another machine never replaces it.

Refresh tokens rotate, so every refresh happens under :func:`oauth_store_lock`
and re-reads the file first: a sibling process may already have rotated it.

Interactive commands (``login``, ``status``, ``logout``) run as
``hindsight-admin chatgpt <command>``. The
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
import threading
import time
import uuid
import webbrowser
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlencode, urlparse

import aiohttp
import jwt
from aiohttp import web
from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError

from ...config import get_config
from ..aiohttp_session import LoopLocalSession, per_phase_timeout, proxy_env_is_set
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
LOGIN_COMMAND = "hindsight-admin chatgpt login"

#: Refresh errors after which only a new login helps (``invalid_client`` included:
#: the registration itself is unusable).
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
_LOCK_LABEL = "chatgpt credential store"


class ChatGPTAuthError(RuntimeError):
    """A credential operation failed; may be transient (network, 5xx)."""


class ChatGPTLoginRequiredError(ChatGPTAuthError):
    """Only a new interactive login can restore service."""


def _lenient_float(value: Any) -> float | None:
    """Numbers (or numeric strings) pass; anything else becomes None instead of failing validation."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _lenient_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


# Optional fields from OpenAI's token endpoint (or a hand-edited file) are parsed
# leniently: a malformed one must not discard a response whose access and refresh
# tokens are valid, because the refresh token it replaces is already spent.
_OptFloat = Annotated[float | None, BeforeValidator(_lenient_float)]
_OptStr = Annotated[str | None, BeforeValidator(_lenient_str)]


class ChatGPTCredential(BaseModel):
    """The stored credential record (OpenAI's documented local format, plus bookkeeping)."""

    model_config = ConfigDict(extra="ignore")

    email: str | None = None
    issuer: str | None = None
    subject: str | None = None
    client_id: str | None = None
    ext_agent_host_id: str | None = None
    id_token: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None
    token_type: _OptStr = None
    expires_in: _OptFloat = None
    # Not specified beyond its name in OpenAI's token reference; honored only as unix seconds.
    earliest_refresh_at: _OptFloat = None
    scopes: list[str] = []
    saved_at: str | None = None
    # Error code of the refresh that invalidated this session, if any.
    last_auth_error: str | None = None

    @property
    def has_plan_scope(self) -> bool:
        return PLAN_SCOPE in self.scopes

    @property
    def is_active(self) -> bool:
        return bool(self.client_id and self.refresh_token)


class _TokenResponse(BaseModel):
    """Token-endpoint reply for both the code exchange and a refresh."""

    model_config = ConfigDict(extra="ignore")

    access_token: str
    refresh_token: _OptStr = None
    id_token: _OptStr = None
    token_type: _OptStr = None
    expires_in: _OptFloat = None
    earliest_refresh_at: _OptFloat = None
    scope: _OptStr = None


@dataclass(frozen=True)
class Pkce:
    verifier: str
    challenge: str


@dataclass(frozen=True)
class CallbackResult:
    code: str
    client_id: str


@dataclass(frozen=True)
class IdTokenClaims:
    sub: str
    email: str | None


@dataclass(frozen=True)
class Endpoints:
    jwks_uri: str | None
    revocation_endpoint: str | None


@dataclass(frozen=True)
class _HttpReply:
    status: int
    body: Any  # parsed JSON, or None

    @property
    def error_code(self) -> str | None:
        return oauth_error_code(self.body)


def _print_now(line: str) -> None:
    # Flushed: the login waits on the callback right after printing the URL, and a
    # piped or redirected stdout would otherwise hold the URL in its buffer.
    print(line, flush=True)


def default_home() -> Path:
    """Credential directory: ``HINDSIGHT_API_LLM_CHATGPT_HOME`` or ``~/.hindsight/chatgpt``."""
    configured = get_config().llm_chatgpt_home
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".hindsight" / "chatgpt"


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


def read_credential(path: Path) -> ChatGPTCredential:
    """Load the credential record; raise :class:`ChatGPTLoginRequiredError` if absent or malformed."""
    try:
        with open(path) as handle:
            return ChatGPTCredential.model_validate(json.load(handle))
    except FileNotFoundError:
        raise ChatGPTLoginRequiredError(
            f"No ChatGPT plan credentials at {path}. Run `{LOGIN_COMMAND}` on a machine with a browser "
            "(see the chatgpt setup docs for headless servers)."
        ) from None
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ChatGPTLoginRequiredError(
            f"ChatGPT plan credentials at {path} are unreadable ({type(exc).__name__}). Run `{LOGIN_COMMAND}`."
        ) from exc


def write_credential(path: Path, credential: ChatGPTCredential) -> None:
    _write_private(path, credential.model_dump_json(indent=2, exclude_none=True) + "\n")


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


def expires_at(credential: ChatGPTCredential) -> float | None:
    """Access-token expiry as unix seconds: the JWT ``exp``, else ``saved_at + expires_in``."""
    if credential.access_token:
        exp = _jwt_exp(credential.access_token)
        if exp is not None:
            return exp
    if credential.expires_in is not None and credential.saved_at:
        try:
            saved = datetime.fromisoformat(credential.saved_at.replace("Z", "+00:00"))
        except ValueError:
            return None
        return saved.timestamp() + credential.expires_in
    return None


def needs_refresh(credential: ChatGPTCredential, now: float | None = None) -> bool:
    """True when there is no access token or it expires within the skew window."""
    if not credential.access_token:
        return True
    exp = expires_at(credential)
    if exp is None:
        # Unknown expiry: use the token and let a 401 trigger the refresh.
        return False
    now = time.time() if now is None else now  # noqa: TID251 - compared with the token's wall-clock expiry
    earliest = credential.earliest_refresh_at
    if earliest is not None and now < earliest and now < exp:
        return False
    return exp - now <= _REFRESH_SKEW_SECONDS


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


async def _post_form(session: aiohttp.ClientSession, url: str, form: dict[str, str]) -> _HttpReply:
    try:
        # No redirects: aiohttp re-sends a POST body on 307/308, which would hand a
        # code or refresh token to wherever the redirect points.
        async with session.post(
            url, data=form, headers={"Accept": "application/json"}, allow_redirects=False
        ) as response:
            status = response.status
            text = await response.text(errors="replace")
    except (aiohttp.ClientError, TimeoutError) as exc:
        raise ChatGPTAuthError(f"Request to {url} failed: {type(exc).__name__}") from exc
    try:
        return _HttpReply(status, json.loads(text) if text else None)
    except json.JSONDecodeError:
        return _HttpReply(status, None)


async def _get_json(session: aiohttp.ClientSession, url: str, what: str) -> dict[str, Any]:
    try:
        async with session.get(url, headers={"Accept": "application/json"}, allow_redirects=False) as response:
            status = response.status
            payload = await response.json(content_type=None)
    except (aiohttp.ClientError, TimeoutError, json.JSONDecodeError) as exc:
        raise ChatGPTAuthError(f"{what} failed: {type(exc).__name__}") from exc
    if status != 200 or not isinstance(payload, dict):
        raise ChatGPTAuthError(f"{what} returned HTTP {status}")
    return payload


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


def _parse_token_response(reply: _HttpReply, what: str) -> _TokenResponse:
    if reply.status == 200:
        with contextlib.suppress(ValidationError):
            return _TokenResponse.model_validate(reply.body)
    raise ChatGPTAuthError(f"{what} failed: HTTP {reply.status} ({reply.error_code or 'no error code'})")


def _same_origin_as_issuer(url: str, field: str) -> str:
    """Pin discovered endpoints to the issuer's origin so a bad discovery reply cannot leak a token."""
    issuer = urlparse(ISSUER)
    parsed = urlparse(url)
    if (parsed.scheme, parsed.netloc) != (issuer.scheme, issuer.netloc):
        raise ChatGPTAuthError(f"OpenAI discovery returned {field} outside {ISSUER}: {url!r}")
    return url


async def discover(session: aiohttp.ClientSession) -> Endpoints:
    """Read ``jwks_uri`` and ``revocation_endpoint`` from OpenAI's OIDC discovery document."""
    payload = await _get_json(session, DISCOVERY_URL, "OpenAI OIDC discovery")

    def endpoint(field: str) -> str | None:
        value = payload.get(field)
        return _same_origin_as_issuer(value, field) if isinstance(value, str) and value else None

    return Endpoints(jwks_uri=endpoint("jwks_uri"), revocation_endpoint=endpoint("revocation_endpoint"))


# ---------------------------------------------------------------------------
# Login pieces
# ---------------------------------------------------------------------------


def new_pkce() -> Pkce:
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode()
    return Pkce(verifier=verifier, challenge=challenge)


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
    prompt: str | None = None,
) -> str:
    """Authorization URL. ``agent_name_hint`` goes only on a first registration.

    ``prompt="consent"`` re-asks for consent, which is how a user who declined
    ChatGPT plan usage turns it on later (OpenAI's errors-and-recovery guide).
    """
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
    if prompt:
        params["prompt"] = prompt
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def _state_matches(received: str, expected: str) -> bool:
    # Bytes, because compare_digest raises TypeError on non-ASCII str input.
    return secrets.compare_digest(received.encode(), expected.encode())


def parse_callback(query: dict[str, str], *, expected_state: str, pending_client_id: str) -> CallbackResult:
    """Validate the loopback callback's query parameters."""
    if not _state_matches(query.get("state", ""), expected_state):
        raise ChatGPTAuthError("Sign-in callback state did not match; aborting.")
    error = query.get("error")
    if error:
        if error == "access_denied":
            raise ChatGPTAuthError("Sign-in was declined in the browser (access_denied).")
        raise ChatGPTAuthError(f"Sign-in failed: {error}")
    code = query.get("code")
    if not code:
        raise ChatGPTAuthError("Sign-in callback carried no authorization code.")
    returned_client_id = query.get("client_id")
    if pending_client_id == DYNAMIC_CLIENT_ID:
        if not returned_client_id or returned_client_id == DYNAMIC_CLIENT_ID:
            raise ChatGPTAuthError("Registration incomplete: the callback carried no issued client_id.")
        return CallbackResult(code=code, client_id=returned_client_id)
    if returned_client_id and returned_client_id != pending_client_id:
        raise ChatGPTAuthError("Sign-in callback returned a different client_id than this registration; rejected.")
    return CallbackResult(code=code, client_id=pending_client_id)


async def exchange_code(
    session: aiohttp.ClientSession, *, client_id: str, code: str, code_verifier: str, redirect_uri: str
) -> _TokenResponse:
    reply = await _post_form(
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
    return _parse_token_response(reply, "Code exchange")


def validate_id_token(id_token: str, *, jwks: dict[str, Any], client_id: str, nonce: str) -> IdTokenClaims:
    """Verify signature, ``iss``, ``aud``, ``exp`` and ``nonce``."""
    try:
        header = jwt.get_unverified_header(id_token)
        keys = jwt.PyJWKSet.from_dict(jwks).keys
    except jwt.PyJWTError as exc:
        raise ChatGPTAuthError(f"ID token or JWKS unreadable: {exc}") from exc
    alg = header.get("alg")
    if alg not in _ID_TOKEN_ALGORITHMS:
        raise ChatGPTAuthError(f"ID token uses an unexpected algorithm: {alg!r}")
    kid = header.get("kid")
    candidates = [k for k in keys if k.key_id == kid] if kid else keys
    if len(candidates) != 1:
        raise ChatGPTAuthError("No unique JWKS key matches the ID token.")
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
        raise ChatGPTAuthError(f"ID token validation failed: {exc}") from exc
    if not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
        raise ChatGPTAuthError("ID token nonce did not match this sign-in attempt.")
    email = claims.get("email")
    return IdTokenClaims(sub=str(claims["sub"]), email=email if isinstance(email, str) else None)


def build_credential(
    tokens: _TokenResponse, *, claims: IdTokenClaims, client_id: str, host_id: str
) -> ChatGPTCredential:
    return ChatGPTCredential(
        email=claims.email,
        issuer=ISSUER,
        subject=claims.sub,
        client_id=client_id,
        ext_agent_host_id=host_id,
        id_token=tokens.id_token,
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        token_type=tokens.token_type or "Bearer",
        expires_in=tokens.expires_in,
        earliest_refresh_at=tokens.earliest_refresh_at,
        scopes=sorted(tokens.scope.split()) if tokens.scope else [],
        saved_at=_utc_now_iso(),
    )


# The callback URL carries the authorization code, so the page loads nothing external
# and sends no referrer.
_CALLBACK_PAGE_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}

_CALLBACK_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hindsight</title>
<style>
  :root {{ --bg: #f6f7f9; --card: #ffffff; --text: #1d2330; --muted: #5d6676; --border: #e3e6eb;
           --accent: {accent_light}; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #111318; --card: #1a1d24; --text: #e8eaef; --muted: #9aa3b2; --border: #2a2e37;
             --accent: {accent_dark}; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; padding: 16px;
         background: var(--bg); color: var(--text);
         font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  main {{ width: 100%; max-width: 380px; padding: 32px 28px; text-align: center;
         background: var(--card); border: 1px solid var(--border); border-radius: 14px; }}
  .mark {{ width: 48px; height: 48px; margin: 0 auto 16px; border-radius: 50%;
          display: grid; place-items: center; color: #fff; background: var(--accent);
          font-size: 24px; font-weight: 600; }}
  h1 {{ margin: 0 0 8px; font-size: 19px; font-weight: 600; }}
  p {{ margin: 0; color: var(--muted); }}
  .brand {{ margin-top: 24px; font-size: 12px; letter-spacing: .04em; color: var(--muted); }}
</style>
</head>
<body>
<main>
  <div class="mark" aria-hidden="true">{symbol}</div>
  <h1>{heading}</h1>
  <p>{body}</p>
  <div class="brand">HINDSIGHT · SIGN IN WITH CHATGPT</div>
</main>
</body>
</html>
"""


def _callback_page(*, ok: bool, heading: str, body: str) -> str:
    """Render the page the browser lands on after the loopback redirect."""
    return _CALLBACK_PAGE.format(
        accent_light="#1f8a5b" if ok else "#b4462f",
        accent_dark="#2fa36f" if ok else "#d0573c",
        symbol="✓" if ok else "!",
        heading=heading,
        body=body,
    )


async def _start_callback_listener(
    port: int, timeout: float, expected_state: str
) -> Callable[[], Awaitable[dict[str, str]]]:
    """Start the loopback listener on 127.0.0.1; return a waiter for the callback query.

    Only a request carrying this attempt's ``state`` completes the wait, so a stray
    or forged request to the port cannot abort a pending sign-in.
    """
    loop = asyncio.get_running_loop()
    received: asyncio.Future[dict[str, str]] = loop.create_future()

    async def handle(request: web.Request) -> web.Response:
        if not _state_matches(request.query.get("state", ""), expected_state):
            return web.Response(status=400, text="Unexpected sign-in response.", content_type="text/plain")
        if not received.done():
            received.set_result(dict(request.query))
        if request.query.get("error"):
            page = _callback_page(
                ok=False,
                heading="Sign-in not completed",
                body="ChatGPT plan usage was not granted. Check the terminal for details.",
            )
        else:
            page = _callback_page(
                ok=True,
                heading="Sign-in received",
                body="Hindsight is finishing setup in your terminal. You can close this tab.",
            )
        return web.Response(text=page, content_type="text/html", headers=_CALLBACK_PAGE_HEADERS)

    app = web.Application()
    app.router.add_get(CALLBACK_PATH, handle)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", port).start()
    except OSError as exc:
        await runner.cleanup()
        raise ChatGPTAuthError(
            f"Cannot bind the sign-in callback to 127.0.0.1:{port} ({exc.strerror}); try --port."
        ) from exc

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
    out: Callable[[str], None] = _print_now,
    enable_plan_usage: bool = False,
) -> ChatGPTCredential:
    """Run the interactive sign-in and write the credential file.

    ``open_browser=None`` prints the URL instead (``--no-browser``). A printed URL
    never carries ``id_token_hint``, so no token lands in terminal scrollback.
    Re-running against an existing record reuses its issued ``client_id`` and
    requires the same ChatGPT account. Consent is requested again when the stored
    grant lacks plan usage (the user declined it before) or ``enable_plan_usage``.
    """
    path = home / CREDENTIAL_FILE
    host_id = load_or_create_host_id(home)
    existing = ChatGPTCredential()
    with contextlib.suppress(ChatGPTLoginRequiredError):
        existing = read_credential(path)

    client_id = existing.client_id or DYNAMIC_CLIENT_ID
    pkce = new_pkce()
    state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    redirect_uri = redirect_uri_for(port)
    declined_before = bool(existing.client_id and existing.scopes and not existing.has_plan_scope)

    def authorize_url(id_token_hint: str | None) -> str:
        return build_authorize_url(
            client_id=client_id,
            host_id=host_id,
            redirect_uri=redirect_uri,
            state=state,
            nonce=nonce,
            code_challenge=pkce.challenge,
            id_token_hint=id_token_hint,
            login_hint=existing.email,
            prompt="consent" if (enable_plan_usage or declined_before) and client_id != DYNAMIC_CLIENT_ID else None,
        )

    wait = await _start_callback_listener(port, _LOGIN_TIMEOUT_SECONDS, state)
    if open_browser is None or not open_browser(authorize_url(existing.id_token)):
        out(f"Open this URL in a browser on this machine to continue:\n\n{authorize_url(None)}\n")
    else:
        out("Opened your browser to sign in with ChatGPT. Waiting for the callback...")
    try:
        query = await wait()
    except TimeoutError:
        raise ChatGPTAuthError("Timed out waiting for the browser sign-in.") from None

    callback = parse_callback(query, expected_state=state, pending_client_id=client_id)
    if client_id == DYNAMIC_CLIENT_ID:
        # Keep the newly issued client id even if a later step fails, so a retry
        # reauthorizes this registration instead of registering a second client.
        async with oauth_store_lock(path, timeout_seconds=_LOCK_TIMEOUT_SECONDS, label=_LOCK_LABEL):
            write_credential(path, ChatGPTCredential(client_id=callback.client_id, ext_agent_host_id=host_id))

    # One-shot CLI work under asyncio.run: a scoped session, not a LoopLocalSession,
    # since nothing reuses it and the loop ends with the command.
    async with aiohttp.ClientSession(
        timeout=per_phase_timeout(_HTTP_TIMEOUT_SECONDS), trust_env=proxy_env_is_set()
    ) as session:
        tokens = await exchange_code(
            session,
            client_id=callback.client_id,
            code=callback.code,
            code_verifier=pkce.verifier,
            redirect_uri=redirect_uri,
        )
        if not tokens.id_token:
            raise ChatGPTAuthError("Token response carried no ID token.")
        endpoints = await discover(session)
        if not endpoints.jwks_uri:
            raise ChatGPTAuthError("OpenAI discovery document has no jwks_uri.")
        jwks = await _get_json(session, endpoints.jwks_uri, "Fetching OpenAI JWKS")

    claims = validate_id_token(tokens.id_token, jwks=jwks, client_id=callback.client_id, nonce=nonce)
    if existing.subject and existing.subject != claims.sub:
        raise ChatGPTAuthError(
            f"Signed in as a different ChatGPT account than the one stored in {path}. "
            "Use a separate HINDSIGHT_API_LLM_CHATGPT_HOME per account."
        )
    if not tokens.refresh_token:
        raise ChatGPTAuthError("Token response carried no refresh token (offline_access not granted).")

    credential = build_credential(tokens, claims=claims, client_id=callback.client_id, host_id=host_id)
    async with oauth_store_lock(path, timeout_seconds=_LOCK_TIMEOUT_SECONDS, label=_LOCK_LABEL):
        write_credential(path, credential)
    return credential


# ---------------------------------------------------------------------------
# Runtime manager
# ---------------------------------------------------------------------------


class ChatGPTAuth:
    """Supplies a fresh access token and serializes rotating refreshes across processes.

    One instance per credential file per process (see :meth:`shared`): Hindsight
    builds a provider per operation and per multi-LLM member, and if each had its
    own manager, a rotation one of them could only keep in memory would leave the
    others refreshing with the spent token.
    """

    _shared: dict[Path, ChatGPTAuth] = {}
    # Guards ``_shared`` only (await-free); instances are reachable from several loops.
    _shared_lock = threading.Lock()

    def __init__(self, path: Path, credential: ChatGPTCredential) -> None:
        self.path = path
        self._credential = credential
        # Refresh token the file still holds after a rotation we could not write;
        # while the file shows it, memory is newer than the file.
        self._unwritten_over: str | None = None
        self._http = LoopLocalSession(timeout=per_phase_timeout(_HTTP_TIMEOUT_SECONDS))

    @classmethod
    def from_home(cls, home: Path) -> ChatGPTAuth:
        path = home / CREDENTIAL_FILE
        credential = read_credential(path)
        if not credential.is_active:
            raise ChatGPTLoginRequiredError(
                f"ChatGPT plan credentials at {path} have no active session. Run `{LOGIN_COMMAND}`."
            )
        if not credential.has_plan_scope:
            raise ChatGPTLoginRequiredError(
                f"ChatGPT plan usage is not enabled for this sign-in ({PLAN_SCOPE} was not granted). "
                f"Run `{LOGIN_COMMAND}` and allow Hindsight to use your ChatGPT plan."
            )
        return cls(path, credential)

    @classmethod
    def shared(cls, home: Path) -> ChatGPTAuth:
        """The process-wide manager for ``home``; the file is validated on every call."""
        fresh = cls.from_home(home)
        key = fresh.path.expanduser().resolve(strict=False)
        with cls._shared_lock:
            return cls._shared.setdefault(key, fresh)

    async def get_access_token(self) -> str:
        if needs_refresh(self._credential):
            await self.refresh()
        token = self._credential.access_token
        if not token:
            raise ChatGPTLoginRequiredError(f"No ChatGPT plan access token available. Run `{LOGIN_COMMAND}`.")
        return token

    def _persist(self, credential: ChatGPTCredential, *, replaces: ChatGPTCredential) -> None:
        """Adopt ``credential`` in memory first, then write it.

        Memory comes first because after a successful rotation the old refresh token
        is already spent: if the write fails (read-only mount, full disk), this
        process keeps working from memory instead of losing the new token set.
        """
        self._credential = credential
        try:
            write_credential(self.path, credential)
            self._unwritten_over = None
        except OSError as exc:
            self._unwritten_over = replaces.refresh_token
            logger.warning(
                f"Could not write ChatGPT plan credentials to {self.path} ({type(exc).__name__}); "
                "continuing with the in-memory token set. Other processes will not see it."
            )

    async def refresh(self, rejected_token: str | None = None) -> None:
        """Rotate the token set unless another holder of the lock already did.

        ``rejected_token`` is the access token the API just refused (reactive
        refresh); a fresh on-disk token equal to it is not reused. Shielded: a
        caller cancelled (request timeout, shutdown) after OpenAI rotated the
        refresh token must not abandon the new set before it is stored.
        """
        await asyncio.shield(self._refresh(rejected_token))

    async def _refresh(self, rejected_token: str | None) -> None:
        async with oauth_store_lock(self.path, timeout_seconds=_LOCK_TIMEOUT_SECONDS, label=_LOCK_LABEL):
            current = read_credential(self.path)
            if self._unwritten_over is not None and current.refresh_token == self._unwritten_over:
                current = self._credential
            self._credential = current
            if current.access_token and current.access_token != rejected_token and not needs_refresh(current):
                return
            if not current.is_active:
                raise ChatGPTLoginRequiredError(
                    f"ChatGPT plan credentials at {self.path} have no refresh token. Run `{LOGIN_COMMAND}`."
                )

            logger.info("Refreshing ChatGPT plan access token")
            reply = await _post_form(
                self._http.get(),
                TOKEN_URL,
                {
                    "grant_type": "refresh_token",
                    "client_id": current.client_id or "",
                    "refresh_token": current.refresh_token or "",
                    "resource": RESOURCE,
                },
            )
            if reply.status != 200 and reply.error_code in TERMINAL_REFRESH_ERROR_CODES:
                # Where the lock cannot hold (no fcntl, a store shared across hosts),
                # another writer may have stored a newer grant meanwhile: keep that one.
                with contextlib.suppress(ChatGPTLoginRequiredError):
                    latest = read_credential(self.path)
                    if latest.is_active and latest.refresh_token != current.refresh_token:
                        self._credential = latest
                        return
                cleared: dict[str, Any] = {
                    "access_token": None,
                    "refresh_token": None,
                    "last_auth_error": reply.error_code,
                }
                self._persist(current.model_copy(update=cleared), replaces=current)
                if reply.error_code == "invalid_client":
                    raise ChatGPTLoginRequiredError(
                        f"OpenAI rejected this app registration (invalid_client, client {current.client_id}). "
                        f"To register again, delete {self.path} and run `{LOGIN_COMMAND}`."
                    )
                raise ChatGPTLoginRequiredError(
                    f"ChatGPT plan refresh token is no longer usable ({reply.error_code}). Run `{LOGIN_COMMAND}`."
                )
            tokens = _parse_token_response(reply, "ChatGPT plan token refresh")

            refreshed = current.model_copy(
                update={
                    "access_token": tokens.access_token,
                    "refresh_token": tokens.refresh_token or current.refresh_token,
                    "token_type": tokens.token_type or current.token_type,
                    "expires_in": tokens.expires_in,
                    "earliest_refresh_at": tokens.earliest_refresh_at,
                    # Omitted scope means the grant is unchanged.
                    "scopes": sorted(tokens.scope.split()) if tokens.scope else current.scopes,
                    "id_token": tokens.id_token or current.id_token,
                    "saved_at": _utc_now_iso(),
                    "last_auth_error": None,
                }
            )
            self._persist(refreshed, replaces=current)
            logger.info("ChatGPT plan access token refreshed")
            if not refreshed.has_plan_scope:
                raise ChatGPTLoginRequiredError(
                    f"ChatGPT plan usage is no longer granted ({PLAN_SCOPE} missing). Run `{LOGIN_COMMAND}`."
                )

    async def close(self) -> None:
        await self._http.close()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def _revoke(session: aiohttp.ClientSession, credential: ChatGPTCredential) -> bool:
    """Revoke the refresh token; retry network failures and 5xx a few times. True on HTTP 200."""
    try:
        endpoint = (await discover(session)).revocation_endpoint
    except ChatGPTAuthError:
        return False
    if not endpoint:
        return False
    for attempt in range(3):
        if attempt:
            await asyncio.sleep(2**attempt)
        try:
            reply = await _post_form(
                session,
                endpoint,
                {
                    "token": credential.refresh_token or "",
                    "token_type_hint": "refresh_token",
                    "client_id": credential.client_id or "",
                },
            )
        except ChatGPTAuthError:
            reply = _HttpReply(0, None)
        if reply.status == 200:
            return True
        if 0 < reply.status < 500:
            return False
    return False


async def logout(home: Path, out: Callable[[str], None] = print, *, revoke: bool = True) -> bool:
    """Revoke the session (unless ``revoke=False``), then clear tokens locally.

    Returns whether revocation was confirmed. The issued client id, account
    identity, granted scopes and host id are kept for a later sign-in. Revocation
    runs before taking the store lock so its retries never hold up a server's
    refresh. ``revoke=False`` is for a laptop whose credentials were moved to a
    server: it must stop using its copy without ending the server's session.
    """
    path = home / CREDENTIAL_FILE
    credential = read_credential(path)
    confirmed = False
    if revoke and credential.is_active:
        # One-shot CLI call (see login): a scoped session is enough.
        async with aiohttp.ClientSession(
            timeout=per_phase_timeout(_HTTP_TIMEOUT_SECONDS), trust_env=proxy_env_is_set()
        ) as session:
            confirmed = await _revoke(session, credential)
    async with oauth_store_lock(path, timeout_seconds=_LOCK_TIMEOUT_SECONDS, label=_LOCK_LABEL):
        current = read_credential(path)
        cleared = current.model_copy(
            update={
                "access_token": None,
                "refresh_token": None,
                "id_token": None,
                "expires_in": None,
                "earliest_refresh_at": None,
            }
        )
        write_credential(path, cleared)
    if not revoke:
        out("Cleared the local tokens without revoking the session.")
    elif confirmed:
        out("Signed out; the session was revoked.")
    else:
        out(
            "Signed out locally, but remote revocation was not confirmed. "
            "You can disconnect Hindsight in ChatGPT Settings."
        )
    return confirmed


async def import_credential(
    home: Path, source: Path, out: Callable[[str], None] = print, *, force: bool = False
) -> None:
    """Install a credential file made on another machine, keeping this host's own host id.

    Written under the store lock so a running server's in-flight refresh cannot
    overwrite the import with the session it was rotating.
    """
    imported = read_credential(source)
    if not imported.is_active:
        raise ChatGPTAuthError(f"{source} is not an active chatgpt credential file.")
    host_id = load_or_create_host_id(home)
    path = home / CREDENTIAL_FILE
    async with oauth_store_lock(path, timeout_seconds=_LOCK_TIMEOUT_SECONDS, label=_LOCK_LABEL):
        existing: ChatGPTCredential | None = None
        with contextlib.suppress(ChatGPTLoginRequiredError):
            existing = read_credential(path)
        if existing is not None and not force:
            for field in ("subject", "client_id"):
                old, new = getattr(existing, field), getattr(imported, field)
                if old and old != new:
                    raise ChatGPTAuthError(
                        f"{path} belongs to a different registration ({field} differs). "
                        "Use a separate --home per account, or pass --force to replace it."
                    )
        write_credential(path, imported.model_copy(update={"ext_agent_host_id": host_id}))
    out(
        f"Imported credentials for {imported.email or imported.subject} into {home / CREDENTIAL_FILE} "
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
        f"Account: {credential.email or '(unknown)'} (subject {credential.subject or '?'})",
        f"Client id: {credential.client_id or '(none)'}",
        f"Host id: {host_id}",
        f"Scopes: {' '.join(credential.scopes) or '(none)'}",
        f"Plan usage enabled: {'yes' if credential.has_plan_scope else 'no'}",
        f"Signed in: {'yes' if credential.is_active else 'no'}",
        "Access token expires: "
        + (datetime.fromtimestamp(exp, timezone.utc).isoformat() if exp is not None else "(unknown)"),
    ]
    if credential.last_auth_error:
        lines.append(f"Last auth error: {credential.last_auth_error}")
    lines.append("Usage and limits: https://chatgpt.com/settings/usage")
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="hindsight-admin chatgpt", description="Manage the chatgpt (Sign in with ChatGPT) credentials."
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help="Credential directory (default: HINDSIGHT_API_LLM_CHATGPT_HOME or ~/.hindsight/chatgpt)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    login_parser = sub.add_parser(
        "login", help="Sign in with ChatGPT in a browser, or install a credential file made on another machine."
    )
    login_parser.add_argument("--port", type=int, default=DEFAULT_CALLBACK_PORT, help="Loopback callback port.")
    login_parser.add_argument("--no-browser", action="store_true", help="Print the sign-in URL instead of opening it.")
    login_parser.add_argument(
        "--enable-plan-usage", action="store_true", help="Ask for consent again, to allow ChatGPT plan usage."
    )
    login_parser.add_argument(
        "--from-file",
        type=Path,
        default=None,
        help="Install a credential file from a machine with a browser instead of signing in here (headless servers).",
    )
    login_parser.add_argument(
        "--force", action="store_true", help="With --from-file: replace a different account's registration."
    )
    sub.add_parser("status", help="Show the stored account, client, scopes and expiry (no token values).")
    logout_parser = sub.add_parser("logout", help="Revoke the session and clear local tokens.")
    logout_parser.add_argument(
        "--local-only",
        action="store_true",
        help="Clear local tokens without revoking (after moving the credentials to another host).",
    )
    args = parser.parse_args(argv)
    home: Path = args.home.expanduser() if args.home else default_home()

    try:
        if args.command == "login" and args.from_file:
            source = args.from_file.expanduser()
            if source.resolve() == (home / CREDENTIAL_FILE).resolve():
                print("Source is already the local credential file.", file=sys.stderr)
                return 1
            asyncio.run(import_credential(home, source, force=args.force))
        elif args.command == "login":
            credential = asyncio.run(
                login(
                    home,
                    port=args.port,
                    open_browser=None if args.no_browser else webbrowser.open,
                    enable_plan_usage=args.enable_plan_usage,
                )
            )
            print(f"Signed in as {credential.email or credential.subject}; saved {home / CREDENTIAL_FILE}.")
            if not credential.has_plan_scope:
                print(
                    f"ChatGPT plan usage was not granted ({PLAN_SCOPE} missing); the chatgpt provider "
                    "will refuse to run. Run login --enable-plan-usage and allow plan usage.",
                    file=sys.stderr,
                )
                return 1
        elif args.command == "status":
            print("\n".join(status_lines(home)))
        elif args.command == "logout":
            asyncio.run(logout(home, revoke=not args.local_only))
    except ChatGPTAuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
