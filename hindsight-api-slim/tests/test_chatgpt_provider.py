"""Tests for the ``chatgpt`` provider (Sign in with ChatGPT for open-source apps).

No live network: OpenAI's auth and Responses endpoints are replaced by a local
aiohttp stub (``tests/aiohttp_stub.py``) by pointing the module URL constants at
it, and the ID token is signed with an RSA key generated in-test and served as a
JWKS. Credential files live under ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import socket
import stat
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import aiohttp
import jwt
import pytest
from aiohttp import web
from cryptography.hazmat.primitives.asymmetric import rsa

from hindsight_api.config import PROVIDER_DEFAULT_MODELS, clear_config_cache
from hindsight_api.engine.llm_interface import ProviderRateLimitResetError
from hindsight_api.engine.llm_wrapper import create_llm_provider, requires_api_key
from hindsight_api.engine.providers import chatgpt_auth as auth
from hindsight_api.engine.providers import chatgpt_llm as plan
from hindsight_api.engine.providers.chatgpt_llm import ChatGPTLLM, ChatGPTRequestError
from hindsight_api.engine.providers.openai_responses_llm import RetryableResponsesError
from tests.aiohttp_stub import stub_server

ISSUED_CLIENT_ID = "oaiapp_test123"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_access_token(exp: float, marker: str = "a") -> str:
    """An unsigned JWT-shaped token: only its ``exp`` is read, for scheduling."""

    def b64(obj: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    return f"{b64({'alg': 'none'})}.{b64({'exp': int(exp), 'jti': marker})}.sig"


def _credential(**overrides: Any) -> dict[str, Any]:
    cred = {
        "email": "user@example.com",
        "issuer": "https://auth.openai.com",
        "subject": "user-sub",
        "client_id": ISSUED_CLIENT_ID,
        "ext_agent_host_id": "urn:uuid:laptop",
        "id_token": "id.token.value",
        "access_token": _fake_access_token(time.time() + 3600),
        "refresh_token": "rt-1",
        "token_type": "Bearer",
        "expires_in": 3600,
        "scopes": sorted(auth.REQUESTED_SCOPE.split()),
        "saved_at": "2026-10-09T00:00:00Z",
    }
    cred.update(overrides)
    return cred


def _write(home: Path, cred: dict[str, Any]) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    path = home / auth.CREDENTIAL_FILE
    auth.write_credential(path, auth.ChatGPTCredential.model_validate(cred))
    return path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def rsa_jwks() -> tuple[Any, dict[str, Any]]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update({"kid": "k1", "alg": "RS256", "use": "sig"})
    return key, {"keys": [jwk]}


def _id_token(key: Any, **claims: Any) -> str:
    payload = {
        "iss": auth.ISSUER,
        "aud": ISSUED_CLIENT_ID,
        "sub": "user-sub",
        "email": "user@example.com",
        "nonce": "n-1",
        "iat": int(time.time()),
        "exp": int(time.time()) + 600,
    }
    payload.update(claims)
    return jwt.encode(payload, key, algorithm="RS256", headers={"kid": "k1"})


@pytest.fixture
def point_auth_at(monkeypatch):
    """Redirect every auth.openai.com URL to a stub base URL."""

    def apply(base: str) -> None:
        monkeypatch.setattr(auth, "ISSUER", base)
        monkeypatch.setattr(auth, "AUTHORIZE_URL", f"{base}/api/accounts/authorize")
        monkeypatch.setattr(auth, "TOKEN_URL", f"{base}/api/accounts/oauth/token")
        monkeypatch.setattr(auth, "DISCOVERY_URL", f"{base}/.well-known/openid-configuration")

    return apply


# ---------------------------------------------------------------------------
# Authorize URL, callback, host id
# ---------------------------------------------------------------------------


def test_authorize_url_first_registration():
    pkce = auth.new_pkce()
    verifier, challenge = pkce.verifier, pkce.challenge
    url = auth.build_authorize_url(
        client_id=auth.DYNAMIC_CLIENT_ID,
        host_id="urn:uuid:h",
        redirect_uri=auth.redirect_uri_for(1455),
        state="s",
        nonce="n",
        code_challenge=challenge,
    )
    parsed = urlparse(url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == "https://auth.openai.com/api/accounts/authorize"
    q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    assert q == {
        "client_id": "dynamic_agent_client",
        "agent_name_hint": "Hindsight",
        "ext_agent_host_id": "urn:uuid:h",
        "response_type": "code",
        "redirect_uri": "http://127.0.0.1:1455/auth/callback",
        "scope": "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct",
        "resource": "https://api.openai.com/v1",
        "state": "s",
        "nonce": "n",
        "code_challenge_method": "S256",
        "code_challenge": challenge,
    }
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert challenge == expected and "=" not in challenge


def test_authorize_url_reauth_uses_issued_client_and_hints():
    url = auth.build_authorize_url(
        client_id=ISSUED_CLIENT_ID,
        host_id="urn:uuid:h",
        redirect_uri=auth.redirect_uri_for(54321),
        state="s",
        nonce="n",
        code_challenge="c",
        id_token_hint="old.id.token",
        login_hint="user@example.com",
    )
    q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
    assert q["client_id"] == ISSUED_CLIENT_ID
    assert "agent_name_hint" not in q
    assert q["id_token_hint"] == "old.id.token"
    assert q["login_hint"] == "user@example.com"
    assert q["redirect_uri"] == "http://127.0.0.1:54321/auth/callback"


def test_parse_callback_rules():
    ok = {"code": "c", "state": "s", "client_id": ISSUED_CLIENT_ID}
    assert auth.parse_callback(ok, expected_state="s", pending_client_id=auth.DYNAMIC_CLIENT_ID) == (
        auth.CallbackResult(code="c", client_id=ISSUED_CLIENT_ID)
    )
    with pytest.raises(auth.ChatGPTAuthError, match="state"):
        auth.parse_callback({**ok, "state": "x"}, expected_state="s", pending_client_id=auth.DYNAMIC_CLIENT_ID)
    with pytest.raises(auth.ChatGPTAuthError, match="access_denied"):
        auth.parse_callback(
            {"error": "access_denied", "state": "s"}, expected_state="s", pending_client_id=auth.DYNAMIC_CLIENT_ID
        )
    with pytest.raises(auth.ChatGPTAuthError, match="incomplete"):
        auth.parse_callback({"code": "c", "state": "s"}, expected_state="s", pending_client_id=auth.DYNAMIC_CLIENT_ID)
    # Reauthorization: a missing client_id keeps the pending one; a different one is rejected.
    reauth = auth.parse_callback({"code": "c", "state": "s"}, expected_state="s", pending_client_id=ISSUED_CLIENT_ID)
    assert reauth.client_id == ISSUED_CLIENT_ID
    with pytest.raises(auth.ChatGPTAuthError, match="different client_id"):
        auth.parse_callback({**ok, "client_id": "oaiapp_other"}, expected_state="s", pending_client_id=ISSUED_CLIENT_ID)


async def test_host_id_is_created_once_and_import_keeps_it(tmp_path):
    home = tmp_path / "vm"
    host_id = auth.load_or_create_host_id(home)
    assert host_id.startswith("urn:uuid:")
    assert auth.load_or_create_host_id(home) == host_id
    assert stat.S_IMODE((home / auth.HOST_ID_FILE).stat().st_mode) == 0o600

    source = tmp_path / "laptop.json"
    source.write_text(json.dumps(_credential()))
    await auth.import_credential(home, source, out=lambda _line: None)
    imported = auth.read_credential(home / auth.CREDENTIAL_FILE)
    assert imported.ext_agent_host_id == host_id
    assert imported.refresh_token == "rt-1"
    assert (home / auth.HOST_ID_FILE).read_text().strip() == host_id
    assert stat.S_IMODE((home / auth.CREDENTIAL_FILE).stat().st_mode) == 0o600


def test_status_never_prints_tokens(tmp_path):
    _write(tmp_path, _credential())
    text = "\n".join(auth.status_lines(tmp_path))
    assert "user@example.com" in text and ISSUED_CLIENT_ID in text
    assert "rt-1" not in text and "id.token.value" not in text
    stored_token = auth.read_credential(tmp_path / auth.CREDENTIAL_FILE).access_token
    assert stored_token and stored_token not in text


def test_hindsight_admin_exposes_the_sign_in_commands(tmp_path):
    from typer.testing import CliRunner

    from hindsight_api.admin import cli

    runner = CliRunner()
    missing = runner.invoke(cli.app, ["chatgpt", "--home", str(tmp_path), "status"])
    assert missing.exit_code == 1
    assert "hindsight-admin chatgpt login" in missing.output

    _write(tmp_path, _credential())
    found = runner.invoke(cli.app, ["chatgpt", "--home", str(tmp_path), "status"])
    assert found.exit_code == 0
    assert "user@example.com" in found.output


# ---------------------------------------------------------------------------
# ID token validation
# ---------------------------------------------------------------------------


def test_validate_id_token_accepts_valid_token(rsa_jwks):
    key, jwks = rsa_jwks
    claims = auth.validate_id_token(_id_token(key), jwks=jwks, client_id=ISSUED_CLIENT_ID, nonce="n-1")
    assert claims.sub == "user-sub" and claims.email == "user@example.com"


@pytest.mark.parametrize(
    "claims, match",
    [
        ({"aud": "oaiapp_other"}, "Audience|audience"),
        ({"iss": "https://evil.example"}, "issuer|Issuer"),
        ({"exp": int(time.time()) - 3600}, "expired"),
        ({"nonce": "other"}, "nonce"),
    ],
)
def test_validate_id_token_rejects_bad_claims(rsa_jwks, claims, match):
    key, jwks = rsa_jwks
    with pytest.raises(auth.ChatGPTAuthError, match=match):
        auth.validate_id_token(_id_token(key, **claims), jwks=jwks, client_id=ISSUED_CLIENT_ID, nonce="n-1")


def test_validate_id_token_rejects_foreign_signature(rsa_jwks):
    _, jwks = rsa_jwks
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(auth.ChatGPTAuthError, match="Signature|signature"):
        auth.validate_id_token(_id_token(other), jwks=jwks, client_id=ISSUED_CLIENT_ID, nonce="n-1")


# ---------------------------------------------------------------------------
# Full login against a stub issuer
# ---------------------------------------------------------------------------


def _issuer_handler(key: Any, jwks: dict[str, Any], seen: dict[str, Any], *, scope: str):
    async def handler(request: web.Request) -> web.StreamResponse:
        base = f"{request.scheme}://{request.host}"
        if request.path == "/.well-known/openid-configuration":
            return web.json_response({"jwks_uri": f"{base}/jwks", "revocation_endpoint": f"{base}/revoke"})
        if request.path == "/jwks":
            return web.json_response(jwks)
        if request.path == "/api/accounts/oauth/token":
            form = dict(await request.post())
            seen["token_form"] = form
            return web.json_response(
                {
                    "access_token": _fake_access_token(time.time() + 3600),
                    "refresh_token": "rt-new",
                    "id_token": _id_token(key, iss=base, nonce=seen["nonce"]),
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": scope,
                }
            )
        return web.Response(status=404)

    return handler


async def _drive_login(
    tmp_path,
    point_auth_at,
    rsa_jwks,
    *,
    scope: str,
    browser_opens: bool = True,
    stray_states: tuple[str, ...] = (),
    enable_plan_usage: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Run ``auth.login`` with a fake browser that answers the loopback callback.

    ``stray_states`` are sent to the callback first (each must get a 400 and leave
    the login waiting); the printed lines are collected in ``seen["printed"]``.
    """
    key, jwks = rsa_jwks
    seen: dict[str, Any] = {"printed": [], "stray_statuses": []}
    port = _free_port()
    async with stub_server(_issuer_handler(key, jwks, seen, scope=scope)) as base:
        point_auth_at(base)

        def browser(url: str) -> bool:
            q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
            seen["authorize"] = q
            seen["nonce"] = q["nonce"]

            async def hit() -> None:
                async with aiohttp.ClientSession() as s:
                    for stray in stray_states:
                        async with s.get(q["redirect_uri"], params={"code": "x", "state": stray}) as r:
                            seen["stray_statuses"].append(r.status)
                    await s.get(
                        q["redirect_uri"], params={"code": "c1", "state": q["state"], "client_id": ISSUED_CLIENT_ID}
                    )

            seen["task"] = asyncio.get_running_loop().create_task(hit())
            return browser_opens

        credential = await auth.login(
            tmp_path,
            port=port,
            open_browser=browser,
            out=seen["printed"].append,
            enable_plan_usage=enable_plan_usage,
        )
        await seen["task"]
    return credential, seen


async def test_login_registers_exchanges_and_saves(tmp_path, point_auth_at, rsa_jwks):
    credential, seen = await _drive_login(tmp_path, point_auth_at, rsa_jwks, scope=auth.REQUESTED_SCOPE)

    assert seen["authorize"]["client_id"] == auth.DYNAMIC_CLIENT_ID
    host_id = (tmp_path / auth.HOST_ID_FILE).read_text().strip()
    assert seen["authorize"]["ext_agent_host_id"] == host_id
    form = seen["token_form"]
    assert form["grant_type"] == "authorization_code"
    assert form["client_id"] == ISSUED_CLIENT_ID
    assert form["code"] == "c1"
    assert form["redirect_uri"] == seen["authorize"]["redirect_uri"]
    assert form["resource"] == "https://api.openai.com/v1"
    challenge = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest()).rstrip(b"=")
    assert challenge.decode() == seen["authorize"]["code_challenge"]
    assert "client_secret" not in form

    path = tmp_path / auth.CREDENTIAL_FILE
    stored = json.loads(path.read_text())
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stored == json.loads(credential.model_dump_json(exclude_none=True))
    assert stored["client_id"] == ISSUED_CLIENT_ID
    assert stored["subject"] == "user-sub"
    assert stored["ext_agent_host_id"] == host_id
    assert stored["refresh_token"] == "rt-new"
    assert credential.has_plan_scope
    auth.ChatGPTAuth.from_home(tmp_path)  # usable for inference


async def test_login_keeps_issued_client_id_when_exchange_fails(tmp_path, point_auth_at, rsa_jwks):
    """A failed first registration must not leave the next login registering a second client."""
    port = _free_port()

    async def handler(request: web.Request) -> web.StreamResponse:
        return web.json_response({"error": "invalid_grant"}, status=400)

    async with stub_server(handler) as base:
        point_auth_at(base)
        task: dict[str, Any] = {}

        def browser(url: str) -> bool:
            q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}

            async def hit() -> None:
                async with aiohttp.ClientSession() as s:
                    await s.get(
                        q["redirect_uri"], params={"code": "c", "state": q["state"], "client_id": ISSUED_CLIENT_ID}
                    )

            task["t"] = asyncio.get_running_loop().create_task(hit())
            return True

        with pytest.raises(auth.ChatGPTAuthError, match="Code exchange failed"):
            await auth.login(tmp_path, port=port, open_browser=browser, out=lambda _line: None)
        await task["t"]
    stored = auth.read_credential(tmp_path / auth.CREDENTIAL_FILE)
    assert stored.client_id == ISSUED_CLIENT_ID and not stored.is_active


def test_cli_status_import_and_no_browser_url(tmp_path, capsys, monkeypatch):
    home = tmp_path / "server"
    assert auth.main(["--home", str(home), "status"]) == 1
    assert "No ChatGPT plan credentials" in capsys.readouterr().err

    source = tmp_path / "laptop.json"
    source.write_text(json.dumps(_credential()))
    assert auth.main(["--home", str(home), "login", "--from-file", str(source)]) == 0
    assert auth.main(["--home", str(home), "status"]) == 0
    out = capsys.readouterr().out
    assert "Plan usage enabled: yes" in out and "rt-1" not in out


async def test_login_without_plan_scope_keeps_sign_in_but_disables_inference(tmp_path, point_auth_at, rsa_jwks):
    credential, _ = await _drive_login(
        tmp_path, point_auth_at, rsa_jwks, scope="openid profile email offline_access resource.invoke"
    )
    assert credential.client_id == ISSUED_CLIENT_ID
    assert not credential.has_plan_scope
    with pytest.raises(auth.ChatGPTLoginRequiredError, match="not enabled"):
        auth.ChatGPTAuth.from_home(tmp_path)


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


def _refresh_handler(seen: dict[str, Any], *, status: int = 200, body: dict[str, Any] | None = None, delay=0.0):
    async def handler(request: web.Request) -> web.StreamResponse:
        if request.path != "/api/accounts/oauth/token":
            return web.Response(status=404)
        seen.setdefault("forms", []).append(dict(await request.post()))
        if delay:
            await asyncio.sleep(delay)
        payload = body or {
            "access_token": _fake_access_token(time.time() + 3600, marker="new"),
            "refresh_token": "rt-2",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": auth.REQUESTED_SCOPE,
        }
        return web.json_response(payload, status=status)

    return handler


async def test_proactive_refresh_rotates_and_persists(tmp_path, point_auth_at):
    path = _write(tmp_path, _credential(access_token=_fake_access_token(time.time() + 10)))
    seen: dict[str, Any] = {}
    async with stub_server(_refresh_handler(seen)) as base:
        point_auth_at(base)
        manager = auth.ChatGPTAuth.from_home(tmp_path)
        token = await manager.get_access_token()
        await manager.close()

    assert seen["forms"] == [
        {
            "grant_type": "refresh_token",
            "client_id": ISSUED_CLIENT_ID,
            "refresh_token": "rt-1",
            "resource": "https://api.openai.com/v1",
        }
    ]
    stored = json.loads(path.read_text())
    assert stored["access_token"] == token
    assert stored["refresh_token"] == "rt-2"
    assert stored["client_id"] == ISSUED_CLIENT_ID and stored["subject"] == "user-sub"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".credentials.json.*"))  # no stray temp file


async def test_refresh_adopts_token_rotated_by_another_process(tmp_path, point_auth_at):
    _write(tmp_path, _credential(access_token=_fake_access_token(time.time() + 10)))
    seen: dict[str, Any] = {}
    async with stub_server(_refresh_handler(seen)) as base:
        point_auth_at(base)
        manager = auth.ChatGPTAuth.from_home(tmp_path)
        rotated = _fake_access_token(time.time() + 3600, marker="sibling")
        _write(tmp_path, _credential(access_token=rotated, refresh_token="rt-sibling"))
        assert await manager.get_access_token() == rotated
        await manager.close()
    assert "forms" not in seen


async def test_concurrent_refreshes_are_serialized(tmp_path, point_auth_at):
    _write(tmp_path, _credential(access_token=_fake_access_token(time.time() + 10)))
    seen: dict[str, Any] = {}
    async with stub_server(_refresh_handler(seen, delay=0.1)) as base:
        point_auth_at(base)
        managers = [auth.ChatGPTAuth.from_home(tmp_path) for _ in range(3)]
        tokens = await asyncio.gather(*(m.get_access_token() for m in managers))
        for m in managers:
            await m.close()
    assert len(seen["forms"]) == 1
    assert len(set(tokens)) == 1


async def test_reactive_refresh_replaces_rejected_but_unexpired_token(tmp_path, point_auth_at):
    _write(tmp_path, _credential())
    seen: dict[str, Any] = {}
    async with stub_server(_refresh_handler(seen)) as base:
        point_auth_at(base)
        manager = auth.ChatGPTAuth.from_home(tmp_path)
        rejected = await manager.get_access_token()
        await manager.refresh(rejected_token=rejected)
        assert await manager.get_access_token() != rejected
        await manager.close()
    assert len(seen["forms"]) == 1


async def test_rotation_survives_unwritable_store(tmp_path, point_auth_at, monkeypatch):
    """The old refresh token is spent once OpenAI rotates it, so a failed write must not lose the new set."""
    _write(tmp_path, _credential(access_token=_fake_access_token(time.time() + 10)))
    seen: dict[str, Any] = {}

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(30, "Read-only file system")

    async with stub_server(_refresh_handler(seen)) as base:
        point_auth_at(base)
        manager = auth.ChatGPTAuth.from_home(tmp_path)
        monkeypatch.setattr(auth, "write_credential", fail)
        token = await manager.get_access_token()
        assert "new" in base64.urlsafe_b64decode(token.split(".")[1] + "==").decode()
        assert await manager.get_access_token() == token
        # A reactive refresh must send the rotated refresh token, not the spent one on disk.
        await manager.refresh(rejected_token=token)
        await manager.close()
    assert [f["refresh_token"] for f in seen["forms"]] == ["rt-1", "rt-2"]


@pytest.mark.parametrize("code", sorted(auth.TERMINAL_REFRESH_ERROR_CODES))
async def test_terminal_refresh_error_requires_login_and_clears_tokens(tmp_path, point_auth_at, code):
    path = _write(tmp_path, _credential(access_token=_fake_access_token(time.time() - 10)))
    async with stub_server(_refresh_handler({}, status=400, body={"error": code})) as base:
        point_auth_at(base)
        manager = auth.ChatGPTAuth.from_home(tmp_path)
        match = "delete .* and run" if code == "invalid_client" else code
        with pytest.raises(auth.ChatGPTLoginRequiredError, match=match):
            await manager.get_access_token()
        await manager.close()
    stored = json.loads(path.read_text())
    assert "refresh_token" not in stored and "access_token" not in stored
    assert stored["last_auth_error"] == code
    assert stored["client_id"] == ISSUED_CLIENT_ID  # registration kept; invalid_client says how to re-register


async def test_terminal_error_does_not_clear_a_grant_written_meanwhile(tmp_path, point_auth_at):
    """Without a working lock another writer may have stored a newer grant; it must survive."""
    path = _write(tmp_path, _credential(access_token=_fake_access_token(time.time() - 10)))

    async def handler(request: web.Request) -> web.StreamResponse:
        await request.post()
        _write(tmp_path, _credential(refresh_token="rt-other-writer"))
        return web.json_response({"error": "refresh_token_reused"}, status=400)

    async with stub_server(handler) as base:
        point_auth_at(base)
        manager = auth.ChatGPTAuth.from_home(tmp_path)
        assert await manager.get_access_token()
        await manager.close()
    assert json.loads(path.read_text())["refresh_token"] == "rt-other-writer"


async def test_transient_refresh_error_keeps_tokens(tmp_path, point_auth_at):
    path = _write(tmp_path, _credential(access_token=_fake_access_token(time.time() - 10)))
    async with stub_server(_refresh_handler({}, status=503, body={"error": "temporarily_unavailable"})) as base:
        point_auth_at(base)
        manager = auth.ChatGPTAuth.from_home(tmp_path)
        with pytest.raises(auth.ChatGPTAuthError) as excinfo:
            await manager.get_access_token()
        assert not isinstance(excinfo.value, auth.ChatGPTLoginRequiredError)
        await manager.close()
    assert json.loads(path.read_text())["refresh_token"] == "rt-1"


async def test_logout_revokes_and_keeps_registration(tmp_path, point_auth_at):
    path = _write(tmp_path, _credential())
    seen: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.StreamResponse:
        base = f"{request.scheme}://{request.host}"
        if request.path == "/.well-known/openid-configuration":
            return web.json_response({"jwks_uri": f"{base}/jwks", "revocation_endpoint": f"{base}/revoke"})
        if request.path == "/revoke":
            seen["revoke"] = dict(await request.post())
            return web.Response(status=200)
        return web.Response(status=404)

    async with stub_server(handler) as base:
        point_auth_at(base)
        assert await auth.logout(tmp_path, out=lambda _line: None)
    assert seen["revoke"] == {"token": "rt-1", "token_type_hint": "refresh_token", "client_id": ISSUED_CLIENT_ID}
    stored = json.loads(path.read_text())
    for field in ("access_token", "refresh_token", "id_token"):
        assert field not in stored
    assert stored["client_id"] == ISSUED_CLIENT_ID and stored["ext_agent_host_id"] == "urn:uuid:laptop"


def test_discovery_endpoints_are_pinned_to_issuer():
    with pytest.raises(auth.ChatGPTAuthError, match="outside"):
        auth._same_origin_as_issuer("https://evil.example/revoke", "revocation_endpoint")


# ---------------------------------------------------------------------------
# Provider: request shaping and SSE handling
# ---------------------------------------------------------------------------


def _sse(*events: dict[str, Any]) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


def _response(output: list[dict[str, Any]], status: str = "completed", **extra: Any) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": status,
        "model": "gpt-6-luna",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 7,
            "total_tokens": 17,
            "input_tokens_details": {"cached_tokens": 4},
            "output_tokens_details": {"reasoning_tokens": 3},
        },
        **extra,
    }


_TEXT_ITEM = {
    "type": "message",
    "id": "m1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "hello", "annotations": []}],
}


def _completed(output: list[dict[str, Any]] | None = None) -> bytes:
    return _sse(
        {"type": "response.created", "sequence_number": 0, "response": _response([], status="in_progress")},
        {"type": "response.completed", "sequence_number": 1, "response": _response(output or [_TEXT_ITEM])},
    )


def _failed(code: str, message: str = "nope") -> bytes:
    return _sse(
        {
            "type": "response.failed",
            "sequence_number": 1,
            "response": _response([], status="failed", error={"code": code, "message": message}),
        }
    )


class _Upstream:
    """Scripted /responses replies; records each request's JSON body and auth header."""

    def __init__(self, replies: list[tuple[int, bytes | dict[str, Any]]]):
        self.replies = replies
        self.requests: list[dict[str, Any]] = []
        self.auth_headers: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.refresh_forms: list[dict[str, Any]] = []

    async def handler(self, request: web.Request) -> web.StreamResponse:
        if request.path.endswith("/oauth/token"):
            self.refresh_forms.append(dict(await request.post()))
            return web.json_response(
                {
                    "access_token": _fake_access_token(time.time() + 3600, marker="refreshed"),
                    "refresh_token": "rt-2",
                    "expires_in": 3600,
                    "scope": auth.REQUESTED_SCOPE,
                }
            )
        assert request.path.endswith("/responses")
        self.requests.append(await request.json())
        self.auth_headers.append(request.headers.get("Authorization", ""))
        self.headers.append(dict(request.headers))
        status, body = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if status == -1:
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            response.enable_chunked_encoding()
            await response.prepare(request)
            await response.write(b"event: response.created\ndata: {")
            assert request.transport is not None
            request.transport.close()
            return response
        if isinstance(body, dict):
            return web.json_response(body, status=status)
        return web.Response(status=status, body=body, content_type="text/event-stream")


@pytest.fixture
async def make_llm(tmp_path, monkeypatch, point_auth_at):
    """Build a ChatGPTLLM whose credentials, auth and API all point at one stub."""
    created: list[ChatGPTLLM] = []
    servers: list[Any] = []

    async def build(upstream: _Upstream, cred: dict[str, Any] | None = None) -> ChatGPTLLM:
        _write(tmp_path, cred or _credential())
        monkeypatch.setenv("HINDSIGHT_API_LLM_CHATGPT_HOME", str(tmp_path))
        clear_config_cache()
        cm = stub_server(upstream.handler)
        base = await cm.__aenter__()
        servers.append(cm)
        point_auth_at(base)
        monkeypatch.setattr(plan, "_API_BASE_URL", f"{base}/v1")
        llm = create_llm_provider(
            provider="chatgpt",
            api_key="",
            base_url="https://proxy.example/v1",
            model="gpt-6-luna",
            reasoning_effort="high",
        )
        created.append(llm)
        return llm

    yield build
    for llm in created:
        await llm.cleanup()
    for cm in servers:
        await cm.__aexit__(None, None, None)
    clear_config_cache()


def test_request_body_strips_unsupported_fields():
    params = {field: "x" for field in plan._UNSUPPORTED_FIELDS}
    params.update(
        model="m",
        input=[{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}],
        reasoning={"effort": "high"},
        extra_body={"temperature": 0.1, "keep": 1},
        store=True,
    )
    body = plan._plan_request_body(params)
    assert not plan._UNSUPPORTED_FIELDS & body.keys()
    assert body["extra_body"] == {"keep": 1}
    assert body["store"] is False and body["stream"] is True
    assert [i["role"] for i in body["input"]] == ["developer", "user"]
    assert body["reasoning"] == {"effort": "high"}


async def test_call_streams_to_completion_and_shapes_request(make_llm):
    upstream = _Upstream([(200, _completed())])
    llm = await make_llm(upstream)
    result = await llm.call(
        messages=[{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        temperature=0.3,
        max_completion_tokens=100,
        max_retries=0,
    )
    assert result.content == "hello"
    assert result.usage.input_tokens == 10 and result.usage.output_tokens == 4
    assert result.usage.cached_tokens == 4 and result.usage.thoughts_tokens == 3

    body = upstream.requests[0]
    assert body["store"] is False and body["stream"] is True
    for field in ("temperature", "max_output_tokens", "user", "previous_response_id"):
        assert field not in body
    assert all(item.get("role") != "system" for item in body["input"])
    assert body["input"][0] == {"role": "developer", "content": "be brief"}
    assert body["reasoning"] == {"effort": "high"}
    assert body["model"] == "gpt-6-luna"
    stored_token = auth.read_credential(llm._auth.path).access_token
    assert upstream.auth_headers[0] == f"Bearer {stored_token}"


async def test_call_with_tools_parses_function_calls(make_llm):
    call_item = {
        "type": "function_call",
        "id": "fc1",
        "call_id": "call_1",
        "name": "recall",
        "arguments": '{"query": "x"}',
        "status": "completed",
    }
    upstream = _Upstream([(200, _completed([call_item]))])
    llm = await make_llm(upstream)
    tools = [{"type": "function", "function": {"name": "recall", "parameters": {"type": "object"}}}]
    result = await llm.call_with_tools(messages=[{"role": "user", "content": "q"}], tools=tools, max_retries=0)
    assert [(c.id, c.name, c.arguments) for c in result.tool_calls] == [("call_1", "recall", {"query": "x"})]
    assert upstream.requests[0]["tools"] == [{"type": "function", "name": "recall", "parameters": {"type": "object"}}]


async def test_usage_limit_in_stream_defers(make_llm):
    upstream = _Upstream([(200, _failed("subscription_sharing_usage_limit_exceeded"))])
    llm = await make_llm(upstream)
    before = datetime.now(UTC)
    with pytest.raises(ProviderRateLimitResetError) as excinfo:
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=3, initial_backoff=0.01)
    after = datetime.now(UTC)
    retry_at = excinfo.value.retry_at
    assert before + timedelta(seconds=plan.USAGE_LIMIT_DEFER_SECONDS) <= retry_at
    assert retry_at <= after + timedelta(seconds=plan.USAGE_LIMIT_DEFER_SECONDS)
    assert "chatgpt.com/settings/usage" in str(excinfo.value)
    assert len(upstream.requests) == 1


async def test_usage_limit_http_429_defers(make_llm):
    body = {"error": {"code": "subscription_sharing_usage_limit_exceeded", "message": "limit", "type": "x"}}
    upstream = _Upstream([(429, body)])
    llm = await make_llm(upstream)
    with pytest.raises(ProviderRateLimitResetError):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=3, initial_backoff=0.01)
    assert len(upstream.requests) == 1


async def test_unsupported_capability_is_not_retried(make_llm):
    body = {
        "error": {
            "code": "subscription_sharing_unsupported_capability",
            "message": "unsupported",
            "param": "tools",
            "type": "invalid_request_error",
        }
    }
    upstream = _Upstream([(400, body)])
    llm = await make_llm(upstream)
    with pytest.raises(ChatGPTRequestError, match="param='tools'"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=3, initial_backoff=0.01)
    assert len(upstream.requests) == 1


async def test_usage_unavailable_in_stream_retries(make_llm):
    upstream = _Upstream([(200, _failed("subscription_sharing_usage_unavailable")), (200, _completed())])
    llm = await make_llm(upstream)
    result = await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=2, initial_backoff=0.01)
    assert result.content == "hello"
    assert len(upstream.requests) == 2


async def test_stream_without_completed_is_retried_then_fails(make_llm):
    truncated = _sse({"type": "response.created", "sequence_number": 0, "response": _response([], "in_progress")})
    upstream = _Upstream([(200, truncated)])
    llm = await make_llm(upstream)
    with pytest.raises(RetryableResponsesError, match="without response.completed"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=1, initial_backoff=0.01)
    assert len(upstream.requests) == 2


async def test_401_refreshes_once_and_retries(make_llm):
    upstream = _Upstream([(401, {"detail": "token rejected"}), (200, _completed())])
    llm = await make_llm(upstream)
    result = await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)
    assert result.content == "hello"
    assert len(upstream.refresh_forms) == 1
    assert upstream.auth_headers[0] != upstream.auth_headers[1]


async def test_non_transient_stream_failure_is_not_retried(make_llm):
    upstream = _Upstream([(200, _failed("context_length_exceeded"))])
    llm = await make_llm(upstream)
    with pytest.raises(ChatGPTRequestError, match="context_length_exceeded"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=3, initial_backoff=0.01)
    assert len(upstream.requests) == 1


async def test_bare_error_payload_in_stream_defers(make_llm):
    """A ``{"error": {...}}`` stream payload (raised by the SDK as a bare APIError) still maps to a defer."""
    payload = {"error": {"code": "subscription_sharing_usage_limit_exceeded", "message": "limit"}}
    upstream = _Upstream([(200, f"event: error\ndata: {json.dumps(payload)}\n\n".encode())])
    llm = await make_llm(upstream)
    with pytest.raises(ProviderRateLimitResetError):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=3, initial_backoff=0.01)
    assert len(upstream.requests) == 1


async def test_connection_dropped_mid_stream_is_retried(make_llm):
    upstream = _Upstream([(-1, b""), (200, _completed())])
    llm = await make_llm(upstream)
    result = await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=1, initial_backoff=0.01)
    assert result.content == "hello"
    assert len(upstream.requests) == 2


async def test_providers_for_one_home_share_a_manager(make_llm):
    upstream = _Upstream([(200, _completed())])
    first = await make_llm(upstream)
    second = await make_llm(upstream)
    assert first._auth is second._auth


async def test_admission_403_is_not_retried(make_llm):
    upstream = _Upstream([(403, {"detail": "region not permitted"})])
    llm = await make_llm(upstream)
    with pytest.raises(Exception, match="region not permitted"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=3, initial_backoff=0.01)
    assert len(upstream.requests) == 1


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_provider_registration():
    assert not requires_api_key("chatgpt")
    assert PROVIDER_DEFAULT_MODELS["chatgpt"] == "gpt-6-luna"


def test_missing_credentials_fail_with_login_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_CHATGPT_HOME", str(tmp_path / "empty"))
    clear_config_cache()
    try:
        with pytest.raises(RuntimeError, match="hindsight-admin chatgpt login"):
            create_llm_provider(provider="chatgpt", api_key="", base_url="", model="gpt-6-luna", reasoning_effort=None)
    finally:
        clear_config_cache()


# ---------------------------------------------------------------------------
# Review follow-ups
# ---------------------------------------------------------------------------


async def test_refresh_with_malformed_optional_fields_still_persists_rotation(tmp_path, point_auth_at):
    path = _write(tmp_path, _credential(access_token=_fake_access_token(time.time() + 10)))
    body = {
        "access_token": _fake_access_token(time.time() + 3600, marker="new"),
        "refresh_token": "rt-2",
        "expires_in": "soon",
        "earliest_refresh_at": "2026-10-09T00:00:00Z",
        "token_type": 7,
        "scope": auth.REQUESTED_SCOPE,
    }
    async with stub_server(_refresh_handler({}, body=body)) as base:
        point_auth_at(base)
        manager = auth.ChatGPTAuth.from_home(tmp_path)
        token = await manager.get_access_token()
        await manager.close()
    stored = auth.read_credential(path)
    assert stored.access_token == token == body["access_token"]
    assert stored.refresh_token == "rt-2"
    assert stored.earliest_refresh_at is None and stored.expires_in is None


async def test_login_after_declined_plan_usage_requests_consent(tmp_path, point_auth_at, rsa_jwks):
    _write(tmp_path, _credential(scopes=["email", "offline_access", "openid", "profile", "resource.invoke"]))
    credential, seen = await _drive_login(tmp_path, point_auth_at, rsa_jwks, scope=auth.REQUESTED_SCOPE)
    assert seen["authorize"]["prompt"] == "consent"
    assert seen["authorize"]["client_id"] == ISSUED_CLIENT_ID
    assert seen["authorize"]["scope"] == auth.REQUESTED_SCOPE
    assert "agent_name_hint" not in seen["authorize"]
    assert credential.has_plan_scope


async def test_ordinary_reauth_does_not_force_consent(tmp_path, point_auth_at, rsa_jwks):
    _write(tmp_path, _credential())
    _, seen = await _drive_login(tmp_path, point_auth_at, rsa_jwks, scope=auth.REQUESTED_SCOPE)
    assert "prompt" not in seen["authorize"]


async def test_enable_plan_usage_flag_requests_consent(tmp_path, point_auth_at, rsa_jwks):
    _write(tmp_path, _credential())
    _, seen = await _drive_login(tmp_path, point_auth_at, rsa_jwks, scope=auth.REQUESTED_SCOPE, enable_plan_usage=True)
    assert seen["authorize"]["prompt"] == "consent"


async def test_printed_url_never_carries_id_token_hint(tmp_path, point_auth_at, rsa_jwks):
    _write(tmp_path, _credential(id_token="secret.id.token"))
    _, seen = await _drive_login(tmp_path, point_auth_at, rsa_jwks, scope=auth.REQUESTED_SCOPE, browser_opens=False)
    assert seen["authorize"]["id_token_hint"] == "secret.id.token"  # the browser got the hint
    printed = "\n".join(seen["printed"])
    assert "Open this URL" in printed
    assert "id_token_hint" not in printed and "secret.id.token" not in printed


async def test_stray_callbacks_do_not_abort_pending_login(tmp_path, point_auth_at, rsa_jwks):
    credential, seen = await _drive_login(
        tmp_path, point_auth_at, rsa_jwks, scope=auth.REQUESTED_SCOPE, stray_states=("wrong", "\u00e9t\u00e9")
    )
    assert seen["stray_statuses"] == [400, 400]
    assert credential.client_id == ISSUED_CLIENT_ID


@pytest.mark.parametrize(
    ("query", "heading"),
    [({"code": "c"}, "Sign-in received"), ({"error": "access_denied"}, "Sign-in not completed")],
)
async def test_callback_page_is_self_contained_html(query, heading):
    port = _free_port()
    wait = await auth._start_callback_listener(port, timeout=5, expected_state="s1")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://127.0.0.1:{port}{auth.CALLBACK_PATH}", params={**query, "state": "s1"}) as resp:
            page = await resp.text()
            assert resp.status == 200
            assert resp.content_type == "text/html"
            assert resp.headers["Content-Security-Policy"].startswith("default-src 'none'")
            assert resp.headers["Referrer-Policy"] == "no-referrer"
    assert heading in page
    assert "http" not in page  # nothing loaded from elsewhere
    assert (await wait())["state"] == "s1"


def test_parse_callback_rejects_non_ascii_state_without_type_error():
    with pytest.raises(auth.ChatGPTAuthError, match="state"):
        auth.parse_callback({"code": "c", "state": "\u00e9"}, expected_state="s", pending_client_id=ISSUED_CLIENT_ID)


async def test_token_post_does_not_follow_redirects(tmp_path, point_auth_at):
    _write(tmp_path, _credential(access_token=_fake_access_token(time.time() - 10)))
    hits: list[str] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        hits.append(request.path)
        if request.path == "/api/accounts/oauth/token":
            raise web.HTTPTemporaryRedirect("/elsewhere")
        return web.json_response({"access_token": "leaked"})

    async with stub_server(handler) as base:
        point_auth_at(base)
        manager = auth.ChatGPTAuth.from_home(tmp_path)
        with pytest.raises(auth.ChatGPTAuthError, match="HTTP 307") as excinfo:
            await manager.get_access_token()
        assert not isinstance(excinfo.value, auth.ChatGPTLoginRequiredError)
        await manager.close()
    assert hits == ["/api/accounts/oauth/token"]


async def test_import_refuses_a_different_registration_unless_forced(tmp_path):
    home = tmp_path / "server"
    _write(home, _credential(subject="someone-else", client_id="oaiapp_other"))
    source = tmp_path / "laptop.json"
    source.write_text(json.dumps(_credential()))
    with pytest.raises(auth.ChatGPTAuthError, match="different registration"):
        await auth.import_credential(home, source, out=lambda _line: None)
    assert auth.read_credential(home / auth.CREDENTIAL_FILE).subject == "someone-else"
    await auth.import_credential(home, source, out=lambda _line: None, force=True)
    assert auth.read_credential(home / auth.CREDENTIAL_FILE).subject == "user-sub"


async def test_logout_local_only_does_not_revoke(tmp_path, point_auth_at):
    path = _write(tmp_path, _credential())
    hits: list[str] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        hits.append(request.path)
        return web.Response(status=500)

    async with stub_server(handler) as base:
        point_auth_at(base)
        assert not await auth.logout(tmp_path, out=lambda _line: None, revoke=False)
    assert hits == []
    stored = auth.read_credential(path)
    assert not stored.is_active and stored.client_id == ISSUED_CLIENT_ID
    assert stored.has_plan_scope  # the grant record is kept, so the next login is an ordinary reauth


async def test_401_then_401_raises_after_exactly_one_refresh(make_llm):
    upstream = _Upstream([(401, {"detail": "rejected"})])
    llm = await make_llm(upstream)
    with pytest.raises(Exception, match="rejected"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=3, initial_backoff=0.01)
    assert len(upstream.refresh_forms) == 1
    assert len(upstream.requests) == 2


async def test_401_with_plan_code_does_not_refresh(make_llm):
    body = {"error": {"code": "subscription_sharing_invalid_user", "message": "bad user", "type": "x"}}
    upstream = _Upstream([(401, body)])
    llm = await make_llm(upstream)
    with pytest.raises(ChatGPTRequestError, match="subscription_sharing_invalid_user"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=3, initial_backoff=0.01)
    assert upstream.refresh_forms == []
    assert len(upstream.requests) == 1


async def test_openai_org_and_project_env_are_not_sent(make_llm, monkeypatch):
    monkeypatch.setenv("OPENAI_ORG_ID", "org-from-env")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-from-env")
    upstream = _Upstream([(200, _completed())])
    llm = await make_llm(upstream)
    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)
    sent = {k.lower() for k in upstream.headers[0]}
    assert "openai-organization" not in sent and "openai-project" not in sent


async def test_provider_cleanup_keeps_shared_manager_usable(make_llm):
    upstream = _Upstream([(401, {"detail": "rejected"}), (200, _completed())])
    # One credential for both builds: _credential() stamps the token with the current
    # second, so a second call can write a different token, which the 401 path then
    # adopts from the file instead of refreshing.
    cred = _credential()
    first = await make_llm(upstream, cred)
    second = await make_llm(upstream, cred)
    await first.cleanup()
    result = await second.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)
    assert result.content == "hello"
    assert len(upstream.refresh_forms) == 1
