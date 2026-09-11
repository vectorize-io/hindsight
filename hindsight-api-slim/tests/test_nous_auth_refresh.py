"""Tests for the native Nous Portal OAuth provider.

The Nous provider mirrors the Codex provider: it reads OAuth state from the
Hermes auth store (``~/.hermes/auth.json``, ``providers.nous``) and refreshes
the inference JWT itself, with no dependency on the ``hermes_cli`` package.

These tests pin that behaviour against a fake auth store on disk and a local
stub of the refresh endpoint (``tests/aiohttp_stub.py``) — no external network,
no Hermes install required:

- ``from_file`` loads access/refresh tokens (and raises a clear "logged out"
  error when ``providers.nous`` is absent / has no access_token).
- A loaded static-shaped store still surfaces the access_token as the bearer.
- Proactive refresh fires when the JWT ``exp`` claim is near/past expiry.
- The refresh request matches Hermes' shape: POST {portal}/api/oauth/token with
  an ``x-nous-refresh-token`` header and a ``grant_type=refresh_token`` body.
- The rotated refresh_token is persisted atomically back into
  ``providers.nous`` (mode 0600) without clobbering sibling fields.
- Terminal refresh errors raise ``NousRefreshExpiredError`` and do not loop.
- Single-use safety: refresh re-reads the latest refresh_token from disk under
  the lock before exchanging.
- The provider registers in ``create_llm_provider`` and needs no api_key.
"""

from __future__ import annotations

import base64
import json
import stat
import time
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

from hindsight_api.engine.providers.nous_auth import (
    _NOUS_TOKEN_REFRESH_SKEW_SECONDS,
    NousAuthManager,
    NousNotLoggedInError,
    NousRefreshExpiredError,
)
from tests.aiohttp_stub import stub_server

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jwt_with_exp(exp_unixtime: int) -> str:
    """Build an unsigned JWT whose payload carries the given ``exp`` claim."""

    def b64(obj: dict) -> str:
        raw = json.dumps(obj).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    return f"{b64({'alg': 'none'})}.{b64({'exp': exp_unixtime})}.sig"


def _write_store(path: Path, state: dict | None, *, extra: dict | None = None) -> None:
    store: dict = {"version": 1, "providers": {}, "credential_pool": {"openai-codex": {"keep": "me"}}}
    if extra:
        store.update(extra)
    if state is not None:
        store["providers"]["nous"] = state
    path.write_text(json.dumps(store, indent=2))


class _PortalTokenEndpoint:
    """Stub of the Portal's ``/api/oauth/token``: records requests, answers with one reply."""

    def __init__(self, status: int, body: dict[str, Any]) -> None:
        self._status = status
        self._body = body
        self.requests: list[dict[str, Any]] = []

    async def handle(self, request: web.Request) -> web.StreamResponse:
        form = await request.post()
        self.requests.append({"path": request.path, "headers": dict(request.headers), "data": dict(form)})
        return web.json_response(self._body, status=self._status)


def _fresh_state(**overrides) -> dict:
    state = {
        "access_token": _jwt_with_exp(int(time.time()) + 3600),
        "refresh_token": "rt-original",
        "portal_base_url": "https://portal.nousresearch.com",
        "inference_base_url": "https://inference-api.nousresearch.com/v1",
        "client_id": "hermes-cli",
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# from_file
# ---------------------------------------------------------------------------


async def test_from_file_loads_nous_oauth_state(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    _write_store(auth, _fresh_state())

    mgr = NousAuthManager.from_file(auth)

    assert mgr.refresh_token == "rt-original"
    assert mgr.base_url == "https://inference-api.nousresearch.com/v1"
    assert await mgr.ensure_fresh_token() == mgr.access_token  # fresh JWT → no refresh


def test_from_file_missing_file_raises_not_logged_in(tmp_path: Path) -> None:
    with pytest.raises(NousNotLoggedInError, match="hermes portal"):
        NousAuthManager.from_file(tmp_path / "nope.json")


def test_from_file_without_nous_provider_raises_not_logged_in(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    _write_store(auth, None)  # has other providers/pool, but no providers.nous
    with pytest.raises(NousNotLoggedInError, match="not logged into Nous Portal"):
        NousAuthManager.from_file(auth)


def test_from_file_without_access_token_raises(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    _write_store(auth, {"refresh_token": "rt"})
    with pytest.raises(NousNotLoggedInError, match="no access_token"):
        NousAuthManager.from_file(auth)


# ---------------------------------------------------------------------------
# Proactive refresh + request shape + persistence
# ---------------------------------------------------------------------------


async def test_stale_token_triggers_refresh_with_hermes_request_shape(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    near_exp = int(time.time()) + (_NOUS_TOKEN_REFRESH_SKEW_SECONDS - 5)  # within skew → stale
    new_access = _jwt_with_exp(int(time.time()) + 3600)
    endpoint = _PortalTokenEndpoint(
        200, {"access_token": new_access, "refresh_token": "rt-rotated", "expires_in": 3600}
    )

    async with stub_server(endpoint.handle) as portal:
        _write_store(auth, _fresh_state(access_token=_jwt_with_exp(near_exp), portal_base_url=portal))
        mgr = NousAuthManager.from_file(auth)
        token = await mgr.ensure_fresh_token()
        await mgr.close()

    assert token == new_access
    [seen] = endpoint.requests
    assert seen["path"] == "/api/oauth/token"
    assert seen["headers"]["x-nous-refresh-token"] == "rt-original"
    assert seen["headers"]["Content-Type"] == "application/x-www-form-urlencoded"
    assert seen["data"] == {"grant_type": "refresh_token", "client_id": "hermes-cli"}

    # Rotated tokens persisted back into providers.nous, pool preserved.
    on_disk = json.loads(auth.read_text())
    assert on_disk["providers"]["nous"]["access_token"] == new_access
    assert on_disk["providers"]["nous"]["refresh_token"] == "rt-rotated"
    assert on_disk["providers"]["nous"]["agent_key"] == new_access  # bearer == access_token
    assert on_disk["credential_pool"]["openai-codex"] == {"keep": "me"}  # not clobbered
    assert stat.S_IMODE(auth.stat().st_mode) == 0o600


async def test_refresh_rereads_latest_refresh_token_from_disk(tmp_path: Path) -> None:
    """Single-use safety: a token Hermes rotated on disk is used, not the stale
    in-memory one the manager loaded at startup."""
    auth = tmp_path / "auth.json"
    near_exp = int(time.time()) + 10
    endpoint = _PortalTokenEndpoint(200, {"access_token": _jwt_with_exp(int(time.time()) + 3600), "expires_in": 3600})

    async with stub_server(endpoint.handle) as portal:
        _write_store(auth, _fresh_state(access_token=_jwt_with_exp(near_exp), portal_base_url=portal))
        mgr = NousAuthManager.from_file(auth)
        assert mgr.refresh_token == "rt-original"

        # Simulate a concurrent Hermes refresh that rotated the RT on disk.
        rotated = _fresh_state(
            access_token=_jwt_with_exp(near_exp), refresh_token="rt-from-hermes", portal_base_url=portal
        )
        _write_store(auth, rotated)

        await mgr.refresh_tokens(force=True)
        await mgr.close()

    # The disk value, not the stale in-memory "rt-original".
    assert endpoint.requests[0]["headers"]["x-nous-refresh-token"] == "rt-from-hermes"


# ---------------------------------------------------------------------------
# Terminal errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status,body", [(400, {"error": "invalid_grant"}), (401, {"error": "refresh_token_reused"})])
async def test_terminal_refresh_error_raises_and_does_not_loop(tmp_path: Path, status, body) -> None:
    auth = tmp_path / "auth.json"
    endpoint = _PortalTokenEndpoint(status, body)

    async with stub_server(endpoint.handle) as portal:
        _write_store(auth, _fresh_state(access_token=_jwt_with_exp(int(time.time()) - 10), portal_base_url=portal))
        mgr = NousAuthManager.from_file(auth)
        with pytest.raises(NousRefreshExpiredError, match="hermes portal"):
            await mgr.refresh_tokens(force=True)
        await mgr.close()

    assert len(endpoint.requests) == 1  # one attempt, no retry loop


async def test_server_error_is_a_runtime_error_not_a_terminal_one(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    endpoint = _PortalTokenEndpoint(503, {"error": "unavailable"})

    async with stub_server(endpoint.handle) as portal:
        _write_store(auth, _fresh_state(access_token=_jwt_with_exp(int(time.time()) - 10), portal_base_url=portal))
        mgr = NousAuthManager.from_file(auth)
        with pytest.raises(RuntimeError, match="HTTP 503") as exc_info:
            await mgr.refresh_tokens(force=True)
        await mgr.close()

    assert not isinstance(exc_info.value, NousRefreshExpiredError)


async def test_missing_refresh_token_raises_runtime_error(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    state = _fresh_state(access_token=_jwt_with_exp(int(time.time()) - 10))
    del state["refresh_token"]
    _write_store(auth, state)
    mgr = NousAuthManager.from_file(auth)

    with pytest.raises(RuntimeError, match="no refresh_token"):
        await mgr.refresh_tokens(force=True)


# ---------------------------------------------------------------------------
# JWT exp decoding
# ---------------------------------------------------------------------------


def test_unparseable_exp_is_not_treated_as_stale(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    _write_store(auth, _fresh_state(access_token="not-a-jwt"))
    mgr = NousAuthManager.from_file(auth)
    # exp can't be determined → prefer reactive 401 recovery over aggressive refresh.
    assert mgr._token_is_stale() is False


def test_load_refresh_token_from_file_missing_returns_none(tmp_path: Path) -> None:
    assert NousAuthManager.load_refresh_token_from_file(tmp_path / "absent.json") is None
