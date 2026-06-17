"""Tests for the native Nous Portal OAuth provider.

The Nous provider mirrors the Codex provider: it reads OAuth state from the
Hermes auth store (``~/.hermes/auth.json``, ``providers.nous``) and refreshes
the inference JWT itself, with no dependency on the ``hermes_cli`` package.

These tests pin that behaviour against a fake auth store on disk and a stubbed
refresh endpoint — no network, no Hermes install required:

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
import os
import stat
import time
from pathlib import Path

import httpx
import pytest

from hindsight_api.engine.providers.nous_auth import (
    _NOUS_TOKEN_REFRESH_SKEW_SECONDS,
    NousAuthManager,
    NousNotLoggedInError,
    NousRefreshExpiredError,
)

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


def test_from_file_loads_nous_oauth_state(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    _write_store(auth, _fresh_state())

    mgr = NousAuthManager.from_file(auth)

    assert mgr.refresh_token == "rt-original"
    assert mgr.base_url == "https://inference-api.nousresearch.com/v1"
    assert mgr.ensure_fresh_token() == mgr.access_token  # fresh JWT → no refresh


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


def test_stale_token_triggers_refresh_with_hermes_request_shape(tmp_path: Path, monkeypatch) -> None:
    auth = tmp_path / "auth.json"
    near_exp = int(time.time()) + (_NOUS_TOKEN_REFRESH_SKEW_SECONDS - 5)  # within skew → stale
    _write_store(auth, _fresh_state(access_token=_jwt_with_exp(near_exp)))
    mgr = NousAuthManager.from_file(auth)

    new_access = _jwt_with_exp(int(time.time()) + 3600)
    captured: dict = {}

    def fake_post(url, *, headers=None, data=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["data"] = data
        return httpx.Response(
            200,
            json={"access_token": new_access, "refresh_token": "rt-rotated", "expires_in": 3600},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(mgr._http_client, "post", fake_post)

    token = mgr.ensure_fresh_token()

    assert token == new_access
    assert captured["url"] == "https://portal.nousresearch.com/api/oauth/token"
    assert captured["headers"]["x-nous-refresh-token"] == "rt-original"
    assert captured["data"] == {"grant_type": "refresh_token", "client_id": "hermes-cli"}

    # Rotated tokens persisted back into providers.nous, pool preserved.
    on_disk = json.loads(auth.read_text())
    assert on_disk["providers"]["nous"]["access_token"] == new_access
    assert on_disk["providers"]["nous"]["refresh_token"] == "rt-rotated"
    assert on_disk["providers"]["nous"]["agent_key"] == new_access  # bearer == access_token
    assert on_disk["credential_pool"]["openai-codex"] == {"keep": "me"}  # not clobbered
    assert stat.S_IMODE(auth.stat().st_mode) == 0o600


def test_refresh_rereads_latest_refresh_token_from_disk(tmp_path: Path, monkeypatch) -> None:
    """Single-use safety: a token Hermes rotated on disk is used, not the stale
    in-memory one the manager loaded at startup."""
    auth = tmp_path / "auth.json"
    near_exp = int(time.time()) + 10
    _write_store(auth, _fresh_state(access_token=_jwt_with_exp(near_exp)))
    mgr = NousAuthManager.from_file(auth)
    assert mgr.refresh_token == "rt-original"

    # Simulate a concurrent Hermes refresh that rotated the RT on disk.
    rotated = _fresh_state(access_token=_jwt_with_exp(near_exp), refresh_token="rt-from-hermes")
    _write_store(auth, rotated)

    sent_rt: dict = {}

    def fake_post(url, *, headers=None, data=None, timeout=None):
        sent_rt["value"] = headers["x-nous-refresh-token"]
        return httpx.Response(
            200,
            json={"access_token": _jwt_with_exp(int(time.time()) + 3600), "expires_in": 3600},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(mgr._http_client, "post", fake_post)
    mgr.refresh_tokens(force=True)

    assert sent_rt["value"] == "rt-from-hermes"  # disk value, not the stale in-memory "rt-original"


# ---------------------------------------------------------------------------
# Terminal errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status,body", [(400, {"error": "invalid_grant"}), (401, {"error": "refresh_token_reused"})])
def test_terminal_refresh_error_raises_and_does_not_loop(tmp_path: Path, monkeypatch, status, body) -> None:
    auth = tmp_path / "auth.json"
    _write_store(auth, _fresh_state(access_token=_jwt_with_exp(int(time.time()) - 10)))
    mgr = NousAuthManager.from_file(auth)

    calls = {"n": 0}

    def fake_post(url, *, headers=None, data=None, timeout=None):
        calls["n"] += 1
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(mgr._http_client, "post", fake_post)

    with pytest.raises(NousRefreshExpiredError, match="hermes portal"):
        mgr.refresh_tokens(force=True)
    assert calls["n"] == 1  # one attempt, no retry loop


def test_missing_refresh_token_raises_runtime_error(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    state = _fresh_state(access_token=_jwt_with_exp(int(time.time()) - 10))
    del state["refresh_token"]
    _write_store(auth, state)
    mgr = NousAuthManager.from_file(auth)

    with pytest.raises(RuntimeError, match="no refresh_token"):
        mgr.refresh_tokens(force=True)


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
