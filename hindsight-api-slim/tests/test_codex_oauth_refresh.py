"""Tests for Codex OAuth token refresh (issue #1637).

The Codex provider was originally a startup-only credential loader: it read
``~/.codex/auth.json`` once and used the cached access_token forever. These
tests pin the new automatic-refresh behavior:

- ``refresh_token`` is now actually loaded from auth.json.
- The provider proactively refreshes ~60s before the JWT ``exp`` claim.
- It reactively refreshes once on a 401/403 from the Codex backend.
- The OAuth refresh request shape mirrors the canonical ``@openai/codex``
  CLI (POST https://auth.openai.com/oauth/token, JSON body with hardcoded
  client_id, grant_type=refresh_token).
- Terminal error codes (refresh_token_expired/reused/invalidated) raise a
  permanent error and do not loop.
- Concurrent callers serialize through a single-flight lock.
- ``auth.json`` is persisted atomically via tempfile+rename with mode 0600.

Tests construct ``CodexLLM`` with ``_load_codex_auth`` mocked, then drive
JWT exp / network / persistence paths through targeted patches.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hindsight_api.engine.providers.codex_llm import (
    _CODEX_CLIENT_ID,
    _CODEX_REFRESH_TOKEN_URL,
    CodexLLM,
    CodexRefreshExpiredError,
)


def _make_jwt(exp_unixtime: int | None) -> str:
    """Build a minimal JWT-shaped token with the given ``exp`` claim.

    Signature segment is a placeholder — we don't verify, we only decode
    the payload to read ``exp``.
    """
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).rstrip(b"=").decode()
    payload_dict: dict[str, object] = {}
    if exp_unixtime is not None:
        payload_dict["exp"] = exp_unixtime
    payload = base64.urlsafe_b64encode(json.dumps(payload_dict).encode()).rstrip(b"=").decode()
    signature = "sig"
    return f"{header}.{payload}.{signature}"


def _build_llm(refresh_token: str | None = "rt-initial", access_token: str | None = None) -> CodexLLM:
    """Construct a CodexLLM with patched auth-file reads."""
    if access_token is None:
        access_token = _make_jwt(int(time.time()) + 3600)  # fresh by default
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=(access_token, "acct-123")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=refresh_token),
    ):
        return CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.4-mini",
        )


# ---------------------------------------------------------------------------
# JWT exp decode
# ---------------------------------------------------------------------------


def test_jwt_exp_decode_returns_int_for_valid_token():
    token = _make_jwt(1_800_000_000)
    assert CodexLLM._decode_jwt_exp_unixtime(token) == 1_800_000_000


def test_jwt_exp_decode_returns_none_when_exp_missing():
    token = _make_jwt(None)
    assert CodexLLM._decode_jwt_exp_unixtime(token) is None


def test_jwt_exp_decode_returns_none_for_malformed_token():
    assert CodexLLM._decode_jwt_exp_unixtime("not.a.real.jwt") is None
    assert CodexLLM._decode_jwt_exp_unixtime("only-one-segment") is None
    assert CodexLLM._decode_jwt_exp_unixtime("a.!!notbase64!!.c") is None


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


def test_token_is_stale_true_when_expired():
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(access_token=expired)
    assert llm._token_is_stale() is True


def test_token_is_stale_true_within_skew_window():
    # 30s before expiry, default skew is 60s → should be considered stale.
    soon = _make_jwt(int(time.time()) + 30)
    llm = _build_llm(access_token=soon)
    assert llm._token_is_stale() is True


def test_token_is_stale_false_when_far_from_expiry():
    far = _make_jwt(int(time.time()) + 3600)
    llm = _build_llm(access_token=far)
    assert llm._token_is_stale() is False


def test_token_is_stale_false_when_exp_unparseable():
    # When we can't decide, we'd rather use a possibly-expired token and
    # recover via the reactive 401 path than refresh aggressively.
    llm = _build_llm(access_token="opaque-token-no-jwt-structure")
    assert llm._token_is_stale() is False


# ---------------------------------------------------------------------------
# refresh_token loading
# ---------------------------------------------------------------------------


def test_refresh_token_loaded_from_auth_file(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "at",
                    "refresh_token": "rt-from-disk",
                    "account_id": "acct",
                },
            }
        )
    )
    with patch.object(CodexLLM, "_load_codex_auth", return_value=("at", "acct")):
        llm = CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.4-mini",
        )
    # Now point the auth_file at our tmp file and reload.
    llm._auth_file = auth_file
    assert llm._load_codex_refresh_token() == "rt-from-disk"


def test_refresh_token_returns_none_when_field_absent(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "at"}}))
    llm = _build_llm()
    llm._auth_file = auth_file
    assert llm._load_codex_refresh_token() is None


def test_refresh_token_returns_none_when_file_missing(tmp_path: Path):
    llm = _build_llm()
    llm._auth_file = tmp_path / "definitely-not-here.json"
    assert llm._load_codex_refresh_token() is None


# ---------------------------------------------------------------------------
# Atomic persistence
# ---------------------------------------------------------------------------


def test_persist_auth_atomic_writes_mode_0600_and_preserves_fields(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "OPENAI_API_KEY": None,
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "old",
                    "refresh_token": "rt-old",
                    "account_id": "acct-keep",
                    "id_token": {"email": "user@example.com"},
                },
                "last_refresh": "2026-01-01T00:00:00Z",
            }
        )
    )

    llm = _build_llm()
    llm._auth_file = auth_file
    llm._persist_auth_atomic({"access_token": "new", "refresh_token": "rt-new"})

    written = json.loads(auth_file.read_text())
    assert written["tokens"]["access_token"] == "new"
    assert written["tokens"]["refresh_token"] == "rt-new"
    # Untouched fields are preserved (account_id, id_token, auth_mode).
    assert written["tokens"]["account_id"] == "acct-keep"
    assert written["tokens"]["id_token"] == {"email": "user@example.com"}
    assert written["auth_mode"] == "chatgpt"
    # last_refresh got bumped to a new ISO-8601 UTC timestamp.
    assert written["last_refresh"] != "2026-01-01T00:00:00Z"
    assert written["last_refresh"].endswith("Z")

    if sys.platform != "win32":
        mode = stat.S_IMODE(auth_file.stat().st_mode)
        assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_persist_auth_atomic_does_not_leak_tempfile_on_success(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"tokens": {"access_token": "old"}}))

    llm = _build_llm()
    llm._auth_file = auth_file
    llm._persist_auth_atomic({"access_token": "new"})

    # No sibling tempfile should remain — atomic rename consumed it.
    siblings = [p.name for p in tmp_path.iterdir()]
    assert siblings == ["auth.json"], f"unexpected leftover files: {siblings}"


# ---------------------------------------------------------------------------
# _refresh_oauth_tokens — request shape, in-memory update, rotation
# ---------------------------------------------------------------------------


def _refresh_response(status_code: int, body: dict | str) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    if isinstance(body, dict):
        response.json.return_value = body
        response.text = json.dumps(body)
    else:
        response.json.side_effect = json.JSONDecodeError("nope", body, 0)
        response.text = body
    return response


@pytest.mark.asyncio
async def test_refresh_sends_canonical_request_shape(tmp_path: Path):
    """POST JSON body with client_id + grant_type=refresh_token + refresh_token."""
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(refresh_token="rt-current", access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": "x", "refresh_token": "rt-current"}}))

    fresh_access = _make_jwt(int(time.time()) + 3600)
    refresh_resp = _refresh_response(200, {"access_token": fresh_access, "refresh_token": "rt-rotated"})

    with patch.object(llm._auth_manager._http_client, "post", return_value=refresh_resp) as mock_post:
        await llm._refresh_oauth_tokens()

    call_args = mock_post.call_args
    assert call_args.args[0] == _CODEX_REFRESH_TOKEN_URL
    assert call_args.kwargs["headers"]["Content-Type"] == "application/json"
    assert call_args.kwargs["json"] == {
        "client_id": _CODEX_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": "rt-current",
    }


@pytest.mark.asyncio
async def test_refresh_updates_in_memory_credentials(tmp_path: Path):
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(refresh_token="rt-old", access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": "x", "refresh_token": "rt-old"}}))

    new_access = _make_jwt(int(time.time()) + 3600)
    refresh_resp = _refresh_response(200, {"access_token": new_access, "refresh_token": "rt-new"})

    with patch.object(llm._auth_manager._http_client, "post", return_value=refresh_resp):
        await llm._refresh_oauth_tokens()

    assert llm.access_token == new_access
    assert llm.refresh_token == "rt-new"


@pytest.mark.asyncio
async def test_refresh_keeps_existing_refresh_token_when_server_omits_one(tmp_path: Path):
    """If the OAuth response has no ``refresh_token`` field, keep the one we have."""
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(refresh_token="rt-keep", access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": "x", "refresh_token": "rt-keep"}}))

    new_access = _make_jwt(int(time.time()) + 3600)
    refresh_resp = _refresh_response(200, {"access_token": new_access})

    with patch.object(llm._auth_manager._http_client, "post", return_value=refresh_resp):
        await llm._refresh_oauth_tokens()

    assert llm.refresh_token == "rt-keep"


@pytest.mark.asyncio
async def test_refresh_raises_permanent_error_on_terminal_oauth_code(tmp_path: Path):
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(refresh_token="rt-stale", access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": "x", "refresh_token": "rt-stale"}}))

    bad_resp = _refresh_response(401, {"error": {"code": "refresh_token_expired"}})

    with patch.object(llm._auth_manager._http_client, "post", return_value=bad_resp):
        with pytest.raises(CodexRefreshExpiredError):
            await llm._refresh_oauth_tokens()


@pytest.mark.asyncio
async def test_refresh_raises_permanent_error_on_unknown_401(tmp_path: Path):
    """Any 401 from the refresh endpoint is treated as permanent — matches upstream Rust classification."""
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(refresh_token="rt-stale", access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": "x", "refresh_token": "rt-stale"}}))

    bad_resp = _refresh_response(401, {"error": "something_else"})

    with patch.object(llm._auth_manager._http_client, "post", return_value=bad_resp):
        with pytest.raises(CodexRefreshExpiredError):
            await llm._refresh_oauth_tokens()


@pytest.mark.asyncio
async def test_refresh_raises_runtime_error_on_5xx(tmp_path: Path):
    """5xx is transient from the caller's perspective — surface as RuntimeError, not CodexRefreshExpiredError."""
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(refresh_token="rt-current", access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": "x", "refresh_token": "rt-current"}}))

    bad_resp = _refresh_response(503, "service unavailable")

    with patch.object(llm._auth_manager._http_client, "post", return_value=bad_resp):
        with pytest.raises(RuntimeError) as exc_info:
            await llm._refresh_oauth_tokens()
    assert not isinstance(exc_info.value, CodexRefreshExpiredError)


@pytest.mark.asyncio
async def test_refresh_does_not_log_token_values(tmp_path: Path, caplog):
    expired = _make_jwt(int(time.time()) - 60)
    secret_rt = "rt-DO-NOT-LEAK-THIS"
    llm = _build_llm(refresh_token=secret_rt, access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": "x", "refresh_token": secret_rt}}))

    new_access = _make_jwt(int(time.time()) + 3600)
    refresh_resp = _refresh_response(200, {"access_token": new_access, "refresh_token": "rt-also-secret"})

    with patch.object(llm._auth_manager._http_client, "post", return_value=refresh_resp):
        with caplog.at_level("DEBUG"):
            await llm._refresh_oauth_tokens()

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert secret_rt not in log_text
    assert new_access not in log_text
    assert "rt-also-secret" not in log_text


# ---------------------------------------------------------------------------
# Single-flight under concurrent callers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_ensure_fresh_token_calls_produce_one_refresh(tmp_path: Path):
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(refresh_token="rt", access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": "x", "refresh_token": "rt"}}))

    new_access = _make_jwt(int(time.time()) + 3600)
    call_count = 0

    def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Simulate non-zero refresh latency so concurrent callers actually queue.
        time.sleep(0.01)
        return _refresh_response(200, {"access_token": new_access, "refresh_token": "rt-new"})

    with patch.object(llm._auth_manager._http_client, "post", new=fake_post):
        await asyncio.gather(*(llm._ensure_fresh_token() for _ in range(10)))

    assert call_count == 1, f"expected 1 network refresh under contention, got {call_count}"


# ---------------------------------------------------------------------------
# Reactive 401 retry on the request path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_reactively_refreshes_on_401_and_retries(tmp_path: Path):
    """A backend 401 triggers one refresh + retry instead of immediately raising."""
    fresh = _make_jwt(int(time.time()) + 3600)  # not stale; the 401 is the trigger
    llm = _build_llm(refresh_token="rt", access_token=fresh)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": fresh, "refresh_token": "rt"}}))

    new_access = _make_jwt(int(time.time()) + 3600)

    success_resp = MagicMock()
    success_resp.status_code = 200
    success_resp.raise_for_status = MagicMock(return_value=None)

    fail_response = MagicMock()
    fail_response.status_code = 401
    fail_response.text = "unauthorized"

    refresh_resp = _refresh_response(200, {"access_token": new_access, "refresh_token": "rt-new"})

    call_count = {"refresh": 0, "post": 0}

    # Sync mock for the auth manager's HTTP client (used for token refresh).
    def fake_refresh_post(*args, **kwargs):
        call_count["refresh"] += 1
        return refresh_resp

    # Async mock for the LLM's HTTP client (used for backend calls).
    async def fake_backend_post(url, **kwargs):
        call_count["post"] += 1
        if call_count["post"] == 1:
            raise httpx.HTTPStatusError("401", request=MagicMock(), response=fail_response)
        return success_resp

    with (
        patch.object(llm._auth_manager._http_client, "post", new=fake_refresh_post),
        patch.object(llm._client, "post", new=fake_backend_post),
        patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock, return_value="ok"),
    ):
        result = await llm.call(
            messages=[{"role": "user", "content": "ping"}],
            max_retries=0,
            initial_backoff=0.0,
            max_backoff=0.0,
        )

    assert result == "ok"
    assert call_count["refresh"] == 1
    assert call_count["post"] == 2  # one 401, one success after refresh
    assert llm.access_token == new_access


@pytest.mark.asyncio
async def test_call_proactively_refreshes_when_token_is_stale(tmp_path: Path):
    """A near-expiry token triggers refresh BEFORE the request is sent."""
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(refresh_token="rt", access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": expired, "refresh_token": "rt"}}))

    new_access = _make_jwt(int(time.time()) + 3600)
    refresh_resp = _refresh_response(200, {"access_token": new_access, "refresh_token": "rt-new"})

    success_resp = MagicMock()
    success_resp.status_code = 200
    success_resp.raise_for_status.return_value = None

    call_order: list[str] = []

    def fake_refresh_post(*args, **kwargs):
        call_order.append("refresh")
        return refresh_resp

    async def fake_backend_post(url, **kwargs):
        call_order.append("backend")
        # Assert that by the time the backend is called, the new token is in use.
        assert kwargs["headers"]["Authorization"] == f"Bearer {new_access}"
        return success_resp

    with (
        patch.object(llm._auth_manager._http_client, "post", new=fake_refresh_post),
        patch.object(llm._client, "post", new=fake_backend_post),
        patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock, return_value="ok"),
    ):
        await llm.call(
            messages=[{"role": "user", "content": "ping"}],
            max_retries=0,
            initial_backoff=0.0,
            max_backoff=0.0,
        )

    assert call_order == ["refresh", "backend"], "expected proactive refresh BEFORE the backend call"


@pytest.mark.asyncio
async def test_call_does_not_refresh_when_token_is_fresh(tmp_path: Path):
    fresh = _make_jwt(int(time.time()) + 3600)
    llm = _build_llm(refresh_token="rt", access_token=fresh)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": fresh, "refresh_token": "rt"}}))

    success_resp = MagicMock()
    success_resp.status_code = 200
    success_resp.raise_for_status.return_value = None

    call_count = {"backend": 0}

    async def fake_backend_post(url, **kwargs):
        call_count["backend"] += 1
        return success_resp

    with (
        patch.object(llm._client, "post", new=fake_backend_post),
        patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock, return_value="ok"),
    ):
        await llm.call(
            messages=[{"role": "user", "content": "ping"}],
            max_retries=0,
            initial_backoff=0.0,
            max_backoff=0.0,
        )

    assert call_count == {"backend": 1}


# ---------------------------------------------------------------------------
# CodexOAuthEmbeddings — proactive + reactive token refresh
# ---------------------------------------------------------------------------


def _make_codex_auth_file(tmp_path: Path, access_token: str, refresh_token: str = "rt-initial") -> Path:
    """Write a minimal ~/.codex/auth.json in tmp_path and return its path."""
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    auth_file = codex_dir / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "account_id": "acct-test",
                },
            }
        )
    )
    return auth_file


def test_codex_oauth_embeddings_picks_up_refreshed_token_on_encode(tmp_path: Path, monkeypatch):
    """encode() calls ensure_fresh_token() and updates api_key when the token rotated."""
    from hindsight_api.engine.embeddings import CodexOAuthEmbeddings

    expired = _make_jwt(int(time.time()) - 60)
    new_access = _make_jwt(int(time.time()) + 3600)

    _make_codex_auth_file(tmp_path, expired, refresh_token="rt-embed")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    emb = CodexOAuthEmbeddings(model="text-embedding-3-small", batch_size=10)

    refresh_resp = _refresh_response(200, {"access_token": new_access, "refresh_token": "rt-new"})

    fake_embeddings = [SimpleNamespace(index=0, embedding=[0.1] * 1536)]
    fake_create_resp = SimpleNamespace(data=fake_embeddings)

    with patch.object(emb._auth_manager._http_client, "post", return_value=refresh_resp):
        emb._client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: fake_create_resp))
        emb._dimension = 1536
        result = emb.encode(["hello"])

    assert result == [[0.1] * 1536]
    # After proactive refresh the manager's token should be the new one.
    assert emb._auth_manager.access_token == new_access
    # api_key on the embeddings object should also be updated.
    assert emb.api_key == new_access


def test_codex_oauth_embeddings_reactive_refresh_on_401(tmp_path: Path, monkeypatch):
    """On AuthenticationError from OpenAI, encode() refreshes and retries once."""
    from openai import AuthenticationError as OAIAuthError

    from hindsight_api.engine.embeddings import CodexOAuthEmbeddings

    fresh = _make_jwt(int(time.time()) + 3600)
    new_access = _make_jwt(int(time.time()) + 7200)

    _make_codex_auth_file(tmp_path, fresh, refresh_token="rt-embed")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    emb = CodexOAuthEmbeddings(model="text-embedding-3-small", batch_size=10)
    emb._dimension = 1536

    refresh_resp = _refresh_response(200, {"access_token": new_access, "refresh_token": "rt-rotated"})

    call_count = {"create": 0}

    def fake_create(**kwargs):
        call_count["create"] += 1
        if call_count["create"] == 1:
            # Simulate OpenAI returning 401.
            mock_response = MagicMock()
            mock_response.status_code = 401
            raise OAIAuthError(
                message="invalid api key",
                response=mock_response,
                body={"error": {"message": "invalid api key"}},
            )
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.2] * 1536)])

    with patch.object(emb._auth_manager._http_client, "post", return_value=refresh_resp):
        emb._client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))
        result = emb.encode(["world"])

    assert result == [[0.2] * 1536]
    assert call_count["create"] == 2  # first failed with 401, second succeeded
    assert emb._auth_manager.access_token == new_access
    assert emb.api_key == new_access
