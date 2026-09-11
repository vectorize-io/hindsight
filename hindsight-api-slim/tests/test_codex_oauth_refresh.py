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
JWT exp / persistence paths through targeted patches and the network through a
local stub of the OAuth refresh endpoint (``tests/aiohttp_stub.py``).
"""

from __future__ import annotations

import asyncio
import base64
import json
import stat
import sys
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from hindsight_api.engine.providers import codex_auth
from hindsight_api.engine.providers.codex_llm import (
    _CODEX_CLIENT_ID,
    CodexAuthManager,
    CodexLLM,
    CodexRefreshExpiredError,
)
from tests.aiohttp_stub import stub_server
from tests.codex_stream_stub import CodexReply, stub_codex_stream_with


@pytest.fixture(autouse=True)
def _isolate_codex_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Codex auth tests off the developer's (or CI's) real Codex home.

    ``default_codex_auth_file()`` reads ``$CODEX_HOME`` and falls back to
    ``Path.home() / ".codex"``, so any test that constructs a provider without
    an explicit auth path would otherwise pick up real credentials on a machine
    where Codex is actually logged in. Both inputs are redirected at a tmp dir.
    """
    home = tmp_path_factory.mktemp("codex-home")
    codex_home = home / ".codex"
    codex_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))


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
# Refresh endpoint stub
# ---------------------------------------------------------------------------

# A scripted reply: (status, JSON body dict or raw text), or a callable that
# receives the request's JSON body and returns one — for tests that rotate
# auth.json "while the request is in flight".
_Reply = tuple[int, dict | str] | Callable[[dict], tuple[int, dict | str]]


@dataclass
class _SeenRequest:
    path: str
    headers: dict[str, str]
    json: dict


class _RefreshEndpoint:
    """Stub of the Codex OAuth token endpoint: records requests, answers from a script.

    The last reply repeats once the script runs out.
    """

    def __init__(self, *replies: _Reply, delay: float = 0.0) -> None:
        self._replies = replies
        self._delay = delay
        self.requests: list[_SeenRequest] = []

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests.append(_SeenRequest(path=request.path, headers=dict(request.headers), json=body))
        reply = self._replies[min(len(self.requests) - 1, len(self._replies) - 1)]
        if self._delay:
            # Non-zero refresh latency so concurrent callers actually queue.
            await asyncio.sleep(self._delay)
        status, payload = reply(body) if callable(reply) else reply
        if isinstance(payload, dict):
            return web.json_response(payload, status=status)
        return web.Response(status=status, text=payload)

    @property
    def sent_refresh_tokens(self) -> list[str]:
        return [seen.json["refresh_token"] for seen in self.requests]


@asynccontextmanager
async def _serve_refresh(monkeypatch: pytest.MonkeyPatch, endpoint: _RefreshEndpoint) -> AsyncIterator[None]:
    """Point every CodexAuthManager at ``endpoint`` for the duration of the block."""
    async with stub_server(endpoint.handle) as base_url:
        monkeypatch.setattr(codex_auth, "_CODEX_REFRESH_TOKEN_URL", f"{base_url}/oauth/token")
        yield


def _auth_json(access_token: str, refresh_token: str) -> str:
    return json.dumps(
        {
            "auth_mode": "chatgpt",
            "tokens": {"access_token": access_token, "refresh_token": refresh_token, "account_id": "acct-test"},
        }
    )


# ---------------------------------------------------------------------------
# _refresh_oauth_tokens — request shape, in-memory update, rotation
# ---------------------------------------------------------------------------


async def test_refresh_sends_canonical_request_shape(tmp_path: Path, monkeypatch):
    """POST JSON body with client_id + grant_type=refresh_token + refresh_token."""
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(refresh_token="rt-current", access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": "x", "refresh_token": "rt-current"}}))

    fresh_access = _make_jwt(int(time.time()) + 3600)
    endpoint = _RefreshEndpoint((200, {"access_token": fresh_access, "refresh_token": "rt-rotated"}))

    async with _serve_refresh(monkeypatch, endpoint):
        await llm._refresh_oauth_tokens()

    [seen] = endpoint.requests
    assert seen.path == "/oauth/token"
    assert seen.headers["Content-Type"] == "application/json"
    assert seen.json == {
        "client_id": _CODEX_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": "rt-current",
    }


async def test_refresh_updates_in_memory_credentials(tmp_path: Path, monkeypatch):
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(refresh_token="rt-old", access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": "x", "refresh_token": "rt-old"}}))

    new_access = _make_jwt(int(time.time()) + 3600)
    endpoint = _RefreshEndpoint((200, {"access_token": new_access, "refresh_token": "rt-new"}))

    async with _serve_refresh(monkeypatch, endpoint):
        await llm._refresh_oauth_tokens()

    assert llm.access_token == new_access
    assert llm.refresh_token == "rt-new"


async def test_refresh_keeps_existing_refresh_token_when_server_omits_one(tmp_path: Path, monkeypatch):
    """If the OAuth response has no ``refresh_token`` field, keep the one we have."""
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(refresh_token="rt-keep", access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": "x", "refresh_token": "rt-keep"}}))

    new_access = _make_jwt(int(time.time()) + 3600)
    endpoint = _RefreshEndpoint((200, {"access_token": new_access}))

    async with _serve_refresh(monkeypatch, endpoint):
        await llm._refresh_oauth_tokens()

    assert llm.refresh_token == "rt-keep"


async def test_refresh_raises_permanent_error_on_terminal_oauth_code(tmp_path: Path, monkeypatch):
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(refresh_token="rt-stale", access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": "x", "refresh_token": "rt-stale"}}))

    endpoint = _RefreshEndpoint((401, {"error": {"code": "refresh_token_expired"}}))

    async with _serve_refresh(monkeypatch, endpoint):
        with pytest.raises(CodexRefreshExpiredError):
            await llm._refresh_oauth_tokens()


async def test_refresh_raises_permanent_error_on_unknown_401(tmp_path: Path, monkeypatch):
    """Any 401 from the refresh endpoint is treated as permanent — matches upstream Rust classification."""
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(refresh_token="rt-stale", access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": "x", "refresh_token": "rt-stale"}}))

    endpoint = _RefreshEndpoint((401, {"error": "something_else"}))

    async with _serve_refresh(monkeypatch, endpoint):
        with pytest.raises(CodexRefreshExpiredError):
            await llm._refresh_oauth_tokens()


async def test_refresh_raises_runtime_error_on_5xx(tmp_path: Path, monkeypatch):
    """5xx is transient from the caller's perspective — surface as RuntimeError, not CodexRefreshExpiredError."""
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(refresh_token="rt-current", access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": "x", "refresh_token": "rt-current"}}))

    endpoint = _RefreshEndpoint((503, "service unavailable"))

    async with _serve_refresh(monkeypatch, endpoint):
        with pytest.raises(RuntimeError) as exc_info:
            await llm._refresh_oauth_tokens()
    assert not isinstance(exc_info.value, CodexRefreshExpiredError)


async def test_refresh_raises_runtime_error_on_network_failure(tmp_path: Path, monkeypatch):
    """An unreachable refresh endpoint is a RuntimeError naming the transport failure."""
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(refresh_token="rt-current", access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": "x", "refresh_token": "rt-current"}}))

    async with stub_server(_RefreshEndpoint((200, {})).handle) as base_url:
        dead_url = f"{base_url}/oauth/token"
    # The server is closed: the connection is refused.
    monkeypatch.setattr(codex_auth, "_CODEX_REFRESH_TOKEN_URL", dead_url)

    with pytest.raises(RuntimeError, match="network error") as exc_info:
        await llm._refresh_oauth_tokens()
    assert not isinstance(exc_info.value, CodexRefreshExpiredError)


async def test_refresh_does_not_log_token_values(tmp_path: Path, caplog, monkeypatch):
    expired = _make_jwt(int(time.time()) - 60)
    secret_rt = "rt-DO-NOT-LEAK-THIS"
    llm = _build_llm(refresh_token=secret_rt, access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": "x", "refresh_token": secret_rt}}))

    new_access = _make_jwt(int(time.time()) + 3600)
    endpoint = _RefreshEndpoint((200, {"access_token": new_access, "refresh_token": "rt-also-secret"}))

    async with _serve_refresh(monkeypatch, endpoint):
        with caplog.at_level("DEBUG"):
            await llm._refresh_oauth_tokens()

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert secret_rt not in log_text
    assert new_access not in log_text
    assert "rt-also-secret" not in log_text


# ---------------------------------------------------------------------------
# Single-flight under concurrent callers
# ---------------------------------------------------------------------------


async def test_concurrent_ensure_fresh_token_calls_produce_one_refresh(tmp_path: Path, monkeypatch):
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(refresh_token="rt", access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": "x", "refresh_token": "rt"}}))

    new_access = _make_jwt(int(time.time()) + 3600)
    endpoint = _RefreshEndpoint((200, {"access_token": new_access, "refresh_token": "rt-new"}), delay=0.01)

    async with _serve_refresh(monkeypatch, endpoint):
        await asyncio.gather(*(llm._ensure_fresh_token() for _ in range(10)))

    assert len(endpoint.requests) == 1, f"expected 1 network refresh under contention, got {len(endpoint.requests)}"


async def test_sibling_auth_manager_adopts_rotated_codex_credentials(tmp_path: Path, monkeypatch):
    """A stale sibling manager should adopt auth.json rotation before reusing the old RT."""
    expired = _make_jwt(int(time.time()) - 60)
    new_access = _make_jwt(int(time.time()) + 3600)
    auth_file = _make_codex_auth_file(tmp_path, expired, refresh_token="rt-old")

    first = CodexAuthManager.from_file(auth_file)
    sibling = CodexAuthManager.from_file(auth_file)

    endpoint = _RefreshEndpoint((200, {"access_token": new_access, "refresh_token": "rt-new"}))
    async with _serve_refresh(monkeypatch, endpoint):
        await first.refresh_tokens(reason="test")
        await sibling.refresh_tokens(reason="test", force=True)

    # The stale sibling must not call the refresh endpoint with the old refresh_token.
    assert endpoint.sent_refresh_tokens == ["rt-old"]
    assert sibling.access_token == new_access
    assert sibling.refresh_token == "rt-new"


async def test_refresh_token_reused_adopts_fresh_disk_access_token_without_second_refresh(tmp_path: Path, monkeypatch):
    """If auth.json has a fresh access token after reuse, adopt it without refreshing again."""
    expired = _make_jwt(int(time.time()) - 60)
    disk_access = _make_jwt(int(time.time()) + 3600)
    auth_file = _make_codex_auth_file(tmp_path, expired, refresh_token="rt-old")

    manager = CodexAuthManager.from_file(auth_file)

    def rotate_then_reject(body: dict) -> tuple[int, dict]:
        auth_file.write_text(_auth_json(disk_access, "rt-disk-new"))
        return 401, {"error": {"code": "refresh_token_reused"}}

    endpoint = _RefreshEndpoint(rotate_then_reject)
    async with _serve_refresh(monkeypatch, endpoint):
        await manager.refresh_tokens(reason="test")

    assert endpoint.sent_refresh_tokens == ["rt-old"]
    assert manager.access_token == disk_access
    assert manager.refresh_token == "rt-disk-new"


async def test_refresh_token_reused_retries_with_newer_disk_refresh_token_when_disk_access_is_stale(
    tmp_path: Path, monkeypatch
):
    """If auth.json moves but disk access is stale, retry once with disk refresh token."""
    expired = _make_jwt(int(time.time()) - 60)
    replacement_access = _make_jwt(int(time.time()) - 30)
    final_access = _make_jwt(int(time.time()) + 3600)
    auth_file = _make_codex_auth_file(tmp_path, expired, refresh_token="rt-old")

    manager = CodexAuthManager.from_file(auth_file)

    def first_reply(body: dict) -> tuple[int, dict]:
        assert body["refresh_token"] == "rt-old"
        # Simulate another Codex process rotating auth.json while this
        # request is in flight, leaving an access token that is still stale
        # for this caller but a newer refresh token that can recover.
        auth_file.write_text(_auth_json(replacement_access, "rt-disk-new"))
        return 401, {"error": {"code": "refresh_token_reused"}}

    def second_reply(body: dict) -> tuple[int, dict]:
        assert body["refresh_token"] == "rt-disk-new"
        return 200, {"access_token": final_access, "refresh_token": "rt-final"}

    endpoint = _RefreshEndpoint(first_reply, second_reply)
    async with _serve_refresh(monkeypatch, endpoint):
        await manager.refresh_tokens(reason="test")

    assert endpoint.sent_refresh_tokens == ["rt-old", "rt-disk-new"]
    assert manager.access_token == final_access
    assert manager.refresh_token == "rt-final"


async def test_force_refresh_retries_with_newer_disk_refresh_token_after_reuse_401(tmp_path: Path, monkeypatch):
    """The reuse-401 retry is reachable from the reactive path too.

    ``force=True`` bypasses the JWT-clock gate, so a token the client still
    considers fresh (but the server rejected) reaches the refresh call and then
    the same rotate-under-us recovery as the proactive path.
    """
    old_fresh_access = _make_jwt(int(time.time()) + 3600)
    disk_stale_access = _make_jwt(int(time.time()) - 30)
    final_access = _make_jwt(int(time.time()) + 3600)
    auth_file = _make_codex_auth_file(tmp_path, old_fresh_access, refresh_token="rt-old")

    manager = CodexAuthManager.from_file(auth_file)

    def first_reply(body: dict) -> tuple[int, dict]:
        auth_file.write_text(_auth_json(disk_stale_access, "rt-disk-new"))
        return 401, {"error": {"code": "refresh_token_reused"}}

    def second_reply(body: dict) -> tuple[int, dict]:
        assert body["refresh_token"] == "rt-disk-new"
        return 200, {"access_token": final_access, "refresh_token": "rt-final"}

    endpoint = _RefreshEndpoint(first_reply, second_reply)
    async with _serve_refresh(monkeypatch, endpoint):
        await manager.refresh_tokens(reason="reactive", force=True)

    assert endpoint.sent_refresh_tokens == ["rt-old", "rt-disk-new"]
    assert manager.access_token == final_access
    assert manager.refresh_token == "rt-final"


async def test_force_refresh_does_not_skip_when_disk_already_rotated_to_a_stale_access_token(
    tmp_path: Path, monkeypatch
):
    """Adopting a *different* on-disk token is not enough to skip the refresh.

    Reactive callers get exactly one retry (``CodexOAuthEmbeddings.encode``
    retries ``super().encode`` once on ``AuthenticationError``). If auth.json
    was rotated by another Codex process long enough ago that its access token
    has itself expired, returning it spends that retry on a second 401 and
    fails the whole call. Only a token that is fresh by the JWT clock may
    short-circuit the network refresh.
    """
    memory_access = _make_jwt(int(time.time()) + 3600)  # clock-fresh, but server rejected it
    disk_stale_access = _make_jwt(int(time.time()) - 30)  # newer on disk, already expired
    final_access = _make_jwt(int(time.time()) + 3600)

    auth_file = _make_codex_auth_file(tmp_path, memory_access, refresh_token="rt-old")
    manager = CodexAuthManager.from_file(auth_file)

    # Another Codex process rotated auth.json before this caller takes the lock.
    auth_file.write_text(_auth_json(disk_stale_access, "rt-disk-new"))

    endpoint = _RefreshEndpoint((200, {"access_token": final_access, "refresh_token": "rt-final"}))
    async with _serve_refresh(monkeypatch, endpoint):
        await manager.refresh_tokens(reason="reactive (401 from embeddings API)", force=True)

    assert endpoint.sent_refresh_tokens == ["rt-disk-new"], "force refresh returned without refreshing"
    assert manager.access_token == final_access
    assert manager.refresh_token == "rt-final"


async def test_parallel_auth_managers_share_one_refresh_for_same_auth_file(tmp_path: Path, monkeypatch):
    """Separate managers on separate event loops single-flight per canonical auth path.

    Each manager refreshes from its own thread under its own ``asyncio.run``, so
    no asyncio lock is shared between them: only the auth store's file lock can
    serialise the two, and the second must then adopt the first's rotation.
    """
    expired = _make_jwt(int(time.time()) - 60)
    new_access = _make_jwt(int(time.time()) + 3600)
    auth_file = _make_codex_auth_file(tmp_path, expired, refresh_token="rt-old")

    managers = [CodexAuthManager.from_file(auth_file), CodexAuthManager.from_file(auth_file)]
    endpoint = _RefreshEndpoint((200, {"access_token": new_access, "refresh_token": "rt-new"}), delay=0.05)

    async def refresh_and_close(manager: CodexAuthManager) -> None:
        try:
            await manager.refresh_tokens("test")
        finally:
            await manager.close()

    async with _serve_refresh(monkeypatch, endpoint):
        await asyncio.gather(*(asyncio.to_thread(asyncio.run, refresh_and_close(m)) for m in managers))

    assert len(endpoint.requests) == 1
    assert [manager.access_token for manager in managers] == [new_access, new_access]
    assert [manager.refresh_token for manager in managers] == ["rt-new", "rt-new"]


async def test_one_manager_refreshes_from_two_event_loops(tmp_path: Path, monkeypatch):
    """A manager shared across loops keeps working: its session and lock are per loop."""
    auth_file = _make_codex_auth_file(tmp_path, _make_jwt(int(time.time()) - 60), refresh_token="rt-old")
    manager = CodexAuthManager.from_file(auth_file)
    endpoint = _RefreshEndpoint(
        lambda body: (200, {"access_token": _make_jwt(int(time.time()) + 3600), "refresh_token": body["refresh_token"]})
    )

    async def force_refresh() -> None:
        try:
            await manager.refresh_tokens("test", force=True)
        finally:
            await manager.close()

    async with _serve_refresh(monkeypatch, endpoint):
        await asyncio.to_thread(asyncio.run, force_refresh())
        await asyncio.to_thread(asyncio.run, force_refresh())

    assert len(endpoint.requests) == 2


# ---------------------------------------------------------------------------
# Reactive 401 retry on the request path
# ---------------------------------------------------------------------------


async def test_call_reactively_refreshes_on_401_and_retries(tmp_path: Path, monkeypatch):
    """A backend 401 triggers one refresh + retry instead of immediately raising."""
    fresh = _make_jwt(int(time.time()) + 3600)  # not stale; the 401 is the trigger
    llm = _build_llm(refresh_token="rt", access_token=fresh)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": fresh, "refresh_token": "rt"}}))

    new_access = _make_jwt(int(time.time()) + 3600)
    endpoint = _RefreshEndpoint((200, {"access_token": new_access, "refresh_token": "rt-new"}))

    posts = 0
    sent_headers: list[Any] = []

    def fake_backend_stream(url, **kwargs):
        nonlocal posts
        posts += 1
        sent_headers.append(kwargs["headers"])
        return CodexReply(401, "unauthorized") if posts == 1 else CodexReply()

    async with _serve_refresh(monkeypatch, endpoint):
        with (
            stub_codex_stream_with(llm, fake_backend_stream),
            patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock, return_value="ok"),
        ):
            result = (
                await llm.call(
                    messages=[{"role": "user", "content": "ping"}],
                    max_retries=0,
                    initial_backoff=0.0,
                    max_backoff=0.0,
                )
            ).content

    assert result == "ok"
    assert len(endpoint.requests) == 1
    assert posts == 2  # one 401, one success after refresh
    assert llm.access_token == new_access
    assert sent_headers[0]["Authorization"] == f"Bearer {fresh}"
    assert sent_headers[1]["Authorization"] == f"Bearer {new_access}"
    for header_name in ("Content-Type", "OpenAI-Account-ID", "User-Agent", "Origin", "originator"):
        assert sent_headers[1][header_name] == sent_headers[0][header_name]


async def test_call_with_tools_reactively_refreshes_on_401_and_retries(tmp_path: Path, monkeypatch):
    """The tool-call path gets the same single refresh + retry on a backend 401."""
    fresh = _make_jwt(int(time.time()) + 3600)
    llm = _build_llm(refresh_token="rt", access_token=fresh)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": fresh, "refresh_token": "rt"}}))

    new_access = _make_jwt(int(time.time()) + 3600)
    endpoint = _RefreshEndpoint((200, {"access_token": new_access, "refresh_token": "rt-new"}))
    replies = [CodexReply(401, "unauthorized"), CodexReply()]

    async with _serve_refresh(monkeypatch, endpoint):
        with (
            stub_codex_stream_with(llm, lambda url, **kwargs: replies.pop(0)) as stream,
            patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock, return_value=(None, [])),
        ):
            await llm.call_with_tools(messages=[{"role": "user", "content": "ping"}], tools=[], max_retries=0)

    assert stream.call_count == 2
    assert len(endpoint.requests) == 1
    assert stream.call_args.kwargs["headers"]["Authorization"] == f"Bearer {new_access}"


async def test_call_proactively_refreshes_when_token_is_stale(tmp_path: Path, monkeypatch):
    """A near-expiry token triggers refresh BEFORE the request is sent."""
    expired = _make_jwt(int(time.time()) - 60)
    llm = _build_llm(refresh_token="rt", access_token=expired)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": expired, "refresh_token": "rt"}}))

    new_access = _make_jwt(int(time.time()) + 3600)
    call_order: list[str] = []

    def refresh_reply(body: dict) -> tuple[int, dict]:
        call_order.append("refresh")
        return 200, {"access_token": new_access, "refresh_token": "rt-new"}

    def fake_backend_stream(url, **kwargs):
        call_order.append("backend")
        # Assert that by the time the backend is called, the new token is in use.
        assert kwargs["headers"]["Authorization"] == f"Bearer {new_access}"
        return CodexReply()

    async with _serve_refresh(monkeypatch, _RefreshEndpoint(refresh_reply)):
        with (
            stub_codex_stream_with(llm, fake_backend_stream),
            patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock, return_value="ok"),
        ):
            await llm.call(
                messages=[{"role": "user", "content": "ping"}],
                max_retries=0,
                initial_backoff=0.0,
                max_backoff=0.0,
            )

    assert call_order == ["refresh", "backend"], "expected proactive refresh BEFORE the backend call"


async def test_call_does_not_refresh_when_token_is_fresh(tmp_path: Path, monkeypatch):
    fresh = _make_jwt(int(time.time()) + 3600)
    llm = _build_llm(refresh_token="rt", access_token=fresh)
    llm._auth_file = tmp_path / "auth.json"
    llm._auth_file.write_text(json.dumps({"tokens": {"access_token": fresh, "refresh_token": "rt"}}))

    endpoint = _RefreshEndpoint((500, "must not be called"))
    async with _serve_refresh(monkeypatch, endpoint):
        with (
            stub_codex_stream_with(llm, lambda url, **kwargs: CodexReply()) as stream,
            patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock, return_value="ok"),
        ):
            await llm.call(
                messages=[{"role": "user", "content": "ping"}],
                max_retries=0,
                initial_backoff=0.0,
                max_backoff=0.0,
            )

    assert stream.call_count == 1
    assert endpoint.requests == []


# ---------------------------------------------------------------------------
# CodexOAuthEmbeddings — proactive + reactive token refresh
# ---------------------------------------------------------------------------


def _make_codex_auth_file(tmp_path: Path, access_token: str, refresh_token: str = "rt-initial") -> Path:
    """Write a minimal ~/.codex/auth.json in tmp_path and return its path."""
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    auth_file = codex_dir / "auth.json"
    auth_file.write_text(_auth_json(access_token, refresh_token))
    return auth_file


async def test_codex_oauth_embeddings_picks_up_refreshed_token_on_encode(tmp_path: Path, monkeypatch):
    """encode() awaits ensure_fresh_token() and updates api_key when the token rotated.

    The OpenAI embeddings call itself (the parent ``encode``) is stubbed: what is
    under test is the token handling around it.
    """
    from hindsight_api.engine.embeddings import CodexOAuthEmbeddings, OpenAIEmbeddings

    expired = _make_jwt(int(time.time()) - 60)
    new_access = _make_jwt(int(time.time()) + 3600)

    _make_codex_auth_file(tmp_path, expired, refresh_token="rt-embed")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))

    emb = CodexOAuthEmbeddings(model="text-embedding-3-small", batch_size=10)
    seen_keys: list[str] = []

    async def fake_parent_encode(self, texts):
        seen_keys.append(self.api_key)
        return [[0.1] * 1536]

    endpoint = _RefreshEndpoint((200, {"access_token": new_access, "refresh_token": "rt-new"}))
    async with _serve_refresh(monkeypatch, endpoint):
        with patch.object(OpenAIEmbeddings, "encode", new=fake_parent_encode):
            result = await emb.encode(["hello"])

    assert result == [[0.1] * 1536]
    # After proactive refresh the manager's token should be the new one...
    assert emb._auth_manager.access_token == new_access
    # ...and the embeddings call already went out carrying it.
    assert emb.api_key == new_access
    assert seen_keys == [new_access]


async def test_codex_oauth_embeddings_reactive_refresh_on_401(tmp_path: Path, monkeypatch):
    """On AuthenticationError from OpenAI, encode() refreshes and retries once."""
    from openai import AuthenticationError as OAIAuthError

    from hindsight_api.engine.embeddings import CodexOAuthEmbeddings, OpenAIEmbeddings

    fresh = _make_jwt(int(time.time()) + 3600)
    new_access = _make_jwt(int(time.time()) + 7200)

    _make_codex_auth_file(tmp_path, fresh, refresh_token="rt-embed")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))

    emb = CodexOAuthEmbeddings(model="text-embedding-3-small", batch_size=10)
    seen_keys: list[str] = []

    async def fake_parent_encode(self, texts):
        seen_keys.append(self.api_key)
        if len(seen_keys) == 1:
            # Simulate OpenAI returning 401.
            mock_response = MagicMock()
            mock_response.status_code = 401
            raise OAIAuthError(
                message="invalid api key",
                response=mock_response,
                body={"error": {"message": "invalid api key"}},
            )
        return [[0.2] * 1536]

    endpoint = _RefreshEndpoint((200, {"access_token": new_access, "refresh_token": "rt-rotated"}))
    async with _serve_refresh(monkeypatch, endpoint):
        with patch.object(OpenAIEmbeddings, "encode", new=fake_parent_encode):
            result = await emb.encode(["world"])

    assert result == [[0.2] * 1536]
    assert seen_keys == [fresh, new_access]  # first failed with 401, second succeeded
    assert len(endpoint.requests) == 1
    assert emb._auth_manager.access_token == new_access
    assert emb.api_key == new_access
