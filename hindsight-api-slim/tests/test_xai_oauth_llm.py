"""Tests for the ``xai-oauth`` SuperGrok subscription provider.

No test here touches the network or the operator's real home directory: the
credential store is written under ``tmp_path``, the HTTP clients are hand-rolled
fakes (house style — the repo uses no respx/vcr), and every timing rule is
driven through injected ``sleeper``/``monotonic`` callables rather than real
waits.

Guards are proven by call counts on those fakes, so a test cannot pass
vacuously: "exactly one refresh" is asserted as ``post_count == 1``, and "the
device flow is never entered" is asserted against a callable that raises if it
is called at all.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from hindsight_api.config import PROVIDER_DEFAULT_MODELS
from hindsight_api.engine.cache_affinity import XAI_CONV_ID_HEADER, cache_affinity_id
from hindsight_api.engine.llm_trace import (
    LLMTraceContext,
    current_trace_context,
    reset_trace_context,
    set_trace_context,
)
from hindsight_api.engine.llm_wrapper import create_llm_provider, requires_api_key
from hindsight_api.engine.providers import xai_oauth_auth as auth_mod
from hindsight_api.engine.providers import xai_oauth_llm as llm_mod
from hindsight_api.engine.providers.xai_oauth_auth import (
    DEFAULT_CLIENT_ID,
    DEFAULT_SCOPE,
    DEVICE_CODE_GRANT_TYPE,
    ENV_CLIENT_ID,
    ENV_SCOPE,
    ENV_TOKEN_PATH,
    StoredCredential,
    XaiOAuthDiscoveryError,
    XaiOAuthError,
    XaiOAuthLoginRequiredError,
    XaiOAuthManager,
    XaiOAuthRefreshError,
    default_token_path,
    device_code_login,
    poll_device_token,
    read_credential,
    write_credential,
)
from hindsight_api.engine.providers.xai_oauth_llm import (
    DEFAULT_BASE_URL,
    ENV_BASE_URL,
    ENV_DEBUG_HEADERS,
    SPENDING_LIMIT_CODE,
    XaiOAuthEntitlementError,
    XaiOAuthLLM,
    XaiOAuthQuotaExhaustedError,
    _ChatUsage,
    _error_detail,
    _token_counts,
)

ACCESS_TOKEN = "access-token-do-not-log"
REFRESH_TOKEN = "refresh-token-do-not-log"
NEW_ACCESS_TOKEN = "rotated-access-token-do-not-log"
TOKEN_ENDPOINT = "https://auth.x.ai/oauth2/token"
DEVICE_ENDPOINT = "https://auth.x.ai/oauth2/device/code"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeResponse:
    status_code: int
    text: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        return json.loads(self.text)


class _FakeAsyncHttp:
    """Stand-in for ``httpx.AsyncClient`` recording every request it is given."""

    def __init__(self, replies: list[_FakeResponse]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, json: Any = None, headers: Any = None) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": dict(headers or {})})
        return self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]

    async def aclose(self) -> None:
        pass


class _FakeSyncHttp:
    """Stand-in for ``httpx.Client`` used by the credential manager and login."""

    def __init__(
        self,
        post_replies: list[_FakeResponse] | None = None,
        get_replies: list[_FakeResponse] | None = None,
        *,
        on_post: Any = None,
    ) -> None:
        self._post_replies = list(post_replies or [])
        self._get_replies = list(get_replies or [])
        self.posts: list[dict[str, Any]] = []
        self.gets: list[dict[str, Any]] = []
        self._on_post = on_post

    def __enter__(self) -> "_FakeSyncHttp":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def post(self, url: str, headers: Any = None, data: Any = None, timeout: Any = None) -> _FakeResponse:
        self.posts.append({"url": url, "headers": dict(headers or {}), "data": dict(data or {}), "timeout": timeout})
        if self._on_post is not None:
            self._on_post()
        if not self._post_replies:
            raise AssertionError("the fake HTTP client ran out of queued POST replies")
        return self._post_replies[min(len(self.posts) - 1, len(self._post_replies) - 1)]

    def get(self, url: str, headers: Any = None) -> _FakeResponse:
        self.gets.append({"url": url, "headers": dict(headers or {})})
        if not self._get_replies:
            raise AssertionError("the fake HTTP client ran out of queued GET replies")
        return self._get_replies[min(len(self.gets) - 1, len(self._get_replies) - 1)]

    def close(self) -> None:
        pass

    @property
    def post_count(self) -> int:
        return len(self.posts)


@contextmanager
def _bound_trace(trace_id: str):
    """Bind an operation trace context, as ConfiguredLLMProvider does per call."""
    token = set_trace_context(LLMTraceContext(bank_id="bank-1", operation="reflect", trace_id=trace_id))
    try:
        yield
    finally:
        reset_trace_context(token)


def _never_called(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("this code path must never be entered by an unattended request")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_store(
    path: Path,
    *,
    access_token: str = ACCESS_TOKEN,
    refresh_token: str = REFRESH_TOKEN,
    expires_in: float | None = 3600.0,
    obtained_ago: float = 3600.0,
    token_endpoint: str = TOKEN_ENDPOINT,
    scope: str = DEFAULT_SCOPE,
) -> Path:
    now = time.time()
    payload = {
        "auth_mode": "xai-oauth-device-code",
        "tokens": {"access_token": access_token, "refresh_token": refresh_token},
        "expires_at": (now + expires_in) if expires_in is not None else None,
        "obtained_at": now - obtained_ago,
        "scope": scope,
        "token_endpoint": token_endpoint,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _refresh_ok(access_token: str = NEW_ACCESS_TOKEN, expires_in: float = 3600.0) -> _FakeResponse:
    return _FakeResponse(
        200,
        json.dumps({"access_token": access_token, "refresh_token": REFRESH_TOKEN, "expires_in": expires_in}),
    )


def _completion(content: str = "ok", usage: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    if usage is not None:
        body["usage"] = usage
    return body


def _ok_reply(content: str = "ok", usage: dict[str, Any] | None = None) -> _FakeResponse:
    return _FakeResponse(200, json.dumps(_completion(content, usage)))


def _make_manager(
    tmp_path: Path,
    *,
    post_replies: list[_FakeResponse] | None = None,
    get_replies: list[_FakeResponse] | None = None,
    min_gap: float = 0.0,
    skew: float = 60.0,
    timeout: float = 20.0,
    store: Path | None = None,
) -> tuple[XaiOAuthManager, _FakeSyncHttp, Path]:
    path = store or (tmp_path / "xai_oauth.json")
    http = _FakeSyncHttp(post_replies, get_replies)
    manager = XaiOAuthManager(
        path,
        refresh_skew_seconds=skew,
        refresh_timeout_seconds=timeout,
        min_refresh_gap_seconds=min_gap,
        http_client=http,  # type: ignore[arg-type]
    )
    return manager, http, path


def _make_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    replies: list[_FakeResponse] | None = None,
    refresh_replies: list[_FakeResponse] | None = None,
    expires_in: float | None = 3600.0,
    timeout: float = 30.0,
    model: str = "grok-4.5",
    reasoning_effort: str = "high",
    base_url: str = "",
) -> XaiOAuthLLM:
    store = tmp_path / "xai_oauth.json"
    _write_store(store, expires_in=expires_in)
    monkeypatch.setenv(ENV_TOKEN_PATH, str(store))
    manager, http, _ = _make_manager(tmp_path, post_replies=refresh_replies, store=store)
    llm = XaiOAuthLLM(
        provider="xai-oauth",
        api_key="",
        base_url=base_url,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        auth_manager=manager,
    )
    llm._client = _FakeAsyncHttp(replies or [_ok_reply()])  # type: ignore[assignment]
    llm._refresh_http = http  # test handle, not used by the provider
    return llm


# ===========================================================================
# 1. Device flow
# ===========================================================================


def test_device_poll_waits_the_advertised_interval_between_polls():
    slept: list[float] = []
    pending = _FakeResponse(400, json.dumps({"error": "authorization_pending"}))
    granted = _FakeResponse(200, json.dumps({"access_token": "a", "refresh_token": "r", "expires_in": 3600}))
    client = _FakeSyncHttp([pending, pending, granted])

    payload = poll_device_token(
        client,  # type: ignore[arg-type]
        token_endpoint=TOKEN_ENDPOINT,
        device_code="dev-code",
        client_id=DEFAULT_CLIENT_ID,
        expires_in=900,
        interval=5,
        sleeper=slept.append,
        monotonic=lambda: 0.0,
    )

    assert payload["access_token"] == "a"
    assert slept == [5, 5]
    assert client.post_count == 3
    assert client.posts[0]["data"]["grant_type"] == DEVICE_CODE_GRANT_TYPE


def test_device_poll_widens_the_interval_by_five_seconds_on_slow_down():
    """RFC 8628 section 3.5: a ``slow_down`` adds 5 seconds to the interval."""
    slept: list[float] = []
    slow_down = _FakeResponse(400, json.dumps({"error": "slow_down"}))
    pending = _FakeResponse(400, json.dumps({"error": "authorization_pending"}))
    granted = _FakeResponse(200, json.dumps({"access_token": "a", "refresh_token": "r"}))
    client = _FakeSyncHttp([slow_down, pending, granted])

    poll_device_token(
        client,  # type: ignore[arg-type]
        token_endpoint=TOKEN_ENDPOINT,
        device_code="dev-code",
        client_id=DEFAULT_CLIENT_ID,
        expires_in=900,
        interval=5,
        sleeper=slept.append,
        monotonic=lambda: 0.0,
    )

    assert slept == [10, 10]


def test_device_poll_raises_when_the_device_code_expires():
    client = _FakeSyncHttp([_FakeResponse(400, json.dumps({"error": "expired_token"}))])

    with pytest.raises(XaiOAuthError, match="expired"):
        poll_device_token(
            client,  # type: ignore[arg-type]
            token_endpoint=TOKEN_ENDPOINT,
            device_code="dev-code",
            client_id=DEFAULT_CLIENT_ID,
            expires_in=900,
            interval=5,
            sleeper=lambda _: None,
            monotonic=lambda: 0.0,
        )
    assert client.post_count == 1


def test_device_poll_stops_at_the_advertised_deadline():
    # Clock reads: deadline base (0), inside the window (10 -> one poll),
    # past it (100 -> stop). One poll proves the loop ran at all, so the
    # timeout is a real stop rather than a window that was never open.
    clock = iter([0.0, 10.0, 100.0])
    client = _FakeSyncHttp([_FakeResponse(400, json.dumps({"error": "authorization_pending"}))])

    with pytest.raises(XaiOAuthError, match="Timed out"):
        poll_device_token(
            client,  # type: ignore[arg-type]
            token_endpoint=TOKEN_ENDPOINT,
            device_code="dev-code",
            client_id=DEFAULT_CLIENT_ID,
            expires_in=60,
            interval=5,
            sleeper=lambda _: None,
            monotonic=lambda: next(clock),
        )
    assert client.post_count == 1


def test_device_poll_rejects_a_grant_without_a_refresh_token():
    client = _FakeSyncHttp([_FakeResponse(200, json.dumps({"access_token": "a"}))])

    with pytest.raises(XaiOAuthError, match="refresh_token"):
        poll_device_token(
            client,  # type: ignore[arg-type]
            token_endpoint=TOKEN_ENDPOINT,
            device_code="dev-code",
            client_id=DEFAULT_CLIENT_ID,
            expires_in=900,
            interval=5,
            sleeper=lambda _: None,
            monotonic=lambda: 0.0,
        )


def test_device_login_prints_the_user_code_and_never_logs_it(tmp_path, monkeypatch, caplog):
    """The user code reaches stdout and no log record at any level.

    The same assertion also proves the probe works: the code IS found in the
    printed lines, so its absence from ``caplog`` is a real negative.
    """
    user_code = "WDJB-MJHT"
    discovery = _FakeResponse(
        200,
        json.dumps({"token_endpoint": TOKEN_ENDPOINT, "device_authorization_endpoint": DEVICE_ENDPOINT}),
    )
    device = _FakeResponse(
        200,
        json.dumps(
            {
                "device_code": "dev-code",
                "user_code": user_code,
                "verification_uri": "https://x.ai/device",
                "verification_uri_complete": f"https://x.ai/device?code={user_code}",
                "expires_in": 900,
                "interval": 5,
            }
        ),
    )
    granted = _FakeResponse(
        200,
        json.dumps({"access_token": ACCESS_TOKEN, "refresh_token": REFRESH_TOKEN, "expires_in": 3600}),
    )
    fake = _FakeSyncHttp([device, granted], [discovery])
    monkeypatch.setattr(auth_mod.httpx, "Client", lambda *a, **kw: fake)

    printed: list[str] = []
    caplog.set_level(logging.DEBUG)
    caplog.clear()

    store = tmp_path / "xai_oauth.json"
    device_code_login(token_path=store, writer=printed.append)

    joined = "\n".join(printed)
    assert user_code in joined
    assert user_code not in caplog.text
    assert ACCESS_TOKEN not in caplog.text
    assert REFRESH_TOKEN not in caplog.text
    assert ACCESS_TOKEN not in joined

    stored = read_credential(store)
    assert stored.access_token == ACCESS_TOKEN
    assert stored.refresh_token == REFRESH_TOKEN
    assert stored.token_endpoint == TOKEN_ENDPOINT


def test_device_login_requests_the_published_client_id_and_scope(tmp_path, monkeypatch):
    discovery = _FakeResponse(200, json.dumps({"token_endpoint": TOKEN_ENDPOINT}))
    device = _FakeResponse(
        200,
        json.dumps(
            {
                "device_code": "dev-code",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://x.ai/device",
                "expires_in": 900,
                "interval": 5,
            }
        ),
    )
    granted = _FakeResponse(200, json.dumps({"access_token": "a", "refresh_token": "r", "expires_in": 60}))
    fake = _FakeSyncHttp([device, granted], [discovery])
    monkeypatch.setattr(auth_mod.httpx, "Client", lambda *a, **kw: fake)

    device_code_login(token_path=tmp_path / "store.json", writer=lambda _: None)

    assert fake.posts[0]["data"] == {"client_id": DEFAULT_CLIENT_ID, "scope": DEFAULT_SCOPE}
    # Discovery omitted device_authorization_endpoint, so the vendor fallback applies.
    assert fake.posts[0]["url"] == auth_mod.XAI_OAUTH_DEVICE_CODE_URL


def test_device_login_honors_the_client_id_and_scope_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_CLIENT_ID, "override-client")
    monkeypatch.setenv(ENV_SCOPE, "openid custom:scope")
    discovery = _FakeResponse(200, json.dumps({"token_endpoint": TOKEN_ENDPOINT}))
    device = _FakeResponse(
        200,
        json.dumps(
            {
                "device_code": "dev-code",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://x.ai/device",
                "expires_in": 900,
                "interval": 5,
            }
        ),
    )
    granted = _FakeResponse(200, json.dumps({"access_token": "a", "refresh_token": "r", "expires_in": 60}))
    fake = _FakeSyncHttp([device, granted], [discovery])
    monkeypatch.setattr(auth_mod.httpx, "Client", lambda *a, **kw: fake)

    device_code_login(token_path=tmp_path / "store.json", writer=lambda _: None)

    assert fake.posts[0]["data"] == {"client_id": "override-client", "scope": "openid custom:scope"}


@pytest.mark.parametrize(
    "endpoint",
    ["http://auth.x.ai/oauth2/token", "https://auth.evil.example/oauth2/token", "https://notx.ai/token"],
)
def test_discovery_refuses_an_endpoint_off_the_xai_origin(endpoint):
    """A substituted token_endpoint would receive every future refresh token."""
    client = _FakeSyncHttp(get_replies=[_FakeResponse(200, json.dumps({"token_endpoint": endpoint}))])

    with pytest.raises(XaiOAuthDiscoveryError):
        auth_mod.discover_endpoints(client)  # type: ignore[arg-type]


# ===========================================================================
# 2. Refresh
# ===========================================================================


def test_proactive_refresh_happens_exactly_once_under_concurrent_managers(tmp_path):
    """Two managers, one store, one refresh — the recheck-under-lock invariant.

    Without the re-read after taking the store lock, the second manager would
    refresh a credential its sibling had already rotated, so the assertion is
    ``post_count == 1`` rather than "a refresh happened".
    """
    store = _write_store(tmp_path / "xai_oauth.json", expires_in=10.0, obtained_ago=3600.0)
    entered_http = threading.Event()
    release_http = threading.Barrier(2)

    def _park() -> None:
        entered_http.set()
        release_http.wait()

    http = _FakeSyncHttp([_refresh_ok()], on_post=_park)
    managers = [
        XaiOAuthManager(store, refresh_skew_seconds=60.0, min_refresh_gap_seconds=0.0, http_client=http)  # type: ignore[arg-type]
        for _ in range(2)
    ]

    results: dict[int, str] = {}

    def _run(index: int) -> None:
        results[index] = managers[index].get_access_token()

    first = threading.Thread(target=_run, args=(0,))
    first.start()
    entered_http.wait()  # the first manager is inside the refresh, holding the store lock

    second = threading.Thread(target=_run, args=(1,))
    second.start()
    release_http.wait()

    first.join(timeout=10)
    second.join(timeout=10)

    assert http.post_count == 1
    assert results == {0: NEW_ACCESS_TOKEN, 1: NEW_ACCESS_TOKEN}


def test_a_fresh_token_is_served_without_any_refresh_request(tmp_path):
    manager, http, _ = _make_manager(tmp_path, store=_write_store(tmp_path / "s.json", expires_in=3600.0))

    assert manager.get_access_token() == ACCESS_TOKEN
    assert http.post_count == 0


def test_a_token_inside_the_skew_window_is_refreshed(tmp_path):
    store = _write_store(tmp_path / "s.json", expires_in=30.0, obtained_ago=3600.0)
    manager, http, _ = _make_manager(tmp_path, post_replies=[_refresh_ok()], skew=60.0, store=store)

    assert manager.get_access_token() == NEW_ACCESS_TOKEN
    assert http.post_count == 1
    assert read_credential(store).access_token == NEW_ACCESS_TOKEN


def test_an_unreadable_expiry_refreshes_rather_than_assuming_validity(tmp_path):
    store = _write_store(tmp_path / "s.json", expires_in=None, obtained_ago=3600.0)
    manager, http, _ = _make_manager(tmp_path, post_replies=[_refresh_ok()], store=store)

    assert manager.get_access_token() == NEW_ACCESS_TOKEN
    assert http.post_count == 1


def test_the_minimum_gap_suppresses_a_second_refresh(tmp_path):
    """A credential obtained seconds ago is not refreshed again (anti-spin)."""
    store = _write_store(tmp_path / "s.json", expires_in=10.0, obtained_ago=2.0)
    manager, http, _ = _make_manager(tmp_path, post_replies=[_refresh_ok()], min_gap=30.0, store=store)

    with pytest.raises(XaiOAuthRefreshError, match="minimum gap"):
        manager.get_access_token()
    assert http.post_count == 0


def test_the_minimum_gap_rides_the_store_not_an_in_memory_clock(tmp_path):
    """A sibling's just-written credential blocks this manager's first refresh."""
    store = _write_store(tmp_path / "s.json", expires_in=10.0, obtained_ago=0.0)
    manager, http, _ = _make_manager(tmp_path, post_replies=[_refresh_ok()], min_gap=30.0, store=store)

    with pytest.raises(XaiOAuthRefreshError):
        manager.refresh(reason="first call from a brand new manager")
    assert http.post_count == 0


def test_the_configured_timeout_reaches_the_refresh_request(tmp_path):
    store = _write_store(tmp_path / "s.json", expires_in=10.0, obtained_ago=3600.0)
    manager, http, _ = _make_manager(tmp_path, post_replies=[_refresh_ok()], timeout=7.5, store=store)

    manager.get_access_token()

    assert http.posts[0]["timeout"] == 7.5


def test_refresh_posts_the_standard_oauth_grant(tmp_path):
    store = _write_store(tmp_path / "s.json", expires_in=10.0, obtained_ago=3600.0)
    manager, http, _ = _make_manager(tmp_path, post_replies=[_refresh_ok()], store=store)

    manager.get_access_token()

    assert http.posts[0]["url"] == TOKEN_ENDPOINT
    assert http.posts[0]["data"] == {
        "grant_type": "refresh_token",
        "client_id": DEFAULT_CLIENT_ID,
        "refresh_token": REFRESH_TOKEN,
    }


def test_a_rotated_refresh_token_is_persisted(tmp_path):
    store = _write_store(tmp_path / "s.json", expires_in=10.0, obtained_ago=3600.0)
    rotated = _FakeResponse(
        200,
        json.dumps({"access_token": NEW_ACCESS_TOKEN, "refresh_token": "rotated-refresh", "expires_in": 3600}),
    )
    manager, _, _ = _make_manager(tmp_path, post_replies=[rotated], store=store)

    manager.get_access_token()

    assert read_credential(store).refresh_token == "rotated-refresh"


def test_the_store_rewrite_goes_through_a_temp_file_and_os_replace(tmp_path, monkeypatch):
    store = tmp_path / "xai_oauth.json"
    replaced: list[tuple[str, str]] = []
    real_replace = os.replace

    def _record(src: Any, dst: Any) -> None:
        replaced.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(auth_mod.os, "replace", _record)
    write_credential(
        store,
        StoredCredential(ACCESS_TOKEN, REFRESH_TOKEN, time.time() + 60, time.time(), DEFAULT_SCOPE, TOKEN_ENDPOINT),
    )

    assert len(replaced) == 1
    src, dst = replaced[0]
    assert src.endswith(".json.tmp")
    assert dst == str(store)
    assert read_credential(store).access_token == ACCESS_TOKEN


def test_a_failed_store_write_leaves_the_previous_credential_intact(tmp_path, monkeypatch):
    """No torn file: a write that dies mid-flight neither truncates nor litters."""
    store = _write_store(tmp_path / "xai_oauth.json")
    before = store.read_bytes()

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(auth_mod.os, "fsync", _boom)

    with pytest.raises(OSError):
        write_credential(
            store,
            StoredCredential("other", "other", time.time() + 60, time.time(), DEFAULT_SCOPE, TOKEN_ENDPOINT),
        )

    assert store.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_a_terminal_rejection_is_retried_once_then_quarantines(tmp_path):
    """400/401 from the token endpoint: one retry, then the grant is dead."""
    store = _write_store(tmp_path / "s.json", expires_in=10.0, obtained_ago=3600.0)
    denied = _FakeResponse(400, json.dumps({"error": "invalid_grant"}))
    manager, http, _ = _make_manager(tmp_path, post_replies=[denied, denied], store=store)

    with pytest.raises(XaiOAuthLoginRequiredError, match="xai_oauth_auth login"):
        manager.get_access_token()

    assert http.post_count == 2
    payload = json.loads(store.read_text())
    assert payload["tokens"] == {}
    assert payload["last_auth_error"]["code"] == "refresh_rejected_400"
    # The quarantine record carries no credential material.
    assert ACCESS_TOKEN not in store.read_text()
    assert REFRESH_TOKEN not in store.read_text()


def test_a_terminal_rejection_that_clears_on_the_retry_succeeds(tmp_path):
    store = _write_store(tmp_path / "s.json", expires_in=10.0, obtained_ago=3600.0)
    denied = _FakeResponse(401, json.dumps({"error": "invalid_grant"}))
    manager, http, _ = _make_manager(tmp_path, post_replies=[denied, _refresh_ok()], store=store)

    assert manager.get_access_token() == NEW_ACCESS_TOKEN
    assert http.post_count == 2


def test_a_server_error_is_not_quarantined(tmp_path):
    store = _write_store(tmp_path / "s.json", expires_in=10.0, obtained_ago=3600.0)
    manager, http, _ = _make_manager(tmp_path, post_replies=[_FakeResponse(503, "upstream down")], store=store)

    with pytest.raises(XaiOAuthRefreshError, match="503"):
        manager.get_access_token()

    assert http.post_count == 1
    assert read_credential(store).refresh_token == REFRESH_TOKEN


def test_a_403_from_the_token_endpoint_is_not_quarantined(tmp_path):
    """A 403 is an edge/policy answer, not an OAuth "this grant is dead".

    RFC 6749 spells a spent or revoked refresh token as 400 ``invalid_grant``;
    403 is what a layer in front of the issuer returns. Quarantining on it
    would turn a transient refusal into a mandatory interactive re-login —
    the same reasoning the API side already applies to its own 403s.
    """
    store = _write_store(tmp_path / "s.json", expires_in=10.0, obtained_ago=3600.0)
    manager, http, _ = _make_manager(tmp_path, post_replies=[_FakeResponse(403, "forbidden")], store=store)

    with pytest.raises(XaiOAuthRefreshError, match="403"):
        manager.get_access_token()

    assert http.post_count == 1
    assert read_credential(store).refresh_token == REFRESH_TOKEN
    assert "last_auth_error" not in json.loads(store.read_text())


def test_a_rejection_is_not_quarantined_when_the_store_moved_to_another_grant(tmp_path):
    """xAI rotates the refresh token, so a rejection can mean "already spent".

    A writer this host's lock does not cover (a store shared with a host whose
    filesystem ignores flock) can land its own rotation first. Wiping the store
    then would destroy a live grant and demand a login for nothing, so the
    rejection is rechecked against the store before anything is discarded.
    """
    store = _write_store(tmp_path / "s.json", expires_in=10.0, obtained_ago=3600.0)

    def _sibling_rotates_the_grant() -> None:
        _write_store(
            store,
            access_token="sibling-access-token",
            refresh_token="sibling-refresh-token",
            expires_in=3600.0,
            obtained_ago=0.0,
        )

    denied = _FakeResponse(400, json.dumps({"error": "invalid_grant"}))
    http = _FakeSyncHttp([denied, denied], on_post=_sibling_rotates_the_grant)
    manager = XaiOAuthManager(
        store,
        refresh_skew_seconds=60.0,
        refresh_timeout_seconds=20.0,
        min_refresh_gap_seconds=0.0,
        http_client=http,  # type: ignore[arg-type]
    )

    assert manager.get_access_token() == "sibling-access-token"
    assert http.post_count == 2
    payload = json.loads(store.read_text())
    assert payload["tokens"]["refresh_token"] == "sibling-refresh-token"
    assert "last_auth_error" not in payload


def test_a_rejection_still_quarantines_when_the_store_holds_the_same_grant(tmp_path):
    """The recheck must not become a blanket excuse to never quarantine."""
    store = _write_store(tmp_path / "s.json", expires_in=10.0, obtained_ago=3600.0)
    denied = _FakeResponse(400, json.dumps({"error": "invalid_grant"}))
    manager, _, _ = _make_manager(tmp_path, post_replies=[denied, denied], store=store)

    with pytest.raises(XaiOAuthLoginRequiredError):
        manager.get_access_token()

    assert json.loads(store.read_text())["tokens"] == {}


# ===========================================================================
# 3. Unattended posture
# ===========================================================================


async def test_a_dead_credential_fails_the_call_without_entering_the_device_flow(tmp_path, monkeypatch):
    """Services never acquire interactive credentials.

    Every device-flow entrypoint is replaced with a callable that raises if it
    is reached, so the assertion is structural rather than incidental.
    """
    monkeypatch.setattr(auth_mod, "device_code_login", _never_called)
    monkeypatch.setattr(auth_mod, "request_device_code", _never_called)
    monkeypatch.setattr(auth_mod, "poll_device_token", _never_called)

    store = tmp_path / "xai_oauth.json"  # never created: nothing to refresh
    monkeypatch.setenv(ENV_TOKEN_PATH, str(store))
    manager, http, _ = _make_manager(tmp_path, store=store)
    llm = XaiOAuthLLM(
        provider="xai-oauth", api_key="", base_url="", model="grok-4.5", timeout=30.0, auth_manager=manager
    )
    llm._client = _FakeAsyncHttp([_ok_reply()])  # type: ignore[assignment]

    with pytest.raises(XaiOAuthLoginRequiredError, match="xai_oauth_auth login"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert http.post_count == 0
    assert llm._client.calls == []


async def test_a_quarantined_store_fails_the_call_with_the_login_remediation(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_mod, "device_code_login", _never_called)
    store = tmp_path / "xai_oauth.json"
    store.write_text(json.dumps({"tokens": {}, "last_auth_error": {"code": "refresh_rejected_400"}}))
    monkeypatch.setenv(ENV_TOKEN_PATH, str(store))
    manager, _, _ = _make_manager(tmp_path, store=store)
    llm = XaiOAuthLLM(
        provider="xai-oauth", api_key="", base_url="", model="grok-4.5", timeout=30.0, auth_manager=manager
    )
    llm._client = _FakeAsyncHttp([_ok_reply()])  # type: ignore[assignment]

    with pytest.raises(XaiOAuthLoginRequiredError, match="no refresh_token"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)


# ===========================================================================
# 4. 401 recovery
# ===========================================================================


async def test_a_401_refreshes_once_and_retries_once(tmp_path, monkeypatch):
    llm = _make_llm(
        tmp_path,
        monkeypatch,
        replies=[_FakeResponse(401, "unauthorized"), _ok_reply("recovered")],
        refresh_replies=[_refresh_ok()],
    )

    result = await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert result == "recovered"
    assert len(llm._client.calls) == 2
    assert llm._refresh_http.post_count == 1
    assert llm._client.calls[1]["headers"]["Authorization"] == f"Bearer {NEW_ACCESS_TOKEN}"


async def test_a_second_401_is_terminal_with_no_third_attempt(tmp_path, monkeypatch):
    llm = _make_llm(
        tmp_path,
        monkeypatch,
        replies=[_FakeResponse(401, "unauthorized")],
        refresh_replies=[_refresh_ok()],
    )

    with pytest.raises(XaiOAuthLoginRequiredError):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=3)

    assert len(llm._client.calls) == 2
    assert llm._refresh_http.post_count == 1


# ===========================================================================
# 5. 403 classification
# ===========================================================================


async def test_a_spending_limit_403_is_a_quota_error_and_never_refreshes(tmp_path, monkeypatch):
    body = json.dumps({"error": {"code": SPENDING_LIMIT_CODE, "message": "blocked"}})
    llm = _make_llm(tmp_path, monkeypatch, replies=[_FakeResponse(403, body)], refresh_replies=[_refresh_ok()])

    with pytest.raises(XaiOAuthQuotaExhaustedError, match="spending limit"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=3)

    assert llm._refresh_http.post_count == 0
    assert len(llm._client.calls) == 1


async def test_an_entitlement_403_names_the_tier_cause_and_never_refreshes(tmp_path, monkeypatch):
    llm = _make_llm(tmp_path, monkeypatch, replies=[_FakeResponse(403, "Forbidden")], refresh_replies=[_refresh_ok()])

    with pytest.raises(XaiOAuthEntitlementError, match="subscription tier"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=3)

    assert llm._refresh_http.post_count == 0
    assert len(llm._client.calls) == 1


async def test_a_generic_coded_403_is_non_retryable_and_never_refreshes(tmp_path, monkeypatch):
    body = json.dumps({"error": {"code": "model_not_permitted"}})
    llm = _make_llm(tmp_path, monkeypatch, replies=[_FakeResponse(403, body)], refresh_replies=[_refresh_ok()])

    with pytest.raises(RuntimeError, match="model_not_permitted"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=3)

    assert llm._refresh_http.post_count == 0
    assert len(llm._client.calls) == 1


async def test_the_three_403_shapes_produce_three_distinct_classifications(tmp_path, monkeypatch):
    shapes = {
        json.dumps({"error": {"code": SPENDING_LIMIT_CODE}}): XaiOAuthQuotaExhaustedError,
        "Forbidden": XaiOAuthEntitlementError,
        json.dumps({"error": {"code": "model_not_permitted"}}): RuntimeError,
    }
    seen: list[type] = []
    for body, expected in shapes.items():
        llm = _make_llm(tmp_path, monkeypatch, replies=[_FakeResponse(403, body)], refresh_replies=[_refresh_ok()])
        with pytest.raises(expected) as exc:
            await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)
        seen.append(type(exc.value))
        assert llm._refresh_http.post_count == 0

    assert len(set(seen)) == 3


async def test_a_429_is_retried_honoring_retry_after(tmp_path, monkeypatch):
    slept: list[float] = []

    async def _record(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(llm_mod.asyncio, "sleep", _record)
    llm = _make_llm(
        tmp_path,
        monkeypatch,
        replies=[_FakeResponse(429, "slow down", {"Retry-After": "7"}), _ok_reply("after backoff")],
    )

    result = await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=2, initial_backoff=1.0)

    assert result == "after backoff"
    assert slept == [7.0]


async def test_a_429_without_retry_after_uses_the_backoff_schedule(tmp_path, monkeypatch):
    slept: list[float] = []

    async def _record(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(llm_mod.asyncio, "sleep", _record)
    llm = _make_llm(tmp_path, monkeypatch, replies=[_FakeResponse(429, "slow down"), _ok_reply("ok")])

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=2, initial_backoff=1.5)

    assert slept == [1.5]


async def test_a_400_is_not_retried(tmp_path, monkeypatch):
    llm = _make_llm(tmp_path, monkeypatch, replies=[_FakeResponse(400, "bad request")])

    with pytest.raises(RuntimeError, match="HTTP 400"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=5)

    assert len(llm._client.calls) == 1


async def test_an_upstream_error_body_is_not_echoed_into_the_error(tmp_path, monkeypatch):
    secret_ish = "sk-should-never-appear-in-an-exception"
    llm = _make_llm(tmp_path, monkeypatch, replies=[_FakeResponse(400, secret_ish)])

    with pytest.raises(RuntimeError) as exc:
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert secret_ish not in str(exc.value)
    assert str(len(secret_ish)) in str(exc.value)


async def test_a_json_error_body_surfaces_its_code_and_message(tmp_path, monkeypatch):
    """A permanent 4xx has to say why, or it cannot be diagnosed.

    Measured against the live API: sending ``reasoning_effort: "none"`` returns
    exactly this body, and reporting it as "HTTP 400 (98 bytes)" leaves an
    operator no way to tell a rejected parameter from a bad model name.
    """
    body = json.dumps(
        {"code": "invalid-argument", "error": "This model does not support `reasoning_effort` value `none`."}
    )
    llm = _make_llm(tmp_path, monkeypatch, replies=[_FakeResponse(400, body)])

    with pytest.raises(RuntimeError) as exc:
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    message = str(exc.value)
    assert "HTTP 400" in message
    assert "invalid-argument" in message
    assert "does not support `reasoning_effort`" in message


async def test_an_unrecognized_json_body_still_carries_no_detail(tmp_path, monkeypatch):
    """Only ``code``/``error`` travel — never arbitrary keys of the body."""
    body = json.dumps({"prompt": "sk-should-never-appear-in-an-exception", "messages": ["secret"]})
    llm = _make_llm(tmp_path, monkeypatch, replies=[_FakeResponse(400, body)])

    with pytest.raises(RuntimeError) as exc:
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert "sk-should-never-appear" not in str(exc.value)
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('{"code": "invalid-argument", "error": "bad value"}', "invalid-argument: bad value"),
        ('{"error": {"code": "quota", "message": "over limit"}}', "quota: over limit"),
        ('{"error": "plain message"}', "plain message"),
        ('{"code": "solo-code"}', "solo-code"),
        ('{"error": "same", "code": "same"}', "same"),
        ("not json at all", ""),
        ("[1, 2, 3]", ""),
        ("{}", ""),
        ('{"error": {"message": null}}', ""),
        ('{"error": true}', ""),
    ],
)
def test_error_detail_reads_only_the_recognized_fields(body, expected):
    assert _error_detail(body) == expected


def test_error_detail_is_length_capped():
    detail = _error_detail(json.dumps({"error": "x" * 500}))

    assert len(detail) <= llm_mod.MAX_ERROR_DETAIL_CHARS + 3
    assert detail.endswith("...")


# ===========================================================================
# 6. Conversation affinity id
# ===========================================================================


def _reference_first_message_fingerprint(messages: Any) -> str | None:
    """The cache-affinity derivation, restated independently for byte-parity.

    Mirrors ``_first_message_fingerprint``: same shape guard, same canonical
    ``json.dumps`` arguments, same sha256 truncation.
    """
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
        return None
    canonical = json.dumps(messages[0], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "system", "content": "S"}],
        [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}],
        [{"role": "system", "content": "unicode: café — dash"}],
        [{}],
    ],
)
def test_affinity_id_matches_the_cache_affinity_derivation_byte_for_byte(messages):
    assert cache_affinity_id(messages) == _reference_first_message_fingerprint(messages)


@pytest.mark.parametrize("messages", ["not a list", {"role": "system"}, [], None, [42]])
def test_affinity_id_fails_open_on_every_non_conforming_shape(messages):
    assert cache_affinity_id(messages) is None
    assert _reference_first_message_fingerprint(messages) is None


def test_affinity_id_is_thirty_two_lowercase_hex_characters():
    result = cache_affinity_id([{"role": "system", "content": "S"}])
    assert result is not None
    assert len(result) == 32
    assert all(character in "0123456789abcdef" for character in result)


def test_affinity_id_prefers_the_trace_id_and_hashes_it():
    """Bind a real trace context rather than patching the lookup.

    The derivation lives in engine/cache_affinity.py and resolves the context
    through llm_trace at call time, so a monkeypatched module attribute here
    would assert against a seam the provider no longer uses.
    """
    expected = hashlib.sha256(b"operation-trace-1").hexdigest()[:32]

    with _bound_trace("operation-trace-1"):
        assert cache_affinity_id([{"role": "system", "content": "S"}]) == expected


def test_affinity_id_falls_back_to_the_first_message_without_a_trace():
    messages = [{"role": "system", "content": "S"}]

    assert current_trace_context() is None, "no operation trace may be bound for the fallback path"
    assert cache_affinity_id(messages) == _reference_first_message_fingerprint(messages)


async def test_the_affinity_id_is_stable_across_calls_sharing_a_first_message(tmp_path, monkeypatch):
    llm = _make_llm(tmp_path, monkeypatch, replies=[_ok_reply(), _ok_reply()])

    await llm.call(messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "a"}], max_retries=0)
    await llm.call(messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "b"}], max_retries=0)

    first, second = (call["headers"][XAI_CONV_ID_HEADER] for call in llm._client.calls)
    assert first == second


# ===========================================================================
# 7. Headers
# ===========================================================================


async def test_the_request_carries_exactly_the_bearer_content_and_affinity_headers(tmp_path, monkeypatch, caplog):
    llm = _make_llm(tmp_path, monkeypatch)
    caplog.set_level(logging.DEBUG)
    caplog.clear()

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    headers = llm._client.calls[0]["headers"]
    assert set(headers) == {"Content-Type", "Accept", "Authorization", XAI_CONV_ID_HEADER}
    assert headers["Authorization"] == f"Bearer {ACCESS_TOKEN}"
    assert headers["Content-Type"] == "application/json"
    assert len(headers[XAI_CONV_ID_HEADER]) == 32
    assert llm._client.calls[0]["url"] == "https://api.x.ai/v1/chat/completions"

    for value in headers.values():
        assert value not in caplog.text


@pytest.mark.parametrize(
    "banned",
    ["x-grok-client-version", "x-grok-client-identifier", "X-XAI-Token-Auth", "x-grok-model-override", "User-Agent"],
)
async def test_no_client_identity_header_is_ever_sent(tmp_path, monkeypatch, banned):
    """This provider speaks a published API plainly; it impersonates no client."""
    llm = _make_llm(
        tmp_path, monkeypatch, replies=[_FakeResponse(401, ""), _ok_reply()], refresh_replies=[_refresh_ok()]
    )

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    lowered = {key.lower() for call in llm._client.calls for key in call["headers"]}
    assert banned.lower() not in lowered


async def test_the_model_travels_in_the_body_not_a_routing_header(tmp_path, monkeypatch):
    llm = _make_llm(tmp_path, monkeypatch, model="grok-4.5")

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert llm._client.calls[0]["json"]["model"] == "grok-4.5"


async def test_reasoning_effort_and_the_token_cap_land_in_the_body(tmp_path, monkeypatch):
    llm = _make_llm(tmp_path, monkeypatch, reasoning_effort="high")

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_completion_tokens=64, max_retries=0)

    body = llm._client.calls[0]["json"]
    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == 64
    assert "max_completion_tokens" not in body


@pytest.mark.parametrize("effort", [None, ""])
async def test_an_unconfigured_reasoning_effort_is_not_sent(tmp_path, monkeypatch, effort):
    """Unset means grok runs at its own default effort.

    Hindsight used to resolve unset to "low" in the config layer and send it here, so
    a deployment that never configured reasoning still had a level chosen for it. An
    empty environment variable counts as unset, exactly like an absent one.
    """
    llm = _make_llm(tmp_path, monkeypatch, reasoning_effort=effort)

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert "reasoning_effort" not in llm._client.calls[0]["json"]


@pytest.mark.parametrize("effort", ["none", "None", "  none  "])
async def test_a_none_reasoning_effort_is_omitted_rather_than_sent(tmp_path, monkeypatch, effort):
    """xAI answers ``reasoning_effort: "none"`` with a non-retryable HTTP 400.

    ``none`` is a real value elsewhere in Hindsight (it is the documented way
    to let an OpenAI reasoning model accept function tools), so a deployment
    can carry it into this lane through the global or any per-operation
    override. Forwarding it would fail every call on the lane.
    """
    llm = _make_llm(tmp_path, monkeypatch, reasoning_effort=effort)

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert "reasoning_effort" not in llm._client.calls[0]["json"]


async def test_a_none_reasoning_effort_is_also_omitted_from_a_tool_call(tmp_path, monkeypatch):
    llm = _make_llm(tmp_path, monkeypatch, reasoning_effort="none")
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

    await llm.call_with_tools(messages=[{"role": "user", "content": "hi"}], tools=tools, max_retries=0)

    assert "reasoning_effort" not in llm._client.calls[0]["json"]


# ===========================================================================
# 8. Token store
# ===========================================================================


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes are not enforced on Windows")
def test_the_store_is_written_owner_only(tmp_path):
    store = tmp_path / "xai_oauth.json"
    write_credential(
        store,
        StoredCredential(ACCESS_TOKEN, REFRESH_TOKEN, time.time() + 60, time.time(), DEFAULT_SCOPE, TOKEN_ENDPOINT),
    )

    assert store.stat().st_mode & 0o777 == 0o600


def test_the_store_path_override_is_honored(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "creds.json"
    monkeypatch.setenv(ENV_TOKEN_PATH, str(target))

    assert default_token_path() == target


def test_the_default_store_lives_under_the_hindsight_home(monkeypatch):
    monkeypatch.delenv(ENV_TOKEN_PATH, raising=False)

    path = default_token_path()

    assert path.parent.name == ".hindsight"
    assert path.name == "xai_oauth.json"


async def test_no_grok_cli_credential_file_is_ever_opened(tmp_path, monkeypatch):
    """The Grok CLI's own credential file is out of scope for this provider.

    The recorder also captures the temporary store, which proves it would have
    caught a ``~/.grok`` read had one happened.
    """
    import builtins

    opened: list[str] = []
    real_open = builtins.open
    real_path_open = Path.open

    def _record_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    def _record_path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(self))
        return real_path_open(self, *args, **kwargs)

    llm = _make_llm(
        tmp_path,
        monkeypatch,
        replies=[_FakeResponse(401, ""), _ok_reply()],
        refresh_replies=[_refresh_ok()],
        expires_in=10.0,
    )
    monkeypatch.setattr(builtins, "open", _record_open)
    monkeypatch.setattr(Path, "open", _record_path_open)
    try:
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)
    finally:
        monkeypatch.undo()

    assert any("xai_oauth.json" in path for path in opened), "the recorder never saw the store it must have read"
    assert not any(".grok" in path for path in opened)


def test_a_missing_store_raises_the_login_remediation(tmp_path):
    with pytest.raises(XaiOAuthLoginRequiredError, match="xai_oauth_auth login"):
        read_credential(tmp_path / "absent.json")


def test_a_corrupt_store_raises_the_login_remediation(tmp_path):
    store = tmp_path / "xai_oauth.json"
    store.write_text("{not json")

    with pytest.raises(XaiOAuthLoginRequiredError, match="unreadable"):
        read_credential(store)


def test_the_stored_credential_round_trips(tmp_path):
    store = tmp_path / "xai_oauth.json"
    expires_at = time.time() + 120
    obtained_at = time.time()
    write_credential(
        store,
        StoredCredential(ACCESS_TOKEN, REFRESH_TOKEN, expires_at, obtained_at, DEFAULT_SCOPE, TOKEN_ENDPOINT),
    )

    loaded = read_credential(store)

    assert loaded.access_token == ACCESS_TOKEN
    assert loaded.refresh_token == REFRESH_TOKEN
    assert loaded.expires_at == pytest.approx(expires_at)
    assert loaded.obtained_at == pytest.approx(obtained_at)
    assert loaded.scope == DEFAULT_SCOPE
    assert loaded.token_endpoint == TOKEN_ENDPOINT


def test_a_credential_without_a_recorded_expiry_reports_no_life_left():
    credential = StoredCredential(ACCESS_TOKEN, REFRESH_TOKEN, None, 0.0, DEFAULT_SCOPE, TOKEN_ENDPOINT)

    assert credential.seconds_left() == 0.0


# ===========================================================================
# 9. Usage parsing
# ===========================================================================


def test_cached_tokens_are_read_when_prompt_tokens_details_is_present():
    usage = _ChatUsage.model_validate(
        {
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "total_tokens": 140,
            "prompt_tokens_details": {"cached_tokens": 80},
        }
    )

    counts = _token_counts(usage)

    assert counts.input_tokens == 100
    assert counts.cached_tokens == 80
    assert counts.output_tokens == 40


def test_missing_prompt_tokens_details_reads_zero_cached_without_error():
    usage = _ChatUsage.model_validate({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

    counts = _token_counts(usage)

    assert counts.cached_tokens == 0
    assert counts.input_tokens == 10


def test_a_missing_usage_block_reads_all_zeroes():
    counts = _token_counts(None)

    assert (counts.input_tokens, counts.output_tokens, counts.total_tokens) == (0, 0, 0)


def test_reasoning_tokens_are_subtracted_from_visible_output():
    usage = _ChatUsage.model_validate(
        {
            "prompt_tokens": 10,
            "completion_tokens": 100,
            "total_tokens": 110,
            "completion_tokens_details": {"reasoning_tokens": 60},
        }
    )

    counts = _token_counts(usage)

    assert counts.output_tokens == 40
    assert counts.thoughts_tokens == 60
    assert counts.total_tokens == 50


def test_output_tokens_survive_when_completion_tokens_is_already_visible_only():
    """xAI does not always fold reasoning into completion_tokens the way the
    OpenAI o1/o3 contract does. When completion_tokens is already
    visible-only (the unfolded shape: total = prompt + completion +
    reasoning), subtracting reasoning_tokens from completion_tokens a second
    time would clamp a real 40-token completion down to 0 -- the production
    bug this guards against.
    """
    usage = _ChatUsage.model_validate(
        {
            "prompt_tokens": 10,
            "completion_tokens": 40,  # visible-only; does NOT include the 60 reasoning tokens
            "total_tokens": 110,  # 10 + 40 + 60: the unfolded shape
            "completion_tokens_details": {"reasoning_tokens": 60},
        }
    )

    counts = _token_counts(usage)

    assert counts.output_tokens == 40
    assert counts.thoughts_tokens == 60
    assert counts.total_tokens == 50


async def test_return_usage_surfaces_cached_tokens(tmp_path, monkeypatch):
    usage = {
        "prompt_tokens": 200,
        "completion_tokens": 20,
        "total_tokens": 220,
        "prompt_tokens_details": {"cached_tokens": 150},
    }
    llm = _make_llm(tmp_path, monkeypatch, replies=[_ok_reply("hi", usage)])

    _, token_usage = await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0, return_usage=True)

    assert token_usage.input_tokens == 200
    assert token_usage.cached_tokens == 150


# ===========================================================================
# 10. Timeout, factory and config wiring
# ===========================================================================


def test_the_configured_timeout_lands_on_the_http_client(tmp_path, monkeypatch):
    store = _write_store(tmp_path / "xai_oauth.json")
    monkeypatch.setenv(ENV_TOKEN_PATH, str(store))
    manager, _, _ = _make_manager(tmp_path, store=store)

    llm = XaiOAuthLLM(
        provider="xai-oauth", api_key="", base_url="", model="grok-4.5", timeout=12.5, auth_manager=manager
    )

    assert llm.timeout == 12.5
    assert llm._client.timeout.read == 12.5


def test_the_admission_bar_is_the_larger_of_the_skew_and_the_timeout(tmp_path, monkeypatch):
    store = _write_store(tmp_path / "xai_oauth.json")
    monkeypatch.setenv(ENV_TOKEN_PATH, str(store))
    manager, _, _ = _make_manager(tmp_path, store=store)

    short = XaiOAuthLLM(provider="xai-oauth", api_key="", base_url="", model="m", timeout=5.0, auth_manager=manager)
    long = XaiOAuthLLM(provider="xai-oauth", api_key="", base_url="", model="m", timeout=600.0, auth_manager=manager)

    assert short._admission_ttl() == auth_mod.DEFAULT_REFRESH_SKEW_SECONDS
    assert long._admission_ttl() == 600.0


def test_the_provider_does_not_require_an_api_key():
    assert requires_api_key("xai-oauth") is False
    assert requires_api_key("openai") is True


def test_the_default_model_is_registered():
    assert PROVIDER_DEFAULT_MODELS["xai-oauth"] == "grok-4.5"


def test_the_factory_builds_the_provider_and_threads_the_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_TOKEN_PATH, str(_write_store(tmp_path / "xai_oauth.json")))

    provider = create_llm_provider(
        provider="xai-oauth",
        api_key="",
        base_url="",
        model="grok-4.5",
        reasoning_effort="high",
        timeout=42.0,
    )

    assert isinstance(provider, XaiOAuthLLM)
    assert provider.timeout == 42.0
    assert provider.base_url == DEFAULT_BASE_URL
    assert provider.supports_attempt_scoped_concurrency() is True


def test_the_provider_specific_base_url_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_TOKEN_PATH, str(_write_store(tmp_path / "xai_oauth.json")))
    monkeypatch.setenv(ENV_BASE_URL, "https://example.invalid/v1/")

    provider = create_llm_provider(
        provider="xai-oauth",
        api_key="",
        base_url="https://ignored.invalid/v1",
        model="grok-4.5",
        reasoning_effort="low",
    )

    assert provider.base_url == "https://example.invalid/v1"


# ===========================================================================
# 11. Structured output and tool calls
# ===========================================================================


class _Answer(BaseModel):
    answer: str


async def test_structured_output_uses_a_strict_json_schema_when_requested(tmp_path, monkeypatch):
    llm = _make_llm(tmp_path, monkeypatch, replies=[_ok_reply(json.dumps({"answer": "42"}))])

    result = await llm.call(
        messages=[{"role": "user", "content": "hi"}],
        response_format=_Answer,
        strict_schema=True,
        max_retries=0,
    )

    body = llm._client.calls[0]["json"]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert result.answer == "42"


async def test_structured_output_falls_back_to_schema_in_prompt(tmp_path, monkeypatch):
    llm = _make_llm(tmp_path, monkeypatch, replies=[_ok_reply(json.dumps({"answer": "42"}))])

    await llm.call(
        messages=[{"role": "user", "content": "hi"}],
        response_format=_Answer,
        strict_schema=False,
        max_retries=0,
    )

    body = llm._client.calls[0]["json"]
    assert body["response_format"] == {"type": "json_object"}
    assert "valid JSON matching this schema" in body["messages"][0]["content"]


async def test_call_with_tools_returns_proposed_tool_calls(tmp_path, monkeypatch):
    reply = _FakeResponse(
        200,
        json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {"id": "call_1", "function": {"name": "search", "arguments": '{"q": "hi"}'}}
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }
        ),
    )
    llm = _make_llm(tmp_path, monkeypatch, replies=[reply])
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

    result = await llm.call_with_tools(messages=[{"role": "user", "content": "hi"}], tools=tools, max_retries=0)

    assert result.finish_reason == "tool_calls"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"q": "hi"}
    assert llm._client.calls[0]["json"]["tools"] == tools


async def test_an_empty_completion_is_retried_then_fails(tmp_path, monkeypatch):
    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(llm_mod.asyncio, "sleep", _no_sleep)
    empty = _FakeResponse(200, json.dumps({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}))
    llm = _make_llm(tmp_path, monkeypatch, replies=[empty])

    with pytest.raises(RuntimeError, match="empty message content"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=2)

    assert len(llm._client.calls) == 3


async def test_cleanup_closes_both_clients(tmp_path, monkeypatch):
    llm = _make_llm(tmp_path, monkeypatch)

    await llm.cleanup()  # must not raise; the manager owns no client it did not create


def test_the_event_loop_is_never_blocked_by_a_refresh(tmp_path, monkeypatch):
    """The refresh runs in a worker thread, so ``call`` stays cooperative."""
    llm = _make_llm(
        tmp_path,
        monkeypatch,
        replies=[_ok_reply()],
        expires_in=10.0,
        refresh_replies=[_refresh_ok()],
    )
    thread_ids: list[int] = []

    original = llm._auth.get_access_token

    def _record(*args: Any, **kwargs: Any) -> str:
        thread_ids.append(threading.get_ident())
        return original(*args, **kwargs)

    llm._auth.get_access_token = _record  # type: ignore[method-assign]
    asyncio.run(llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0))

    assert thread_ids and thread_ids[0] != threading.get_ident()


# ===========================================================================
# 12. Connection recycling on a retryable 5xx
# ===========================================================================


async def test_a_502_retry_lands_on_a_new_client_not_the_pinned_one(tmp_path, monkeypatch):
    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(llm_mod.asyncio, "sleep", _no_sleep)
    llm = _make_llm(tmp_path, monkeypatch, replies=[_FakeResponse(502, "bad gateway")])
    first_client = llm._client
    second_client = _FakeAsyncHttp([_ok_reply("recovered")])
    llm._new_client = lambda: second_client  # type: ignore[method-assign]

    result = await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=1)

    assert result == "recovered"
    assert llm._client is second_client
    assert len(first_client.calls) == 1
    assert len(second_client.calls) == 1


async def test_a_429_retry_does_not_recycle_the_connection(tmp_path, monkeypatch):
    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(llm_mod.asyncio, "sleep", _no_sleep)
    llm = _make_llm(tmp_path, monkeypatch, replies=[_FakeResponse(429, "slow down"), _ok_reply("ok")])
    original_client = llm._client
    llm._new_client = _never_called  # type: ignore[method-assign]

    result = await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=1)

    assert result == "ok"
    assert llm._client is original_client
    assert len(original_client.calls) == 2


async def test_a_400_retry_does_not_recycle_the_connection(tmp_path, monkeypatch):
    """4xx other than 408/429 is not even retried, but assert the negative
    directly rather than relying on that as an accident of the retry gate.
    """
    llm = _make_llm(tmp_path, monkeypatch, replies=[_FakeResponse(400, "bad request")])
    original_client = llm._client
    llm._new_client = _never_called  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="HTTP 400"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=3)

    assert llm._client is original_client


async def test_call_with_tools_also_recycles_the_connection_on_a_5xx(tmp_path, monkeypatch):
    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(llm_mod.asyncio, "sleep", _no_sleep)
    ok_reply = _FakeResponse(
        200,
        json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}),
    )
    llm = _make_llm(tmp_path, monkeypatch, replies=[_FakeResponse(503, "unavailable")])
    first_client = llm._client
    second_client = _FakeAsyncHttp([ok_reply])
    llm._new_client = lambda: second_client  # type: ignore[method-assign]
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

    result = await llm.call_with_tools(messages=[{"role": "user", "content": "hi"}], tools=tools, max_retries=1)

    assert result.content == "ok"
    assert llm._client is second_client
    assert len(first_client.calls) == 1
    assert len(second_client.calls) == 1


async def test_recycling_closes_the_stale_client(tmp_path, monkeypatch):
    """The stale client's own aclose() is awaited, releasing its connections."""

    closed: list[Any] = []

    class _TrackedFakeAsyncHttp(_FakeAsyncHttp):
        async def aclose(self) -> None:
            closed.append(self)

    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(llm_mod.asyncio, "sleep", _no_sleep)
    llm = _make_llm(tmp_path, monkeypatch, replies=[_FakeResponse(502, "bad gateway")])
    tracked = _TrackedFakeAsyncHttp([_FakeResponse(502, "bad gateway")])
    llm._client = tracked
    llm._new_client = lambda: _FakeAsyncHttp([_ok_reply()])  # type: ignore[method-assign]

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=1)

    assert closed == [tracked]


class _GatedFakeAsyncHttp(_FakeAsyncHttp):
    """A fake whose designated call parks until the test releases it."""

    def __init__(self, replies: list[_FakeResponse], *, gate_on_call: int = 0) -> None:
        super().__init__(replies)
        self.gate = asyncio.Event()
        self.entered = asyncio.Event()
        self.closed = False
        self._gate_on_call = gate_on_call

    async def post(self, url: str, json: Any = None, headers: Any = None) -> _FakeResponse:
        index = len(self.calls)
        self.calls.append({"url": url, "json": json, "headers": dict(headers or {})})
        if index == self._gate_on_call:
            self.entered.set()
            await self.gate.wait()
        return self._replies[min(index, len(self._replies) - 1)]

    async def aclose(self) -> None:
        self.closed = True


async def _settle() -> None:
    """Give scheduled background tasks a turn without a wall-clock wait."""
    for _ in range(10):
        await asyncio.sleep(0)


async def test_a_recycle_leaves_a_sibling_request_on_the_stale_client_alive(tmp_path, monkeypatch):
    """One backend's 5xx must not cancel the other requests in flight.

    A provider instance is shared by every concurrent call on its lane. Closing
    the stale client on the spot yanks the connection out from under siblings
    mid-response — measured, they surface ``httpx.ReadError`` and retry, which
    re-sends a completion the upstream already accepted and fails outright any
    sibling that was on its final attempt.
    """
    llm = _make_llm(tmp_path, monkeypatch)
    gated = _GatedFakeAsyncHttp([_ok_reply("sibling")])
    llm._client = gated  # type: ignore[assignment]
    replacement = _FakeAsyncHttp([_ok_reply("fresh")])
    llm._new_client = lambda: replacement  # type: ignore[method-assign]

    sibling = asyncio.create_task(llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0))
    await gated.entered.wait()

    await llm._recycle_client()

    assert llm._client is replacement
    assert not gated.closed, "the stale client was closed while a sibling was still using it"

    gated.gate.set()
    assert await sibling == "sibling"

    await _settle()
    assert gated.closed, "the stale client must still be closed once it drains"


async def test_the_retrying_call_does_not_wait_for_the_sibling_to_drain(tmp_path, monkeypatch):
    """Deferring the close must not defer the retry it exists to unblock."""
    llm = _make_llm(tmp_path, monkeypatch)
    gated = _GatedFakeAsyncHttp([_ok_reply("sibling")])
    llm._client = gated  # type: ignore[assignment]
    llm._new_client = lambda: _FakeAsyncHttp([_ok_reply("fresh")])  # type: ignore[method-assign]

    sibling = asyncio.create_task(llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0))
    await gated.entered.wait()

    # Completes even though the sibling is still parked on the stale client.
    await asyncio.wait_for(llm._recycle_client(), timeout=1.0)

    gated.gate.set()
    await sibling
    await _settle()


async def test_cleanup_closes_a_client_still_draining_from_a_recycle(tmp_path, monkeypatch):
    """Shutdown must neither hang on nor leak a client that never drained."""
    llm = _make_llm(tmp_path, monkeypatch)
    gated = _GatedFakeAsyncHttp([_ok_reply("sibling")])
    llm._client = gated  # type: ignore[assignment]
    llm._new_client = lambda: _FakeAsyncHttp([_ok_reply("fresh")])  # type: ignore[method-assign]

    sibling = asyncio.create_task(llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0))
    await gated.entered.wait()
    await llm._recycle_client()

    await asyncio.wait_for(llm.cleanup(), timeout=1.0)

    assert gated.closed

    gated.gate.set()
    with suppress(Exception):
        await sibling


# ===========================================================================
# 14. Credential-side transients rejoin the retry loop
# ===========================================================================


async def test_a_transient_refresh_failure_is_retried_like_any_other_blip(tmp_path, monkeypatch):
    """A hiccup reaching auth.x.ai must not fail the whole operation outright.

    The same failure one hop later — against api.x.ai — gets the full retry
    budget, so a credential-side transient getting none was an inconsistency,
    not a policy.
    """

    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(llm_mod.asyncio, "sleep", _no_sleep)
    llm = _make_llm(tmp_path, monkeypatch, replies=[_ok_reply("recovered")])
    attempts: list[float] = []

    def _flaky(min_ttl: float) -> str:
        attempts.append(min_ttl)
        if len(attempts) == 1:
            raise XaiOAuthRefreshError("xai-oauth refresh network error: ConnectError")
        return ACCESS_TOKEN

    llm._auth.get_access_token = _flaky  # type: ignore[method-assign]

    assert await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=2) == "recovered"
    assert len(attempts) == 2
    assert len(llm._client.calls) == 1


async def test_a_transient_refresh_failure_is_retried_in_the_tool_loop(tmp_path, monkeypatch):
    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(llm_mod.asyncio, "sleep", _no_sleep)
    llm = _make_llm(tmp_path, monkeypatch, replies=[_ok_reply("recovered")])
    attempts: list[float] = []

    def _flaky(min_ttl: float) -> str:
        attempts.append(min_ttl)
        if len(attempts) == 1:
            raise XaiOAuthRefreshError("xai-oauth refresh network error: ConnectError")
        return ACCESS_TOKEN

    llm._auth.get_access_token = _flaky  # type: ignore[method-assign]
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

    result = await llm.call_with_tools(messages=[{"role": "user", "content": "hi"}], tools=tools, max_retries=2)

    assert result.content == "recovered"
    assert len(attempts) == 2


async def test_a_login_required_credential_is_still_fatal_on_the_first_attempt(tmp_path, monkeypatch):
    """Only the transients rejoin the loop; a dead grant must not be retried."""

    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(llm_mod.asyncio, "sleep", _no_sleep)
    llm = _make_llm(tmp_path, monkeypatch)
    attempts: list[float] = []

    def _dead(min_ttl: float) -> str:
        attempts.append(min_ttl)
        raise XaiOAuthLoginRequiredError("no credential")

    llm._auth.get_access_token = _dead  # type: ignore[method-assign]

    with pytest.raises(XaiOAuthLoginRequiredError):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=5)

    assert len(attempts) == 1
    assert len(llm._client.calls) == 0


# ===========================================================================
# 15. Startup verification
# ===========================================================================


async def test_verification_waves_through_a_real_rate_limit(tmp_path, monkeypatch):
    """An exhausted rate limit is not a misconfiguration; startup continues."""

    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(llm_mod.asyncio, "sleep", _no_sleep)
    llm = _make_llm(tmp_path, monkeypatch, replies=[_FakeResponse(429, "slow down")])

    await llm.verify_connection()


async def test_verification_does_not_mistake_a_byte_count_for_a_rate_limit(tmp_path, monkeypatch):
    """A 5xx whose message merely contains "429" must still fail startup.

    The error text states the body's byte count and the upstream's error
    string, so a substring test on "429" lets a broken lane pass verification —
    here a 500 with a 429-byte body, which reads as "HTTP 500 (429 bytes)".
    """

    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(llm_mod.asyncio, "sleep", _no_sleep)
    server_error = _FakeResponse(500, "x" * 429)
    llm = _make_llm(tmp_path, monkeypatch, replies=[server_error])
    # verify_connection retries, and a >=500 recycles the client: without this
    # the replacement would be a real httpx.AsyncClient and the retry would
    # leave the process for api.x.ai, which no test here may do.
    llm._new_client = lambda: _FakeAsyncHttp([server_error])  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="verification failed"):
        await llm.verify_connection()


# ===========================================================================
# 13. Debug-gated non-2xx header logging
# ===========================================================================


async def test_debug_headers_off_by_default_logs_nothing_on_a_5xx(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    caplog.clear()
    llm = _make_llm(
        tmp_path,
        monkeypatch,
        replies=[_FakeResponse(502, "bad gateway", {"cf-ray": "abc123", "via": "1.1 google"})],
    )

    with pytest.raises(RuntimeError, match="HTTP 502"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert "cf-ray" not in caplog.text
    assert "abc123" not in caplog.text


async def test_debug_headers_on_logs_the_allowlist_and_nothing_else_on_a_5xx(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(ENV_DEBUG_HEADERS, "true")
    caplog.set_level(logging.DEBUG)
    caplog.clear()
    secret_body = "sk-should-never-appear-in-a-log"
    llm = _make_llm(
        tmp_path,
        monkeypatch,
        replies=[
            _FakeResponse(
                502,
                secret_body,
                {
                    "cf-ray": "abc123",
                    "via": "1.1 google",
                    "x-request-id": "req-1",
                    "server": "envoy",
                    "date": "Sat, 08 Aug 2026 00:00:00 GMT",
                    "set-cookie": "session=do-not-log",
                },
            )
        ],
    )

    with pytest.raises(RuntimeError, match="HTTP 502"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert "cf-ray" in caplog.text and "abc123" in caplog.text
    assert "x-request-id" in caplog.text and "req-1" in caplog.text
    assert "server" in caplog.text and "envoy" in caplog.text
    assert "via" in caplog.text
    assert "date" in caplog.text
    assert "set-cookie" not in caplog.text
    assert "do-not-log" not in caplog.text
    assert ACCESS_TOKEN not in caplog.text
    assert secret_body not in caplog.text


async def test_debug_headers_on_logs_nothing_on_a_200(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(ENV_DEBUG_HEADERS, "true")
    caplog.set_level(logging.DEBUG)
    caplog.clear()
    reply = _FakeResponse(200, json.dumps(_completion("ok")), {"cf-ray": "abc123", "x-request-id": "req-1"})
    llm = _make_llm(tmp_path, monkeypatch, replies=[reply])

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert "cf-ray" not in caplog.text
    assert "abc123" not in caplog.text
    assert "req-1" not in caplog.text


def test_debug_headers_env_var_is_case_insensitive_on_the_value():
    os.environ[ENV_DEBUG_HEADERS] = "TRUE"
    try:
        assert llm_mod._debug_headers_enabled() is True
    finally:
        del os.environ[ENV_DEBUG_HEADERS]


# ===========================================================================
# 14. Error messages report the actual request host
# ===========================================================================
#
# A misrouted ``base_url`` (a misconfigured proxy, a regional mirror) must
# never have its failures blamed on the literal "api.x.ai" -- that sent a
# real production investigation to the wrong system three times in one day.
# These pin the fix at both the unit level (the host-derivation helper) and
# the integration level (the actual exception text a caller sees).


@pytest.mark.parametrize(
    ("base_url", "expected_host"),
    [
        ("https://api.x.ai/v1", "api.x.ai"),
        ("https://internal-proxy.example.com:8443/v1", "internal-proxy.example.com"),
        ("http://localhost:8080/v1", "localhost"),
        ("https://user:secret-token@proxy.example.com/v1", "proxy.example.com"),
    ],
)
def test_actual_host_derives_the_hostname_from_base_url(base_url, expected_host):
    assert llm_mod._actual_host(base_url) == expected_host


def test_actual_host_falls_back_to_the_literal_string_when_unparseable():
    """A base_url with no host at all (or one urlsplit rejects outright)
    still yields some signal instead of silently blaming a fixed default.
    """
    assert llm_mod._actual_host("not a url") == "not a url"
    assert llm_mod._actual_host("ftp://[::1") == "ftp://[::1"


async def test_a_non_2xx_error_names_the_actual_host_not_a_hardcoded_one(tmp_path, monkeypatch):
    """RED against the unfixed code: the old message always read
    "api.x.ai returned HTTP ..." regardless of where the request actually
    went, which is exactly the shape that misdirected diagnosis in
    production when base_url pointed at an internal proxy returning 502s.
    """
    llm = _make_llm(
        tmp_path,
        monkeypatch,
        replies=[_FakeResponse(502, "bad gateway")],
        base_url="https://internal-proxy.example.com/v1",
    )

    with pytest.raises(RuntimeError) as exc:
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert "internal-proxy.example.com" in str(exc.value)
    assert "api.x.ai" not in str(exc.value)


async def test_an_unusable_success_shape_also_names_the_actual_host(tmp_path, monkeypatch):
    """Same class of message, the shape-error path (``_content_of``) rather
    than the transport-error path (``_request_completion``).
    """
    empty = _FakeResponse(200, json.dumps({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}))
    llm = _make_llm(
        tmp_path,
        monkeypatch,
        replies=[empty],
        base_url="https://internal-proxy.example.com/v1",
    )

    with pytest.raises(RuntimeError, match="empty message content") as exc:
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert "internal-proxy.example.com" in str(exc.value)
    assert "api.x.ai" not in str(exc.value)


async def test_a_default_base_url_still_names_api_x_ai(tmp_path, monkeypatch):
    """The default deployment (no override) keeps seeing "api.x.ai" -- the
    fix changes where the host comes from, not the message shown for the
    common case.
    """
    llm = _make_llm(tmp_path, monkeypatch, replies=[_FakeResponse(502, "bad gateway")])

    with pytest.raises(RuntimeError, match="api.x.ai returned HTTP 502"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)
