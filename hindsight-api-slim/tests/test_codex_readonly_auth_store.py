"""Read-only consumers adopt external credentials, never rotate them without a lock.

Use injected errnos rather than chmod, which does not constrain privileged CI.
All credentials are synthetic; HTTP and persistence are forbidden in fallback.
"""

from __future__ import annotations

import base64
import builtins
import errno
import json
import multiprocessing
import os
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest

from hindsight_api.engine.providers import codex_auth as auth

pytestmark = pytest.mark.skipif(auth.fcntl is None, reason="POSIX auth-store locking")
NOW = 1_800_000_000


def jwt(offset: int, label: str = "external") -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"exp": NOW + offset, "label": label}).encode()).decode()
    return f"synthetic.{payload}.signature"


def publish(path: Path, access: str, refresh: str = "external-refresh") -> None:
    replacement = path.with_suffix(".new")
    replacement.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": access,
                    "refresh_token": refresh,
                    "account_id": "external-account",
                },
            }
        )
    )
    os.replace(replacement, path)


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[auth.CodexAuthManager]:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr(auth.time, "time", lambda: NOW)
    monkeypatch.setattr(auth.httpx.Client, "post", Mock(side_effect=AssertionError("OAuth forbidden")))
    monkeypatch.setattr(
        auth.CodexAuthManager, "_persist_auth_atomic", Mock(side_effect=AssertionError("write forbidden"))
    )
    path = tmp_path / "auth.json"
    publish(path, jwt(-120, "old"), "old-refresh")
    mgr = auth.CodexAuthManager.from_file(path)
    try:
        yield mgr
    finally:
        mgr.close()


def deny_lock(monkeypatch: pytest.MonkeyPatch, path: Path, error: int = errno.EROFS, stage: str = "open") -> None:
    if stage == "mkdir":
        monkeypatch.setattr(Path, "mkdir", Mock(side_effect=OSError(error, "injected mkdir")))
    else:
        real_open = builtins.open

        def open_file(file, *args, **kwargs):
            if Path(file) == path.with_suffix(".lock"):
                raise OSError(error, "injected lock open")
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(auth, "open", open_file, raising=False)


@pytest.mark.parametrize("error", [errno.EROFS, errno.EACCES])
@pytest.mark.parametrize("stage", ["mkdir", "open"])
@pytest.mark.parametrize("force", [False, True])
def test_adopts_fresh_external_token(manager, monkeypatch, error, stage, force):
    publish(manager._auth_file, jwt(3600))
    before = manager._auth_file.read_bytes()
    deny_lock(monkeypatch, manager._auth_file, error, stage)
    manager.refresh_tokens(force=force)
    assert manager.access_token == jwt(3600)
    assert manager.refresh_token == "external-refresh"
    assert manager.account_id == "external-account"
    assert manager._auth_file.read_bytes() == before
    auth.httpx.Client.post.assert_not_called()
    manager._persist_auth_atomic.assert_not_called()


@pytest.mark.parametrize(
    "kind",
    [
        "unchanged",
        "unchanged_fresh",
        "refresh_only",
        "account_only",
        "expired",
        "skew",
        "opaque",
        "malformed",
        "wrong_mode",
        "wrong_schema",
        "wrong_tokens",
        "wrong_access",
        "wrong_refresh",
        "wrong_account",
    ],
)
def test_rejects_unusable_snapshot_without_mutating(manager, monkeypatch, kind):
    path = manager._auth_file
    if kind == "unchanged_fresh":
        manager.access_token = jwt(3600)
        publish(path, manager.access_token)
    elif kind in ("refresh_only", "account_only"):
        publish(path, manager.access_token, "different-refresh" if kind == "refresh_only" else manager.refresh_token)
    elif kind in ("expired", "skew", "opaque"):
        publish(path, {"expired": jwt(-1), "skew": jwt(60), "opaque": "opaque"}[kind])
    elif kind == "malformed":
        path.write_text("{private-synthetic-marker")
    elif kind == "wrong_mode":
        path.write_text(json.dumps({"auth_mode": "apikey", "tokens": {"access_token": jwt(3600)}}))
    elif kind == "wrong_schema":
        path.write_text("[]")
    elif kind == "wrong_tokens":
        path.write_text('{"auth_mode":"chatgpt","tokens":[]}')
    elif kind in ("wrong_access", "wrong_refresh", "wrong_account"):
        field = {"wrong_access": "access_token", "wrong_refresh": "refresh_token", "wrong_account": "account_id"}[kind]
        path.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": jwt(3600), field: 123}}))
    before = (manager.access_token, manager.refresh_token, manager.account_id)
    deny_lock(monkeypatch, path)
    with pytest.raises(RuntimeError, match="external writer") as caught:
        manager.refresh_tokens(force=True)
    assert "private-synthetic-marker" not in str(caught.value)
    assert (manager.access_token, manager.refresh_token, manager.account_id) == before
    auth.httpx.Client.post.assert_not_called()
    manager._persist_auth_atomic.assert_not_called()


@pytest.mark.parametrize("stage", ["mkdir", "open"])
@pytest.mark.parametrize("error", [errno.EIO, errno.EMFILE, errno.EPERM])
def test_unexpected_lock_errors_propagate(manager, monkeypatch, stage, error):
    publish(manager._auth_file, jwt(3600))
    deny_lock(monkeypatch, manager._auth_file, error, stage)
    with pytest.raises(OSError) as caught:
        manager.ensure_fresh_token()
    assert caught.value.errno == error
    assert manager.access_token == jwt(-120, "old")


@pytest.mark.parametrize("error", [errno.EIO, errno.EACCES, errno.ENOENT])
def test_auth_read_errors_propagate(manager, monkeypatch, error):
    def open_file(path, *args, **kwargs):
        raise OSError(errno.EROFS if Path(path).suffix == ".lock" else error, "injected")

    monkeypatch.setattr(auth, "open", open_file, raising=False)
    with pytest.raises(OSError) as caught:
        manager.ensure_fresh_token()
    assert caught.value.errno == error


def test_retry_after_external_publication(manager, monkeypatch):
    deny_lock(monkeypatch, manager._auth_file)
    with pytest.raises(RuntimeError, match="external writer"):
        manager.ensure_fresh_token()
    publish(manager._auth_file, jwt(3600))
    manager.ensure_fresh_token()
    assert manager.access_token == jwt(3600)


@pytest.mark.parametrize("same_manager", [False, True])
def test_concurrent_consumers_adopt(manager, monkeypatch, same_manager):
    managers = [manager] if same_manager else [auth.CodexAuthManager.from_file(manager._auth_file) for _ in range(8)]
    publish(manager._auth_file, jwt(3600))
    deny_lock(monkeypatch, manager._auth_file)
    barrier = threading.Barrier(8)

    def consume(index: int) -> None:
        barrier.wait(timeout=5)
        managers[index % len(managers)].ensure_fresh_token()

    try:
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(consume, range(8)))
        assert all(m.access_token == jwt(3600) for m in managers)
        auth.httpx.Client.post.assert_not_called()
    finally:
        for m in managers:
            if m is not manager:
                m.close()


def test_path_lock_retained_during_fallback(manager, monkeypatch):
    deny_lock(monkeypatch, manager._auth_file)
    with auth._codex_auth_lock(manager._auth_file) as allowed:
        assert allowed is False
        lock = auth._path_scoped_lock(manager._auth_file)
        assert not lock.acquire(blocking=False)


@pytest.mark.parametrize("readonly", [False, True])
def test_body_permission_error_is_not_reinterpreted(manager, monkeypatch, readonly):
    if readonly:
        deny_lock(monkeypatch, manager._auth_file)
    with pytest.raises(OSError) as caught:
        with auth._codex_auth_lock(manager._auth_file):
            raise OSError(errno.EACCES, "body failure")
    assert caught.value.errno == errno.EACCES


def test_unexpected_flock_error_propagates(manager, monkeypatch):
    monkeypatch.setattr(auth.fcntl, "flock", Mock(side_effect=OSError(errno.EIO, "injected")))
    monkeypatch.setattr(auth.time, "monotonic", Mock(side_effect=[0, 100]))
    with pytest.raises(OSError) as caught:
        manager.ensure_fresh_token()
    assert caught.value.errno == errno.EIO


def hold_lock(path, ready, release):
    import fcntl

    with open(path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        ready.set()
        release.wait(10)


@pytest.mark.parametrize("payload", [[], None, {"exp": float("inf")}])
def test_malformed_jwt_payload_fails_closed(manager, monkeypatch, payload):
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    publish(manager._auth_file, f"synthetic.{encoded}.signature")
    deny_lock(monkeypatch, manager._auth_file)
    with pytest.raises(RuntimeError, match="external writer"):
        manager.refresh_tokens(force=True)
    assert manager.access_token == jwt(-120, "old")
    auth.httpx.Client.post.assert_not_called()


def test_busy_process_lock_times_out_then_releases(manager):
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    process = ctx.Process(target=hold_lock, args=(manager._auth_file.with_suffix(".lock"), ready, release))
    process.start()
    try:
        assert ready.wait(5)
        with pytest.raises(TimeoutError):
            with auth._codex_auth_lock(manager._auth_file, timeout_seconds=1):
                pytest.fail("busy lock must not permit adoption or refresh")
    finally:
        release.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
    assert process.exitcode == 0
    with auth._codex_auth_lock(manager._auth_file) as allowed:
        assert allowed is True
