"""Tests for `oauth_store_lock` against a store that cannot hold a lock file.

The lock is taken by creating ``<store>.lock`` next to the credential store. That
write is impossible when the store is read-only, which is the normal shape on
Kubernetes: a Secret volume is mounted read-only whatever ``volumeMount.readOnly``
says, so a store fed by an external secret manager (ESO, a sidecar, a ConfigMap
projection) raises ``EROFS`` when the lock file is created.

The lock guards the whole refresh, including the read of the on-disk store that
lets a credential published by another writer reach this process. A hard failure
there did not just skip the lock — it disabled the read path, so the provider
answered every call with the token it booted with until it restarted. Three
providers share this lock (Codex, Nous, xai-oauth), so the failure mode and the
fallback are shared too.

**`EACCES` is ambiguous, which is why the errno alone is not the condition.**
``open(lock_path, "a+")`` raises ``EACCES`` in two different worlds:

* the store directory cannot be written — degrade, nothing can hold a lock here;
* the directory is writable but a pre-existing ``<store>.lock`` owned by another
  uid denies this process — the shared-credential-directory shape (a container
  that once ran as root and now runs non-root over the same volume). There the
  store IS writable and a peer holds the lock, so degrading would admit a
  concurrent refresh of a rotating token.

``os.access(lock_path.parent, os.W_OK)`` separates them, and the tests below
cover both the branch logic (injection, works as root) and the real condition
truly exercised (real permissions, skipped for root because root bypasses them).
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import errno
import os
import stat
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from hindsight_api.engine.providers.oauth_store_lock import oauth_store_lock

# Root defeats every permission-based setup below (it opens 0444 files and
# creates files in 0500 directories), so those tests would pass for free.
requires_nonroot = pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses the permission bits these tests rely on")


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A credential store file, as the OAuth managers write it."""
    path = tmp_path / "provider-home" / "auth.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"tokens": {"access_token": "a", "refresh_token": "r"}}')
    return path


@contextlib.contextmanager
def _fail_lock_creation(store: Path, error_number: int, *, parent_writable: bool) -> Iterator[None]:
    """Make creating ``<store>.lock`` fail with ``error_number``.

    ``parent_writable`` states whether the store directory can be written, which
    is the condition under test — the errno alone cannot express it. Reads and
    writes of the store itself keep working, so the store behaves like a
    projection that accepts nothing new beside the credential.
    """
    lock_path = store.with_suffix(".lock")
    real_open = builtins.open

    def fake_open(file, *args, **kwargs):
        if Path(file) == lock_path:
            raise OSError(error_number, f"injected {errno.errorcode.get(error_number)}")
        return real_open(file, *args, **kwargs)

    real_mkdir = Path.mkdir

    def fake_mkdir(self, *args, **kwargs):
        if self == lock_path.parent:
            raise OSError(error_number, f"injected {errno.errorcode.get(error_number)}")
        return real_mkdir(self, *args, **kwargs)

    real_access = os.access

    def fake_access(path, mode, **kwargs):
        if Path(path) == lock_path.parent and mode == os.W_OK:
            return parent_writable
        return real_access(path, mode, **kwargs)

    with (
        patch("builtins.open", fake_open),
        patch.object(Path, "mkdir", fake_mkdir),
        patch("os.access", fake_access),
    ):
        yield


# ---------------------------------------------------------------------------
# The real condition, with real permissions.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@requires_nonroot
async def test_degrades_on_a_directory_that_cannot_be_written(store: Path):
    """A read-only store directory: no lock file is possible, so degrade."""
    assert os.access(store.parent, os.W_OK) is True
    store.parent.chmod(stat.S_IRUSR | stat.S_IXUSR)  # r-x: read the store, create nothing
    try:
        assert os.access(store.parent, os.W_OK) is False
        entered = False
        async with oauth_store_lock(store, timeout_seconds=1.0, label="codex auth"):
            entered = True
        assert entered
    finally:
        store.parent.chmod(stat.S_IRWXU)


@pytest.mark.asyncio
@requires_nonroot
async def test_propagates_when_only_the_lock_file_is_unwritable(store: Path):
    """Writable directory, foreign/read-only lock file: the peer holds the lock.

    This is the shape the errno cannot distinguish. Degrading here would run a
    second refresh body concurrently with whoever owns the lock file, which for
    a rotating token means one rotation is necessarily lost.
    """
    lock_path = store.with_suffix(".lock")
    lock_path.write_text("")
    lock_path.chmod(stat.S_IRUSR)  # 0400: not writable by us

    # The store itself remains writable — that is the point.
    tmp = store.parent / ".probe.tmp"
    tmp.write_text("x")
    os.replace(tmp, store)

    try:
        with pytest.raises(OSError) as excinfo:
            async with oauth_store_lock(store, timeout_seconds=1.0, label="codex auth"):
                pass
    finally:
        lock_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    assert excinfo.value.errno == errno.EACCES


# ---------------------------------------------------------------------------
# Branch logic, injected so it also runs as root.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("error_number", [errno.EROFS, errno.EACCES, errno.EPERM])
async def test_degrades_when_the_store_directory_cannot_be_written(store: Path, error_number: int):
    """Errno in the set + directory not writable => degrade to per-loop only."""
    entered = False
    with _fail_lock_creation(store, error_number, parent_writable=False):
        async with oauth_store_lock(store, timeout_seconds=1.0, label="codex auth"):
            entered = True

    assert entered, f"{errno.errorcode[error_number]} on an unwritable dir must not prevent entry"
    assert not store.with_suffix(".lock").exists()


@pytest.mark.asyncio
async def test_propagates_when_the_directory_is_writable(store: Path):
    """Same errno, writable directory => the failure is not about the store."""
    with _fail_lock_creation(store, errno.EACCES, parent_writable=True), pytest.raises(OSError) as excinfo:
        async with oauth_store_lock(store, timeout_seconds=1.0, label="codex auth"):
            pass

    assert excinfo.value.errno == errno.EACCES


@pytest.mark.asyncio
@pytest.mark.parametrize("error_number", [errno.ENOSPC, errno.EMFILE, errno.ENFILE])
async def test_propagates_on_a_writable_store(store: Path, error_number: int):
    """A full disk or an exhausted descriptor table is not a read-only store.

    Those hit a perfectly writable store, and they arrive when the box is under
    pressure. Swallowing them would drop the cross-process lock and admit two
    concurrent refreshes of a rotating token, one of which is necessarily lost.
    """
    with _fail_lock_creation(store, error_number, parent_writable=True), pytest.raises(OSError) as excinfo:
        async with oauth_store_lock(store, timeout_seconds=1.0, label="codex auth"):
            pass

    assert excinfo.value.errno == error_number


# ---------------------------------------------------------------------------
# The fallback must still serialise this interpreter's own callers.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_loop_lock_still_serialises_when_degraded(store: Path):
    """Degrading drops the FILE lock only — one loop's callers still queue."""
    concurrent = 0
    max_concurrent = 0

    async def hold() -> None:
        nonlocal concurrent, max_concurrent
        async with oauth_store_lock(store, timeout_seconds=1.0, label="codex auth"):
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            for _ in range(3):
                await asyncio.sleep(0)
            concurrent -= 1

    with _fail_lock_creation(store, errno.EROFS, parent_writable=False):
        await asyncio.gather(*(hold() for _ in range(4)))

    assert max_concurrent == 1, f"{max_concurrent} callers entered the guarded section at once"


@pytest.mark.asyncio
async def test_body_errors_are_not_chained_to_the_degrade_oserror(store: Path):
    """An error from the refresh body must not read as raised while handling EROFS."""
    with _fail_lock_creation(store, errno.EROFS, parent_writable=False), pytest.raises(RuntimeError) as excinfo:
        async with oauth_store_lock(store, timeout_seconds=1.0, label="codex auth"):
            raise RuntimeError("refresh failed")

    assert excinfo.value.__context__ is None
