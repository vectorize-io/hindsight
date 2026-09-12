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

Failures are injected into the lock path's own ``open``/``mkdir`` rather than
produced with ``chmod``: a restrictive mode is a no-op for root, which CI
containers routinely run as, so a ``chmod``-based test would pass whether or not
the code handled the read-only case. Injection is deterministic for every user.
"""

from __future__ import annotations

import builtins
import errno
from pathlib import Path
from unittest.mock import patch

import pytest

from hindsight_api.engine.providers.oauth_store_lock import oauth_store_lock


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A credential store file, as the OAuth managers write it."""
    path = tmp_path / "provider-home" / "auth.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"tokens": {"access_token": "a", "refresh_token": "r"}}')
    return path


def _fail_lock_creation(store: Path, error_number: int):
    """Make creating ``<store>.lock`` fail with ``error_number``.

    Reads and writes of the store itself keep working, so the store behaves like
    a projection that accepts nothing new beside the credential.
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

    return patch("builtins.open", fake_open), patch.object(Path, "mkdir", fake_mkdir)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_number", [errno.EROFS, errno.EACCES, errno.EPERM])
async def test_degrades_when_the_store_cannot_be_written(store: Path, error_number: int):
    """A store that cannot be written degrades to per-loop-only instead of raising."""
    p_open, p_mkdir = _fail_lock_creation(store, error_number)
    entered = False
    with p_open, p_mkdir:
        async with oauth_store_lock(store, timeout_seconds=1.0, label="codex auth"):
            entered = True

    assert entered, f"{errno.errorcode[error_number]} must not prevent the guarded section"
    assert not store.with_suffix(".lock").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_number", [errno.ENOSPC, errno.EMFILE, errno.ENFILE])
async def test_propagates_when_the_store_is_writable_but_the_create_fails(store: Path, error_number: int):
    """A full disk or an exhausted descriptor table is not a read-only store.

    Those failures hit a perfectly writable store, and they arrive when the box
    is under pressure. Swallowing them would drop the cross-process lock and
    admit two concurrent refreshes of a rotating token, where one is necessarily
    lost.
    """
    p_open, p_mkdir = _fail_lock_creation(store, error_number)
    with p_open, p_mkdir, pytest.raises(OSError) as excinfo:
        async with oauth_store_lock(store, timeout_seconds=1.0, label="codex auth"):
            pass

    assert excinfo.value.errno == error_number


@pytest.mark.asyncio
async def test_per_loop_lock_still_serialises_when_degraded(store: Path):
    """Degrading drops the FILE lock only — one loop's callers still queue.

    That is what makes the fallback safe: the fallback's justification is that
    the file lock was never the protection against an external writer (such a
    writer does not take it), while same-process callers remain serialised.
    """
    p_open, p_mkdir = _fail_lock_creation(store, errno.EROFS)
    concurrent = 0
    max_concurrent = 0

    async def hold() -> None:
        nonlocal concurrent, max_concurrent
        async with oauth_store_lock(store, timeout_seconds=1.0, label="codex auth"):
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            # Yield control so a second caller would overlap if unsynchronised.
            for _ in range(3):
                import asyncio

                await asyncio.sleep(0)

            concurrent -= 1

    import asyncio

    with p_open, p_mkdir:
        await asyncio.gather(*(hold() for _ in range(4)))

    assert max_concurrent == 1, f"{max_concurrent} callers entered the guarded section at once"
