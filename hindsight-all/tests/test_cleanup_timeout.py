"""
Unit test for _cleanup lock timeout behavior.

Verifies that _cleanup completes even when the lock is held by another thread,
instead of hanging indefinitely (fixes #952).
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest


def test_cleanup_completes_when_lock_held():
    """
    _cleanup should complete (best-effort) even when self._lock is held
    by another thread, e.g. during a long _ensure_started call.
    """
    with patch.dict("sys.modules", {
        "hindsight_client": MagicMock(),
        "hindsight_embed": MagicMock(),
        "hindsight.api_namespaces": MagicMock(),
    }):
        from hindsight.embedded import HindsightEmbedded

        client = HindsightEmbedded.__new__(HindsightEmbedded)
        client.profile = "test"
        client._lock = threading.Lock()
        client._closed = False
        client._client = None
        client._started = False
        client._ui = False

        # Simulate another thread holding the lock
        client._lock.acquire()

        cleanup_done = threading.Event()

        def run_cleanup():
            client._cleanup()
            cleanup_done.set()

        t = threading.Thread(target=run_cleanup)
        t.start()

        # Cleanup should complete within the timeout (5s) + margin
        assert cleanup_done.wait(timeout=8.0), (
            "_cleanup hung instead of timing out on lock acquisition"
        )

        # Release the lock from the simulating thread
        client._lock.release()
        t.join(timeout=1.0)

        assert client._closed, "Client should be marked as closed after cleanup"
