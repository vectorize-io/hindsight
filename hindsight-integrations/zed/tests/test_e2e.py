"""End-to-end: drive the daemon pipeline against a live Hindsight server.

Exercises the real recall + retain HTTP path (no real Zed needed — we build a
thread in memory). Gated behind HINDSIGHT_API_URL and marked requires_real_llm,
so it is excluded from the deterministic PR-CI bucket.
"""

import os
import time
import uuid

import pytest

from hindsight_zed.client import HindsightClient
from hindsight_zed.config import ZedConfig
from hindsight_zed.daemon import process_thread
from hindsight_zed.rules_file import BEGIN_MARKER
from hindsight_zed.state import DaemonState
from hindsight_zed.threads_db import ThreadMessage, ZedThread

pytestmark = pytest.mark.requires_real_llm

API_URL = os.environ.get("HINDSIGHT_API_URL")


@pytest.mark.skipif(not API_URL, reason="HINDSIGHT_API_URL not set")
def test_retain_then_recall_roundtrip(tmp_path):
    project = tmp_path / f"zed-e2e-{uuid.uuid4().hex[:8]}"
    project.mkdir()
    client = HindsightClient(API_URL, os.environ.get("HINDSIGHT_API_TOKEN"))
    cfg = ZedConfig(bank_prefix=f"zed-e2e-{uuid.uuid4().hex[:6]}")
    state = DaemonState(path=tmp_path / "s.json")

    # Turn 1 — retain a clear, recallable fact (recall has nothing yet).
    t1 = ZedThread(
        id="e2e-1",
        title="prefs",
        updated_at="2026-06-10T10:00:00Z",
        messages=[
            ThreadMessage("user", "Remember that my favorite language is Haskell."),
            ThreadMessage("assistant", "Got it — Haskell."),
        ],
        folder_paths=[str(project)],
    )
    process_thread(t1, client, cfg, state)

    # Turn 2 — after extraction settles, a new thread's recall should surface it
    # into the project's instruction file.
    found = False
    for _ in range(20):
        time.sleep(3)
        t2 = ZedThread(
            id="e2e-2",
            title="q",
            updated_at="2026-06-10T10:05:00Z",
            messages=[ThreadMessage("user", "What is my favorite language?")],
            folder_paths=[str(project)],
        )
        process_thread(t2, client, cfg, state)
        rules = project / ".rules"
        if rules.is_file() and "haskell" in rules.read_text().lower():
            assert BEGIN_MARKER in rules.read_text()
            found = True
            break

    assert found, "retained memory was not recalled into the project's .rules within the timeout"
