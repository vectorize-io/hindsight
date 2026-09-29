"""Smoke test for the Hermes Agent ↔ Hindsight integration in embedded mode.

Drives the `HindsightMemoryProvider` plugin shipped with Hermes Agent against
a locally-spawned Hindsight Embedded daemon, exercising the full retain →
recall roundtrip end-to-end.

Run on demand via the installed Hermes venv (it already has every dep —
hermes-agent's plugin code, hindsight_embed, hindsight_client, pytest):

    HINDSIGHT_LLM_API_KEY=... \
        ~/.hermes/hermes-agent/venv/bin/python -m pytest \
        hindsight-integration-tests/tests/test_hermes_embedded_smoke.py -v -s

Skipped automatically if `HINDSIGHT_LLM_API_KEY` (or `OPENAI_API_KEY`) is not
set, since embedded mode needs an LLM to extract facts during retain.

Defaults: openai / gpt-4o-mini. Override via `HINDSIGHT_LLM_PROVIDER` and
`HINDSIGHT_LLM_MODEL`.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytest


HERMES_VENV_SITE = Path.home() / ".hermes" / "hermes-agent"
LLM_API_KEY = os.environ.get("HINDSIGHT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
LLM_PROVIDER = os.environ.get("HINDSIGHT_LLM_PROVIDER", "openai")
LLM_MODEL = os.environ.get("HINDSIGHT_LLM_MODEL", "gpt-4o-mini")

pytestmark = [
    pytest.mark.skipif(
        not LLM_API_KEY,
        reason="HINDSIGHT_LLM_API_KEY (or OPENAI_API_KEY) not set",
    ),
    pytest.mark.skipif(
        not (HERMES_VENV_SITE / "plugins" / "memory" / "hindsight" / "__init__.py").exists(),
        reason=f"Hermes plugin not found at {HERMES_VENV_SITE} — run `hermes update` first",
    ),
]


@pytest.fixture(scope="module")
def hermes_path():
    """Make the installed hermes-agent importable in this process."""
    if str(HERMES_VENV_SITE) not in sys.path:
        sys.path.insert(0, str(HERMES_VENV_SITE))


@pytest.fixture
def embedded_provider(tmp_path, monkeypatch, hermes_path):
    """Spin up a HindsightMemoryProvider in local_embedded mode.

    Uses a temp HERMES_HOME so we never touch the user's real ~/.hermes.
    The Hindsight daemon stores its data under that temp dir too.
    """
    profile_name = f"hermes-smoke-{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HINDSIGHT_LLM_API_KEY", LLM_API_KEY)

    config_dir = tmp_path / "hindsight"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "mode": "local_embedded",
        "profile": profile_name,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "llm_api_key": LLM_API_KEY,
        "bank_id": f"smoke-{uuid.uuid4().hex[:8]}",
        "recall_budget": "low",
        "auto_retain": True,
        "auto_recall": True,
        "retain_async": False,
        "retain_every_n_turns": 1,
    }
    (config_dir / "config.json").write_text(json.dumps(config, indent=2))

    from plugins.memory.hindsight import HindsightMemoryProvider

    provider = HindsightMemoryProvider()
    provider.initialize(session_id=f"smoke-{uuid.uuid4().hex[:8]}", platform="cli")

    # The plugin starts the daemon on a background thread. Force a synchronous
    # boot here so the test isn't racing it. First-run setup can take ~2 min
    # because the embedded daemon installs its own deps into a profile venv.
    deadline = time.time() + 240.0
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            client = provider._get_client()
            client._ensure_started()
            if client.is_running:
                break
        except Exception as exc:
            last_err = exc
        time.sleep(2.0)
    else:
        provider.shutdown()
        pytest.fail(f"Hindsight embedded daemon did not start within 240s (last error: {last_err!r})")

    yield provider

    try:
        provider.shutdown()
    except Exception:
        pass
    try:
        from hindsight_embed.daemon_embed_manager import DaemonEmbedManager

        DaemonEmbedManager().stop(profile_name)
    except Exception:
        pass


def test_retain_then_recall_roundtrip(embedded_provider):
    """Store a memorable fact, then verify recall finds it.

    This exercises the full Hermes plugin path: sync_turn -> aretain_batch ->
    daemon -> LLM fact extraction -> indexing -> recall -> prefetch.
    """
    fact = "The user's favorite programming language is Rust"
    embedded_provider.sync_turn(
        user_content="What's my favorite programming language?",
        assistant_content=fact,
    )
    if embedded_provider._sync_thread:
        embedded_provider._sync_thread.join(timeout=60.0)

    deadline = time.time() + 60.0
    last_result = ""
    while time.time() < deadline:
        embedded_provider._prefetch_result = ""
        embedded_provider.queue_prefetch("favorite programming language")
        if embedded_provider._prefetch_thread:
            embedded_provider._prefetch_thread.join(timeout=30.0)
        last_result = embedded_provider._prefetch_result
        if "rust" in last_result.lower():
            return
        time.sleep(2.0)

    pytest.fail(
        f"recall did not surface the stored fact within 60s. "
        f"Last prefetch result: {last_result!r}"
    )
