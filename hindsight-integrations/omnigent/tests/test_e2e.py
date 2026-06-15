"""End-to-end tests for the Hindsight-Omnigent integration.

Exercises the retain/recall/reflect tool callables against a live Hindsight
server. The callables talk to Hindsight directly (the server's LLM does fact
extraction), so only a running Hindsight instance is required — no provider key.
By default these tests are skipped; point ``HINDSIGHT_API_URL`` at a reachable
server to enable them.

The whole module is the real-LLM bucket (``requires_real_llm``).
"""

from __future__ import annotations

import os
import time
import urllib.request
import uuid

import pytest
from hindsight_client import Hindsight
from hindsight_omnigent import configure, recall, reflect, reset_config, retain
from hindsight_omnigent.tools import _reset_created_banks

HINDSIGHT_API_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
_NO_MEMORIES_RECALL = "No relevant memories found."


def _hindsight_available() -> bool:
    try:
        with urllib.request.urlopen(f"{HINDSIGHT_API_URL}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


requires_hindsight = pytest.mark.skipif(
    not _hindsight_available(),
    reason=f"Hindsight not reachable at {HINDSIGHT_API_URL}",
)

pytestmark = [requires_hindsight, pytest.mark.requires_real_llm]


def _recall_until_nonempty(query, attempts=12, delay=1.0):
    for _ in range(attempts):
        out = recall(query=query)
        if out and out != _NO_MEMORIES_RECALL:
            return out
        time.sleep(delay)
    pytest.fail(
        f"recall({query!r}) returned no memories after {attempts * delay:.0f}s — "
        "either retain failed to surface or the query no longer matches."
    )


@pytest.fixture
def live():
    reset_config()
    _reset_created_banks()
    bank_id = f"omnigent-e2e-{uuid.uuid4().hex[:8]}"
    configure(hindsight_api_url=HINDSIGHT_API_URL, bank_id=bank_id)
    client = Hindsight(base_url=HINDSIGHT_API_URL)
    try:
        yield bank_id
    finally:
        try:
            client.delete_bank(bank_id)
        except Exception:
            pass
        reset_config()
        _reset_created_banks()


class TestE2ETools:
    def test_retain_and_recall_roundtrip(self, live):
        assert retain(content="The team uses PostgreSQL 16 and deploys to us-east-1.") == (
            "Stored to long-term memory."
        )
        recalled = _recall_until_nonempty("What technologies does the team use?")
        lowered = recalled.lower()
        assert "postgresql" in lowered or "us-east-1" in lowered, (
            f"recall surfaced results but none referenced the stored content: {recalled}"
        )

    def test_reflect_synthesizes_from_memory(self, live):
        retain(content="The team uses PostgreSQL 16 and deploys to us-east-1.")
        _recall_until_nonempty("What technologies does the team use?")
        answer = reflect(query="What do I know about the team's tech stack?")
        assert answer and answer != _NO_MEMORIES_RECALL, "reflect should synthesise non-empty text"
        lowered = answer.lower()
        assert "postgresql" in lowered or "us-east" in lowered, (
            f"reflect text didn't reference the stored memory: {answer[:300]}"
        )

    def test_recall_empty_bank(self, live):
        assert recall(query="anything at all") == _NO_MEMORIES_RECALL
