"""End-to-end tests for the Hindsight-Haystack integration.

Exercises the retain/recall/reflect tools against a live Hindsight server. The
tools talk to Hindsight directly (the server's LLM does fact extraction), so
only a running Hindsight instance is required — no provider key. By default
these tests are skipped; point ``HINDSIGHT_API_URL`` at a reachable server to
enable them.

Run with::

    uv run pytest tests/test_e2e.py -v

The whole module is the real-LLM bucket (``requires_real_llm``): it depends on
the Hindsight server's LLM-backed fact extraction, so it is excluded from the
deterministic PR-CI bucket and run on its own / nightly.
"""

from __future__ import annotations

import logging
import os
import time
import uuid

import pytest
import requests
from hindsight_client import Hindsight

from hindsight_haystack import create_hindsight_tools
from hindsight_haystack.tools import _run_sync

HINDSIGHT_API_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
_NO_MEMORIES = "No relevant memories found."


def _hindsight_available() -> bool:
    try:
        return requests.get(f"{HINDSIGHT_API_URL}/health", timeout=3).status_code == 200
    except Exception:
        return False


requires_hindsight = pytest.mark.skipif(
    not _hindsight_available(),
    reason=f"Hindsight not reachable at {HINDSIGHT_API_URL}",
)

# Real-LLM / real-service bucket: depends on a live Hindsight server. Excluded
# from PR CI via `-m "not requires_real_llm"`; the skipif still gates runtime.
pytestmark = [requires_hindsight, pytest.mark.requires_real_llm]


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


def _recall_until_nonempty(recall_tool, query, attempts=12, delay=1.0):
    """Poll the recall tool until it surfaces a memory (retain takes a moment
    to flow through fact extraction + indexing)."""
    for _ in range(attempts):
        result = recall_tool.invoke(query=query)
        if result and result != _NO_MEMORIES:
            return result
        time.sleep(delay)
    pytest.fail(
        f"recall({query!r}) returned no memories after {attempts * delay:.0f}s — "
        "either retain failed to surface or the query no longer matches."
    )


@pytest.fixture
def live():
    client = Hindsight(base_url=HINDSIGHT_API_URL)
    bank_id = f"haystack-e2e-{uuid.uuid4().hex[:8]}"
    # Run all client I/O on the tools' background loop so the aiohttp session is
    # created and closed on the same loop (avoids unclosed-connector warnings).
    _run_sync(client.acreate_bank(bank_id, name=f"Haystack E2E {bank_id}"))
    try:
        yield client, bank_id
    finally:
        try:
            _run_sync(client.adelete_bank(bank_id))
        except Exception as e:
            logging.getLogger(__name__).warning("E2E bank cleanup failed: %s", e)
        try:
            _run_sync(client.aclose())
        except Exception as e:
            logging.getLogger(__name__).warning("E2E client close failed: %s", e)


class TestE2ETools:
    def test_retain_and_recall_roundtrip(self, live):
        client, bank_id = live
        tools = create_hindsight_tools(bank_id=bank_id, client=client)
        retain, recall = _tool(tools, "retain_memory"), _tool(tools, "recall_memory")

        assert retain.invoke(content="The team uses PostgreSQL 16 and deploys to us-east-1.") == (
            "Memory stored successfully."
        )
        result = _recall_until_nonempty(recall, "What technologies does the team use?")
        lowered = result.lower()
        assert "postgresql" in lowered or "us-east-1" in lowered, (
            f"recall surfaced results but none referenced the stored content: {result}"
        )

    def test_reflect_synthesizes_from_memory(self, live):
        client, bank_id = live
        tools = create_hindsight_tools(bank_id=bank_id, client=client)
        retain, recall, reflect = (
            _tool(tools, "retain_memory"),
            _tool(tools, "recall_memory"),
            _tool(tools, "reflect_on_memory"),
        )

        retain.invoke(content="The team uses PostgreSQL 16 and deploys to us-east-1.")
        _recall_until_nonempty(recall, "What technologies does the team use?")

        result = reflect.invoke(query="What do I know about the team's tech stack?")
        assert result and result != _NO_MEMORIES, "reflect should synthesise non-empty text"
        lowered = result.lower()
        assert "postgresql" in lowered or "us-east" in lowered, (
            f"reflect text didn't reference the stored memory: {result[:300]}"
        )

    def test_recall_empty_bank(self, live):
        client, bank_id = live
        tools = create_hindsight_tools(
            bank_id=bank_id, client=client, include_retain=False, include_reflect=False
        )
        result = _tool(tools, "recall_memory").invoke(query="anything at all")
        assert result == _NO_MEMORIES
