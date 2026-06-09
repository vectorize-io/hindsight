"""End-to-end tests for the Hindsight-Claude-Agent-SDK integration.

Exercises the retain/recall/reflect MCP tools against a live Hindsight server.
The tools talk to Hindsight directly (the server's LLM does fact extraction),
so only a running Hindsight instance is required — no provider key. By default
these tests are skipped; point ``HINDSIGHT_API_URL`` at a reachable server to
enable them.

Run with::

    uv run pytest tests/test_e2e.py -v

The whole module is the real-LLM bucket (``requires_real_llm``): it depends on
the Hindsight server's LLM-backed fact extraction, so it is excluded from the
deterministic PR-CI bucket and run on its own / nightly.
"""

from __future__ import annotations

import asyncio
import os
import urllib.request
import uuid

import pytest
import pytest_asyncio
from hindsight_client import Hindsight

from hindsight_claude_agent_sdk import create_hindsight_tools

HINDSIGHT_API_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
_NO_MEMORIES = "No relevant memories found."


def _hindsight_available() -> bool:
    # Use stdlib urllib (no `requests` dependency in this integration).
    try:
        with urllib.request.urlopen(f"{HINDSIGHT_API_URL}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


requires_hindsight = pytest.mark.skipif(
    not _hindsight_available(),
    reason=f"Hindsight not reachable at {HINDSIGHT_API_URL}",
)

# Real-LLM / real-service bucket: depends on a live Hindsight server. Excluded
# from PR CI via `-m "not requires_real_llm"`; the skipif still gates runtime.
pytestmark = [requires_hindsight, pytest.mark.requires_real_llm]


def _text(result) -> str:
    """Extract the text payload from an MCP tool result."""
    return result["content"][0]["text"]


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


@pytest_asyncio.fixture
async def live():
    client = Hindsight(base_url=HINDSIGHT_API_URL)
    bank_id = f"claude-agent-sdk-e2e-{uuid.uuid4().hex[:8]}"
    await client.acreate_bank(bank_id, name=f"Claude Agent SDK E2E {bank_id}")
    try:
        yield client, bank_id
    finally:
        try:
            await client.adelete_bank(bank_id)
        except Exception:
            pass
        await client.aclose()


async def _recall_until_nonempty(recall_tool, query, attempts=12, delay=1.0):
    """Poll the recall tool until it surfaces a memory (retain takes a moment
    to flow through fact extraction + indexing)."""
    for _ in range(attempts):
        text = _text(await recall_tool.handler({"query": query}))
        if text and text != _NO_MEMORIES:
            return text
        await asyncio.sleep(delay)
    pytest.fail(
        f"recall({query!r}) returned no memories after {attempts * delay:.0f}s — "
        "either retain failed to surface or the query no longer matches."
    )


class TestE2ETools:
    @pytest.mark.asyncio
    async def test_retain_and_recall_roundtrip(self, live):
        client, bank_id = live
        tools = create_hindsight_tools(bank_id=bank_id, client=client)
        retain, recall = _tool(tools, "hindsight_retain"), _tool(tools, "hindsight_recall")

        stored = _text(await retain.handler({"content": "The team uses PostgreSQL 16 and deploys to us-east-1."}))
        assert stored == "Memory stored successfully."

        result = await _recall_until_nonempty(recall, "What technologies does the team use?")
        lowered = result.lower()
        assert "postgresql" in lowered or "us-east-1" in lowered, (
            f"recall surfaced results but none referenced the stored content: {result}"
        )

    @pytest.mark.asyncio
    async def test_reflect_synthesizes_from_memory(self, live):
        client, bank_id = live
        tools = create_hindsight_tools(bank_id=bank_id, client=client)
        retain = _tool(tools, "hindsight_retain")
        recall = _tool(tools, "hindsight_recall")
        reflect = _tool(tools, "hindsight_reflect")

        await retain.handler({"content": "The team uses PostgreSQL 16 and deploys to us-east-1."})
        await _recall_until_nonempty(recall, "What technologies does the team use?")

        result = _text(await reflect.handler({"query": "What do I know about the team's tech stack?"}))
        assert result and result != _NO_MEMORIES, "reflect should synthesise non-empty text"
        lowered = result.lower()
        assert "postgresql" in lowered or "us-east" in lowered, (
            f"reflect text didn't reference the stored memory: {result[:300]}"
        )

    @pytest.mark.asyncio
    async def test_recall_empty_bank(self, live):
        client, bank_id = live
        tools = create_hindsight_tools(bank_id=bank_id, client=client, include_retain=False, include_reflect=False)
        result = _text(await _tool(tools, "hindsight_recall").handler({"query": "anything at all"}))
        assert result == _NO_MEMORIES
