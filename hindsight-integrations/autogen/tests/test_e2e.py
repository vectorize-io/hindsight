"""End-to-end tests for the Hindsight-AutoGen integration.

Exercises the retain/recall/reflect tools against a live Hindsight server. The
tools talk to Hindsight directly (the server's LLM does fact extraction), so
only a running Hindsight instance is required — no provider key. By default
these tests are skipped; point ``HINDSIGHT_API_URL`` at a reachable server to
enable them.

The whole module is the real-LLM bucket (``requires_real_llm``).
"""

from __future__ import annotations

import asyncio
import os
import urllib.request
import uuid

import pytest
import pytest_asyncio
from autogen_core import CancellationToken
from hindsight_client import Hindsight

from hindsight_autogen import create_hindsight_tools

HINDSIGHT_API_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
_NO_MEMORIES = "No relevant memories found."


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


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


async def _run(tool, args: dict) -> str:
    return str(await tool.run_json(args, CancellationToken()))


async def _recall_until_nonempty(recall_tool, query, attempts=12, delay=1.0):
    for _ in range(attempts):
        text = await _run(recall_tool, {"query": query})
        if text and text != _NO_MEMORIES:
            return text
        await asyncio.sleep(delay)
    pytest.fail(
        f"recall({query!r}) returned no memories after {attempts * delay:.0f}s — "
        "either retain failed to surface or the query no longer matches."
    )


@pytest_asyncio.fixture
async def live():
    client = Hindsight(base_url=HINDSIGHT_API_URL)
    bank_id = f"autogen-e2e-{uuid.uuid4().hex[:8]}"
    await client.acreate_bank(bank_id, name=f"AutoGen E2E {bank_id}")
    try:
        yield client, bank_id
    finally:
        try:
            await client.adelete_bank(bank_id)
        except Exception:
            pass
        await client.aclose()


class TestE2ETools:
    @pytest.mark.asyncio
    async def test_retain_and_recall_roundtrip(self, live):
        client, bank_id = live
        tools = create_hindsight_tools(bank_id=bank_id, client=client)
        retain, recall = _tool(tools, "hindsight_retain"), _tool(tools, "hindsight_recall")

        result = await _run(retain, {"content": "The team uses PostgreSQL 16 and deploys to us-east-1."})
        assert result == "Memory stored successfully."

        recalled = await _recall_until_nonempty(recall, "What technologies does the team use?")
        lowered = recalled.lower()
        assert "postgresql" in lowered or "us-east-1" in lowered, (
            f"recall surfaced results but none referenced the stored content: {recalled}"
        )

    @pytest.mark.asyncio
    async def test_reflect_synthesizes_from_memory(self, live):
        client, bank_id = live
        tools = create_hindsight_tools(bank_id=bank_id, client=client)
        retain = _tool(tools, "hindsight_retain")
        recall = _tool(tools, "hindsight_recall")
        reflect = _tool(tools, "hindsight_reflect")

        await _run(retain, {"content": "The team uses PostgreSQL 16 and deploys to us-east-1."})
        await _recall_until_nonempty(recall, "What technologies does the team use?")

        result = await _run(reflect, {"query": "What do I know about the team's tech stack?"})
        assert result and result != _NO_MEMORIES, "reflect should synthesise non-empty text"
        lowered = result.lower()
        assert "postgresql" in lowered or "us-east" in lowered, (
            f"reflect text didn't reference the stored memory: {result[:300]}"
        )

    @pytest.mark.asyncio
    async def test_recall_empty_bank(self, live):
        client, bank_id = live
        tools = create_hindsight_tools(bank_id=bank_id, client=client, include_retain=False, include_reflect=False)
        result = await _run(_tool(tools, "hindsight_recall"), {"query": "anything at all"})
        assert result == _NO_MEMORIES
