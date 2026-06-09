"""End-to-end tests for the Hindsight-LlamaIndex integration.

Exercises the retain/recall/reflect tools and the ``HindsightMemory`` adapter
against a live Hindsight server. The integration talks to Hindsight directly
(the server's LLM does fact extraction), so only a running Hindsight instance
is required — no provider key. Set ``HINDSIGHT_API_URL`` to enable.

The whole module is the real-LLM bucket (``requires_real_llm``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import urllib.request
import uuid

import pytest
from hindsight_client import Hindsight

from hindsight_llamaindex import HindsightMemory, create_hindsight_tools

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
    return next(t for t in tools if t.metadata.name == name)


def _recall_until_nonempty(recall_tool, query, attempts=12, delay=1.0):
    for _ in range(attempts):
        result = str(recall_tool.call(query=query))
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
    bank_id = f"llamaindex-e2e-{uuid.uuid4().hex[:8]}"
    client.create_bank(bank_id, name=f"LlamaIndex E2E {bank_id}")
    try:
        yield client, bank_id
    finally:
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(client.adelete_bank(bank_id))
            loop.run_until_complete(client.aclose())
            loop.close()
        except Exception as e:
            logging.getLogger(__name__).warning("E2E teardown failed: %s", e)


class TestE2ETools:
    def test_retain_and_recall_roundtrip(self, live):
        client, bank_id = live
        tools = create_hindsight_tools(bank_id=bank_id, client=client)
        retain, recall = _tool(tools, "retain_memory"), _tool(tools, "recall_memory")

        result = str(retain.call(content="The team uses PostgreSQL 16 and deploys to us-east-1."))
        assert "stored successfully" in result, f"unexpected retain result: {result}"

        recalled = _recall_until_nonempty(recall, "What technologies does the team use?")
        lowered = recalled.lower()
        assert "postgresql" in lowered or "us-east-1" in lowered, (
            f"recall surfaced results but none referenced the stored content: {recalled}"
        )

    def test_reflect_synthesizes_from_memory(self, live):
        client, bank_id = live
        tools = create_hindsight_tools(bank_id=bank_id, client=client)
        retain = _tool(tools, "retain_memory")
        recall = _tool(tools, "recall_memory")
        reflect = _tool(tools, "reflect_on_memory")

        retain.call(content="The team uses PostgreSQL 16 and deploys to us-east-1.")
        _recall_until_nonempty(recall, "What technologies does the team use?")

        result = str(reflect.call(query="What do I know about the team's tech stack?"))
        assert result and result != _NO_MEMORIES, "reflect should synthesise non-empty text"
        lowered = result.lower()
        assert "postgresql" in lowered or "us-east" in lowered, (
            f"reflect text didn't reference the stored memory: {result[:300]}"
        )

    def test_recall_empty_bank(self, live):
        client, bank_id = live
        tools = create_hindsight_tools(bank_id=bank_id, client=client, include_retain=False, include_reflect=False)
        result = str(_tool(tools, "recall_memory").call(query="anything at all"))
        assert result == _NO_MEMORIES


class TestE2EHindsightMemory:
    def test_put_and_get_roundtrip(self, live):
        from llama_index.core.llms import ChatMessage, MessageRole

        client, bank_id = live
        memory = HindsightMemory(bank_id=bank_id, client=client, chat_history_limit=10)

        memory.put(
            ChatMessage(
                role=MessageRole.USER,
                content="My favourite database is PostgreSQL 16 for the new platform.",
            )
        )
        # Allow fact extraction to surface; aget queries Hindsight via recall.
        last = None
        for _ in range(12):
            last = memory.aget(input="favourite database")
            if asyncio.iscoroutine(last):
                last = asyncio.new_event_loop().run_until_complete(last)
            joined = " ".join(getattr(m, "content", "") or "" for m in (last or [])).lower()
            if "postgresql" in joined:
                return
            time.sleep(1.0)
        pytest.fail("HindsightMemory.aget never surfaced the retained PostgreSQL fact after ~12s")
