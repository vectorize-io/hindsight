"""End-to-end tests for the Hindsight-Composio integration.

Exercises the retain/recall/reflect custom tools against a live Hindsight
server, driving them through the same ``(input, ctx)`` call path Composio uses
when it invokes a custom tool. The tools talk to Hindsight directly (the
server's LLM does fact extraction), so only a running Hindsight instance is
required — no provider key. By default these tests are skipped; point
``HINDSIGHT_API_URL`` at a reachable server to enable them.

The whole module is the real-LLM bucket (``requires_real_llm``).
"""

from __future__ import annotations

import os
import time
import urllib.request
import uuid

import pytest
from hindsight_client import Hindsight
from hindsight_composio import (
    RecallInput,
    ReflectInput,
    RetainInput,
    register_hindsight_tools,
)

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


class _Ctx:
    """Stand-in for the Composio SessionContext injected into a tool call."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id


class _NoopComposio:
    """Identity ``tool`` decorator — returns the bare function unchanged."""

    class _Experimental:
        def tool(self, fn):
            return fn

    def __init__(self) -> None:
        self.experimental = self._Experimental()


def _by_name(client):
    tools = register_hindsight_tools(_NoopComposio(), client=client)
    return {t.__name__: t for t in tools}


def _recall_until_nonempty(recall, ctx, query, attempts=12, delay=1.0):
    for _ in range(attempts):
        result = recall(RecallInput(query=query), ctx)
        if result["count"] > 0:
            return result["memories"]
        time.sleep(delay)
    pytest.fail(
        f"recall({query!r}) returned no memories after {attempts * delay:.0f}s — "
        "either retain failed to surface or the query no longer matches."
    )


@pytest.fixture
def live():
    client = Hindsight(base_url=HINDSIGHT_API_URL)
    bank_id = f"composio-e2e-{uuid.uuid4().hex[:8]}"
    try:
        yield client, bank_id
    finally:
        try:
            client.delete_bank(bank_id)
        except Exception:
            pass


class TestE2ETools:
    def test_retain_and_recall_roundtrip(self, live):
        client, bank_id = live
        tools = _by_name(client)
        ctx = _Ctx(user_id=bank_id)

        result = tools["hindsight_retain"](
            RetainInput(content="The team uses PostgreSQL 16 and deploys to us-east-1."),
            ctx,
        )
        assert result == {"status": "stored", "bank": bank_id}

        memories = _recall_until_nonempty(tools["hindsight_recall"], ctx, "What technologies does the team use?")
        joined = " ".join(memories).lower()
        assert "postgresql" in joined or "us-east-1" in joined, (
            f"recall surfaced results but none referenced the stored content: {memories}"
        )

    def test_reflect_synthesizes_from_memory(self, live):
        client, bank_id = live
        tools = _by_name(client)
        ctx = _Ctx(user_id=bank_id)

        tools["hindsight_retain"](
            RetainInput(content="The team uses PostgreSQL 16 and deploys to us-east-1."),
            ctx,
        )
        _recall_until_nonempty(tools["hindsight_recall"], ctx, "What technologies does the team use?")

        result = tools["hindsight_reflect"](
            ReflectInput(query="What do I know about the team's tech stack?"),
            ctx,
        )
        answer = result["answer"]
        assert answer and answer != _NO_MEMORIES, "reflect should synthesise non-empty text"
        lowered = answer.lower()
        assert "postgresql" in lowered or "us-east" in lowered, (
            f"reflect text didn't reference the stored memory: {answer[:300]}"
        )

    def test_recall_empty_bank(self, live):
        client, bank_id = live
        tools = _by_name(client)
        result = tools["hindsight_recall"](RecallInput(query="anything at all"), _Ctx(user_id=bank_id))
        assert result == {"memories": [], "count": 0}
