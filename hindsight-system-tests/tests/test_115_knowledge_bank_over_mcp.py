"""An agent reaches a knowledge bank over MCP: it finds what was ingested over HTTP, and what it
ingests is found over HTTP.

A knowledge bank has its own MCP endpoint, /mcp/kb/{bank_id}/, beside the memory banks'
/mcp/{bank_id}/. The story is the seam: that endpoint must hand an agent the knowledge-bank tools
and nothing else, and the two transports must read and write the same documents.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from hindsight_system_tests import KnowledgeBanks

pytestmark = pytest.mark.asyncio


class _McpSession:
    """The JSON-RPC calls any MCP client makes, over streamable HTTP."""

    def __init__(self, http: httpx.AsyncClient, path: str) -> None:
        self._http = http
        self._path = path
        self._headers = {"Accept": "application/json, text/event-stream"}
        self._id = 0

    async def _post(self, method: str, params: dict[str, Any] | None = None, *, notify: bool = False) -> Any:
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notify:
            self._id += 1
            body["id"] = self._id
        response = await self._http.post(self._path, json=body, headers=self._headers)
        response.raise_for_status()
        if "mcp-session-id" in response.headers:
            self._headers["Mcp-Session-Id"] = response.headers["mcp-session-id"]
        if notify:
            return None
        if response.headers.get("content-type", "").startswith("text/event-stream"):
            data = [line[len("data:") :].strip() for line in response.text.splitlines() if line.startswith("data:")]
            message = json.loads(data[-1])
        else:
            message = response.json()
        assert "result" in message, message
        return message["result"]

    async def open(self) -> None:
        await self._post(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "system-test", "version": "1"},
            },
        )
        await self._post("notifications/initialized", notify=True)

    async def tools(self) -> set[str]:
        return {t["name"] for t in (await self._post("tools/list"))["tools"]}

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self._post("tools/call", {"name": name, "arguments": arguments})
        assert not result.get("isError"), result
        return json.loads(result["content"][0]["text"])


@pytest.fixture
async def knowledge_bank(client) -> AsyncIterator[str]:
    kb = KnowledgeBanks(client)
    bank_id = f"systest-kb-mcp-{uuid.uuid4().hex[:12]}"
    r = await kb.create(bank_id)
    assert r.status == 200, r.body
    yield bank_id
    await kb.delete(bank_id)


async def _session(http: httpx.AsyncClient, path: str) -> _McpSession:
    session = _McpSession(http, path)
    await session.open()
    return session


async def test_an_agent_on_the_banks_endpoint_gets_only_knowledge_tools(hindsight_server, knowledge_bank):
    async with httpx.AsyncClient(base_url=hindsight_server.url, timeout=60) as http:
        tools = await (await _session(http, f"/mcp/kb/{knowledge_bank}/")).tools()

    assert "search_knowledge_bank" in tools and "ingest_knowledge_text" in tools
    assert "recall" not in tools and "retain" not in tools


async def test_a_memory_banks_endpoint_has_no_knowledge_tools(hindsight_server, bank_id):
    async with httpx.AsyncClient(base_url=hindsight_server.url, timeout=60) as http:
        tools = await (await _session(http, f"/mcp/{bank_id}/")).tools()

    assert "recall" in tools
    assert not {t for t in tools if "knowledge" in t}


async def test_http_and_mcp_read_and_write_the_same_documents(client, hindsight_server, knowledge_bank):
    kb = KnowledgeBanks(client)
    r = await kb.ingest(
        knowledge_bank,
        [{"document_id": "outage", "text": "During an outage the status page shows code ZX-4417."}],
    )
    assert r.status == 200, r.body

    async with httpx.AsyncClient(base_url=hindsight_server.url, timeout=60) as http:
        agent = await _session(http, f"/mcp/kb/{knowledge_bank}/")

        # Written over HTTP, found by the agent.
        question = "what code does the status page show"
        found = await agent.call("search_knowledge_bank", {"query": question})
        assert found["results"][0]["document_id"] == "outage", found
        # The same search over both transports is the same answer, passage for passage.
        r = await kb.search(knowledge_bank, query=question)
        assert r.status == 200, r.body
        assert found["results"] == r.body["results"]

        # Written by the agent, found over HTTP.
        await agent.call(
            "ingest_knowledge_text",
            {"document_id": "keys", "text": "Rotate every api key each month from the security settings page."},
        )
    r = await kb.search(knowledge_bank, query="how do I rotate an api key")
    assert r.status == 200, r.body
    assert r.body["results"][0]["document_id"] == "keys", r.body
