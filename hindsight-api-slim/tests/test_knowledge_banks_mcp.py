"""Knowledge banks over MCP, through a real MCP client against the app.

A knowledge bank has its own endpoint, /mcp/kb/{kb}/, which lists the knowledge-bank tools and
nothing else; a memory bank's /mcp/{bank}/ never lists them. The multi-bank endpoint lists both
and reaches a knowledge bank by its id.
"""

import json
import uuid

import httpx
import pytest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from hindsight_api.extensions import KnowledgeBankReadOperation
from hindsight_api.mcp_tools import _KNOWLEDGE_TOOLS
from hindsight_api.models import RequestContext
from tests.test_knowledge_banks import fake_embedder  # noqa: F401  (fixture)


def _payload(result) -> dict:
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
async def test_knowledge_bank_over_mcp(memory, fake_embedder):  # noqa: F811
    from hindsight_api.api import create_app

    kb = f"kb-mcp-{uuid.uuid4().hex[:8]}"
    app = create_app(memory, mcp_api_enabled=True, initialize_memory=False)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:

            async def session(url: str):
                return streamable_http_client(url, http_client=http)

            # Multi-bank: both families, and the knowledge bank is created and filled by id.
            async with await session("http://test/mcp/") as (r, w, _), ClientSession(r, w) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                assert _KNOWLEDGE_TOOLS <= names and "recall" in names

                created = _payload(await s.call_tool("create_knowledge_bank", {"bank_id": kb}))
                assert created["bank_id"] == kb, created
                ingested = _payload(
                    await s.call_tool(
                        "ingest_knowledge_text",
                        {"bank_id": kb, "document_id": "pw", "text": "The vault password is kept in the red box."},
                    )
                )
                assert ingested["document_ids"] == ["pw"], ingested
                listed = _payload(await s.call_tool("list_knowledge_banks", {"query": kb}))
                assert [b["bank_id"] for b in listed["banks"]] == [kb]

                # A memory bank id is not a knowledge bank: an error payload, not a crash.
                missing = _payload(await s.call_tool("get_knowledge_bank", {"bank_id": f"nope-{kb}"}))
                assert "not found" in missing["error"]
                # Each family refuses the other kind of bank by id.
                assert "error" in _payload(await s.call_tool("recall", {"bank_id": kb, "query": "x"}))
                await memory.get_bank_profile(f"mem-{kb}", request_context=RequestContext())
                assert "error" in _payload(await s.call_tool("get_knowledge_bank", {"bank_id": f"mem-{kb}"}))

            # The knowledge bank's own endpoint: only its tools, scoped to it.
            async with await session(f"http://test/mcp/kb/{kb}/") as (r, w, _), ClientSession(r, w) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                assert names == _KNOWLEDGE_TOOLS - {"list_knowledge_banks", "create_knowledge_bank"}

                found = (await s.call_tool("search_knowledge_bank", {"query": "password"})).structuredContent
                assert found["results"][0]["document_id"] == "pw", found
                doc = (await s.call_tool("get_knowledge_document", {"document_id": "pw"})).structuredContent
                assert doc["status"] == "ready" and "red box" in doc["passages"][0]["text"]
                deleted = (await s.call_tool("delete_knowledge_document", {"document_id": "pw"})).structuredContent
                assert deleted == {"deleted": True}
                docs = (await s.call_tool("list_knowledge_documents", {})).structuredContent
                assert docs["total"] == 0

                # A memory tool is not reachable on a knowledge bank's connection.
                refused = await s.call_tool("recall", {"query": "x"})
                assert refused.isError

            # Single-bank on a memory bank: no knowledge tools.
            async with await session(f"http://test/mcp/mem-{kb}/") as (r, w, _), ClientSession(r, w) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                assert "recall" in names and not (names & _KNOWLEDGE_TOOLS)

            # A memory bank named "kb" keeps /mcp/kb/: only a further segment is a knowledge bank.
            async with await session("http://test/mcp/kb/") as (r, w, _), ClientSession(r, w) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                assert "recall" in names and not (names & _KNOWLEDGE_TOOLS)

            async with await session(f"http://test/mcp/kb/{kb}/") as (r, w, _), ClientSession(r, w) as s:
                await s.initialize()
                assert (await s.call_tool("delete_knowledge_bank", {})).structuredContent == {"deleted": True}


@pytest.mark.asyncio
async def test_knowledge_bank_mcp_goes_through_the_same_auth(memory, fake_embedder):  # noqa: F811
    """The tenant extension authenticates /mcp/kb/ like every MCP path, and each tool call reaches
    the knowledge-bank hooks of the operation validator, as the HTTP routes do."""
    from hindsight_api.api import create_app
    from hindsight_api.extensions.builtin.tenant import ApiKeyTenantExtension
    from hindsight_api.knowledge.service import get_knowledge_service
    from hindsight_api.models import RequestContext
    from tests.test_knowledge_bank_extensions import HidesSecretKnowledgeBanks

    # Created before the validator and the tenant key are installed; the test's bank is left to
    # the per-run database, as a delete after the app's shutdown would find the engine closed.
    secret = f"kb-secret-{uuid.uuid4().hex[:8]}"
    await get_knowledge_service(memory).create_bank(secret, RequestContext())
    original_tenant, original_validator = memory._tenant_extension, memory._operation_validator
    validator = HidesSecretKnowledgeBanks()
    memory._operation_validator = validator
    # The MCP transport takes the tenant extension when the app is built.
    memory._tenant_extension = ApiKeyTenantExtension({"api_key": "right-key"})
    try:
        app = create_app(memory, mcp_api_enabled=True, initialize_memory=False)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                headers={"Authorization": "Bearer right-key"},
            ) as http:
                async with streamable_http_client(f"http://test/mcp/kb/{secret}/", http_client=http) as (r, w, _):
                    async with ClientSession(r, w) as s:
                        await s.initialize()
                        refused = (await s.call_tool("search_knowledge_bank", {"query": "x"})).structuredContent
                assert "not yours" in refused["error"], refused
                assert (secret, KnowledgeBankReadOperation.SEARCH_KNOWLEDGE_BANK) in validator.seen

                r = await http.post(
                    f"/mcp/kb/{secret}/",
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                    headers={"Authorization": "Bearer wrong-key", "Accept": "application/json, text/event-stream"},
                )
                assert r.status_code == 401, r.text
    finally:
        memory._tenant_extension, memory._operation_validator = original_tenant, original_validator
