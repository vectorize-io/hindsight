"""A bank tells every MCP client how to use it, the moment the client connects.

The bank owner writes `mcp_server_instructions` through the config API; an MCP
client connecting to the bank's own endpoint receives that text as the server
instructions of its `initialize` response. The two meet nowhere else, so the story
is the composition: config written through the published client, read back over
the MCP wire by a client that knows nothing of Hindsight.
"""

from __future__ import annotations

import json

import httpx
import pytest
from hindsight_client_api.exceptions import ApiException
from hindsight_client_api.models.bank_config_update import BankConfigUpdate

pytestmark = pytest.mark.asyncio

INSTRUCTIONS = "Read the document house-rules before anything else. Save new facts with the tag inbox."


async def _initialize(server_url: str, path: str, headers: dict[str, str] | None = None) -> dict:
    """A bare MCP initialize: the JSON-RPC request any client sends first."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "system-test", "version": "1"},
        },
    }
    async with httpx.AsyncClient(base_url=server_url, timeout=30) as http:
        response = await http.post(
            path,
            json=request,
            headers={"Accept": "application/json, text/event-stream", **(headers or {})},
        )
    response.raise_for_status()
    # Streamable HTTP answers with plain JSON or with one SSE message, server's choice.
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        data = [line[len("data:") :].strip() for line in response.text.splitlines() if line.startswith("data:")]
        assert len(data) == 1, response.text
        body = json.loads(data[0])
    else:
        body = response.json()
    assert "result" in body, body
    return body["result"]


async def test_a_connecting_client_receives_the_banks_instructions(client, hindsight_server, bank_id):
    await client.aupdate_bank_config(bank_id, mcp_server_instructions=INSTRUCTIONS)

    result = await _initialize(hindsight_server.url, f"/mcp/{bank_id}/")

    assert result["instructions"] == INSTRUCTIONS


async def test_a_bank_without_instructions_sends_none(client, hindsight_server, bank_id):
    await client.aupdate_bank_config(bank_id, mcp_enabled_tools=["recall"])

    result = await _initialize(hindsight_server.url, f"/mcp/{bank_id}/")

    assert result.get("instructions") is None


async def test_the_multi_bank_endpoint_does_not_speak_for_one_bank(client, hindsight_server, bank_id):
    await client.aupdate_bank_config(bank_id, mcp_server_instructions=INSTRUCTIONS)

    result = await _initialize(hindsight_server.url, "/mcp/", headers={"X-Bank-Id": bank_id})

    assert result.get("instructions") is None


async def test_an_oversized_text_is_refused_and_the_old_one_stays(client, hindsight_server, bank_id):
    await client.aupdate_bank_config(bank_id, mcp_server_instructions=INSTRUCTIONS)

    with pytest.raises(ApiException) as refused:
        await client.banks.update_bank_config(
            bank_id, BankConfigUpdate(updates={"mcp_server_instructions": "x" * 4001})
        )
    assert refused.value.status == 400
    assert "mcp_server_instructions must be at most 4000 characters" in str(refused.value.body)

    result = await _initialize(hindsight_server.url, f"/mcp/{bank_id}/")
    assert result["instructions"] == INSTRUCTIONS
