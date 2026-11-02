"""Client mistakes answer 4xx, not a 500 carrying the database's error (#6231).

A non-UUID directive or memory id, a directive priority the INT column cannot
hold, and text Postgres cannot store (a NUL byte, a lone surrogate) all used to
reach the database and come back as 500, which a retrying client cannot tell
apart from a server fault.
"""

import json
import uuid

import pytest
from pydantic import ValidationError

from hindsight_api.engine.bank_templates import BankTemplateDirective

BANK = "malformed-input-4xx"
B = f"/v1/default/banks/{BANK}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", f"{B}/directives/not-a-uuid"),
        ("PATCH", f"{B}/directives/not-a-uuid"),
        ("DELETE", f"{B}/directives/not-a-uuid"),
        ("DELETE", f"{B}/memories/not-a-uuid/observations"),
    ],
)
async def test_non_uuid_id_is_400(api_client, method, path):
    response = await api_client.request(method, path, json={"name": "n"} if method == "PATCH" else None)
    assert response.status_code == 400, response.text
    assert "is not a valid UUID" in response.json()["detail"]


@pytest.mark.asyncio
async def test_other_uuid_spellings_reach_the_lookup(api_client):
    """`uuid.UUID` accepts these but the uuid column does not, so the id is passed on canonical."""
    await api_client.put(B, json={})
    raw = uuid.uuid4()
    for spelling in (f"{{{raw}}}", f"urn:uuid:{raw}"):
        response = await api_client.get(f"{B}/directives/{spelling}")
        assert response.status_code == 404, response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("priority", [2**31, -(2**31) - 1])
async def test_out_of_range_priority_is_422(api_client, priority):
    response = await api_client.post(f"{B}/directives", json={"name": "n", "content": "c", "priority": priority})
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_out_of_range_priority_is_rejected_by_the_engine(memory, request_context):
    """The MCP tools pass a plain int and skip the request models."""
    with pytest.raises(ValueError, match="Invalid priority"):
        await memory.create_directive(BANK, "n", "c", priority=2**31, request_context=request_context)


def test_out_of_range_priority_fails_template_validation():
    with pytest.raises(ValidationError):
        BankTemplateDirective(name="n", content="c", priority=2**31)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["a\x00b", "a\ud800b"])
async def test_unstorable_text_is_400(api_client, bad):
    await api_client.put(B, json={})
    requests = [
        ("POST", f"{B}/directives", {"name": bad, "content": "c"}),
        ("POST", f"{B}/memories/recall", {"query": "alice", "tags": [bad]}),
        ("PUT", B, {"name": bad}),
    ]
    for method, path, body in requests:
        # Escaped, as a JSON client sends it: httpx cannot encode a lone surrogate itself.
        content = json.dumps(body, ensure_ascii=True)
        response = await api_client.request(method, path, content=content, headers={"Content-Type": "application/json"})
        assert response.status_code == 400, f"{method} {path}: {response.text}"


@pytest.mark.asyncio
async def test_nul_byte_in_query_and_path_is_400(api_client):
    await api_client.put(B, json={})
    for path in (f"{B}/memories/list?q=a%00b", f"{B}/documents/a%00b"):
        response = await api_client.get(path)
        assert response.status_code == 400, f"{path}: {response.text}"
