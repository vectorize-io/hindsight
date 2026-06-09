"""End-to-end: retain runs through the Memory Defense extension before fact extraction.

Covers the ALLOW and REDACT actions. BLOCK enforcement (422 on full-block) is
covered in test_retain_memory_defense_regex.py.
"""

import pytest

STRICT_POLICY = {
    "memory_defense": {
        "enabled": True,
        "rules": [
            {"on": "sensitive_data", "action": "redact"},
        ],
    }
}


async def _set_strict_policy(api_client, bank: str) -> None:
    r = await api_client.patch(f"/v1/default/banks/{bank}/config", json={"updates": STRICT_POLICY})
    assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_allowed_content_writes_normally(api_client) -> None:
    await api_client.put("/v1/default/banks/mg11-1", json={})
    await _set_strict_policy(api_client, "mg11-1")
    r = await api_client.post(
        "/v1/default/banks/mg11-1/memories",
        json={
            "items": [{"content": "the meeting is friday"}],
        },
    )
    assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_redacted_content_stores_redacted_text(api_client, memory) -> None:
    await api_client.put("/v1/default/banks/mg11-3", json={})
    await _set_strict_policy(api_client, "mg11-3")
    secret = "ghp_" + "A" * 36
    await api_client.post(
        "/v1/default/banks/mg11-3/memories",
        json={
            "items": [{"content": f"my token is {secret}"}],
        },
    )
    async with memory._pool.acquire() as conn:
        texts = [r["text"] for r in await conn.fetch("SELECT text FROM memory_units WHERE bank_id = 'mg11-3'")]
    assert all(secret not in t for t in texts), texts
