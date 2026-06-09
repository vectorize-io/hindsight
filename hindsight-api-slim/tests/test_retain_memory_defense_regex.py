"""End-to-end retain with the regex extension — verify redaction, block, webhook."""

import json

import pytest


@pytest.mark.asyncio
async def test_regex_redacts_during_retain(api_client) -> None:
    await api_client.put("/v1/default/banks/md-regex-1", json={})
    await api_client.patch(
        "/v1/default/banks/md-regex-1/config",
        json={
            "updates": {"memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", "action": "redact"}]}}
        },
    )

    secret = "ghp_" + "A" * 36
    r = await api_client.post(
        "/v1/default/banks/md-regex-1/memories",
        json={
            "items": [{"content": f"rotate {secret}"}],
        },
    )
    assert r.status_code == 200, r.text

    r2 = await api_client.get("/v1/default/banks/md-regex-1/memories/list", params={"limit": 50})
    body = r2.json()
    for m in body["items"]:
        assert secret not in m["text"], m


@pytest.mark.asyncio
async def test_regex_blocks_secret_item(api_client) -> None:
    await api_client.put("/v1/default/banks/md-regex-2", json={})
    await api_client.patch(
        "/v1/default/banks/md-regex-2/config",
        json={
            "updates": {
                "memory_defense": {
                    "enabled": True,
                    "rules": [
                        {"on": "sensitive_data", "action": "block"},
                    ],
                }
            }
        },
    )

    # A single item that contains a secret is fully blocked → 422.
    secret = "sk-ant-" + "B" * 40
    r = await api_client.post(
        "/v1/default/banks/md-regex-2/memories",
        json={
            "items": [{"content": f"key={secret}"}],
        },
    )
    assert r.status_code == 422, r.text

    # Content with no sensitive_data hit still passes (nothing to block).
    r2 = await api_client.post(
        "/v1/default/banks/md-regex-2/memories",
        json={
            "items": [{"content": "the roadmap meeting is on friday"}],
        },
    )
    assert r2.status_code == 200, r2.text


@pytest.mark.asyncio
async def test_regex_fires_webhook_on_redact(api_client, memory) -> None:
    """A redact decision queues a memory_defense.triggered webhook delivery
    when a webhook is subscribed to that event type."""
    bank = "md-regex-wh"
    await api_client.put(f"/v1/default/banks/{bank}", json={})

    wr = await api_client.post(
        f"/v1/default/banks/{bank}/webhooks",
        json={"url": "https://example.com/hook", "event_types": ["memory_defense.triggered"]},
    )
    assert wr.status_code in {200, 201}, wr.text

    await api_client.patch(
        f"/v1/default/banks/{bank}/config",
        json={
            "updates": {"memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", "action": "redact"}]}}
        },
    )

    secret = "ghp_" + "A" * 36
    rr = await api_client.post(
        f"/v1/default/banks/{bank}/memories",
        json={"items": [{"content": f"rotate {secret}"}]},
    )
    assert rr.status_code == 200, rr.text

    async with memory._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT task_payload FROM async_operations
            WHERE operation_type = 'webhook_delivery' AND bank_id = $1
            """,
            bank,
        )
    event_types = []
    for row in rows:
        payload = row["task_payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        event_types.append(payload.get("event_type"))
    assert "memory_defense.triggered" in event_types, event_types
