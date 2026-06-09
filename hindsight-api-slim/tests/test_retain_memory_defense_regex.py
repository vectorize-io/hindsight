"""End-to-end retain with the regex extension — verify redaction and block."""

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
