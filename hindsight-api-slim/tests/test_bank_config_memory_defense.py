import pytest


@pytest.mark.asyncio
async def test_patch_accepts_memory_defense_policy(api_client) -> None:
    r1 = await api_client.put("/v1/default/banks/mg-15-1", json={})
    assert r1.status_code in {200, 201}, r1.text
    r = await api_client.patch(
        "/v1/default/banks/mg-15-1/config",
        json={
            "updates": {"memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", "action": "redact"}]}}
        },
    )
    assert r.status_code == 200, r.text

    r2 = await api_client.get("/v1/default/banks/mg-15-1/config")
    body = r2.json()
    # `config` field on BankConfigResponse holds the merged effective config
    assert body["config"]["memory_defense"]["enabled"] is True


@pytest.mark.asyncio
async def test_patch_rejects_invalid_policy_action(api_client) -> None:
    r1 = await api_client.put("/v1/default/banks/mg-15-2", json={})
    assert r1.status_code in {200, 201}
    # Use a valid ``on`` so the parser progresses to ``action`` validation —
    # otherwise the unknown-``on`` check fires first.
    r = await api_client.patch(
        "/v1/default/banks/mg-15-2/config",
        json={
            "updates": {
                "memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", "action": "delete_everything"}]}
            }
        },
    )
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert "action" in str(detail).lower()


@pytest.mark.asyncio
async def test_patch_rejects_unknown_detector(api_client) -> None:
    """``on`` must be one of the canonical detector names."""
    r1 = await api_client.put("/v1/default/banks/mg-15-4", json={})
    assert r1.status_code in {200, 201}
    r = await api_client.patch(
        "/v1/default/banks/mg-15-4/config",
        json={"updates": {"memory_defense": {"enabled": True, "rules": [{"on": "nope", "action": "redact"}]}}},
    )
    assert r.status_code == 422, r.text
    assert "on" in str(r.json()["detail"]).lower()
