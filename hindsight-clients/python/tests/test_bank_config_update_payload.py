from hindsight_client import Hindsight


def test_update_bank_config_can_set_retain_structured_chunk_size(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_update(self, bank_id, updates):
        captured["bank_id"] = bank_id
        captured["updates"] = updates
        return {"bank_id": bank_id, "config": {}, "overrides": updates}

    monkeypatch.setattr(Hindsight, "_aupdate_bank_config", fake_update)

    client = Hindsight(base_url="http://example.invalid")
    result = client.update_bank_config(
        "test-bank",
        retain_structured_chunk_size=12000,
    )

    assert result["bank_id"] == "test-bank"
    assert captured["updates"] == {"retain_structured_chunk_size": 12000}


def test_update_bank_config_omits_retain_structured_chunk_size_when_unset(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_update(self, bank_id, updates):
        captured["bank_id"] = bank_id
        captured["updates"] = updates
        return {"bank_id": bank_id, "config": {}, "overrides": updates}

    monkeypatch.setattr(Hindsight, "_aupdate_bank_config", fake_update)

    client = Hindsight(base_url="http://example.invalid")
    result = client.update_bank_config("test-bank")

    assert result["bank_id"] == "test-bank"
    assert captured["updates"] == {}
