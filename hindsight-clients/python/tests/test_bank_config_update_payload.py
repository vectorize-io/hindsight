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


def test_update_bank_config_forwards_recall_pipeline_toggles(monkeypatch):
    """The recall stage toggles must reach the request body.

    The wrapper enumerates config fields rather than passing a dict through, so a
    field added to the API but not to this method is silently dropped for every
    consumer of the SDK.
    """
    captured: dict[str, object] = {}

    async def fake_update(self, bank_id, updates):
        captured["updates"] = updates
        return {"bank_id": bank_id, "config": {}, "overrides": updates}

    monkeypatch.setattr(Hindsight, "_aupdate_bank_config", fake_update)

    client = Hindsight(base_url="http://example.invalid")
    client.update_bank_config(
        "test-bank",
        enable_text_search=False,
        enable_temporal_retrieval=False,
        enable_graph_retrieval=False,
        enable_reranking=False,
    )

    assert captured["updates"] == {
        "enable_text_search": False,
        "enable_temporal_retrieval": False,
        "enable_graph_retrieval": False,
        "enable_reranking": False,
    }


def test_create_bank_forwards_recall_pipeline_toggles(monkeypatch):
    """Same toggles, set through the bank create/update (PUT) endpoint.

    Asserts the real request body, not just the signature: create_bank enumerates
    fields into `body`, so a missing mapping would drop the toggle silently.
    """
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        async def json(self):
            return {
                "bank_id": "test-bank",
                "name": "test-bank",
                "mission": "",
                "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
            }

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        def put(self, url, json=None, headers=None, timeout=None):
            captured["body"] = json
            return FakeResponse()

    # aiohttp is imported inside the method, so it resolves from sys.modules at
    # call time — patch the module itself, not the wrapper's namespace.
    import aiohttp

    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **k: FakeSession())

    client = Hindsight(base_url="http://example.invalid")
    client.create_bank(
        "test-bank",
        retain_extraction_mode="chunks",
        enable_observations=False,
        enable_text_search=False,
        enable_temporal_retrieval=False,
        enable_graph_retrieval=False,
        enable_reranking=False,
    )

    body = captured["body"]
    assert body["retain_extraction_mode"] == "chunks"
    assert body["enable_observations"] is False
    assert body["enable_text_search"] is False
    assert body["enable_temporal_retrieval"] is False
    assert body["enable_graph_retrieval"] is False
    assert body["enable_reranking"] is False


def test_create_bank_omits_recall_pipeline_toggles_when_unset(monkeypatch):
    """Unset toggles stay out of the body so the bank inherits the server default."""
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        async def json(self):
            return {
                "bank_id": "test-bank",
                "name": "test-bank",
                "mission": "",
                "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
            }

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        def put(self, url, json=None, headers=None, timeout=None):
            captured["body"] = json
            return FakeResponse()

    # aiohttp is imported inside the method, so it resolves from sys.modules at
    # call time — patch the module itself, not the wrapper's namespace.
    import aiohttp

    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **k: FakeSession())

    client = Hindsight(base_url="http://example.invalid")
    client.create_bank("test-bank")

    body = captured["body"]
    for name in ("enable_text_search", "enable_temporal_retrieval", "enable_graph_retrieval", "enable_reranking"):
        assert name not in body
