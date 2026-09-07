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


def test_update_bank_config_forwards_entity_labels(monkeypatch):
    """Entity-label groups must reach the request body unflattened.

    The wrapper enumerates config fields, so a label group that never lands in
    `updates` is silently dropped — the bank keeps extracting nothing. The
    TypeScript wrapper has the mirror of this test in
    tests/bank_config_entity_labels_mapping.test.ts; the two must stay in step.
    """
    captured: dict[str, object] = {}

    async def fake_update(self, bank_id, updates):
        captured["updates"] = updates
        return {"bank_id": bank_id, "config": {}, "overrides": updates}

    monkeypatch.setattr(Hindsight, "_aupdate_bank_config", fake_update)

    labels = [
        {
            "key": "name",
            "type": "multi-text",
            "tag": True,
            "description": "Every name the subject of this fact is known by.",
        },
        {"key": "topic", "type": "value", "values": [{"value": "infra"}]},
    ]

    client = Hindsight(base_url="http://example.invalid")
    client.update_bank_config("test-bank", entity_labels=labels, entities_allow_free_form=False)

    assert captured["updates"] == {
        "entity_labels": labels,
        "entities_allow_free_form": False,
    }


def test_update_bank_config_omits_entity_labels_when_unset(monkeypatch):
    """An unset entity_labels must not clear the bank's existing vocabulary."""
    captured: dict[str, object] = {}

    async def fake_update(self, bank_id, updates):
        captured["updates"] = updates
        return {"bank_id": bank_id, "config": {}, "overrides": updates}

    monkeypatch.setattr(Hindsight, "_aupdate_bank_config", fake_update)

    client = Hindsight(base_url="http://example.invalid")
    client.update_bank_config("test-bank", retain_chunk_size=1000)

    assert "entity_labels" not in captured["updates"]
    assert "entities_allow_free_form" not in captured["updates"]


def test_update_bank_config_forwards_the_full_configurable_surface(monkeypatch):
    """Every server-configurable field must reach the request body.

    The wrapper enumerates config fields, so one accepted as a keyword but never
    written into `updates` is silently dropped — the caller sees a 200 and no
    change. This covers the groups that were unreachable before #4029: the recall
    budget, auto-consolidation, memory defense and audit settings. The TypeScript
    wrapper has the mirror of this test in
    tests/bank_config_full_surface_mapping.test.ts; the two must stay in step.
    """
    captured: dict[str, object] = {}

    async def fake_update(self, bank_id, updates):
        captured["updates"] = updates
        return {"bank_id": bank_id, "config": {}, "overrides": updates}

    monkeypatch.setattr(Hindsight, "_aupdate_bank_config", fake_update)

    client = Hindsight(base_url="http://example.invalid")
    client.update_bank_config(
        "test-bank",
        recall_max_tokens=4096,
        recall_include_chunks=False,
        recall_chunks_max_tokens=500,
        recall_budget_function="adaptive",
        recall_budget_fixed_low=50,
        recall_budget_fixed_mid=150,
        recall_budget_fixed_high=500,
        recall_budget_adaptive_low=0.01,
        recall_budget_adaptive_mid=0.05,
        recall_budget_adaptive_high=0.2,
        recall_budget_min=10,
        recall_budget_max=1000,
        enable_auto_consolidation=False,
        consolidation_llm_parallelism=2,
        consolidation_max_memories_per_round=50,
        mental_model_min_refresh_interval_seconds=3600,
        max_observations_per_scope=25,
        observation_scope_limits=[{"scope": "team", "limit": 10}],
        retain_chunk_batch_size=64,
        store_document_text=False,
        memory_defense={"redact_secrets": True},
        audit_log_enabled=True,
    )

    assert captured["updates"] == {
        "recall_max_tokens": 4096,
        "recall_include_chunks": False,
        "recall_chunks_max_tokens": 500,
        "recall_budget_function": "adaptive",
        "recall_budget_fixed_low": 50,
        "recall_budget_fixed_mid": 150,
        "recall_budget_fixed_high": 500,
        "recall_budget_adaptive_low": 0.01,
        "recall_budget_adaptive_mid": 0.05,
        "recall_budget_adaptive_high": 0.2,
        "recall_budget_min": 10,
        "recall_budget_max": 1000,
        "enable_auto_consolidation": False,
        "consolidation_llm_parallelism": 2,
        "consolidation_max_memories_per_round": 50,
        "mental_model_min_refresh_interval_seconds": 3600,
        "max_observations_per_scope": 25,
        "observation_scope_limits": [{"scope": "team", "limit": 10}],
        "retain_chunk_batch_size": 64,
        "store_document_text": False,
        "memory_defense": {"redact_secrets": True},
        "audit_log_enabled": True,
    }


def test_update_bank_config_gemini_safety_settings_is_a_list(monkeypatch):
    """The provider takes a list of {category, threshold}, not a category->threshold map.

    The wrapper used to type this `dict[str, str]`; a caller following that hint
    sent a shape the Gemini provider rejects.
    """
    captured: dict[str, object] = {}

    async def fake_update(self, bank_id, updates):
        captured["updates"] = updates
        return {"bank_id": bank_id, "config": {}, "overrides": updates}

    monkeypatch.setattr(Hindsight, "_aupdate_bank_config", fake_update)

    settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]
    client = Hindsight(base_url="http://example.invalid")
    client.update_bank_config("test-bank", llm_gemini_safety_settings=settings)

    assert captured["updates"] == {"llm_gemini_safety_settings": settings}
