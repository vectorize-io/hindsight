"""Tests for AI-002 — Provider Registry & Model Inventory Hardening.

Covers:
- model_inventory table: upsert-on-refresh, deactivation, staleness
- provider enable/disable (PUT /api/ai/providers/{id}/enabled)
- provider registration (POST /api/ai/providers)
- inventory-backed GET /api/ai/models (query params: provider_id, family, capability, health)
- GET /api/ai/models/{provider_id}/{model_id}  (single model detail)
- POST /api/ai/models/refresh  (full refresh)
- POST /api/ai/models/{provider_id}/refresh  (per-provider refresh)
- GET /api/ai/models/inventory/stats
- health propagation to model rows
- route/preview falling back to live if inventory empty
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


# ── Helpers ───────────────────────────────────────────────────────────────────

def client() -> TestClient:
    return TestClient(app)


def _make_model(provider_id: str, model_id: str, **caps) -> dict:
    capabilities = {
        "chat": caps.get("chat", True),
        "completion": caps.get("completion", True),
        "embedding": caps.get("embedding", False),
        "audio": caps.get("audio", False),
        "tools": caps.get("tools", False),
        "streaming": caps.get("streaming", True),
        "vision": caps.get("vision", False),
    }
    return {
        "provider_id": provider_id,
        "model_id": model_id,
        "display_name": model_id.title(),
        "family": "llama" if "llama" in model_id else "unknown",
        "capabilities": capabilities,
        "context_window": None,
        "cost": {"input_per_1m": None, "output_per_1m": None, "currency": "USD"},
        "latency_ms": None,
        "health": "unknown",
        "metadata": {},
    }


def _make_provider(pid: str, health: str = "unknown", enabled: bool = True) -> dict:
    from datetime import datetime
    return {
        "id": f"{pid}-id",
        "provider_id": pid,
        "display_name": pid.title(),
        "base_url": f"http://{pid}.internal",
        "provider_type": "local",
        "api_style": "openai_compatible",
        "auth_type": "none",
        "enabled": enabled,
        "supports_chat": True,
        "supports_completion": True,
        "supports_embeddings": True,
        "supports_audio": False,
        "supports_tools": True,
        "supports_streaming": True,
        "health_status": health,
        "last_health_check": None,
        "metadata": {},
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


# ── Inventory service: unit tests (no HTTP layer) ─────────────────────────────

class TestInventoryService:
    """Unit tests against inventory.py directly (in-memory SQLite)."""

    @pytest.mark.asyncio
    async def test_refresh_upserts_new_models(self):
        from app.ai import inventory
        models = [_make_model("localai", "llama-3-8b"), _make_model("localai", "mistral-7b")]
        summary = await inventory.refresh_provider("localai", models)
        assert summary["upserted"] == 2
        assert summary["deactivated"] == 0
        assert summary["total_active"] == 2

    @pytest.mark.asyncio
    async def test_refresh_updates_existing_model(self):
        from app.ai import inventory
        m = _make_model("localai", "gpt-neo")
        await inventory.refresh_provider("localai", [m])

        # Second refresh with updated health
        m2 = {**m, "health": "healthy", "latency_ms": 150}
        summary = await inventory.refresh_provider("localai", [m2])
        assert summary["upserted"] == 1
        assert summary["deactivated"] == 0

        result = await inventory.get_model("localai", "gpt-neo")
        assert result is not None
        assert result["health"] == "healthy"
        assert result["latency_ms"] == 150

    @pytest.mark.asyncio
    async def test_refresh_deactivates_missing_models(self):
        from app.ai import inventory
        models = [_make_model("deactivate-test", "m1"), _make_model("deactivate-test", "m2")]
        await inventory.refresh_provider("deactivate-test", models)

        # Second refresh: only m1 returned → m2 should be deactivated
        summary = await inventory.refresh_provider("deactivate-test", [models[0]])
        assert summary["deactivated"] == 1

        m2 = await inventory.get_model("deactivate-test", "m2")
        assert m2 is not None
        assert m2["is_active"] is False

    @pytest.mark.asyncio
    async def test_refresh_with_empty_list_deactivates_all(self):
        from app.ai import inventory
        await inventory.refresh_provider("empty-test", [_make_model("empty-test", "m1")])
        summary = await inventory.refresh_provider("empty-test", [])
        assert summary["upserted"] == 0
        assert summary["deactivated"] == 1
        assert summary["total_active"] == 0

    @pytest.mark.asyncio
    async def test_propagate_health_updates_active_models(self):
        from app.ai import inventory
        models = [_make_model("health-test", "m1"), _make_model("health-test", "m2")]
        await inventory.refresh_provider("health-test", models)

        updated = await inventory.propagate_provider_health("health-test", "healthy")
        assert updated == 2

        m1 = await inventory.get_model("health-test", "m1")
        assert m1 is not None
        assert m1["health"] == "healthy"

    @pytest.mark.asyncio
    async def test_propagate_health_ok_maps_to_healthy(self):
        """'ok' (ollama/litellm style) should be stored as 'healthy' in models."""
        from app.ai import inventory
        await inventory.refresh_provider("ok-test", [_make_model("ok-test", "m")])
        await inventory.propagate_provider_health("ok-test", "ok")
        m = await inventory.get_model("ok-test", "m")
        assert m is not None
        assert m["health"] == "healthy"

    @pytest.mark.asyncio
    async def test_propagate_health_does_not_affect_inactive_models(self):
        from app.ai import inventory
        await inventory.refresh_provider("inactive-health-test", [_make_model("inactive-health-test", "m1")])
        await inventory.refresh_provider("inactive-health-test", [])  # deactivates m1
        updated = await inventory.propagate_provider_health("inactive-health-test", "healthy")
        assert updated == 0  # m1 is inactive → should not be updated

    @pytest.mark.asyncio
    async def test_query_models_by_provider(self):
        from app.ai import inventory
        await inventory.refresh_provider("query-test", [
            _make_model("query-test", "m1"),
            _make_model("query-test", "m2"),
        ])
        results = await inventory.query_models(provider_id="query-test")
        assert len(results) == 2
        assert all(r["provider_id"] == "query-test" for r in results)

    @pytest.mark.asyncio
    async def test_query_models_by_capability(self):
        from app.ai import inventory
        pid = "cap-test"
        await inventory.refresh_provider(pid, [
            _make_model(pid, "embed-model", chat=False, embedding=True),
            _make_model(pid, "chat-model", chat=True, embedding=False),
        ])
        embed_only = await inventory.query_models(provider_id=pid, capability="embedding")
        assert len(embed_only) == 1
        assert embed_only[0]["model_id"] == "embed-model"

    @pytest.mark.asyncio
    async def test_query_models_active_only_true(self):
        from app.ai import inventory
        pid = "active-test"
        await inventory.refresh_provider(pid, [_make_model(pid, "active"), _make_model(pid, "inactive")])
        await inventory.refresh_provider(pid, [_make_model(pid, "active")])  # deactivates "inactive"
        active = await inventory.query_models(provider_id=pid, active_only=True)
        assert len(active) == 1
        assert active[0]["model_id"] == "active"

    @pytest.mark.asyncio
    async def test_query_models_active_only_false_returns_all(self):
        from app.ai import inventory
        pid = "all-test"
        await inventory.refresh_provider(pid, [_make_model(pid, "a"), _make_model(pid, "b")])
        await inventory.refresh_provider(pid, [_make_model(pid, "a")])  # b deactivated
        all_models = await inventory.query_models(provider_id=pid, active_only=False)
        assert len(all_models) == 2

    @pytest.mark.asyncio
    async def test_query_models_pagination(self):
        from app.ai import inventory
        pid = "page-test"
        models = [_make_model(pid, f"m{i}") for i in range(10)]
        await inventory.refresh_provider(pid, models)
        page1 = await inventory.query_models(provider_id=pid, limit=4, offset=0)
        page2 = await inventory.query_models(provider_id=pid, limit=4, offset=4)
        assert len(page1) == 4
        assert len(page2) == 4
        ids1 = {m["model_id"] for m in page1}
        ids2 = {m["model_id"] for m in page2}
        assert ids1.isdisjoint(ids2)

    @pytest.mark.asyncio
    async def test_get_model_not_found(self):
        from app.ai import inventory
        result = await inventory.get_model("nonexistent", "ghost")
        assert result is None

    @pytest.mark.asyncio
    async def test_inventory_stats(self):
        from app.ai import inventory
        pid = "stats-test"
        await inventory.refresh_provider(pid, [
            _make_model(pid, "s1"),
            _make_model(pid, "s2"),
        ])
        stats = await inventory.inventory_stats()
        assert stats["total"] >= 2
        assert stats["active"] >= 2
        assert isinstance(stats["by_provider"], dict)
        assert stats["by_provider"].get(pid, 0) >= 2

    @pytest.mark.asyncio
    async def test_inventory_stats_healthy_count(self):
        from app.ai import inventory
        pid = "stats-healthy"
        await inventory.refresh_provider(pid, [_make_model(pid, "h1"), _make_model(pid, "h2")])
        await inventory.propagate_provider_health(pid, "healthy")
        stats = await inventory.inventory_stats()
        assert stats["healthy"] >= 2


# ── HTTP routes: POST /api/ai/providers (register) ────────────────────────────

class TestProviderRegistration:
    def test_register_new_provider_returns_201(self):
        with patch("app.ai.providers.register_provider", new=AsyncMock(return_value=_make_provider("custom-llm"))):
            r = client().post("/api/ai/providers", json={
                "provider_id": "custom-llm",
                "base_url": "http://custom-llm.internal:8080",
            })
        assert r.status_code == 201
        data = r.json()
        assert data["provider_id"] == "custom-llm"

    def test_register_provider_id_invalid_chars_rejected(self):
        r = client().post("/api/ai/providers", json={
            "provider_id": "Custom LLM!!",
            "base_url": "http://custom-llm.internal",
        })
        assert r.status_code == 422

    def test_register_provider_empty_id_rejected(self):
        r = client().post("/api/ai/providers", json={
            "provider_id": "",
            "base_url": "http://custom.internal",
        })
        assert r.status_code == 422

    def test_register_provider_no_secrets_in_response(self):
        provider_data = _make_provider("clean-provider")
        with patch("app.ai.providers.register_provider", new=AsyncMock(return_value=provider_data)):
            r = client().post("/api/ai/providers", json={
                "provider_id": "clean-provider",
                "base_url": "http://clean.internal",
            })
        body = r.text
        assert "api_key" not in body
        assert "secret" not in body

    def test_register_updates_existing_provider(self):
        """Registering an existing provider_id is idempotent (200 on GET after)."""
        updated = _make_provider("localai")
        updated["base_url"] = "http://new-localai.internal"
        with patch("app.ai.providers.register_provider", new=AsyncMock(return_value=updated)):
            r = client().post("/api/ai/providers", json={
                "provider_id": "localai",
                "base_url": "http://new-localai.internal",
            })
        assert r.status_code == 201

    def test_register_provider_enabled_field_propagated(self):
        disabled = _make_provider("disabled-provider", enabled=False)
        with patch("app.ai.providers.register_provider", new=AsyncMock(return_value=disabled)):
            r = client().post("/api/ai/providers", json={
                "provider_id": "disabled-provider",
                "base_url": "http://disabled.internal",
                "enabled": False,
            })
        assert r.status_code == 201
        assert r.json()["enabled"] is False


# ── HTTP routes: PUT /api/ai/providers/{id}/enabled ───────────────────────────

class TestProviderEnableDisable:
    def test_disable_provider(self):
        disabled = _make_provider("localai", enabled=False)
        with patch("app.ai.providers.set_enabled", new=AsyncMock(return_value=disabled)):
            r = client().put("/api/ai/providers/localai/enabled", json={"enabled": False})
        assert r.status_code == 200
        assert r.json()["enabled"] is False

    def test_enable_provider(self):
        enabled = _make_provider("localai", enabled=True)
        with patch("app.ai.providers.set_enabled", new=AsyncMock(return_value=enabled)):
            r = client().put("/api/ai/providers/localai/enabled", json={"enabled": True})
        assert r.status_code == 200
        assert r.json()["enabled"] is True

    def test_enable_nonexistent_provider_404(self):
        with patch("app.ai.providers.set_enabled", new=AsyncMock(return_value=None)):
            r = client().put("/api/ai/providers/ghost/enabled", json={"enabled": True})
        assert r.status_code == 404

    def test_enable_missing_body_422(self):
        r = client().put("/api/ai/providers/localai/enabled", json={})
        assert r.status_code == 422


# ── HTTP routes: POST /api/ai/models/refresh ─────────────────────────────────

class TestInventoryRefresh:
    def test_refresh_all_returns_200(self):
        models = [_make_model("localai", "m1"), _make_model("localai", "m2")]
        with (
            patch("app.ai.providers.list_providers", new=AsyncMock(
                return_value=[_make_provider("localai")]
            )),
            patch("app.ai.localai.list_models", new=AsyncMock(return_value=models)),
            patch("app.ai.inventory.refresh_provider", new=AsyncMock(
                return_value={"upserted": 2, "deactivated": 0, "total_active": 2}
            )),
        ):
            r = client().post("/api/ai/models/refresh")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 2
        assert data["source"] == "live"

    def test_refresh_per_provider_returns_200(self):
        models = [_make_model("localai", "fresh-model")]
        with (
            patch("app.ai.providers.get_provider", new=AsyncMock(
                return_value=_make_provider("localai")
            )),
            patch("app.ai.localai.list_models", new=AsyncMock(return_value=models)),
            patch("app.ai.inventory.refresh_provider", new=AsyncMock(
                return_value={"upserted": 1, "deactivated": 0, "total_active": 1}
            )),
        ):
            r = client().post("/api/ai/models/localai/refresh")
        assert r.status_code == 200
        assert r.json()["count"] == 1

    def test_refresh_provider_not_found_404(self):
        with patch("app.ai.providers.get_provider", new=AsyncMock(return_value=None)):
            r = client().post("/api/ai/models/ghost/refresh")
        assert r.status_code == 404

    def test_refresh_provider_no_adapter_422(self):
        with patch("app.ai.providers.get_provider", new=AsyncMock(
            return_value=_make_provider("openai")  # openai has no local adapter
        )):
            r = client().post("/api/ai/models/openai/refresh")
        assert r.status_code == 422


# ── HTTP routes: GET /api/ai/models (inventory query) ────────────────────────

class TestModelsInventoryQuery:
    def test_models_returns_200(self):
        with patch("app.ai.inventory.query_models", new=AsyncMock(return_value=[])):
            r = client().get("/api/ai/models")
        assert r.status_code == 200
        data = r.json()
        assert data["source"] == "inventory"
        assert data["count"] == 0

    def test_models_filter_by_provider(self):
        models = [_make_model("localai", "m1")]
        with patch("app.ai.inventory.query_models", new=AsyncMock(return_value=models)):
            r = client().get("/api/ai/models?provider_id=localai")
        assert r.status_code == 200

    def test_models_filter_by_family(self):
        models = [_make_model("localai", "llama-7b")]
        with patch("app.ai.inventory.query_models", new=AsyncMock(return_value=models)):
            r = client().get("/api/ai/models?family=llama")
        assert r.status_code == 200

    def test_models_filter_by_capability(self):
        embed = {**_make_model("localai", "nomic-embed"), "capabilities": {
            "chat": False, "completion": False, "embedding": True,
            "audio": False, "tools": False, "streaming": False, "vision": False,
        }}
        with patch("app.ai.inventory.query_models", new=AsyncMock(return_value=[embed])):
            r = client().get("/api/ai/models?capability=embedding")
        assert r.status_code == 200
        assert r.json()["count"] == 1

    def test_models_source_live_fetches_from_adapter(self):
        models = [_make_model("localai", "live-model")]
        with (
            patch("app.ai.providers.list_providers", new=AsyncMock(
                return_value=[_make_provider("localai")]
            )),
            patch("app.ai.localai.list_models", new=AsyncMock(return_value=models)),
        ):
            r = client().get("/api/ai/models?source=live")
        assert r.status_code == 200
        data = r.json()
        assert data["source"] == "live"
        assert data["count"] == 1

    def test_models_limit_and_offset(self):
        with patch("app.ai.inventory.query_models", new=AsyncMock(return_value=[])):
            r = client().get("/api/ai/models?limit=10&offset=0")
        assert r.status_code == 200

    def test_models_limit_too_large_422(self):
        r = client().get("/api/ai/models?limit=99999")
        assert r.status_code == 422

    def test_models_negative_offset_422(self):
        r = client().get("/api/ai/models?offset=-1")
        assert r.status_code == 422

    def test_models_active_only_false(self):
        with patch("app.ai.inventory.query_models", new=AsyncMock(return_value=[])):
            r = client().get("/api/ai/models?active_only=false")
        assert r.status_code == 200


# ── HTTP routes: GET /api/ai/models/{provider}/{model} ───────────────────────

class TestSingleModelDetail:
    def test_get_model_found(self):
        m = _make_model("localai", "llama-7b")
        m["is_active"] = True
        m["first_seen"] = "2026-01-01T00:00:00Z"
        m["last_seen"] = "2026-01-02T00:00:00Z"
        with (
            patch("app.ai.providers.get_provider", new=AsyncMock(
                return_value=_make_provider("localai")
            )),
            patch("app.ai.inventory.get_model", new=AsyncMock(return_value=m)),
        ):
            r = client().get("/api/ai/models/localai/llama-7b")
        assert r.status_code == 200
        data = r.json()
        assert data["model_id"] == "llama-7b"
        assert data["provider_id"] == "localai"

    def test_get_model_not_found_404(self):
        with (
            patch("app.ai.providers.get_provider", new=AsyncMock(
                return_value=_make_provider("localai")
            )),
            patch("app.ai.inventory.get_model", new=AsyncMock(return_value=None)),
        ):
            r = client().get("/api/ai/models/localai/ghost-model")
        assert r.status_code == 404

    def test_get_model_provider_not_found_404(self):
        with patch("app.ai.providers.get_provider", new=AsyncMock(return_value=None)):
            r = client().get("/api/ai/models/ghost/any-model")
        assert r.status_code == 404

    def test_get_model_includes_inventory_fields(self):
        """Single model detail must include is_active, first_seen, last_seen."""
        m = _make_model("localai", "test-model")
        m["is_active"] = True
        m["first_seen"] = "2026-01-01T00:00:00Z"
        m["last_seen"] = "2026-01-10T00:00:00Z"
        with (
            patch("app.ai.providers.get_provider", new=AsyncMock(
                return_value=_make_provider("localai")
            )),
            patch("app.ai.inventory.get_model", new=AsyncMock(return_value=m)),
        ):
            r = client().get("/api/ai/models/localai/test-model")
        data = r.json()
        assert "is_active" in data
        assert data["is_active"] is True


# ── HTTP routes: GET /api/ai/models/inventory/stats ──────────────────────────

class TestInventoryStats:
    def test_stats_returns_200(self):
        with patch("app.ai.inventory.inventory_stats", new=AsyncMock(return_value={
            "total": 10,
            "active": 8,
            "inactive": 2,
            "healthy": 6,
            "by_provider": {"localai": 5, "ollama": 3},
        })):
            r = client().get("/api/ai/models/inventory/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 10
        assert data["active"] == 8
        assert data["inactive"] == 2
        assert data["healthy"] == 6
        assert data["by_provider"]["localai"] == 5

    def test_stats_by_provider_is_dict(self):
        with patch("app.ai.inventory.inventory_stats", new=AsyncMock(return_value={
            "total": 0, "active": 0, "inactive": 0, "healthy": 0, "by_provider": {},
        })):
            r = client().get("/api/ai/models/inventory/stats")
        assert isinstance(r.json()["by_provider"], dict)


# ── Health propagation via HTTP route ────────────────────────────────────────

class TestHealthPropagation:
    def test_health_check_propagates_to_inventory(self):
        """GET provider health → should call inventory.propagate_provider_health."""
        with (
            patch("app.ai.providers.get_provider", new=AsyncMock(
                return_value=_make_provider("localai")
            )),
            patch("app.ai.providers.update_health", new=AsyncMock()),
            patch("app.ai.localai.health", new=AsyncMock(return_value={"status": "healthy"})),
            patch("app.ai.inventory.propagate_provider_health", new=AsyncMock(return_value=3)) as mock_propagate,
        ):
            r = client().get("/api/ai/providers/localai/health")
        assert r.status_code == 200
        mock_propagate.assert_awaited_once_with("localai", "healthy")

    def test_health_down_propagates_down(self):
        with (
            patch("app.ai.providers.get_provider", new=AsyncMock(
                return_value=_make_provider("localai")
            )),
            patch("app.ai.providers.update_health", new=AsyncMock()),
            patch("app.ai.localai.health", new=AsyncMock(return_value={"status": "down", "error": "refused"})),
            patch("app.ai.inventory.propagate_provider_health", new=AsyncMock(return_value=2)) as mock_propagate,
        ):
            r = client().get("/api/ai/providers/localai/health")
        assert r.status_code == 200
        mock_propagate.assert_awaited_once_with("localai", "down")


# ── Route preview with inventory fallback ────────────────────────────────────

class TestRoutePreviewInventoryIntegration:
    def _post(self, payload: dict):
        return client().post("/api/ai/route/preview", json=payload)

    def test_preview_uses_inventory_when_populated(self):
        inventory_models = [_make_model("localai", "inv-model")]
        with patch("app.ai.inventory.query_models", new=AsyncMock(return_value=inventory_models)):
            r = self._post({"request_type": "chat"})
        assert r.status_code == 200
        assert r.json()["selected_model"] == "inv-model"

    def test_preview_falls_back_to_live_when_inventory_empty(self):
        live_models = [_make_model("localai", "live-fallback")]
        with (
            patch("app.ai.inventory.query_models", new=AsyncMock(return_value=[])),
            patch("app.ai.providers.list_providers", new=AsyncMock(
                return_value=[_make_provider("localai")]
            )),
            patch("app.ai.localai.list_models", new=AsyncMock(return_value=live_models)),
        ):
            r = self._post({"request_type": "chat"})
        assert r.status_code == 200
        assert r.json()["selected_model"] == "live-fallback"

    def test_preview_no_selection_when_both_empty(self):
        with (
            patch("app.ai.inventory.query_models", new=AsyncMock(return_value=[])),
            patch("app.ai.providers.list_providers", new=AsyncMock(return_value=[])),
        ):
            r = self._post({"request_type": "chat"})
        assert r.status_code == 200
        assert r.json()["selected_provider"] is None
