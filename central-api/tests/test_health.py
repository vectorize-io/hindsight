from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_engine_health_lists_all_adapters():
    r = client.get("/api/health/engines")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in {"ok", "degraded"}
    assert isinstance(data["checked_at"], int)
    assert isinstance(data["services"], list)
    backends = {e["key"] for e in data["services"]}
    assert {"internal", "openmemory", "memlord", "coderag"} <= backends
    assert all(e["status"] in {"ok", "down"} for e in data["services"])


def test_dependency_health_is_public_and_structured():
    r = client.get("/health/dependencies")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "services" in data
    assert "providers" in data
    assert isinstance(data["services"], dict)
    assert isinstance(data["providers"], list)
