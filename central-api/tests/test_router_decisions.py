"""Tests for router decisions endpoint."""
from fastapi.testclient import TestClient

from app.main import app


def test_router_decisions_returns_empty_list():
    """GET /api/router/decisions returns 200 with empty list when no decisions exist."""
    with TestClient(app) as client:
        r = client.get("/api/router/decisions")
        assert r.status_code == 200
        data = r.json()
        assert "decisions" in data
        assert "count" in data
        assert isinstance(data["decisions"], list)
        assert data["count"] == 0


def test_router_decisions_accepts_limit_offset():
    """GET /api/router/decisions accepts limit and offset query params."""
    with TestClient(app) as client:
        r = client.get("/api/router/decisions?limit=50&offset=10")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["decisions"], list)


def test_router_decisions_rejects_invalid_limit():
    """GET /api/router/decisions rejects limit > 1000."""
    with TestClient(app) as client:
        r = client.get("/api/router/decisions?limit=2000")
        assert r.status_code == 422  # Validation error


def test_router_decisions_rejects_negative_offset():
    """GET /api/router/decisions rejects negative offset."""
    with TestClient(app) as client:
        r = client.get("/api/router/decisions?offset=-1")
        assert r.status_code == 422  # Validation error
