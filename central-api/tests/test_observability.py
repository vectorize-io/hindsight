"""Tests for observability — runtime health monitoring."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.observability.health import (
    HealthStatus,
    check_service_health,
    check_provider_health,
    get_runtime_health,
)

client = TestClient(app)


@pytest.mark.asyncio
async def test_check_service_health_down():
    """Test checking health of a non-existent service."""
    health = await check_service_health(
        "http://localhost:19999/health",
        "test-service",
        19999,
        timeout=1.0,
    )
    
    # Should be down since service doesn't exist
    assert health.name == "test-service"
    assert health.port == 19999
    assert health.status in [HealthStatus.down, HealthStatus.unknown]


@pytest.mark.asyncio
async def test_check_provider_health_down():
    """Test provider health check with non-existent provider."""
    health = await check_provider_health(
        "test-provider",
        "http://localhost:19999",
    )
    
    # Should fail and report down
    assert health.name == "test-provider"
    assert health.status == HealthStatus.down


@pytest.mark.asyncio
async def test_provider_health_no_secret_leakage():
    """Test that provider health doesn't leak API keys."""
    health = await check_provider_health(
        "test-provider",
        "http://localhost:19999",
        api_key="secret-key-12345",
    )
    
    # Verify no secret in error message
    if health.error_summary:
        assert "secret" not in health.error_summary.lower()
        assert "key" not in health.error_summary.lower()


@pytest.mark.asyncio
async def test_service_health_no_secret_leakage():
    """Test that service health doesn't leak sensitive data."""
    health = await check_service_health(
        "http://localhost:19999/health",
        "test-service",
        19999,
    )
    
    # Verify no secrets in response
    if health.error_summary:
        assert "password" not in health.error_summary.lower()


@pytest.mark.asyncio
async def test_health_status_enum_values():
    """Test that HealthStatus enum has expected values."""
    assert HealthStatus.healthy.value == "healthy"
    assert HealthStatus.degraded.value == "degraded"
    assert HealthStatus.down.value == "down"
    assert HealthStatus.unknown.value == "unknown"


def test_runtime_health_endpoint_exists():
    """Test that /api/observability/runtime-health endpoint exists."""
    r = client.get("/api/observability/runtime-health")
    # TestClient provides default auth, so 200 is expected
    assert r.status_code == 200


def test_service_status_endpoint_exists():
    """Test that /api/observability/service-status endpoint exists."""
    r = client.get("/api/observability/service-status")
    # TestClient provides default auth, so 200 is expected
    assert r.status_code == 200
    data = r.json()
    assert "overall_status" in data
    assert "service_count" in data


def test_public_health_endpoint_no_auth():
    """Test that /api/health is public (no auth required)."""
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "central-api"
    assert "version" in data


@pytest.mark.asyncio
async def test_get_runtime_health_returns_timestamp():
    """Test that get_runtime_health returns valid timestamp."""
    health = await get_runtime_health(tenant_id="test-tenant")
    assert health.timestamp_ms > 0
    assert isinstance(health.timestamp_ms, int)


@pytest.mark.asyncio
async def test_get_runtime_health_has_services_dict():
    """Test that runtime health includes services dictionary."""
    health = await get_runtime_health(tenant_id="test-tenant")
    assert isinstance(health.services, dict)
    # Should have service names as keys
    for name, svc in health.services.items():
        assert isinstance(name, str)
        assert svc.name == name
        assert hasattr(svc, "status")
        assert hasattr(svc, "latency_ms")
