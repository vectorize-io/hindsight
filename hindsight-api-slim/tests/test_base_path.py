"""
Integration test for API base path support.

Tests that the API works correctly when deployed with a base path (e.g., /hindsight)
for reverse proxy deployments.
"""

import os
import pytest
import pytest_asyncio
import httpx
from hindsight_api.api import create_app
from hindsight_api.config import clear_config_cache


@pytest_asyncio.fixture
async def api_client_with_base_path(memory):
    """Create an async test client for the FastAPI app with a base path."""
    # Set base path in environment
    base_path = "/hindsight"
    os.environ["HINDSIGHT_API_BASE_PATH"] = base_path

    # Clear config cache to force reload with new base_path
    clear_config_cache()

    # Memory is already initialized by the conftest fixture (with migrations)
    app = create_app(memory, initialize_memory=False)

    # Use base_url with base path
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=f"http://test{base_path}") as client:
        yield client

    # Cleanup: unset base path
    os.environ.pop("HINDSIGHT_API_BASE_PATH", None)
    clear_config_cache()


@pytest_asyncio.fixture
async def api_client_without_base_path(memory):
    """Create an async test client for the FastAPI app without a base path (root)."""
    # Ensure no base path is set
    os.environ.pop("HINDSIGHT_API_BASE_PATH", None)
    clear_config_cache()

    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_base_path_health_endpoint(api_client_with_base_path):
    """Test that health endpoint works with base path."""
    # With base path set to /hindsight, health should be at /hindsight/health
    # But since our client base_url is already http://test/hindsight, we request /health
    response = await api_client_with_base_path.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["ok", "healthy"]  # Accept both formats


@pytest.mark.asyncio
async def test_base_path_banks_endpoint(api_client_with_base_path):
    """Test that banks endpoint works with base path."""
    response = await api_client_with_base_path.get("/v1/default/banks")
    assert response.status_code == 200
    data = response.json()
    assert "banks" in data


@pytest.mark.asyncio
async def test_base_path_openapi_schema(api_client_with_base_path):
    """Test that OpenAPI schema includes correct base path in servers."""
    response = await api_client_with_base_path.get("/openapi.json")
    assert response.status_code == 200
    openapi_schema = response.json()

    # Check that servers array includes base path
    assert "servers" in openapi_schema
    servers = openapi_schema["servers"]
    assert len(servers) > 0
    # FastAPI should set server URL to the root_path
    assert servers[0]["url"] == "/hindsight"


@pytest.mark.asyncio
async def test_base_path_docs_redirect(api_client_with_base_path):
    """Test that /docs redirects correctly with base path."""
    # FastAPI docs endpoint should work
    response = await api_client_with_base_path.get("/docs", follow_redirects=False)
    # Should either return 200 (direct) or 307 (redirect to trailing slash)
    assert response.status_code in [200, 307]


@pytest.mark.asyncio
async def test_base_path_metrics(api_client_with_base_path):
    """Test that metrics endpoint works with base path."""
    response = await api_client_with_base_path.get("/metrics")
    assert response.status_code == 200
    # Metrics should be in Prometheus format
    assert "# HELP" in response.text or "# TYPE" in response.text


@pytest.mark.asyncio
async def test_base_path_full_workflow(api_client_with_base_path):
    """
    Test a full retain/recall workflow with base path.

    This ensures that all memory operations work correctly when the API
    is deployed with a base path.
    """
    bank_id = "test_base_path_bank"

    # 1. Store a memory (implicitly creates the bank)
    response = await api_client_with_base_path.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                {
                    "content": "The API supports base path deployment for reverse proxy use cases.",
                    "context": "testing base path feature",
                }
            ]
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True

    # 3. Recall the memory
    response = await api_client_with_base_path.post(
        f"/v1/default/banks/{bank_id}/memories/recall", json={"query": "base path support"}
    )
    assert response.status_code == 200
    recall_result = response.json()
    # API returns "results" not "memories"
    assert "results" in recall_result
    assert len(recall_result["results"]) > 0


@pytest.mark.asyncio
async def test_without_base_path_still_works(api_client_without_base_path):
    """
    Regression test: ensure default behavior (no base path) still works.

    This test verifies that when HINDSIGHT_API_BASE_PATH is not set,
    the API works at the root path as before.
    """
    # Health check at root
    response = await api_client_without_base_path.get("/health")
    assert response.status_code == 200

    # Banks endpoint at root
    response = await api_client_without_base_path.get("/v1/default/banks")
    assert response.status_code == 200

    # OpenAPI schema should have empty or "/" server path
    response = await api_client_without_base_path.get("/openapi.json")
    assert response.status_code == 200
    openapi_schema = response.json()
    servers = openapi_schema.get("servers", [])
    if servers:
        # Server URL should be empty string (root) or "/"
        assert servers[0]["url"] in ["", "/"]


@pytest.mark.skip(reason="MCP endpoint routing with base path needs investigation")
@pytest.mark.asyncio
async def test_base_path_mcp_endpoint(api_client_with_base_path):
    """Test that MCP endpoint is accessible with base path."""
    bank_id = "test_mcp_bank"

    # MCP endpoint should be mounted at /mcp/{bank_id}/
    # The MCP server uses a different protocol, so just check the root exists
    response = await api_client_with_base_path.get(f"/mcp/{bank_id}/")
    # MCP may return various status codes, but should not be 404 (not found)
    # Accept 405 (method not allowed), 400 (bad request), etc.
    assert response.status_code != 404, "MCP endpoint should exist"
