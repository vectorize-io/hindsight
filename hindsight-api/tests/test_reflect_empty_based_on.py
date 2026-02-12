"""
Test reflect endpoint with empty based_on (no memories scenario).

This test verifies that the API returns the correct based_on format:
- v0.3.0 (old): returned based_on as list []
- v0.4.0+ (current): returns based_on as object {"memories": [], "mental_models": [], "directives": []}
"""

import pytest
import pytest_asyncio
import httpx
from hindsight_api.api import create_app


@pytest_asyncio.fixture
async def api_client(memory):
    """Create an async test client for the FastAPI app."""
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_reflect_with_no_memories_empty_bank(api_client):
    """Test reflect on an empty bank (no memories) with include.facts enabled."""
    bank_id = "test_empty_bank"

    # Reflect on empty bank with facts requested
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/reflect",
        json={
            "query": "What do you know about machine learning?",
            "budget": "low",
            "include": {
                "facts": {}  # Request facts but bank is empty
            }
        }
    )

    assert response.status_code == 200
    data = response.json()

    # DEBUG: Print what the API actually returned
    import json
    print("\n" + "="*80)
    print("API Response:")
    print(json.dumps(data, indent=2))
    print("="*80 + "\n")

    # Verify response structure
    assert "text" in data
    assert "based_on" in data

    # The API should return based_on as either:
    # 1. null/None (if include.facts not set)
    # 2. {"memories": [], "mental_models": [], "directives": []} (if include.facts set but empty)
    # It should NEVER return based_on: []

    based_on = data.get("based_on")
    if based_on is not None:
        assert isinstance(based_on, dict), f"based_on should be dict or null, got {type(based_on)}: {based_on}"
        assert not isinstance(based_on, list), f"based_on should NEVER be a list! Got: {based_on}"
        assert "memories" in based_on
        assert "mental_models" in based_on
        assert "directives" in based_on
        # All should be empty lists
        assert based_on["memories"] == []
        assert based_on["mental_models"] == []
        assert based_on["directives"] == []

    # Verify the structure is parseable as proper types
    assert isinstance(data["text"], str)
    if based_on is not None:
        # Verify it's the v0.4.0+ format (object with arrays)
        assert isinstance(based_on["memories"], list)
        assert isinstance(based_on["mental_models"], list)
        assert isinstance(based_on["directives"], list)


@pytest.mark.asyncio
async def test_reflect_without_include_facts(api_client):
    """Test reflect without requesting facts (based_on should be None)."""
    bank_id = "test_no_facts"

    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/reflect",
        json={
            "query": "Hello world",
            "budget": "low"
            # No include.facts
        }
    )

    assert response.status_code == 200
    data = response.json()

    # When include.facts is not set, based_on should not be in response (or be null)
    based_on = data.get("based_on")
    assert based_on is None, f"based_on should be None when not requested, got {type(based_on)}: {based_on}"

    # Verify structure
    assert isinstance(data["text"], str)
