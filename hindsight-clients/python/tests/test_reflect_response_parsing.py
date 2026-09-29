"""
Test ReflectResponse parsing for different API versions.

This tests the client's ability to parse reflect responses from:
- v0.3.0 API (based_on as list)
- v0.4.0+ API (based_on as object)
"""

import pytest
from hindsight_client_api.models.reflect_response import ReflectResponse
from hindsight_client_api.models.reflect_based_on import ReflectBasedOn


def test_parse_v4_format_with_empty_based_on():
    """Test parsing v0.4.0+ format with empty based_on object."""
    response_data = {
        "text": "I don't have any information about that.",
        "based_on": {
            "memories": [],
            "mental_models": [],
            "directives": []
        }
    }

    response = ReflectResponse.from_dict(response_data)
    assert response is not None
    assert response.text == "I don't have any information about that."
    assert response.based_on is not None
    assert isinstance(response.based_on, ReflectBasedOn)
    assert response.based_on.memories == []
    assert response.based_on.mental_models == []
    assert response.based_on.directives == []


def test_parse_v4_format_with_null_based_on():
    """Test parsing v0.4.0+ format with null based_on (include.facts not set)."""
    response_data = {
        "text": "Hello!",
        "based_on": None
    }

    response = ReflectResponse.from_dict(response_data)
    assert response is not None
    assert response.text == "Hello!"
    assert response.based_on is None


def test_parse_v4_format_with_populated_based_on():
    """Test parsing v0.4.0+ format with actual facts."""
    response_data = {
        "text": "Based on my knowledge, AI is transformative.",
        "based_on": {
            "memories": [
                {
                    "id": "mem-123",
                    "text": "AI is used in healthcare",
                    "type": "world",
                    "context": None,
                    "occurred_start": None,
                    "occurred_end": None
                }
            ],
            "mental_models": [
                {
                    "id": "mm-456",
                    "text": "AI transforms industries",
                    "context": "technology trends"
                }
            ],
            "directives": [
                {
                    "id": "dir-789",
                    "name": "Be concise",
                    "content": "Keep responses brief"
                }
            ]
        }
    }

    response = ReflectResponse.from_dict(response_data)
    assert response is not None
    assert response.text == "Based on my knowledge, AI is transformative."
    assert response.based_on is not None
    assert len(response.based_on.memories) == 1
    assert response.based_on.memories[0].id == "mem-123"
    assert len(response.based_on.mental_models) == 1
    assert response.based_on.mental_models[0].id == "mm-456"
    assert len(response.based_on.directives) == 1
    assert response.based_on.directives[0].id == "dir-789"


def test_parse_v3_format_with_empty_list_fails():
    """
    Test that v0.3.0 format (based_on as list) fails validation.

    This is a BREAKING CHANGE from v0.3.0 to v0.4.0.
    Clients using v0.4.x SDK cannot parse v0.3.0 API responses.

    Users must either:
    - Upgrade API to v0.4.0+
    - Use v0.3.0 client with v0.3.0 API
    """
    response_data = {
        "text": "No information available.",
        "based_on": []  # v0.3.0 format - incompatible with v0.4.0+ client
    }

    with pytest.raises(Exception) as exc_info:
        ReflectResponse.from_dict(response_data)

    # Should fail with validation error
    assert "ValidationError" in str(type(exc_info.value).__name__) or "validation" in str(exc_info.value).lower()


def test_parse_missing_based_on_field():
    """Test parsing response when based_on field is omitted entirely."""
    response_data = {
        "text": "Hello!"
        # based_on field not present
    }

    response = ReflectResponse.from_dict(response_data)
    assert response is not None
    assert response.text == "Hello!"
    assert response.based_on is None
