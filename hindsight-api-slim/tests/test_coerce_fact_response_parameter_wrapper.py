"""
Test that _coerce_fact_response unwraps the ``{"parameter": {...}}`` envelope
that Vertex AI Claude returns for tool-use structured output via litellm.

Without this fix, retains via ``litellm`` + ``vertex_ai/claude-*`` silently
produce 0 facts because the extracted facts sit inside ``response["parameter"]``
and ``response.get("facts", [])`` returns an empty list.
"""

from hindsight_api.engine.retain.fact_extraction import _coerce_fact_response


def test_unwraps_parameter_envelope():
    """Vertex AI Claude wraps tool output in {"parameter": {"facts": [...]}}."""
    response = {
        "parameter": {
            "facts": [
                {"what": "version is 1.0", "who": "user", "fact_type": "world"},
            ]
        }
    }
    result = _coerce_fact_response(response)
    assert "facts" in result
    assert len(result["facts"]) == 1
    assert result["facts"][0]["what"] == "version is 1.0"


def test_passthrough_normal_response():
    """Normal {"facts": [...]} responses pass through unchanged."""
    response = {"facts": [{"what": "something", "fact_type": "world"}]}
    result = _coerce_fact_response(response)
    assert result is response


def test_passthrough_empty_facts():
    """Empty facts list passes through."""
    response = {"facts": []}
    result = _coerce_fact_response(response)
    assert result is response


def test_coerces_bare_list():
    """A bare list of fact dicts is wrapped into {"facts": [...]}."""
    response = [{"what": "a"}, {"what": "b"}]
    result = _coerce_fact_response(response)
    assert result == {"facts": [{"what": "a"}, {"what": "b"}]}


def test_rejects_non_dict_non_list():
    """Non-dict, non-list values return None."""
    assert _coerce_fact_response("bad") is None
    assert _coerce_fact_response(42) is None
    assert _coerce_fact_response(None) is None


def test_parameter_with_nested_structure():
    """Parameter envelope with additional metadata is still unwrapped."""
    response = {
        "parameter": {
            "facts": [{"what": "a"}],
            "metadata": {"extra": True},
        }
    }
    result = _coerce_fact_response(response)
    assert len(result["facts"]) == 1
    assert result.get("metadata") == {"extra": True}
