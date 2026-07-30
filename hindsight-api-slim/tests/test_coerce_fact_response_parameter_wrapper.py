"""
Test that _coerce_fact_response unwraps single-key envelope dicts whose
inner value contains the expected ``facts`` list.

Vertex AI Claude (via litellm) returns ``{"parameter": {"facts": [...]}}``,
but other providers/versions may use different wrapper keys (``"content"``,
``"data"``, ``"json_input"``).  The fix is key-agnostic: any single-key dict
whose sole value is a dict containing ``"facts"`` is unwrapped.
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


def test_unwraps_any_single_key_envelope():
    """Any single-key wrapper around {"facts": [...]} is unwrapped."""
    for key in ("content", "data", "json_input", "result"):
        response = {key: {"facts": [{"what": "a"}]}}
        result = _coerce_fact_response(response)
        assert "facts" in result, f"failed for wrapper key: {key}"
        assert result["facts"][0]["what"] == "a"


def test_no_unwrap_when_facts_at_top_level():
    """Normal {"facts": [...]} responses pass through unchanged."""
    response = {"facts": [{"what": "something", "fact_type": "world"}]}
    result = _coerce_fact_response(response)
    assert result is response


def test_no_unwrap_multi_key_dict():
    """Multi-key dicts are not treated as envelopes."""
    response = {"parameter": {"facts": [{"what": "a"}]}, "extra": True}
    result = _coerce_fact_response(response)
    assert result is response
    assert "parameter" in result


def test_no_unwrap_single_key_without_facts():
    """Single-key dict whose value lacks 'facts' is returned as-is."""
    response = {"parameter": {"other": "stuff"}}
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
