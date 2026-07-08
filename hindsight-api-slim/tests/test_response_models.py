"""Tests for MemoryFact response-model validation."""

from hindsight_api.engine.response_models import MemoryFact

_BASE = {"id": "u1", "text": "hi", "fact_type": "observation"}


class TestMemoryFactMetadataCoercion:
    """metadata values may arrive as non-strings from JSONB; they must coerce to str.

    Regression for #2622: an integer ``original_id`` stored in ``metadata`` raised
    ``ValidationError`` (``string_type``) and blocked consolidation, because
    ``metadata`` is typed ``dict[str, str]`` and the raw dict passed straight
    through to Pydantic.
    """

    def test_integer_value_is_coerced_to_string(self):
        fact = MemoryFact(**_BASE, metadata={"original_id": 12345})
        assert fact.metadata == {"original_id": "12345"}

    def test_jsonb_string_with_integer_value_is_coerced(self):
        # asyncpg may hand back JSONB as a raw string; ints inside must still coerce.
        fact = MemoryFact(**_BASE, metadata='{"original_id": 12345, "n": 3}')
        assert fact.metadata == {"original_id": "12345", "n": "3"}

    def test_string_values_pass_through_unchanged(self):
        fact = MemoryFact(**_BASE, metadata={"source": "slack", "channel": "eng"})
        assert fact.metadata == {"source": "slack", "channel": "eng"}

    def test_none_metadata_stays_none(self):
        assert MemoryFact(**_BASE, metadata=None).metadata is None
