"""
Tests for format_facts_for_prompt in think_utils.
"""

import json

from hindsight_api.engine.response_models import MemoryFact
from hindsight_api.engine.search.think_utils import format_facts_for_prompt


def test_format_facts_includes_temporal_fields():
    """All temporal fields (occurred_start, occurred_end, mentioned_at) should appear in the JSON."""
    facts = [
        MemoryFact(
            id="fact-1",
            text="Team offsite in February",
            fact_type="experience",
            occurred_start="2024-02-01T00:00:00Z",
            occurred_end="2024-02-28T23:59:59Z",
            mentioned_at="2024-03-05T10:00:00Z",
        )
    ]
    result = json.loads(format_facts_for_prompt(facts))
    assert len(result) == 1
    assert result[0]["text"] == "Team offsite in February"
    assert result[0]["occurred_start"] == "2024-02-01T00:00:00Z"
    assert result[0]["occurred_end"] == "2024-02-28T23:59:59Z"
    assert result[0]["mentioned_at"] == "2024-03-05T10:00:00Z"


def test_format_facts_omits_null_temporal_fields():
    """Null temporal fields should not appear in the JSON."""
    facts = [
        MemoryFact(
            id="fact-2",
            text="The sky is blue",
            fact_type="world",
        )
    ]
    result = json.loads(format_facts_for_prompt(facts))
    assert len(result) == 1
    assert "occurred_start" not in result[0]
    assert "occurred_end" not in result[0]
    assert "mentioned_at" not in result[0]


def test_format_facts_partial_temporal_fields():
    """Only non-null temporal fields should appear."""
    facts = [
        MemoryFact(
            id="fact-3",
            text="Meeting happened",
            fact_type="experience",
            occurred_start="2024-06-01T09:00:00Z",
        )
    ]
    result = json.loads(format_facts_for_prompt(facts))
    assert result[0]["occurred_start"] == "2024-06-01T09:00:00Z"
    assert "occurred_end" not in result[0]
    assert "mentioned_at" not in result[0]


def test_format_facts_empty_list():
    """Empty list should return '[]'."""
    assert format_facts_for_prompt([]) == "[]"
