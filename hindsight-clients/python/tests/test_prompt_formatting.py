"""
Tests for RecallResponse.to_prompt_string.
"""

import json

from hindsight_client import RecallResponse


def test_to_prompt_string_facts_only():
    response = RecallResponse.from_dict({
        "results": [
            {
                "id": "1",
                "text": "Alice works at Google",
                "type": "world",
                "context": "work",
                "occurred_start": "2024-01-15T10:00:00Z",
                "occurred_end": "2024-06-15T10:00:00Z",
                "mentioned_at": "2024-03-01T09:00:00Z",
            },
            {"id": "2", "text": "The sky is blue", "type": "world"},
        ]
    })
    prompt = response.to_prompt_string()
    assert prompt.startswith("FACTS:\n")
    facts = json.loads(prompt[len("FACTS:\n"):])
    assert len(facts) == 2
    assert facts[0] == {
        "text": "Alice works at Google",
        "context": "work",
        "occurred_start": "2024-01-15T10:00:00Z",
        "occurred_end": "2024-06-15T10:00:00Z",
        "mentioned_at": "2024-03-01T09:00:00Z",
    }
    assert facts[1] == {"text": "The sky is blue"}


def test_to_prompt_string_with_chunks():
    response = RecallResponse.from_dict({
        "results": [
            {"id": "1", "text": "Alice works at Google", "type": "world", "chunk_id": "chunk_1"},
        ],
        "chunks": {
            "chunk_1": {"id": "chunk_1", "text": "Alice works at Google on the AI team since 2020.", "chunk_index": 0},
        },
    })
    prompt = response.to_prompt_string()
    facts = json.loads(prompt[len("FACTS:\n"):])
    assert facts[0]["source_chunk"] == "Alice works at Google on the AI team since 2020."


def test_to_prompt_string_with_entities():
    response = RecallResponse.from_dict({
        "results": [
            {"id": "1", "text": "Alice works at Google", "type": "world"},
        ],
        "entities": {
            "Alice": {
                "entity_id": "e1",
                "canonical_name": "Alice",
                "observations": [{"text": "Alice is a senior engineer at Google working on AI."}],
            },
        },
    })
    prompt = response.to_prompt_string()
    assert "ENTITIES:" in prompt
    assert "## Alice" in prompt
    assert "Alice is a senior engineer at Google working on AI." in prompt


def test_to_prompt_string_with_chunks_and_entities():
    response = RecallResponse.from_dict({
        "results": [
            {"id": "1", "text": "Alice works at Google", "type": "world", "chunk_id": "c1"},
        ],
        "chunks": {
            "c1": {"id": "c1", "text": "Full conversation about Alice at Google.", "chunk_index": 0},
        },
        "entities": {
            "Alice": {
                "entity_id": "e1",
                "canonical_name": "Alice",
                "observations": [{"text": "Alice is a senior engineer."}],
            },
        },
    })
    prompt = response.to_prompt_string()
    # Facts with chunk
    facts = json.loads(prompt.split("ENTITIES:")[0].strip()[len("FACTS:\n"):])
    assert facts[0]["source_chunk"] == "Full conversation about Alice at Google."
    # Entities
    assert "## Alice\nAlice is a senior engineer." in prompt


def test_to_prompt_string_empty_results():
    response = RecallResponse.from_dict({"results": []})
    prompt = response.to_prompt_string()
    assert prompt == "FACTS:\n[]"


def test_to_prompt_string_chunk_id_not_in_chunks():
    """chunk_id that doesn't match any chunk should be ignored."""
    response = RecallResponse.from_dict({
        "results": [
            {"id": "1", "text": "Some fact", "type": "world", "chunk_id": "missing_chunk"},
        ],
    })
    prompt = response.to_prompt_string()
    facts = json.loads(prompt[len("FACTS:\n"):])
    assert "source_chunk" not in facts[0]
