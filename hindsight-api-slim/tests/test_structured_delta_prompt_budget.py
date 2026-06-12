"""Tests for structured-delta prompt input budgeting."""

from hindsight_api.engine.reflect.prompts import (
    STRUCTURED_DELTA_SYSTEM_PROMPT,
    _fit_structured_delta_prompt_parts,
    build_structured_delta_prompt,
)
from hindsight_api.engine.reflect.tokenization import count_cl100k_tokens


def test_build_structured_delta_prompt_truncates_huge_document():
    huge_doc = (
        '{"sections": [{"id": "s1", "heading": "H", "level": 1, "blocks": [{"type": "paragraph", "text": "'
        + ("word " * 50_000)
        + '"}]}]}'
    )
    prompt = build_structured_delta_prompt(
        current_document_json=huge_doc,
        candidate_markdown="short synthesis",
        supporting_facts=[{"id": "1", "text": "new fact", "type": "world"}],
        source_query="topic?",
        max_input_tokens=4000,
    )
    total = count_cl100k_tokens(STRUCTURED_DELTA_SYSTEM_PROMPT) + count_cl100k_tokens(prompt)
    assert total < 12_000
    assert "truncated to fit the model" in prompt


def test_fit_structured_delta_keeps_small_prompt_unchanged():
    doc_out, cand_out, facts_out, truncated = _fit_structured_delta_prompt_parts(
        source_query="q",
        current_document_json='{"sections": []}',
        candidate_markdown="hello",
        facts_block="one line",
        budget_hint="",
        task_footer="## Task\nDo it.",
        max_input_tokens=24_000,
    )
    assert not truncated
    assert doc_out == '{"sections": []}'
    assert cand_out == "hello"
    assert facts_out == "one line"
