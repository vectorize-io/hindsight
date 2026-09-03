"""Tests for structured-delta prompt input budgeting."""

from hindsight_api.engine.reflect.prompts import (
    STRUCTURED_DELTA_SYSTEM_PROMPT,
    _fit_structured_delta_prompt_parts,
    build_structured_delta_prompt,
)
from hindsight_api.engine.reflect.tokenization import count_prompt_tokens


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
    total = count_prompt_tokens(STRUCTURED_DELTA_SYSTEM_PROMPT) + count_prompt_tokens(prompt)
    assert total < 12_000
    assert "truncated to fit the model" in prompt


def test_fit_structured_delta_keeps_small_prompt_unchanged():
    fitted = _fit_structured_delta_prompt_parts(
        source_query="q",
        current_document_json='{"sections": []}',
        candidate_markdown="hello",
        facts_block="one line",
        budget_hint="",
        task_footer="## Task\nDo it.",
        max_input_tokens=24_000,
    )
    assert not fitted.truncated
    assert fitted.document_json == '{"sections": []}'
    assert fitted.candidate == "hello"
    assert fitted.facts == "one line"


def test_retraction_prompt_does_not_transpose_surviving_and_retracted():
    """The two fact lists must land under their own headings.

    ``build_structured_retraction_prompt`` reuses ``_fit_structured_delta_prompt_parts``
    with the surviving facts in the ``candidate`` slot and the retracted ones in the
    ``facts`` slot. Both are ``str``, so transposing them type-checks and produces a
    grammatical prompt — one that tells the model to strip content resting on facts
    that are still valid and keep content resting on facts that were withdrawn.
    This asserts the mapping directly, since no type can.
    """
    from hindsight_api.engine.reflect.prompts import build_structured_retraction_prompt

    prompt = build_structured_retraction_prompt(
        current_document_json='{"sections": []}',
        retracted_facts=[{"id": "r1", "text": "WITHDRAWN_MARKER", "type": "world", "context": ""}],
        surviving_facts=[{"id": "v1", "text": "STILL_VALID_MARKER", "type": "world", "context": ""}],
        source_query="topic",
    )

    surviving_heading = prompt.index("## STILL-SUPPORTED FACTS")
    retracted_heading = prompt.index("## RETRACTED FACTS")
    assert surviving_heading < prompt.index("STILL_VALID_MARKER") < retracted_heading, (
        "the surviving fact must appear under STILL-SUPPORTED, before the RETRACTED heading"
    )
    assert prompt.index("WITHDRAWN_MARKER") > retracted_heading, "the retracted fact must appear under RETRACTED FACTS"
