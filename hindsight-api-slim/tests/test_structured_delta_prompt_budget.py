"""Tests for structured-delta prompt input budgeting."""

from hindsight_api.engine.reflect.prompts import (
    STRUCTURED_DELTA_SYSTEM_PROMPT,
    _fit_structured_delta_prompt_parts,
    build_mental_model_refresh_context,
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


def test_delta_prompt_ends_with_the_date_order_rule():
    """The date-order rule is the last thing the model reads.

    Facts arrive out of date order, so a delta batch can bring an ownership change
    OLDER than the owner the page records. Stated only in the system prompt, a
    35B model kept the newer owner in 0 of 8 replays; the same rule at the end of
    the user message, where it is read last, kept it in 7 of 8. So the position is
    part of the contract, not just the wording.
    """
    prompt = build_structured_delta_prompt(
        current_document_json='{"sections": []}',
        candidate_markdown="synthesis",
        supporting_facts=[{"id": "1", "text": "new fact", "type": "world"}],
        source_query="topic?",
        max_output_tokens=2000,
    )
    tail = prompt[prompt.index("## Task") :]
    assert "latest-dated event is the current state" in tail
    assert prompt.rstrip().endswith("only adds history.")
    assert "later-DATED statement" in STRUCTURED_DELTA_SYSTEM_PROMPT


def test_refresh_context_asks_full_refresh_for_dates_and_delta_for_events():
    """Full and delta refreshes get opposite advice about "current".

    A full refresh writes the page, so it must record since when each state holds —
    without that date a backfilled event cannot be ordered against it. A delta
    synthesis sees only the new batch; asked the same thing it writes "X owns it as
    of April 2024", which the delta step reads as superseding a newer owner. So the
    delta synthesis reports dated events and never calls anything current.
    """
    full = build_mental_model_refresh_context("Ownership", delta=False)
    delta = build_mental_model_refresh_context("Ownership", delta=True)

    for context in (full, delta):
        assert context.startswith('You are writing a document called "Ownership".')
    assert "say since when it has been true" in full
    assert "Do NOT say what is current" not in full
    assert "Do NOT say what is current" in delta
    assert "say since when it has been true" not in delta
