"""The document's own token budget on the delta leg.

``max_tokens`` was enforced in exactly one place: a rewrite of the *synthesis*
answer. In delta mode that answer is only context for the operations call, and
the document that actually gets stored was never measured against the budget at
all. A delta refresh only ever adds, so a long-lived page grows every round
(~20 tokens per round, measured over 45 rounds in the document-evolution
benchmark) and drifts past its configured size with nothing noticing.

Truncating it here would delete knowledge nobody asked to delete, so the budget
is stated to the model — which reclaims space with the same operations it uses
for everything else — and the outcome is recorded either way.

These cover the prompt itself; the engine-level tests (that a refresh records the
size, and that going over is reported rather than truncated) live beside the
other refresh tests in ``test_mental_model_delta.py``, where their fixtures are.
"""

from __future__ import annotations

from hindsight_api.engine.reflect.prompts import build_structured_delta_prompt


def _prompt(document_tokens: int, document_budget: int) -> str:
    return build_structured_delta_prompt(
        current_document_json='{"version": 2, "sections": []}',
        candidate_markdown="candidate",
        supporting_facts=[{"id": "o1", "text": "a new fact", "type": "observation"}],
        source_query="Document the API",
        document_tokens=document_tokens,
        document_budget=document_budget,
    )


class TestDocumentBudgetPrompt:
    def test_silent_when_well_under_budget(self):
        assert "Document budget" not in _prompt(100, 2048)

    def test_warns_when_close_to_the_budget(self):
        prompt = _prompt(1700, 2048)
        assert "## Document budget" in prompt
        assert "EXCEEDED" not in prompt
        assert "prefer replacing stale blocks" in prompt

    def test_asks_for_room_when_over_budget(self):
        prompt = _prompt(2500, 2048)
        assert "## Document budget (EXCEEDED)" in prompt
        assert "2500" in prompt and "2048" in prompt
        assert "remove_block" in prompt

    def test_never_asks_to_drop_the_new_facts(self):
        prompt = _prompt(2500, 2048)
        assert "never" in prompt.lower()
        assert "dropping the facts you are integrating" in prompt

    def test_absent_when_the_caller_does_not_pass_a_budget(self):
        prompt = build_structured_delta_prompt(
            current_document_json="{}",
            candidate_markdown="c",
            supporting_facts=[],
            source_query="q",
        )
        assert "Document budget" not in prompt
