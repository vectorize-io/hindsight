"""Which pipeline step an LLM call belongs to, decided in exactly one place.

Nothing on the wire says "this is fact extraction". The scope is a server-side
concept, the ``json_schema`` name is the constant ``"response"``, and the model
name is whatever the deployment configured. All that reaches the stub is the
prompt.

So a step is identified by an anchor: a short, load-bearing phrase from its
prompt. Tests never write those phrases — they say ``on_step("extract_facts")``
and the mapping lives here. That is the difference between a prompt edit costing
one line and costing thirty red tests, which is the failure mode that kills
wiremock-style suites.

When an anchor stops matching, the miss report prints the prompt that arrived;
fix the anchor here, once.
"""

from __future__ import annotations

# Anchor phrases, each unique to one step's prompt. Keep them short: the more of
# a prompt an anchor quotes, the more prompt edits break it.
STEP_ANCHORS: dict[str, str] = {
    "extract_facts": "Extract facts from the following chunk.",
    "consolidate": "### Existing observations",
    # The reflect loop uses two different system prompts: a tool-using role for
    # the search turns, and a separate synthesising role for the turn that writes
    # the answer. They need separate anchors or the final turn matches nothing.
    # The preamble, not the role line: a bank with a reflect mission *replaces*
    # the default role, so anchoring on that made every story break the moment a
    # mission was set. This line is on every reflect turn regardless.
    "reflect": "CRITICAL: You MUST ONLY use information from retrieved tool results.",
    "reflect_answer": "You are a thoughtful assistant that synthesizes answers from retrieved memories.",
    # Structured output is a *second* call after the answer: an extraction pass
    # that reshapes the prose into the caller's schema. Anchored separately
    # because a story about the schema is about this call, not the answer.
    "reflect_structured": "You are a precise data extraction assistant.",
    # An over-long reflect answer gets a *second* call that rewrites it to the
    # caller's token budget. Anchored separately because a story about the budget
    # is about this call, not the one that wrote the answer.
    "reflect_trim": "Rewrite the user's text so it fits within the requested token budget.",
    "connection_probe": "Say 'ok'",
}


class UnknownStep(KeyError):
    """Raised for a step name with no anchor, so a typo fails loudly at declaration."""


def anchor_for(step: str) -> str:
    try:
        return STEP_ANCHORS[step]
    except KeyError:
        known = ", ".join(sorted(STEP_ANCHORS))
        raise UnknownStep(f"unknown step {step!r}; known steps: {known}") from None
