"""Shared utilities for prompt assembly."""

import re

_LONE_OPEN_BRACE = re.compile(r"(?<!\{)\{(?!\{)")
_LONE_CLOSE_BRACE = re.compile(r"(?<!\})\}(?!\})")


def escape_for_prompt(text: str) -> str:
    """Double any lone ``{`` / ``}`` so the text survives ``str.format`` untouched.

    Prompt templates are often passed through ``str.format`` to substitute real
    placeholders like ``{facts_text}``.  Any literal braces in caller-supplied
    text — e.g. a bank mission that contains JSON examples — would otherwise be
    interpreted as format keys and raise ``KeyError``.

    Idempotent: text that already contains escaped ``{{`` / ``}}`` pairs is
    left as-is.  Only lone braces (not adjacent to another brace of the same
    kind) are doubled.
    """
    text = _LONE_OPEN_BRACE.sub("{{", text)
    text = _LONE_CLOSE_BRACE.sub("}}", text)
    return text


def output_language_directive(language: str | None) -> str:
    """Return an LLM directive forcing all output into ``language``.

    Used by retain (fact extraction), consolidation (observations), and reflect
    (response synthesis) so HINDSIGHT_API_LLM_OUTPUT_LANGUAGE applies uniformly
    across every LLM-generated artifact. Returns an empty string when
    ``language`` is unset so the calling prompt stays unchanged.
    """
    if not language:
        return ""
    return (
        f"\n\nIMPORTANT: Respond exclusively in {language}. "
        f"Translate any source content into {language}. "
        f"All output text — including fact text, observations, entity names, "
        f"and the final response — must be in {language}."
    )
