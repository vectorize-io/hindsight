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
