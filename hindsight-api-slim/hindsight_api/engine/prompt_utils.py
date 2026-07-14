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


def minified_json_directive(same_output: str, extra: str = "") -> str:
    """Directive telling the model to emit compact/minified JSON (no pretty-print
    whitespace), for machine-parsed structured outputs.

    Gated by the global ``HINDSIGHT_API_MINIFY_LLM_JSON_OUTPUT`` flag (default on);
    returns an empty string when disabled so the calling prompt is unchanged.

    ``same_output`` states what must stay identical (e.g. "extract exactly the
    same facts") so the directive is understood as formatting-only and does not
    change the model's extraction/decision behaviour. ``extra`` appends an
    operation-specific caveat (e.g. the delta-ops string-escaping note).

    The returned text is deliberately brace-free: callers append it to prompts
    that are later run through ``str.format`` (e.g. the consolidation system
    prompt), where a lone ``{`` / ``}`` would raise ``KeyError``.
    """
    # Lazy import: prompt_utils is imported widely and config is a heavier module;
    # importing at call time avoids any import cycle.
    from ..config import get_config

    if not get_config().minify_llm_json_output:
        return ""
    return (
        "\n\nOUTPUT WHITESPACE: Emit the JSON minified — a single line with no "
        "line breaks, no indentation, and no space after ':' or ','. This is a "
        "formatting instruction only; " + same_output + "." + extra
    )
