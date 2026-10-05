"""Conservative validation of explicit memory citations, not arbitrary UUID prose."""

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from ..response_models import ToolCallTrace
from .models import ToolCall
from .structured_doc import StructuredDocument, render_document

_UUID = r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
_CITATION = re.compile(
    rf"^ {{0,3}}Source:[ \t]+(?P<quote>`?)(?P<source>{_UUID})(?P=quote)(?![\w-])"
    rf"|\[[^\]\n]+\]\((?:https?://[^/\s)]+)?/v1/default/banks/"
    rf"(?P<bank>[^/\s?#)]+)/memories/(?P<link>{_UUID})(?:[?#][^\s)]*)?\)",
    re.MULTILINE,
)
# Examples in fences and inline code are authored text, not live citations.
_CODE = re.compile(r"^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?(?:^ {0,3}\1[ \t]*$|\Z)|(`+)[^\n]*?\2", re.MULTILINE | re.DOTALL)
_UNVERIFIED = " (unverified)"


@dataclass(frozen=True)
class AnnotatedAnswer:
    text: str
    document: StructuredDocument | None


def _mark_text(text: str, retrieved_ids: Collection[str], bank_id: str | None) -> str:
    protected = [match.span() for match in _CODE.finditer(text)]

    def mark(match: re.Match[str]) -> str:
        if any(start <= match.start() < end for start, end in protected):
            return match.group()
        memory_id = match["source"] or match["link"]
        same_bank = match["bank"] is None or match["bank"] == bank_id
        if (same_bank and memory_id in retrieved_ids) or text[match.end() :].startswith(_UNVERIFIED):
            return match.group()
        # A well-formed UUID or a matching prefix proves nothing. Preserve the ID
        # for inspection, but do not present an unretrieved source as verified.
        return match.group() + _UNVERIFIED

    return _CITATION.sub(mark, text)


def annotate_memory_citations(
    text: str,
    document: StructuredDocument | None,
    retrieved_ids: Collection[str],
    bank_id: str | None,
) -> AnnotatedAnswer:
    """Mark explicit, unretrieved references and keep the two document views aligned.

    Recognition deliberately stops at ``Source: UUID`` and REST memory links.
    A UUID elsewhere might describe an external request, so scanning/replacing
    all UUIDs would corrupt ordinary content. This makes no database-existence
    claim: unverified means absent from the supplied retrieval evidence.
    """
    if document is None:
        return AnnotatedAnswer(text=_mark_text(text, retrieved_ids, bank_id), document=None)
    # Preserve section/block IDs, which later delta operations address, and never
    # modify the caller's document or read the rendered markdown back into it.
    annotated = document.model_copy(
        update={
            "sections": [
                section.model_copy(
                    update={
                        "blocks": [
                            block.model_copy(update={"text": _mark_text(block.text, retrieved_ids, bank_id)})
                            for block in section.blocks
                        ]
                    }
                )
                for section in document.sections
            ]
        }
    )
    return AnnotatedAnswer(text=render_document(annotated).strip(), document=annotated)


def retrieved_memory_ids(tool_trace: Sequence[ToolCall | ToolCallTrace]) -> set[str]:
    """Memory/observation IDs returned by tools, rather than IDs declared as used."""
    return {
        item["id"]
        for call in tool_trace
        if "error" not in call.output
        for key in ("memories", "observations")
        for item in call.output.get(key, [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
