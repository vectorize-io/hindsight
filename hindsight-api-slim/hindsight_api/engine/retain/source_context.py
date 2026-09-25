"""Bounded source text for reference resolution, separate from the extraction target."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .attachment_content import PLACEHOLDER_RE


@dataclass(frozen=True)
class ContextualChunk:
    text: str
    previous_source: str


def extend_source_context(previous: str, text: str, budget: int) -> str:
    """Keep the newest source text, measured in characters including separators.

    Attachment bytes are deliberately excluded: resolving a prior placeholder would
    shift the target's attachment numbers and bypass its per-chunk attachment cap.
    A window may start mid-chunk when the preceding chunk exceeds the budget.
    """
    if budget == 0:
        return ""
    text = PLACEHOLDER_RE.sub("[attachment omitted]", text)
    return (previous + "\n\n" + text if previous else text)[-budget:]


def contextual_chunks(chunks: Iterable[str], budget: int, previous_source: str = "") -> Iterator[ContextualChunk]:
    """Prepare windows before extraction without depending on any model outputs."""
    previous = extend_source_context("", previous_source, budget)
    for text in chunks:
        yield ContextualChunk(text=text, previous_source=previous)
        previous = extend_source_context(previous, text, budget)
