"""
Embedding processing for retain pipeline.

Handles augmenting fact texts with temporal information and generating embeddings.
"""

import logging
from collections.abc import Callable, Sequence
from datetime import datetime

from . import embedding_utils
from .types import ExtractedFact

logger = logging.getLogger(__name__)


def format_readable_date(value: datetime) -> str:
    """Format a date for temporal embedding text."""
    return f"{value.strftime('%B')} {value.strftime('%Y')}"


def build_fact_embedding_text(
    *,
    fact_text: str,
    occurred_start: datetime | None,
    occurred_end: datetime | None,
    mentioned_at: datetime | None,
    entities: Sequence[str],
    format_date_fn: Callable[[datetime], str],
) -> str:
    """Build the embedding input for a retained fact."""
    # Use occurred_start as the representative date, fall back to mentioned_at.
    fact_date = occurred_start or mentioned_at
    if fact_date is not None:
        readable_date = format_date_fn(fact_date)
        if occurred_end and occurred_end != occurred_start:
            readable_end = format_date_fn(occurred_end)
            text = f"{fact_text} (happened from {readable_date} to {readable_end})"
        else:
            text = f"{fact_text} (happened in {readable_date})"
    else:
        text = fact_text

    entity_names = [entity for entity in entities if entity]
    if entity_names:
        text = f"{text} [{', '.join(entity_names)}]"
    return text


def build_mental_model_embedding_text(name: str | None, content: str | None) -> str:
    """Build the canonical embedding input for the current mental-model row."""
    return f"{name or ''} {content or ''}"


def augment_texts_with_dates(facts: list[ExtractedFact], format_date_fn) -> list[str]:
    """
    Augment fact texts with readable dates for better temporal matching.

    This allows queries like "camping in June" to match facts that happened in June.

    Args:
        facts: List of ExtractedFact objects
        format_date_fn: Function to format datetime to readable string

    Returns:
        List of augmented text strings (same length as facts)
    """
    augmented_texts = []
    for fact in facts:
        # Entity names (including key:value labels) improve retrieval without polluting stored content.
        augmented_texts.append(
            build_fact_embedding_text(
                fact_text=fact.fact_text,
                occurred_start=fact.occurred_start,
                occurred_end=fact.occurred_end,
                mentioned_at=fact.mentioned_at,
                entities=fact.entities,
                format_date_fn=format_date_fn,
            )
        )
    return augmented_texts


async def generate_embeddings_batch(embeddings_model, texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.

    Args:
        embeddings_model: Embeddings model instance
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (same length as texts)
    """
    if not texts:
        return []

    embeddings = await embedding_utils.generate_embeddings_batch(embeddings_model, texts)

    return embeddings
