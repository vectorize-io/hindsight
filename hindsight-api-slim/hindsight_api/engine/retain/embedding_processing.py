"""
Embedding processing for retain pipeline.

Handles augmenting fact texts with temporal information and generating embeddings.
"""

import logging

from ..temporal_precision import format_embedding_occurrence_date, resolve_stored_occurrence_precision
from . import embedding_utils
from .types import ExtractedFact

logger = logging.getLogger(__name__)


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
        # Augment text with date and entity names for embedding (but store original text in DB)
        # Entity names (including key:value labels) improve retrieval without polluting stored content
        if fact.occurred_start is not None:
            precision = resolve_stored_occurrence_precision(
                metadata=fact.metadata,
                fact_text=fact.fact_text,
                occurred_start=fact.occurred_start,
                occurred_end=fact.occurred_end,
                allow_legacy_recovery=fact.fact_type != "observation",
            )
            readable_date = format_embedding_occurrence_date(fact.occurred_start, precision, format_date_fn)
            if fact.occurred_end and fact.occurred_end != fact.occurred_start:
                readable_end = format_embedding_occurrence_date(fact.occurred_end, precision, format_date_fn)
                augmented_text = f"{fact.fact_text} (happened from {readable_date} to {readable_end})"
            else:
                augmented_text = f"{fact.fact_text} (happened in {readable_date})"
        elif fact.mentioned_at is not None:
            # Mention time is a different clock.  Occurrence precision must not
            # reinterpret it when an otherwise-undated fact falls back here.
            readable_date = format_date_fn(fact.mentioned_at)
            augmented_text = f"{fact.fact_text} (happened in {readable_date})"
        else:
            augmented_text = fact.fact_text
        if fact.entities:
            augmented_text = f"{augmented_text} [{', '.join(fact.entities)}]"
        augmented_texts.append(augmented_text)
    return augmented_texts


async def generate_embeddings_batch(embeddings_model, texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.

    Args:
        embeddings_model: Embeddings backend, or a wrapper around one that batches
            concurrent callers together (``CoalescingEmbedder``, used by the streaming
            retain producer — see issue #3784). Such a wrapper is recognised by its
            ``embed_documents_async`` method; anything else is a plain backend.
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (same length as texts)
    """
    if not texts:
        return []

    embed_batched = getattr(embeddings_model, "embed_documents_async", None)
    if embed_batched is not None:
        return await embed_batched(texts)

    embeddings = await embedding_utils.generate_embeddings_batch(embeddings_model, texts)

    return embeddings
