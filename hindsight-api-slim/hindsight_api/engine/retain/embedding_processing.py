"""
Embedding processing for retain pipeline.

Handles augmenting fact texts with temporal information and generating embeddings.
"""

import logging

from ..embeddings import EmbeddingPurpose
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
        # Use occurred_start as the representative date, fall back to mentioned_at
        fact_date = fact.occurred_start or fact.mentioned_at
        # Augment text with date and entity names for embedding (but store original text in DB)
        # Entity names (including key:value labels) improve retrieval without polluting stored content
        if fact_date is not None:
            readable_date = format_date_fn(fact_date)
            if fact.occurred_end and fact.occurred_end != fact.occurred_start:
                readable_end = format_date_fn(fact.occurred_end)
                augmented_text = f"{fact.fact_text} (happened from {readable_date} to {readable_end})"
            else:
                augmented_text = f"{fact.fact_text} (happened in {readable_date})"
        else:
            augmented_text = fact.fact_text
        if fact.entities:
            augmented_text = f"{augmented_text} [{', '.join(fact.entities)}]"
        augmented_texts.append(augmented_text)
    return augmented_texts


async def generate_embeddings_batch(
    embeddings_model, texts: list[str], purpose: EmbeddingPurpose = "document"
) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.

    Args:
        embeddings_model: Embeddings model instance
        texts: List of text strings to embed
        purpose: "query" or "document" — controls which configured prefix
            (if any) the backend prepends.

    Returns:
        List of embedding vectors (same length as texts)
    """
    if not texts:
        return []

    embeddings = await embedding_utils.generate_embeddings_batch(embeddings_model, texts, purpose=purpose)

    return embeddings
