"""
Utility functions for memory system.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_wrapper import LLMConfig
    from .retain.fact_extraction import Fact

from .retain.fact_extraction import extract_facts_from_text


async def extract_facts(
    text: str,
    event_date: datetime,
    context: str = "",
    llm_config: "LLMConfig" = None,
    agent_name: str = None,
    extract_opinions: bool = False,
) -> tuple[list["Fact"], list[tuple[str, int]]]:
    """
    Extract semantic facts from text using LLM.

    Uses LLM for intelligent fact extraction that:
    - Filters out social pleasantries and filler words
    - Creates self-contained statements with absolute dates
    - Handles conversational text well
    - Resolves relative time expressions to absolute dates

    Args:
        text: Input text (conversation, article, etc.)
        event_date: Reference date for resolving relative times
        context: Context about the conversation/document
        llm_config: LLM configuration to use
        agent_name: Optional agent name to help identify agent-related facts
        extract_opinions: If True, extract ONLY opinions. If False, extract world and agent facts (no opinions)

    Returns:
        Tuple of (facts, chunks) where:
        - facts: List of Fact model instances
        - chunks: List of tuples (chunk_text, fact_count) for each chunk

    Raises:
        Exception: If LLM fact extraction fails
    """
    if not text or not text.strip():
        return [], []

    facts, chunks, _ = await extract_facts_from_text(
        text,
        event_date,
        context=context,
        llm_config=llm_config,
        agent_name=agent_name,
        extract_opinions=extract_opinions,
    )

    if not facts:
        logging.warning(
            f"LLM extracted 0 facts from text of length {len(text)}. This may indicate the text contains no meaningful information, or the LLM failed to extract facts. Full text: {text}"
        )
        return [], chunks

    return facts, chunks


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Similarity score between 0 and 1
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same dimension")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def calculate_recency_weight(days_since: float, half_life_days: float = 365.0) -> float:
    """
    Calculate recency weight using logarithmic decay.

    This provides much better differentiation over long time periods compared to
    exponential decay. Uses a log-based decay where the half-life parameter controls
    when memories reach 50% weight.

    Examples:
        - Today (0 days): 1.0
        - 1 year (365 days): ~0.5 (with default half_life=365)
        - 2 years (730 days): ~0.33
        - 5 years (1825 days): ~0.17
        - 10 years (3650 days): ~0.09

    This ensures that 2-year-old and 5-year-old memories have meaningfully
    different weights, unlike exponential decay which makes them both ~0.

    Args:
        days_since: Number of days since the memory was created
        half_life_days: Number of days for weight to reach 0.5 (default: 1 year)

    Returns:
        Weight between 0 and 1
    """
    import math

    # Logarithmic decay: 1 / (1 + log(1 + days_since/half_life))
    # This decays much slower than exponential, giving better long-term differentiation
    normalized_age = days_since / half_life_days
    return 1.0 / (1.0 + math.log1p(normalized_age))


def calculate_frequency_weight(access_count: int, max_boost: float = 2.0) -> float:
    """
    Calculate frequency weight based on access count.

    Frequently accessed memories are weighted higher.
    Uses logarithmic scaling to avoid over-weighting.

    Args:
        access_count: Number of times the memory was accessed
        max_boost: Maximum multiplier for frequently accessed memories

    Returns:
        Weight between 1.0 and max_boost
    """
    import math

    if access_count <= 0:
        return 1.0

    # Logarithmic scaling: log(access_count + 1) / log(10)
    # This gives: 0 accesses = 1.0, 9 accesses ~= 1.5, 99 accesses ~= 2.0
    normalized = math.log(access_count + 1) / math.log(10)
    return 1.0 + min(normalized, max_boost - 1.0)


def calculate_temporal_anchor(occurred_start: datetime, occurred_end: datetime) -> datetime:
    """
    Calculate a single temporal anchor point from a temporal range.

    Used for spreading activation - we need a single representative date
    to calculate temporal proximity between facts. This simplifies the
    range-to-range distance problem.

    Strategy: Use midpoint of the range for balanced representation.

    Args:
        occurred_start: Start of temporal range
        occurred_end: End of temporal range

    Returns:
        Single datetime representing the temporal anchor (midpoint)

    Examples:
        - Point event (July 14): start=July 14, end=July 14 → anchor=July 14
        - Month range (February): start=Feb 1, end=Feb 28 → anchor=Feb 14
        - Year range (2023): start=Jan 1, end=Dec 31 → anchor=July 1
    """
    # Calculate midpoint
    time_delta = occurred_end - occurred_start
    midpoint = occurred_start + (time_delta / 2)
    return midpoint


def calculate_temporal_proximity(anchor_a: datetime, anchor_b: datetime, half_life_days: float = 30.0) -> float:
    """
    Calculate temporal proximity between two temporal anchors.

    Used for spreading activation to determine how "close" two facts are
    in time. Uses logarithmic decay so that temporal similarity doesn't
    drop off too quickly.

    Args:
        anchor_a: Temporal anchor of first fact
        anchor_b: Temporal anchor of second fact
        half_life_days: Number of days for proximity to reach 0.5
                       (default: 30 days = 1 month)

    Returns:
        Proximity score in [0, 1] where:
        - 1.0 = same day
        - 0.5 = ~half_life days apart
        - 0.0 = very distant in time

    Examples:
        - Same day: 1.0
        - 1 week apart (half_life=30): ~0.7
        - 1 month apart (half_life=30): ~0.5
        - 1 year apart (half_life=30): ~0.2
    """
    import math

    days_apart = abs((anchor_a - anchor_b).days)

    if days_apart == 0:
        return 1.0

    # Logarithmic decay: 1 / (1 + log(1 + days_apart/half_life))
    # Similar to calculate_recency_weight but for proximity between events
    normalized_distance = days_apart / half_life_days
    proximity = 1.0 / (1.0 + math.log1p(normalized_distance))

    return proximity
