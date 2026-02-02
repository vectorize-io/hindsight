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
    )

    if not facts:
        logging.warning(
            f"LLM extracted 0 facts from text of length {len(text)}. This may indicate the text contains no meaningful information, or the LLM failed to extract facts. Full text: {text}"
        )
        return [], chunks

    return facts, chunks
