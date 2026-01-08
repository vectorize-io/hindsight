"""
Mental model summary synthesis.

Uses the same 4-step pipeline as the original observation system:
1. Facet Discovery - LLM generates diverse probe queries
2. Parallel Retrieval - Execute probes against recall
3. Diversity Selection - Clustering + MMR for representative facts
4. Synthesis - LLM generates coherent summary

This module is a thin wrapper around observation_utils, providing
mental-model-specific customization.
"""

import logging
from typing import TYPE_CHECKING, Awaitable, Callable

from ..search.observation_utils import (
    TARGET_SUMMARY_TOKENS,
    TOTAL_TOKEN_BUDGET,
    SummaryResult,
    generate_entity_summary,
)

if TYPE_CHECKING:
    from ..llm_wrapper import LLMConfig
    from ..response_models import MemoryFact

logger = logging.getLogger(__name__)


async def generate_mental_model_summary(
    llm_config: "LLMConfig",
    name: str,
    description: str,
    recall_fn: Callable[[str, int], Awaitable[tuple[list["MemoryFact"], list[list[float]]]]],
    *,
    bank_id: str | None = None,
    total_token_budget: int = TOTAL_TOKEN_BUDGET,
    target_summary_tokens: int = TARGET_SUMMARY_TOKENS,
) -> SummaryResult:
    """
    Generate a summary for a mental model.

    Uses the 4-step pipeline:
    1. FACET DISCOVERY - LLM generates diverse probe queries based on name + description
    2. PARALLEL RETRIEVAL - Execute probes against recall
    3. DIVERSITY SELECTION - Clustering + MMR for representative facts
    4. SYNTHESIS - LLM generates coherent summary

    Args:
        llm_config: LLM configuration
        name: Name of the mental model (used as entity_name in the pipeline)
        description: Description of what to track (used as custom_directions)
        recall_fn: Async function (query, max_tokens) -> (facts, embeddings)
        bank_id: Bank identifier for logging
        total_token_budget: Total tokens for fact retrieval
        target_summary_tokens: Target summary length

    Returns:
        SummaryResult containing summary, log buffer, and timing info
    """
    # Use the existing pipeline with description as custom_directions
    return await generate_entity_summary(
        llm_config,
        name,
        recall_fn,
        bank_id=bank_id,
        total_token_budget=total_token_budget,
        target_summary_tokens=target_summary_tokens,
        custom_directions=description,
    )
