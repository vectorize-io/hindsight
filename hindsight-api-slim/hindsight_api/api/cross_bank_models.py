"""
Cross-bank operation models for Hindsight memory system.

These models define the structure of data returned by cross-bank operations
(recall and reflect across multiple memory banks).
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Type aliases for cross-bank operations
TagsMatch = Literal["any", "all", "any_strict", "all_strict"]


class BankInfo(BaseModel):
    """
    Information about a memory bank participating in cross-bank operations.

    Contains the bank's identity and configuration relevant to
    cross-bank query processing.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bank_id": "personal-notes",
                "name": "Personal Notes",
                "disposition": {"skepticism": 3, "literalism": 2, "empathy": 4},
                "background": "Personal journaling and notes",
                "tags": ["personal", "journal"],
            }
        }
    )

    bank_id: str = Field(description="Unique identifier for the bank")
    name: str | None = Field(default=None, description="Human-readable name for the bank")
    disposition: dict[str, int] = Field(
        default_factory=lambda: {"skepticism": 3, "literalism": 3, "empathy": 3},
        description="Bank disposition traits (skepticism, literalism, empathy 1-5)",
    )
    background: str | None = Field(default=None, description="Bank background context")
    tags: list[str] = Field(default_factory=list, description="Tags for bank categorization")


class CrossBankFact(BaseModel):
    """
    A memory fact with bank attribution for cross-bank operations.

    Extends the base MemoryFact with bank identification to track
    which bank each fact originated from during cross-bank queries.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "text": "Alice works at Google on the AI team",
                "fact_type": "world",
                "bank_id": "work-notes",
                "bank_name": "Work Notes",
                "context": "work info",
                "confidence": 0.95,
                "occurred_start": "2024-01-15T10:30:00Z",
                "entities": ["Alice", "Google"],
            }
        }
    )

    id: str = Field(description="Unique identifier for the memory fact")
    text: str = Field(description="The actual text content of the memory")
    fact_type: str = Field(description="Type of fact: 'world', 'experience', or 'observation'")
    bank_id: str = Field(description="ID of the bank this fact belongs to")
    bank_name: str | None = Field(default=None, description="Human-readable name of the bank")
    context: str | None = Field(default=None, description="Additional context for the memory")
    confidence: float | None = Field(default=None, description="Confidence score (0-1)")
    occurred_start: datetime | None = Field(default=None, description="When the fact occurred")
    entities: list[str] = Field(default_factory=list, description="Entities mentioned in this fact")
    tags: list[str] = Field(default_factory=list, description="Tags associated with this fact")


class MentalModelReference(BaseModel):
    """
    Reference to a mental model used during cross-bank reflection.

    Tracks which mental models from which banks contributed to
    the reflection response.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "team-dynamics-model",
                "name": "Team Dynamics",
                "bank_id": "work-notes",
                "excerpt": "The team has strong collaboration patterns...",
            }
        }
    )

    id: str = Field(description="Unique identifier for the mental model")
    name: str = Field(description="Human-readable name of the mental model")
    bank_id: str = Field(description="ID of the bank this mental model belongs to")
    excerpt: str = Field(description="Relevant portion of the mental model content")


class ReasoningStepModel(BaseModel):
    """
    A single step in a multi-step reasoning chain.

    Used when cross-bank reflect operates with MID/HIGH budget
    and decomposes complex queries into sub-questions.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "step_number": 1,
                "question": "What are the team dynamics?",
                "relevant_banks": ["work-notes", "personal-notes"],
                "conclusion": "The team has strong collaboration...",
                "confidence": 0.85,
                "evidence_count": 5,
                "budget_used": 100,
            }
        }
    )

    step_number: int = Field(description="Step number in the reasoning chain (1-based)")
    question: str = Field(description="The sub-question being answered")
    relevant_banks: list[str] = Field(description="Bank IDs consulted for this step")
    conclusion: str = Field(description="The conclusion drawn for this step")
    confidence: float = Field(description="Confidence in the conclusion (0-1)")
    evidence_count: int = Field(description="Number of facts supporting this conclusion")
    budget_used: int = Field(description="Budget tokens consumed for this step")


class CrossBankRecallResult(BaseModel):
    """
    Result from a cross-bank recall operation.

    Contains fused results from multiple banks with attribution
    and statistics about the cross-bank query.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "id": "fact-123",
                        "text": "Alice works at Google",
                        "fact_type": "world",
                        "bank_id": "work-notes",
                        "bank_name": "Work Notes",
                    }
                ],
                "bank_stats": {"work-notes": 5, "personal-notes": 3},
                "total_results": 8,
                "fusion_metadata": {
                    "strategy": "reciprocal_rank_fusion",
                    "dedup_count": 2,
                },
            }
        }
    )

    results: list[CrossBankFact] = Field(
        default_factory=list,
        description="Fused and ranked facts from all queried banks",
    )
    bank_stats: dict[str, int] = Field(
        default_factory=dict,
        description="Number of facts contributed by each bank (bank_id -> count)",
    )
    total_results: int = Field(default=0, description="Total number of results returned")
    fusion_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the fusion process (strategy, dedup counts, etc.)",
    )


class CrossBankReflectResult(BaseModel):
    """
    Result from a cross-bank reflect operation.

    Contains the synthesized response along with supporting facts,
    mental models used, and attribution to source banks.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Based on my knowledge from your work and personal notes...",
                "based_on": [
                    {
                        "id": "fact-123",
                        "text": "Alice works at Google",
                        "fact_type": "world",
                        "bank_id": "work-notes",
                    }
                ],
                "mental_models_used": [
                    {
                        "id": "team-dynamics",
                        "name": "Team Dynamics",
                        "bank_id": "work-notes",
                        "excerpt": "...",
                    }
                ],
                "bank_dispositions": {
                    "work-notes": {"skepticism": 3, "literalism": 3, "empathy": 3}
                },
                "structured_output": None,
                "reasoning_chain": None,
                "new_opinions": [],
            }
        }
    )

    text: str = Field(description="The synthesized response text")
    based_on: list[CrossBankFact] = Field(
        default_factory=list,
        description="Facts used to formulate the response",
    )
    mental_models_used: list[MentalModelReference] = Field(
        default_factory=list,
        description="Mental models consulted during reflection",
    )
    bank_dispositions: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Disposition traits per bank (bank_id -> disposition dict)",
    )
    structured_output: dict[str, Any] | None = Field(
        default=None,
        description="Structured output if response_schema was provided",
    )
    reasoning_chain: list[ReasoningStepModel] | None = Field(
        default=None,
        description="Reasoning steps if multi-step was used (MID/HIGH budget)",
    )
    new_opinions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="New opinions extracted during reflection",
    )


class CrossBankRecallRequest(BaseModel):
    """
    Request model for cross-bank recall endpoint.

    Allows querying multiple memory banks simultaneously with fused ranking.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What projects is Alice working on?",
                "bank_ids": ["work-notes", "personal-notes"],
                "bank_tags": None,
                "max_results": 20,
                "budget": "mid",
                "tags": None,
                "tags_match": "any",
            }
        }
    )

    query: str = Field(description="The search query to execute across banks")
    bank_ids: list[str] | None = Field(
        default=None,
        description="Specific bank IDs to query. If None, queries all accessible banks.",
    )
    bank_tags: list[str] | None = Field(
        default=None,
        description="Filter banks by tags. Only banks with these tags will be queried.",
    )
    max_results: int = Field(
        default=20,
        description="Maximum total results to return after fusion",
    )
    budget: str = Field(
        default="mid",
        description="Budget level: 'low' (100 tokens), 'mid' (300 tokens), or 'high' (1000 tokens)",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Filter memories by tags within each bank",
    )
    tags_match: TagsMatch = Field(
        default="any",
        description="How to match tags: 'any' (OR), 'all' (AND), 'any_strict', 'all_strict'",
    )


class CrossBankReflectRequest(BaseModel):
    """
    Request model for cross-bank reflect endpoint.

    Allows disposition-aware reflection across multiple memory banks.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What do you think about our team's progress?",
                "bank_ids": ["work-notes", "personal-notes"],
                "bank_tags": None,
                "budget": "mid",
                "context": None,
                "include_mental_models": True,
                "include_reasoning_chain": False,
                "response_schema": None,
                "tags": None,
                "tags_match": "any",
            }
        }
    )

    query: str = Field(description="The question to reflect on across banks")
    bank_ids: list[str] | None = Field(
        default=None,
        description="Specific bank IDs to query. If None, queries all accessible banks.",
    )
    bank_tags: list[str] | None = Field(
        default=None,
        description="Filter banks by tags. Only banks with these tags will be queried.",
    )
    budget: str = Field(
        default="low",
        description="Budget level: 'low' (100 tokens), 'mid' (300 tokens), or 'high' (1000 tokens)",
    )
    context: str | None = Field(
        default=None,
        description="Additional context for the reflection",
    )
    include_mental_models: bool = Field(
        default=True,
        description="Whether to consult mental models during reflection",
    )
    include_reasoning_chain: bool = Field(
        default=False,
        description="Whether to decompose complex queries into reasoning steps",
    )
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON Schema for structured output",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Filter memories by tags within each bank",
    )
    tags_match: TagsMatch = Field(
        default="any",
        description="How to match tags: 'any' (OR), 'all' (AND), 'any_strict', 'all_strict'",
    )
