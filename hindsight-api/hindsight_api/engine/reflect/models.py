"""
Pydantic models for the reflect agent.
"""

from typing import Literal

from pydantic import BaseModel, Field

from ..mental_models.models import MentalModelType


class MentalModelInput(BaseModel):
    """Input for the learn tool to create/update a mental model."""

    name: str = Field(description="Human-readable name for the mental model")
    type: MentalModelType = Field(description="Type: entity, concept, or event")
    description: str = Field(description="One-liner description for quick scanning")
    summary: str = Field(description="Full synthesized understanding")
    triggers: list[str] = Field(default_factory=list, description="Keywords for retrieval matching")
    entity_id: str | None = Field(default=None, description="Optional link to existing entity ID")


class ReflectAction(BaseModel):
    """Single action the reflect agent can take."""

    tool: Literal["lookup", "recall", "learn", "expand", "done"] = Field(
        description="Tool to invoke: lookup, recall, learn, expand, or done"
    )
    # Tool-specific parameters
    model_id: str | None = Field(default=None, description="Mental model ID for lookup (if None, lists all)")
    query: str | None = Field(default=None, description="Search query for recall")
    mental_model: MentalModelInput | None = Field(default=None, description="Mental model to create/update for learn")
    memory_id: str | None = Field(default=None, description="Memory unit ID for expand")
    depth: Literal["chunk", "document"] | None = Field(default=None, description="Expansion depth for expand")
    answer: str | None = Field(default=None, description="Final answer text for done action")
    reasoning: str | None = Field(default=None, description="Brief reasoning for this action")


class ReflectActionBatch(BaseModel):
    """Batch of actions for parallel execution."""

    actions: list[ReflectAction] = Field(description="List of actions to execute in parallel")


class ReflectAgentResult(BaseModel):
    """Result from the reflect agent."""

    text: str = Field(description="Final answer text")
    iterations: int = Field(default=0, description="Number of iterations taken")
    tools_called: int = Field(default=0, description="Total number of tool calls made")
    mental_models_created: list[str] = Field(default_factory=list, description="IDs of mental models created/updated")
