"""
Pydantic models for the reflect agent.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ObservationSection(BaseModel):
    """A section within an observation with its supporting memories."""

    title: str = Field(description="Section header (can be empty for intro)")
    text: str = Field(description="Section content - no headers, use lists/tables/bold")
    memory_ids: list[str] = Field(default_factory=list, description="Memory IDs supporting this section")


class ObservationInput(BaseModel):
    """Input for the learn tool to create an observation placeholder.

    The agent only specifies name and description - the actual content/sections
    are generated during refresh, similar to pinned models.
    """

    name: str = Field(description="Human-readable name for the observation")
    description: str = Field(description="What to track - used as prompt for content generation during refresh")
    entity_id: str | None = Field(default=None, description="Optional link to existing entity ID")


class AnswerSection(BaseModel):
    """A section of the answer with its supporting evidence (DEPRECATED)."""

    title: str = Field(description="Section header/title")
    text: str = Field(description="Section content")
    memory_ids: list[str] = Field(default_factory=list, description="Memory IDs supporting this section")
    model_ids: list[str] = Field(default_factory=list, description="Mental model IDs supporting this section")


class ReflectAction(BaseModel):
    """Single action the reflect agent can take."""

    tool: Literal["list_observations", "get_observation", "recall", "learn", "expand", "done"] = Field(
        description="Tool to invoke: list_observations, get_observation, recall, learn, expand, or done"
    )
    # Tool-specific parameters
    observation_id: str | None = Field(default=None, description="Observation ID for get_observation")
    query: str | None = Field(default=None, description="Search query for recall")
    max_tokens: int | None = Field(default=None, description="Max tokens for recall results (default 2048)")
    observation: ObservationInput | None = Field(default=None, description="Observation to create/update for learn")
    memory_ids: list[str] | None = Field(default=None, description="Memory unit IDs for expand (batched)")
    depth: Literal["chunk", "document"] | None = Field(default=None, description="Expansion depth for expand")
    sections: list[AnswerSection] | None = Field(default=None, description="DEPRECATED: Use answer field instead")
    observation_sections: list[ObservationSection] | None = Field(
        default=None, description="Observation sections for done action (when output_mode=observations)"
    )
    # Plain text answer fields (for output_mode=answer)
    answer: str | None = Field(default=None, description="Plain text answer for done action (no markdown)")
    answer_memory_ids: list[str] | None = Field(
        default=None, description="Memory IDs supporting the answer", alias="memory_ids"
    )
    answer_model_ids: list[str] | None = Field(
        default=None, description="Mental model IDs supporting the answer", alias="model_ids"
    )
    reasoning: str | None = Field(default=None, description="Brief reasoning for this action")


class ReflectActionBatch(BaseModel):
    """Batch of actions for parallel execution."""

    actions: list[ReflectAction] = Field(description="List of actions to execute in parallel")


class ToolCall(BaseModel):
    """A single tool call made during reflect."""

    tool: str = Field(description="Tool name: lookup, recall, learn, expand")
    input: dict = Field(description="Tool input parameters")
    output: dict = Field(description="Tool output/result")
    duration_ms: int = Field(description="Execution time in milliseconds")
    iteration: int = Field(default=0, description="Iteration number (1-based) when this tool was called")


class LLMCall(BaseModel):
    """A single LLM call made during reflect."""

    scope: str = Field(description="Call scope: agent_1, agent_2, final, etc.")
    duration_ms: int = Field(description="Execution time in milliseconds")
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens used")


class DirectiveInfo(BaseModel):
    """Information about a directive that was applied during reflect."""

    id: str = Field(description="Directive mental model ID")
    name: str = Field(description="Directive name")
    rules: list[str] = Field(default_factory=list, description="Directive rules/observations that were applied")


class TokenUsageSummary(BaseModel):
    """Total token usage across all LLM calls."""

    input_tokens: int = Field(default=0, description="Total input tokens used")
    output_tokens: int = Field(default=0, description="Total output tokens used")
    total_tokens: int = Field(default=0, description="Total tokens (input + output)")


class ReflectAgentResult(BaseModel):
    """Result from the reflect agent."""

    text: str = Field(description="Final answer text")
    structured_output: dict[str, Any] | None = Field(
        default=None, description="Structured output parsed according to provided response_schema"
    )
    iterations: int = Field(default=0, description="Number of iterations taken")
    tools_called: int = Field(default=0, description="Total number of tool calls made")
    tool_trace: list[ToolCall] = Field(default_factory=list, description="Trace of all tool calls made")
    llm_trace: list[LLMCall] = Field(default_factory=list, description="Trace of all LLM calls made")
    usage: TokenUsageSummary = Field(
        default_factory=TokenUsageSummary, description="Total token usage across all LLM calls"
    )
    used_memory_ids: list[str] = Field(default_factory=list, description="Validated memory IDs actually used in answer")
    used_mental_model_ids: list[str] = Field(
        default_factory=list, description="Validated mental model IDs actually used in answer"
    )
    used_observation_ids: list[str] = Field(
        default_factory=list, description="Validated observation IDs actually used in answer"
    )
    directives_applied: list[DirectiveInfo] = Field(
        default_factory=list, description="Directive mental models that affected this reflection"
    )
