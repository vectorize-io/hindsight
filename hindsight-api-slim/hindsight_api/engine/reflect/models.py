"""
Pydantic models for the reflect agent.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from .structured_doc import StructuredDocument


class ObservationSection(BaseModel):
    """A section within an observation with its supporting memories."""

    title: str = Field(description="Section header (can be empty for intro)")
    text: str = Field(description="Section content - no headers, use lists/tables/bold")
    memory_ids: list[str] = Field(default_factory=list, description="Memory IDs supporting this section")


class ReflectAction(BaseModel):
    """Single action the reflect agent can take."""

    tool: Literal["list_observations", "get_observation", "recall", "expand", "done"] = Field(
        description="Tool to invoke: list_observations, get_observation, recall, expand, or done"
    )
    # Tool-specific parameters
    observation_id: str | None = Field(default=None, description="Observation ID for get_observation")
    query: str | None = Field(default=None, description="Search query for recall")
    max_tokens: int | None = Field(default=None, description="Max tokens for recall results (default 2048)")
    memory_ids: list[str] | None = Field(default=None, description="Memory unit IDs for expand (batched)")
    depth: Literal["chunk", "document"] | None = Field(default=None, description="Expansion depth for expand")
    observation_sections: list[ObservationSection] | None = Field(
        default=None, description="Observation sections for done action (when output_mode=observations)"
    )
    # Plain text answer fields (for output_mode=answer)
    answer: str | None = Field(default=None, description="Well-formatted markdown answer for done action")
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

    tool: str = Field(description="Tool name: lookup, recall, expand")
    reason: str | None = Field(default=None, description="Agent's reasoning for making this tool call")
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
    detail: str | None = Field(default=None, description="What the call decided, when it is a decision")


class DirectiveInfo(BaseModel):
    """Information about a directive that was applied during reflect."""

    id: str = Field(description="Directive mental model ID")
    name: str = Field(description="Directive name")
    content: str = Field(description="Directive content")


class TokenUsageSummary(BaseModel):
    """Total token usage across all LLM calls."""

    input_tokens: int = Field(default=0, description="Total input tokens used (includes any cached prefix tokens)")
    output_tokens: int = Field(default=0, description="Total visible output tokens used (excludes reasoning/thoughts)")
    total_tokens: int = Field(default=0, description="Total tokens (input + output, excludes thoughts)")
    cached_tokens: int = Field(
        default=0,
        description="Cached/cache-read prompt tokens summed across calls. Subset of input_tokens.",
    )
    thoughts_tokens: int = Field(
        default=0,
        description=(
            "Reasoning/thinking tokens summed across calls. Billed at the output rate by some providers "
            "but not part of visible output."
        ),
    )


class StructuredOutputResult(BaseModel):
    """Result of structured-output generation, including token usage for the call."""

    structured_output: dict[str, Any] | None = Field(
        default=None, description="Generated structured output, or None if generation failed"
    )
    error: str | None = Field(
        default=None,
        description=(
            "Why the extraction call produced no structured output, when it failed. None on success. "
            "Callers cannot otherwise tell a failed extraction from one that had nothing to extract."
        ),
    )
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Visible output tokens used")
    cached_tokens: int = Field(default=0, description="Cached prefix tokens. Subset of input_tokens.")
    thoughts_tokens: int = Field(default=0, description="Reasoning/thinking tokens, when reported by the provider")


class LengthRewrite(BaseModel):
    """Outcome of the post-hoc ``max_tokens`` rewrite of a final answer.

    ``applied`` is False when the answer was already inside the budget (or there
    was no budget), in which case ``markdown``/``structure`` are the inputs
    unchanged and every token count is zero.
    """

    applied: bool = Field(description="Whether a rewrite call was actually made")
    markdown: str = Field(description="The answer text after the rewrite")
    structure: StructuredDocument | None = Field(
        default=None, description="The document ``markdown`` renders from, when the answer is a document"
    )
    duration_ms: int = Field(default=0, description="Rewrite call duration in milliseconds")
    input_tokens: int = Field(default=0, description="Input tokens used by the rewrite call")
    output_tokens: int = Field(default=0, description="Visible output tokens used by the rewrite call")
    cached_tokens: int = Field(default=0, description="Cached prefix tokens. Subset of input_tokens.")
    thoughts_tokens: int = Field(default=0, description="Reasoning/thinking tokens, when reported by the provider")


class AbstainAnswer(BaseModel):
    """An abstain reflect's answer, written from the memories the decision model picked.

    ``check`` comes first so the model compares before it decides: with ``applies`` as the first field
    it committed to "nothing conflicts" before reading the code against the rule, and missed a change
    that broke a recorded rule (an exact-name check where the decision said "ends with").
    """

    check: str = Field(
        default="",
        description="One or two short sentences: what the memories state, held against the context. Never shown.",
    )
    applies: bool = Field(
        description="True only if the memories state what the question asks for; false when the true answer is no or nothing."
    )
    answer: str = Field(default="", description="The answer, when it applies; empty otherwise.")


class SettledAbstainAnswer(BaseModel):
    """The answer, for a pick the decision model was sure of (``settled``), then whether it applies.

    No ``check``: writing it first delayed the watcher's heads-up. ``applies`` comes after the answer,
    so it labels what was written instead of deciding it: without it a settled writer could answer
    "Nothing conflicts." and the watcher delivered that as a note.
    """

    answer: str = Field(description="The answer.")
    applies: bool = Field(
        default=True,
        description="False when the answer you wrote says the memories do not apply: nothing conflicts, "
        "no decision applies, or the answer is no.",
    )


class ReflectAgentResult(BaseModel):
    """Result from the reflect agent."""

    text: str = Field(description="Final answer text")
    abstained: bool = Field(
        default=False, description="The caller allowed no answer, and what was retrieved did not answer it"
    )
    document: StructuredDocument | None = Field(
        default=None,
        description=(
            "The structured document the agent emitted, when it was asked for one. "
            "``text`` is its deterministic render — the model never wrote that markdown."
        ),
    )
    structured_output: dict[str, Any] | None = Field(
        default=None, description="Structured output parsed according to provided response_schema"
    )
    structured_output_error: str | None = Field(
        default=None,
        description=(
            "Why the structured-output extraction failed, when a response_schema was given and the "
            "extraction call errored or returned unparseable output. None when it succeeded or when "
            "no schema was requested."
        ),
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
