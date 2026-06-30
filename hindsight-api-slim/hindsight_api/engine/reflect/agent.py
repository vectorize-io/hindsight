"""
Reflect agent - agentic loop for reflection with native tool calling.

Uses hierarchical retrieval:
1. search_mental_models - User-curated summaries (highest quality)
2. search_observations - Consolidated knowledge with freshness
3. recall - Raw facts as ground truth
"""

import asyncio
import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ...config import get_config
from .models import DirectiveInfo, LLMCall, ReflectAgentResult, StructuredOutputResult, TokenUsageSummary, ToolCall
from .prompts import (
    _extract_directive_rules,
    build_final_prompt,
    build_final_system_prompt,
    build_system_prompt_for_tools,
)
from .tokenization import count_cl100k_tokens
from .tools_schema import get_reflect_tools


def _build_directives_applied(directives: list[dict[str, Any]] | None) -> list[DirectiveInfo]:
    """Build list of DirectiveInfo from directives."""
    if not directives:
        return []

    return [
        DirectiveInfo(
            id=directive.get("id", ""),
            name=directive.get("name", ""),
            content=directive.get("content", ""),
        )
        for directive in directives
    ]


if TYPE_CHECKING:
    from ..llm_wrapper import LLMProvider
    from ..response_models import LLMToolCall

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 10


def _normalize_tool_name(name: str) -> str:
    """Normalize tool name from various LLM output formats.

    Some LLMs output tool names in non-standard formats:
    - 'functions.done' (OpenAI-style prefix)
    - 'call=functions.done' (some models)
    - 'call=done' (some models)
    - 'done<|channel|>commentary' (malformed special tokens appended)

    Returns the normalized tool name (e.g., 'done', 'recall', etc.)
    """
    # Handle 'call=functions.name' or 'call=name' format
    if name.startswith("call="):
        name = name[len("call=") :]

    # Handle 'functions.name' format
    if name.startswith("functions."):
        name = name[len("functions.") :]

    # Handle malformed special tokens appended to tool name
    # e.g., 'done<|channel|>commentary' -> 'done'
    if "<|" in name:
        name = name.split("<|")[0]

    return name


def _is_done_tool(name: str) -> bool:
    """Check if the tool name represents the 'done' tool."""
    return _normalize_tool_name(name) == "done"


# Pattern to match done() call as text - handles done({...}) with nested JSON
_DONE_CALL_PATTERN = re.compile(r"done\s*\(\s*\{.*$", re.DOTALL)

# Patterns for leaked structured output in the answer field
_LEAKED_JSON_SUFFIX = re.compile(
    r'\s*```(?:json)?\s*\{[^}]*(?:"(?:observation_ids|memory_ids|mental_model_ids)"|\})\s*```\s*$',
    re.DOTALL | re.IGNORECASE,
)
_TRAILING_IDS_PATTERN = re.compile(
    r"\s*(?:observation_ids|memory_ids|mental_model_ids)\s*[=:]\s*\[.*?\]\s*$", re.DOTALL | re.IGNORECASE
)
_JSON_CODE_FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*(\{.*\})\s*```\s*$", re.DOTALL | re.IGNORECASE)

_DONE_ARGUMENT_KEYS = frozenset(
    {
        "answer",
        "directive_compliance",
        "memory_ids",
        "mental_model_ids",
        "observation_ids",
        "model_ids",
    }
)
_DONE_ARGUMENT_MARKER_KEYS = _DONE_ARGUMENT_KEYS - {"answer"}
_LEAKED_JSON_ID_KEYS = frozenset({"memory_ids", "mental_model_ids", "observation_ids", "model_ids"})


def _unwrap_leaked_done_arguments(text: str) -> str | None:
    """Return the answer when a done tool call was rendered as JSON text.

    Some providers leak the done tool's argument object instead of surfacing it
    as a native tool call, e.g. {"answer": "...", "memory_ids": [...]}. Only
    unwrap objects that match the done argument shape so normal JSON answers
    stay intact.
    """
    candidate = text.strip()
    if not candidate:
        return None

    fenced = _JSON_CODE_FENCE_PATTERN.match(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None

    keys = set(payload)
    if not keys.intersection(_DONE_ARGUMENT_MARKER_KEYS):
        return None
    if not keys.issubset(_DONE_ARGUMENT_KEYS):
        return None

    for key in ("memory_ids", "mental_model_ids", "observation_ids", "model_ids"):
        value = payload.get(key)
        if value is not None and not isinstance(value, list):
            return None

    return answer.strip()


def _strip_trailing_id_json_object(text: str) -> str:
    stripped = text.rstrip()
    if not stripped.endswith("}"):
        return text.strip()

    start = stripped.rfind("{")
    if start < 0:
        return text.strip()

    try:
        payload = json.loads(stripped[start:])
    except json.JSONDecodeError:
        return text.strip()

    if not isinstance(payload, dict) or not payload:
        return text.strip()
    keys = set(payload)
    if not keys.issubset(_LEAKED_JSON_ID_KEYS):
        return text.strip()

    return stripped[:start].strip()


def _clean_answer_text(text: str) -> str:
    """Clean up answer text by removing any done() tool call syntax.

    Some LLMs output the done() call as text instead of a proper tool call.
    This strips out patterns like: done({"answer": "...", ...})
    """
    unwrapped = _unwrap_leaked_done_arguments(text)
    if unwrapped is not None:
        return unwrapped

    # Remove done() call pattern from the end of the text
    cleaned = _DONE_CALL_PATTERN.sub("", text).strip()
    return cleaned if cleaned else text


def _clean_done_answer(text: str) -> str:
    """Clean up the answer field from a done() tool call.

    Some LLMs leak structured output patterns into the answer text, such as:
    - JSON code blocks with observation_ids/memory_ids at the end
    - Raw JSON objects with these fields
    - Plain text like "observation_ids: [...]"

    This cleans those patterns while preserving the actual answer content.
    """
    if not text:
        return text

    unwrapped = _unwrap_leaked_done_arguments(text)
    if unwrapped is not None:
        return unwrapped

    cleaned = text

    # Remove leaked JSON in code blocks at the end
    cleaned = _LEAKED_JSON_SUFFIX.sub("", cleaned).strip()

    # Remove leaked raw JSON objects at the end
    cleaned = _strip_trailing_id_json_object(cleaned)

    # Remove trailing ID patterns
    cleaned = _TRAILING_IDS_PATTERN.sub("", cleaned).strip()

    return cleaned if cleaned else text


async def _generate_structured_output(
    answer: str,
    response_schema: dict,
    llm_config: "LLMProvider",
    reflect_id: str,
) -> StructuredOutputResult:
    """Generate structured output from an answer using the provided JSON schema.

    Args:
        answer: The text answer to extract structured data from
        response_schema: JSON Schema for the expected output structure
        llm_config: LLM provider for making the extraction call
        reflect_id: Reflect ID for logging

    Returns:
        A StructuredOutputResult carrying the structured output (None if
        generation fails) and the call's token usage.
    """
    try:
        from typing import Any as TypingAny

        from pydantic import create_model

        def _json_schema_type_to_python(field_schema: dict) -> type:
            """Map JSON schema type to Python type for better LLM guidance."""
            json_type = field_schema.get("type", "string")
            if json_type == "array":
                return list
            elif json_type == "object":
                return dict
            elif json_type == "integer":
                return int
            elif json_type == "number":
                return float
            elif json_type == "boolean":
                return bool
            else:
                return str

        # Build fields from JSON schema properties
        schema_props = response_schema.get("properties", {})
        required_fields = set(response_schema.get("required", []))
        fields: dict[str, TypingAny] = {}
        for field_name, field_schema in schema_props.items():
            field_type = _json_schema_type_to_python(field_schema)
            default = ... if field_name in required_fields else None
            fields[field_name] = (field_type, default)

        if not fields:
            logger.warning(f"[REFLECT {reflect_id}] No fields found in response_schema, skipping structured output")
            return StructuredOutputResult()

        DynamicModel = create_model("StructuredResponse", **fields)

        # Include the full schema in the prompt for better LLM guidance
        schema_str = json.dumps(response_schema, indent=2, ensure_ascii=False)

        # Build field descriptions for the prompt
        field_descriptions = []
        for field_name, field_schema in schema_props.items():
            field_type = field_schema.get("type", "string")
            field_desc = field_schema.get("description", "")
            is_required = field_name in required_fields
            req_marker = " (REQUIRED)" if is_required else " (optional)"
            field_descriptions.append(f"- {field_name} ({field_type}){req_marker}: {field_desc}")
        fields_text = "\n".join(field_descriptions)

        # Call LLM with the answer to extract structured data
        structured_prompt = f"""Your task is to extract specific information from the answer below and format it as JSON.

ANSWER TO EXTRACT FROM:
\"\"\"
{answer}
\"\"\"

REQUIRED OUTPUT FORMAT - Extract the following fields from the answer above:
{fields_text}

JSON Schema:
```json
{schema_str}
```

INSTRUCTIONS:
1. Read the answer carefully and identify the information that matches each field
2. Extract the ACTUAL content from the answer - do NOT leave fields empty if information is present
3. For string fields: use the exact text or a clear summary from the answer
4. For array fields: return a JSON array (e.g., ["item1", "item2"]), NOT a string
5. For required fields: you MUST provide a value extracted from the answer
6. Return ONLY the JSON object, no explanation

OUTPUT:"""

        structured_result, usage = await llm_config.call(
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise data extraction assistant. Extract information from text and return it as valid JSON matching the provided schema. Always extract actual content - never return empty strings for required fields if information is available.",
                },
                {"role": "user", "content": structured_prompt},
            ],
            response_format=DynamicModel,
            scope="reflect_structured",
            max_retries=1,
            initial_backoff=0.25,
            max_backoff=1.0,
            skip_validation=True,  # We'll handle the dict ourselves
            return_usage=True,
        )

        # Convert to dict
        if hasattr(structured_result, "model_dump"):
            structured_output = structured_result.model_dump()
        elif isinstance(structured_result, dict):
            structured_output = structured_result
        else:
            # Try to parse as JSON
            structured_output = json.loads(str(structured_result))

        # Validate that required fields have non-empty values
        for field_name in required_fields:
            value = structured_output.get(field_name)
            if value is None or value == "" or value == []:
                logger.warning(f"[REFLECT {reflect_id}] Required field '{field_name}' is empty in structured output")

        logger.info(f"[REFLECT {reflect_id}] Generated structured output with {len(structured_output)} fields")
        return StructuredOutputResult(
            structured_output=structured_output,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cached_tokens,
            thoughts_tokens=usage.thoughts_tokens,
        )

    except Exception as e:
        logger.warning(f"[REFLECT {reflect_id}] Failed to generate structured output: {e}")
        return StructuredOutputResult()


def _count_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate the token count of the messages list using cl100k_base encoding."""
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, str):
            total += count_cl100k_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += count_cl100k_tokens(part["text"])
        # Tool call arguments and results also count
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                total += count_cl100k_tokens(func.get("arguments", ""))
    return total


def _is_context_overflow_error(exc: Exception) -> bool:
    """Return True if the exception signals the LLM context window was exceeded."""
    msg = str(exc).lower()
    return any(
        phrase in msg
        for phrase in (
            "context_length_exceeded",
            "context length exceeded",
            "maximum context length",
            "prompt_too_long",
            "prompt is too long",
            "resource_exhausted",
            "input is too long",
            "too many tokens",
        )
    )


def _all_mental_models_are_usable_and_fresh(tool_output: dict[str, Any]) -> bool:
    """Return whether every retrieved mental model is explicitly fresh and has answerable content.

    Used to decide — without an extra LLM call — whether a forced
    ``search_mental_models`` result is trustworthy enough to hand control back
    to the agent. A model is usable only when it is explicitly ``is_stale ==
    False`` (an unknown/missing staleness flag is treated as unsafe) and has
    non-empty content.
    """
    models = tool_output.get("mental_models") or []
    for model in models:
        if model.get("is_stale") is not False:
            return False
        if not str(model.get("content") or "").strip():
            return False
    return True


async def run_reflect_agent(
    llm_config: "LLMProvider",
    bank_id: str,
    query: str,
    bank_profile: dict[str, Any],
    search_mental_models_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    search_observations_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    recall_fn: Callable[[str, int, int], Awaitable[dict[str, Any]]],
    expand_fn: Callable[[list[str], str], Awaitable[dict[str, Any]]],
    context: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tokens: int | None = None,
    response_schema: dict | None = None,
    directives: list[dict[str, Any]] | None = None,
    has_mental_models: bool = False,
    include_observations: bool = True,
    include_recall: bool = True,
    budget: str | None = None,
    max_context_tokens: int = 100_000,
    llm_output_language: str | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> ReflectAgentResult:
    """
    Execute the reflect agent loop using native tool calling.

    The agent uses hierarchical retrieval:
    1. search_mental_models - User-curated summaries (try first)
    2. search_observations - Consolidated knowledge with freshness
    3. recall - Raw facts as ground truth

    Args:
        llm_config: LLM provider for agent calls
        bank_id: Bank identifier
        query: Question to answer
        bank_profile: Bank profile with name and mission
        search_mental_models_fn: Tool callback for searching mental models (query, max_results) -> result
        search_observations_fn: Tool callback for searching observations (query, max_results) -> result
        recall_fn: Tool callback for recall (query, max_tokens) -> result
        expand_fn: Tool callback for expand (memory_ids, depth) -> result
        context: Optional additional context
        max_iterations: Maximum number of iterations before forcing response
        max_tokens: Maximum tokens for the final response
        response_schema: Optional JSON Schema for structured output in final response
        directives: Optional list of directive mental models to inject as hard rules

    Returns:
        ReflectAgentResult with final answer and metadata
    """
    reflect_id = f"{bank_id[:8]}-{int(time.time() * 1000) % 100000}"
    start_time = time.time()

    # Build directives_applied for the trace
    directives_applied = _build_directives_applied(directives)

    # Extract directive rules for tool schema (if any)
    directive_rules = _extract_directive_rules(directives) if directives else None

    # Get tools for this agent (with directive compliance field if directives exist).
    # The expand tool only reads back raw source text (chunks/documents), so it is
    # useless and excluded when document text storage is disabled.
    include_expand = get_config().store_document_text
    tools = get_reflect_tools(
        directive_rules=directive_rules,
        include_mental_models=has_mental_models,
        include_observations=include_observations,
        include_recall=include_recall,
        include_expand=include_expand,
    )
    # Build set of enabled tool names to guard against LLM hallucinating disabled tool calls
    enabled_tools: frozenset[str] = frozenset(t["function"]["name"] for t in tools if t.get("type") == "function")

    # Build initial messages (directives are injected into system prompt at START and END)
    system_prompt = build_system_prompt_for_tools(
        bank_profile,
        context,
        directives=directives,
        has_mental_models=has_mental_models,
        include_observations=include_observations,
        budget=budget,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    # Opt into context caching for the agentic tool loop. The system
    # prompt and tool definitions are stable for the duration of this
    # reflect call (and across reflects against the same bank), so
    # caching them once and reusing across every iteration of the loop
    # collapses the dominant input cost — the prefix repeated on every
    # turn. ``get_or_create_cached_prefix`` returns None when caching is
    # disabled, unsupported, or the prefix is too small; the
    # ``call_with_tools`` invocation below transparently falls back to
    # the uncached path in that case.
    cached_prefix_name: str | None = None
    provider_impl = getattr(llm_config, "_provider_impl", None)
    if provider_impl is not None and provider_impl.supports_prompt_caching():
        try:
            cached_prefix_name = await provider_impl.get_or_create_cached_prefix(
                system_instruction=system_prompt,
                tools=tools,
            )
        except Exception:
            # Caching is a soft optimisation; never let a cache-side
            # error block a reflect.
            cached_prefix_name = None

    # Tracking
    total_tools_called = 0
    tool_trace: list[ToolCall] = []
    tool_trace_summary: list[dict[str, Any]] = []
    llm_trace: list[dict[str, Any]] = []
    context_history: list[dict[str, Any]] = []  # For final prompt fallback

    # Token usage tracking - accumulate across all LLM calls.
    # cached_tokens and thoughts_tokens are surfaced for cost attribution
    # and prompt-cache tuning. Both are subsets of (or parallel to) the
    # input/output counts and are NOT double-counted in total_tokens.
    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_tokens = 0
    total_thoughts_tokens = 0

    # Track available IDs for validation (prevents hallucinated citations)
    available_memory_ids: set[str] = set()
    available_mental_model_ids: set[str] = set()
    available_observation_ids: set[str] = set()

    def _get_llm_trace() -> list[LLMCall]:
        return [
            LLMCall(
                scope=c["scope"],
                duration_ms=c["duration_ms"],
                input_tokens=c.get("input_tokens", 0),
                output_tokens=c.get("output_tokens", 0),
            )
            for c in llm_trace
        ]

    def _get_usage() -> TokenUsageSummary:
        return TokenUsageSummary(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
            cached_tokens=total_cached_tokens,
            thoughts_tokens=total_thoughts_tokens,
        )

    def _log_completion(answer: str, iterations: int, forced: bool = False):
        elapsed_ms = int((time.time() - start_time) * 1000)
        tools_summary = (
            ", ".join(
                f"{t['tool']}({t['input_summary']})={t['duration_ms']}ms/{t.get('output_chars', 0)}c"
                for t in tool_trace_summary
            )
            or "none"
        )
        llm_summary = ", ".join(f"{c['scope']}={c['duration_ms']}ms" for c in llm_trace) or "none"
        total_llm_ms = sum(c["duration_ms"] for c in llm_trace)
        total_tools_ms = sum(t["duration_ms"] for t in tool_trace_summary)

        answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
        mode = "forced" if forced else "done"
        logger.info(
            f"[REFLECT {reflect_id}] {mode} | "
            f"query='{query[:50]}...' | "
            f"iterations={iterations} | "
            f"llm=[{llm_summary}] ({total_llm_ms}ms) | "
            f"tools=[{tools_summary}] ({total_tools_ms}ms) | "
            f"answer='{answer_preview}' | "
            f"total={elapsed_ms}ms"
        )

    consecutive_errors = 0
    # When a forced ``search_mental_models`` returns fresh, usable models on a
    # low/mid-budget call, we stop forcing the lower retrieval layers from this
    # iteration onward and let the agent answer (or retrieve deeper itself)
    # under ``auto`` tool choice. None means the full forced path still applies.
    stop_forcing_from_iteration: int | None = None
    for iteration in range(max_iterations):
        # Cooperative cancellation checkpoint: abort the agent loop between
        # iterations if the caller (e.g. an HTTP client) has gone away, rather
        # than spending another LLM round-trip on a result nobody will read
        # (issue #2122). Raises OperationCancelledError when fired.
        if cancel_check is not None:
            cancel_check()

        is_last = iteration == max_iterations - 1

        if is_last:
            # Force text response on last iteration - no tools
            prompt = build_final_prompt(
                query, context_history, bank_profile, context, max_context_tokens=max_context_tokens
            )
            llm_start = time.time()
            response, usage = await llm_config.call(
                messages=[
                    {
                        "role": "system",
                        "content": build_final_system_prompt(
                            bank_profile.get("mission"), llm_output_language, directives
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                scope="reflect",
                max_completion_tokens=max_tokens,
                return_usage=True,
            )
            llm_duration = int((time.time() - llm_start) * 1000)
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
            total_cached_tokens += getattr(usage, "cached_tokens", 0) or 0
            total_thoughts_tokens += getattr(usage, "thoughts_tokens", 0) or 0
            llm_trace.append(
                {
                    "scope": "final",
                    "duration_ms": llm_duration,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                }
            )
            answer = _clean_answer_text(response.strip())

            # Generate structured output if schema provided
            structured_output = None
            if response_schema and answer:
                struct = await _generate_structured_output(answer, response_schema, llm_config, reflect_id)
                structured_output = struct.structured_output
                total_input_tokens += struct.input_tokens
                total_output_tokens += struct.output_tokens
                total_cached_tokens += struct.cached_tokens
                total_thoughts_tokens += struct.thoughts_tokens

            _log_completion(answer, iteration + 1, forced=True)
            return ReflectAgentResult(
                text=answer,
                structured_output=structured_output,
                iterations=iteration + 1,
                tools_called=total_tools_called,
                tool_trace=tool_trace,
                llm_trace=_get_llm_trace(),
                usage=_get_usage(),
                directives_applied=directives_applied,
            )

        # Proactive context-window guard: if accumulated messages would exceed the
        # configured token budget, bail out early and synthesize from what we have.
        estimated_tokens = _count_messages_tokens(messages)
        if estimated_tokens >= max_context_tokens and (
            bool(available_memory_ids) or bool(available_mental_model_ids) or bool(available_observation_ids)
        ):
            logger.warning(
                f"[REFLECT {reflect_id}] Context budget exceeded on iteration {iteration + 1}: "
                f"~{estimated_tokens} tokens >= {max_context_tokens} limit. Forcing final synthesis."
            )
            prompt = build_final_prompt(
                query, context_history, bank_profile, context, max_context_tokens=max_context_tokens
            )
            llm_start = time.time()
            response, usage = await llm_config.call(
                messages=[
                    {
                        "role": "system",
                        "content": build_final_system_prompt(
                            bank_profile.get("mission"), llm_output_language, directives
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                scope="reflect",
                max_completion_tokens=max_tokens,
                return_usage=True,
            )
            llm_duration = int((time.time() - llm_start) * 1000)
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
            total_cached_tokens += getattr(usage, "cached_tokens", 0) or 0
            total_thoughts_tokens += getattr(usage, "thoughts_tokens", 0) or 0
            llm_trace.append(
                {
                    "scope": "final",
                    "duration_ms": llm_duration,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                }
            )
            answer = _clean_answer_text(response.strip())

            structured_output = None
            if response_schema and answer:
                struct = await _generate_structured_output(answer, response_schema, llm_config, reflect_id)
                structured_output = struct.structured_output
                total_input_tokens += struct.input_tokens
                total_output_tokens += struct.output_tokens
                total_cached_tokens += struct.cached_tokens
                total_thoughts_tokens += struct.thoughts_tokens

            _log_completion(answer, iteration + 1, forced=True)
            return ReflectAgentResult(
                text=answer,
                structured_output=structured_output,
                iterations=iteration + 1,
                tools_called=total_tools_called,
                tool_trace=tool_trace,
                llm_trace=_get_llm_trace(),
                usage=_get_usage(),
                directives_applied=directives_applied,
            )

        # Call LLM with tools
        llm_start = time.time()

        # Determine tool_choice for this iteration.
        # Force the full hierarchical retrieval path (only for enabled tools) before allowing auto.
        # Build the forced sequence from the tools that are actually enabled.
        forced_sequence = []
        if has_mental_models:
            forced_sequence.append("search_mental_models")
        if include_observations:
            forced_sequence.append("search_observations")
        if include_recall:
            forced_sequence.append("recall")

        if stop_forcing_from_iteration is not None and iteration >= stop_forcing_from_iteration:
            # A fresh mental model already short-circuited the forced path.
            iter_tool_choice: str | dict = "auto"
        elif iteration < len(forced_sequence):
            iter_tool_choice = {"type": "function", "function": {"name": forced_sequence[iteration]}}
        else:
            iter_tool_choice = "auto"

        try:
            ct_kwargs: dict[str, Any] = dict(
                messages=messages,
                tools=tools,
                scope="reflect_tool_call",
                tool_choice=iter_tool_choice,
            )
            # Gemini rejects ``cached_content`` alongside a per-request
            # ``tool_config`` (forced tool choice): "CachedContent can not be used
            # with GenerateContent request setting system_instruction, tools or
            # tool_config." The forced-sequence iterations set tool_config, so only
            # the ``auto`` iterations can reference the cache; forced iterations send
            # the prefix inline. The cache (tools + system prompt) is identical
            # either way, so this just limits *which* iterations are billed cached.
            if cached_prefix_name is not None and iter_tool_choice == "auto":
                ct_kwargs["cached_prefix"] = cached_prefix_name
            result = await llm_config.call_with_tools(**ct_kwargs)
            llm_duration = int((time.time() - llm_start) * 1000)
            consecutive_errors = 0
            total_input_tokens += result.input_tokens
            total_output_tokens += result.output_tokens
            total_cached_tokens += getattr(result, "cached_tokens", 0) or 0
            total_thoughts_tokens += getattr(result, "thoughts_tokens", 0) or 0
            llm_trace.append(
                {
                    "scope": f"agent_{iteration + 1}",
                    "duration_ms": llm_duration,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                }
            )

        except Exception as e:
            err_duration = int((time.time() - llm_start) * 1000)
            consecutive_errors += 1
            logger.warning(f"[REFLECT {reflect_id}] LLM error on iteration {iteration + 1}: {e} ({err_duration}ms)")
            llm_trace.append({"scope": f"agent_{iteration + 1}_err", "duration_ms": err_duration})
            has_gathered_evidence = (
                bool(available_memory_ids) or bool(available_mental_model_ids) or bool(available_observation_ids)
            )
            # Context overflow errors must never be retried — retrying would only make them worse.
            # Skip straight to final synthesis with whatever evidence we have.
            if _is_context_overflow_error(e):
                logger.warning(
                    f"[REFLECT {reflect_id}] Context window exceeded on iteration {iteration + 1}, "
                    "forcing final synthesis from gathered evidence."
                )
            # For other errors: retry if no evidence yet (but cap consecutive errors to avoid long hangs)
            elif not has_gathered_evidence and iteration < max_iterations - 1 and consecutive_errors < 2:
                continue
            prompt = build_final_prompt(
                query, context_history, bank_profile, context, max_context_tokens=max_context_tokens
            )
            llm_start = time.time()
            response, usage = await llm_config.call(
                messages=[
                    {
                        "role": "system",
                        "content": build_final_system_prompt(
                            bank_profile.get("mission"), llm_output_language, directives
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                scope="reflect",
                max_completion_tokens=max_tokens,
                return_usage=True,
            )
            llm_duration = int((time.time() - llm_start) * 1000)
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
            total_cached_tokens += getattr(usage, "cached_tokens", 0) or 0
            total_thoughts_tokens += getattr(usage, "thoughts_tokens", 0) or 0
            llm_trace.append(
                {
                    "scope": "final",
                    "duration_ms": llm_duration,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                }
            )
            answer = _clean_answer_text(response.strip())

            # Generate structured output if schema provided
            structured_output = None
            if response_schema and answer:
                struct = await _generate_structured_output(answer, response_schema, llm_config, reflect_id)
                structured_output = struct.structured_output
                total_input_tokens += struct.input_tokens
                total_output_tokens += struct.output_tokens
                total_cached_tokens += struct.cached_tokens
                total_thoughts_tokens += struct.thoughts_tokens

            _log_completion(answer, iteration + 1, forced=True)
            return ReflectAgentResult(
                text=answer,
                structured_output=structured_output,
                iterations=iteration + 1,
                tools_called=total_tools_called,
                tool_trace=tool_trace,
                llm_trace=_get_llm_trace(),
                usage=_get_usage(),
                directives_applied=directives_applied,
            )

        # No tool calls - LLM wants to respond with text
        if not result.tool_calls:
            # When directives are present but no evidence has been gathered,
            # the LLM tends to echo directive content verbatim as its answer.
            # Fall through to the final-prompt path which doesn't include
            # directives and handles "no data" gracefully.
            has_gathered_evidence = (
                bool(available_memory_ids) or bool(available_mental_model_ids) or bool(available_observation_ids)
            )
            directive_leak_risk = directives and not has_gathered_evidence
            if result.content and not directive_leak_risk:
                answer = _clean_answer_text(result.content.strip())

                # The call_with_tools call above is intentionally uncapped so the
                # LLM has headroom to emit tool-call JSON plus any intermediate
                # reasoning. But when the LLM short-circuits and returns text
                # directly, that text becomes the user-visible final answer and
                # must respect max_tokens like the forced-final paths do. If it
                # overshoots, run one extra capped call to rewrite it within
                # the cap.
                if max_tokens is not None and count_cl100k_tokens(answer) > max_tokens:
                    rewrite_start = time.time()
                    rewritten, rewrite_usage = await llm_config.call(
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Rewrite the user's text so it fits within the requested token "
                                    "budget. Preserve the key facts and structure; drop lower-priority "
                                    "detail. Respond with the rewritten text only, no preamble."
                                ),
                            },
                            {
                                "role": "user",
                                "content": f"Target budget: {max_tokens} tokens.\n\nText to rewrite:\n{answer}",
                            },
                        ],
                        scope="reflect",
                        max_completion_tokens=max_tokens,
                        return_usage=True,
                    )
                    total_input_tokens += rewrite_usage.input_tokens
                    total_output_tokens += rewrite_usage.output_tokens
                    total_cached_tokens += getattr(rewrite_usage, "cached_tokens", 0) or 0
                    total_thoughts_tokens += getattr(rewrite_usage, "thoughts_tokens", 0) or 0
                    llm_trace.append(
                        {
                            "scope": "final_rewrite",
                            "duration_ms": int((time.time() - rewrite_start) * 1000),
                            "input_tokens": rewrite_usage.input_tokens,
                            "output_tokens": rewrite_usage.output_tokens,
                        }
                    )
                    answer = _clean_answer_text(rewritten.strip())

                # Generate structured output if schema provided
                structured_output = None
                if response_schema and answer:
                    struct = await _generate_structured_output(answer, response_schema, llm_config, reflect_id)
                    structured_output = struct.structured_output
                    total_input_tokens += struct.input_tokens
                    total_output_tokens += struct.output_tokens
                    total_cached_tokens += struct.cached_tokens
                    total_thoughts_tokens += struct.thoughts_tokens

                _log_completion(answer, iteration + 1)
                return ReflectAgentResult(
                    text=answer,
                    structured_output=structured_output,
                    iterations=iteration + 1,
                    tools_called=total_tools_called,
                    tool_trace=tool_trace,
                    llm_trace=_get_llm_trace(),
                    usage=_get_usage(),
                    directives_applied=directives_applied,
                )
            # Empty response, force final
            prompt = build_final_prompt(
                query, context_history, bank_profile, context, max_context_tokens=max_context_tokens
            )
            llm_start = time.time()
            response, usage = await llm_config.call(
                messages=[
                    {
                        "role": "system",
                        "content": build_final_system_prompt(
                            bank_profile.get("mission"), llm_output_language, directives
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                scope="reflect",
                max_completion_tokens=max_tokens,
                return_usage=True,
            )
            llm_duration = int((time.time() - llm_start) * 1000)
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
            total_cached_tokens += getattr(usage, "cached_tokens", 0) or 0
            total_thoughts_tokens += getattr(usage, "thoughts_tokens", 0) or 0
            llm_trace.append(
                {
                    "scope": "final",
                    "duration_ms": llm_duration,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                }
            )
            answer = _clean_answer_text(response.strip())

            # Generate structured output if schema provided
            structured_output = None
            if response_schema and answer:
                struct = await _generate_structured_output(answer, response_schema, llm_config, reflect_id)
                structured_output = struct.structured_output
                total_input_tokens += struct.input_tokens
                total_output_tokens += struct.output_tokens
                total_cached_tokens += struct.cached_tokens
                total_thoughts_tokens += struct.thoughts_tokens

            _log_completion(answer, iteration + 1, forced=True)
            return ReflectAgentResult(
                text=answer,
                structured_output=structured_output,
                iterations=iteration + 1,
                tools_called=total_tools_called,
                tool_trace=tool_trace,
                llm_trace=_get_llm_trace(),
                usage=_get_usage(),
                directives_applied=directives_applied,
            )

        # Check for done tool call (handle various LLM output formats)
        done_call = next((tc for tc in result.tool_calls if _is_done_tool(tc.name)), None)
        if done_call:
            # Guardrail: Require evidence before done
            has_gathered_evidence = (
                bool(available_memory_ids) or bool(available_mental_model_ids) or bool(available_observation_ids)
            )
            if not has_gathered_evidence and iteration < max_iterations - 1:
                # Add assistant message and fake tool result asking for evidence
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [_tool_call_to_dict(done_call)],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": done_call.id,
                        "name": done_call.name,  # Required by Gemini
                        "content": json.dumps(
                            {
                                "error": "You must search for information first. Use search_mental_models(), search_observations(), or recall() before providing your final answer."
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
                continue

            # Process done tool - wrap with tool call span
            from hindsight_api.tracing import get_tracer

            tracer = get_tracer()
            span_name = "hindsight.reflect_tool_call"
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("hindsight.scope", "reflect_tool_call")
                span.set_attribute("hindsight.operation", "reflect_tool_call")
                return await _process_done_tool(
                    done_call,
                    available_memory_ids,
                    available_mental_model_ids,
                    available_observation_ids,
                    iteration + 1,
                    total_tools_called,
                    tool_trace,
                    _get_llm_trace(),
                    _get_usage(),
                    _log_completion,
                    reflect_id,
                    directives_applied=directives_applied,
                    llm_config=llm_config,
                    response_schema=response_schema,
                )

        # Execute other tools in parallel (exclude done tool in all its format variants)
        other_tools = [tc for tc in result.tool_calls if not _is_done_tool(tc.name)]
        if other_tools:
            # Partition into enabled vs hallucinated (not in enabled_tools set)
            allowed_tools = []
            hallucinated_tools = []
            for tc in other_tools:
                norm = _normalize_tool_name(tc.name)
                # "done" is always available. "expand" is governed by enabled_tools
                # (it is excluded when text storage is disabled), so it is not hardcoded here.
                if enabled_tools is not None and norm not in enabled_tools and norm != "done":
                    hallucinated_tools.append(tc)
                else:
                    allowed_tools.append(tc)

            # Build assistant message with all tool calls (LLM requires them for history)
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [_tool_call_to_dict(tc) for tc in other_tools],
                }
            )

            # Immediately reject hallucinated tool calls without adding to trace
            for tc in hallucinated_tools:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": json.dumps(
                            {
                                "error": f"Tool '{_normalize_tool_name(tc.name)}' is not available. Use only the tools provided to you."
                            },
                            ensure_ascii=False,
                        ),
                    }
                )

            other_tools = allowed_tools

            # Execute tools in parallel
            tool_tasks = [
                _execute_tool_with_timing(
                    tc,
                    search_mental_models_fn,
                    search_observations_fn,
                    recall_fn,
                    expand_fn,
                    enabled_tools=enabled_tools,
                )
                for tc in other_tools
            ]
            tool_results = await asyncio.gather(*tool_tasks, return_exceptions=True)
            total_tools_called += len(other_tools)

            # Process results and add to messages
            for tc, result_data in zip(other_tools, tool_results):
                if isinstance(result_data, Exception):
                    # Tool execution failed - send error back to LLM so it can try again
                    logger.warning(f"[REFLECT {reflect_id}] Tool {tc.name} failed with exception: {result_data}")
                    output = {"error": f"Tool execution failed: {result_data}"}
                    duration_ms = 0
                else:
                    output, duration_ms = result_data

                # Normalize tool name for consistent tracking
                normalized_tool_name = _normalize_tool_name(tc.name)

                # Check if tool returned an error response - log but continue (LLM will see the error)
                if isinstance(output, dict) and "error" in output:
                    logger.warning(
                        f"[REFLECT {reflect_id}] Tool {normalized_tool_name} returned error: {output['error']}"
                    )

                # Track available IDs from tool results (only for successful responses)
                if (
                    normalized_tool_name == "search_mental_models"
                    and isinstance(output, dict)
                    and "mental_models" in output
                ):
                    for mm in output["mental_models"]:
                        if "id" in mm:
                            available_mental_model_ids.add(mm["id"])
                    # Deterministic short-circuit (no extra LLM call): on a
                    # low/mid-budget call, if every retrieved mental model is
                    # fresh and has usable content, stop forcing the lower
                    # retrieval layers. The next iteration runs under ``auto``
                    # tool choice, so the agent can answer directly when the
                    # mental model suffices, or — having just read it — issue a
                    # targeted ``search_observations``/``recall`` itself. Stale,
                    # empty, or missing mental models keep the full forced path.
                    if (
                        stop_forcing_from_iteration is None
                        and (budget or "low").lower() != "high"
                        and output.get("mental_models")
                        and _all_mental_models_are_usable_and_fresh(output)
                    ):
                        stop_forcing_from_iteration = iteration + 1
                        logger.info(
                            f"[REFLECT {reflect_id}] Fresh mental models sufficient on iteration {iteration + 1}; "
                            "releasing forced lower-level retrieval to auto."
                        )

                if (
                    normalized_tool_name == "search_observations"
                    and isinstance(output, dict)
                    and "observations" in output
                ):
                    for obs in output["observations"]:
                        if "id" in obs:
                            available_observation_ids.add(obs["id"])

                if normalized_tool_name == "recall" and isinstance(output, dict) and "memories" in output:
                    for memory in output["memories"]:
                        if "id" in memory:
                            available_memory_ids.add(memory["id"])

                # Add tool result message
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,  # Required by Gemini
                        "content": json.dumps(output, default=str, ensure_ascii=False),
                    }
                )

                # Track for logging and context history
                input_dict = {"tool": tc.name, **tc.arguments}
                input_summary = _summarize_input(tc.name, tc.arguments)

                # Extract reason from tool arguments (if provided)
                tool_reason = tc.arguments.get("reason")

                tool_trace.append(
                    ToolCall(
                        tool=tc.name,
                        reason=tool_reason,
                        input=input_dict,
                        output=output,
                        duration_ms=duration_ms,
                        iteration=iteration + 1,
                    )
                )

                try:
                    output_chars = len(json.dumps(output, ensure_ascii=False))
                except (TypeError, ValueError):
                    output_chars = len(str(output))

                tool_trace_summary.append(
                    {
                        "tool": tc.name,
                        "input_summary": input_summary,
                        "duration_ms": duration_ms,
                        "output_chars": output_chars,
                    }
                )

                # Keep context history for fallback final prompt
                context_history.append({"tool": tc.name, "input": input_dict, "output": output})

    # Should not reach here
    answer = "I was unable to formulate a complete answer within the iteration limit."
    _log_completion(answer, max_iterations, forced=True)
    return ReflectAgentResult(
        text=answer,
        iterations=max_iterations,
        tools_called=total_tools_called,
        tool_trace=tool_trace,
        llm_trace=_get_llm_trace(),
        usage=_get_usage(),
        directives_applied=directives_applied,
    )


def _tool_call_to_dict(tc: "LLMToolCall") -> dict[str, Any]:
    """Convert LLMToolCall to OpenAI message format."""
    d: dict[str, Any] = {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.name,
            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
        },
    }
    if tc.thought_signature is not None:
        d["thought_signature"] = tc.thought_signature
    return d


async def _process_done_tool(
    done_call: "LLMToolCall",
    available_memory_ids: set[str],
    available_mental_model_ids: set[str],
    available_observation_ids: set[str],
    iterations: int,
    total_tools_called: int,
    tool_trace: list[ToolCall],
    llm_trace: list[LLMCall],
    usage: TokenUsageSummary,
    log_completion: Callable,
    reflect_id: str,
    directives_applied: list[DirectiveInfo],
    llm_config: "LLMProvider | None" = None,
    response_schema: dict | None = None,
) -> ReflectAgentResult:
    """Process the done tool call and return the result."""
    args = done_call.arguments

    # Extract and clean the answer - some LLMs leak structured output into the answer text
    raw_answer = args.get("answer", "").strip()
    answer = _clean_done_answer(raw_answer) if raw_answer else ""
    if not answer:
        answer = "No answer provided."

    # Validate IDs (only include IDs that were actually retrieved)
    used_memory_ids = [mid for mid in (args.get("memory_ids") or []) if mid in available_memory_ids]
    used_mental_model_ids = [mid for mid in (args.get("mental_model_ids") or []) if mid in available_mental_model_ids]
    used_observation_ids = [oid for oid in (args.get("observation_ids") or []) if oid in available_observation_ids]

    # Generate structured output if schema provided
    structured_output = None
    final_usage = usage
    if response_schema and llm_config and answer:
        struct = await _generate_structured_output(answer, response_schema, llm_config, reflect_id)
        structured_output = struct.structured_output
        # Add structured output tokens to usage
        final_usage = TokenUsageSummary(
            input_tokens=usage.input_tokens + struct.input_tokens,
            output_tokens=usage.output_tokens + struct.output_tokens,
            total_tokens=usage.total_tokens + struct.input_tokens + struct.output_tokens,
            cached_tokens=usage.cached_tokens + struct.cached_tokens,
            thoughts_tokens=usage.thoughts_tokens + struct.thoughts_tokens,
        )

    log_completion(answer, iterations)
    return ReflectAgentResult(
        text=answer,
        structured_output=structured_output,
        iterations=iterations,
        tools_called=total_tools_called,
        tool_trace=tool_trace,
        llm_trace=llm_trace,
        usage=final_usage,
        used_memory_ids=used_memory_ids,
        used_mental_model_ids=used_mental_model_ids,
        used_observation_ids=used_observation_ids,
        directives_applied=directives_applied,
    )


async def _execute_tool_with_timing(
    tc: "LLMToolCall",
    search_mental_models_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    search_observations_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    recall_fn: Callable[[str, int, int], Awaitable[dict[str, Any]]],
    expand_fn: Callable[[list[str], str], Awaitable[dict[str, Any]]],
    enabled_tools: frozenset[str] | None = None,
) -> tuple[dict[str, Any], int]:
    """Execute a tool call and return result with timing."""
    from hindsight_api.tracing import get_tracer

    start_time = time.time()

    # Create span for tool execution
    tracer = get_tracer()
    # Normalize tool name for span
    normalized_name = _normalize_tool_name(tc.name)
    span_name = f"hindsight.reflect_tool_exec.{normalized_name}"

    # Calculate timestamps
    start_time_ns = time.time_ns()

    with tracer.start_as_current_span(
        span_name,
        start_time=start_time_ns,
        end_on_exit=False,
    ) as span:
        # Set attributes
        span.set_attribute("hindsight.tool.name", normalized_name)
        span.set_attribute("hindsight.tool.id", tc.id)
        span.set_attribute("hindsight.tool.arguments", json.dumps(tc.arguments, ensure_ascii=False))

        try:
            result = await _execute_tool(
                tc.name,
                tc.arguments,
                search_mental_models_fn,
                search_observations_fn,
                recall_fn,
                expand_fn,
                enabled_tools=enabled_tools,
            )

            # Set success attributes
            if isinstance(result, dict) and "error" in result:
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR, result["error"]))
            else:
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.OK))

            duration_ms = int((time.time() - start_time) * 1000)
            span.set_attribute("hindsight.tool.duration_ms", duration_ms)

            # End span with correct timestamp
            end_time_ns = time.time_ns()
            span.end(end_time=end_time_ns)

            return result, duration_ms
        except Exception as e:
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            duration_ms = int((time.time() - start_time) * 1000)
            span.set_attribute("hindsight.tool.duration_ms", duration_ms)
            end_time_ns = time.time_ns()
            span.end(end_time=end_time_ns)
            raise


async def _execute_tool(
    tool_name: str,
    args: dict[str, Any],
    search_mental_models_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    search_observations_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    recall_fn: Callable[[str, int, int], Awaitable[dict[str, Any]]],
    expand_fn: Callable[[list[str], str], Awaitable[dict[str, Any]]],
    enabled_tools: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Execute a single tool by name."""
    # Normalize tool name for various LLM output formats
    tool_name = _normalize_tool_name(tool_name)

    # Guard against LLMs hallucinating calls to tools that were not provided.
    # "done" is always available; "expand" is governed by enabled_tools (excluded
    # when text storage is disabled), so it is not hardcoded as always-allowed here.
    if enabled_tools is not None and tool_name not in enabled_tools and tool_name != "done":
        return {"error": f"Tool '{tool_name}' is not available. Use only the tools provided to you."}

    if tool_name == "search_mental_models":
        query = args.get("query")
        if not query:
            return {"error": "search_mental_models requires a query parameter"}
        max_results = int(args.get("max_results") or 5)
        return await search_mental_models_fn(query, max_results)

    elif tool_name == "search_observations":
        query = args.get("query")
        if not query:
            return {"error": "search_observations requires a query parameter"}
        max_tokens = max(int(args.get("max_tokens") or 5000), 1000)  # Default 5000, min 1000
        return await search_observations_fn(query, max_tokens)

    elif tool_name == "recall":
        query = args.get("query")
        if not query:
            return {"error": "recall requires a query parameter"}
        max_tokens = max(int(args.get("max_tokens") or 2048), 1000)  # Default 2048, min 1000
        max_chunk_tokens = max(int(args.get("max_chunk_tokens") or 1000), 1000)  # Always enabled, min 1000
        return await recall_fn(query, max_tokens, max_chunk_tokens)

    elif tool_name == "expand":
        memory_ids = args.get("memory_ids", [])
        if not memory_ids:
            return {"error": "expand requires memory_ids"}
        depth = args.get("depth", "chunk")
        return await expand_fn(memory_ids, depth)

    else:
        return {"error": f"Unknown tool: {tool_name}"}


def _summarize_input(tool_name: str, args: dict[str, Any]) -> str:
    """Create a summary of tool input for logging, showing all params."""
    if tool_name == "search_mental_models":
        query = args.get("query", "")
        query_preview = f"'{query[:30]}...'" if len(query) > 30 else f"'{query}'"
        max_results = int(args.get("max_results") or 5)
        return f"(query={query_preview}, max_results={max_results})"
    elif tool_name == "search_observations":
        query = args.get("query", "")
        query_preview = f"'{query[:30]}...'" if len(query) > 30 else f"'{query}'"
        max_tokens = max(int(args.get("max_tokens") or 5000), 1000)
        return f"(query={query_preview}, max_tokens={max_tokens})"
    elif tool_name == "recall":
        query = args.get("query", "")
        query_preview = f"'{query[:30]}...'" if len(query) > 30 else f"'{query}'"
        max_tokens = max(int(args.get("max_tokens") or 2048), 1000)
        max_chunk_tokens = max(int(args.get("max_chunk_tokens") or 1000), 1000)
        return f"(query={query_preview}, max_tokens={max_tokens}, max_chunk_tokens={max_chunk_tokens})"
    elif tool_name == "expand":
        memory_ids = args.get("memory_ids", [])
        depth = args.get("depth", "chunk")
        return f"(memory_ids=[{len(memory_ids)} ids], depth={depth})"
    elif tool_name == "done":
        answer = args.get("answer", "")
        answer_preview = f"'{answer[:30]}...'" if len(answer) > 30 else f"'{answer}'"
        memory_ids = args.get("memory_ids", [])
        mental_model_ids = args.get("mental_model_ids", [])
        observation_ids = args.get("observation_ids", [])
        return (
            f"(answer={answer_preview}, mem={len(memory_ids)}, mm={len(mental_model_ids)}, obs={len(observation_ids)})"
        )
    return str(args)
