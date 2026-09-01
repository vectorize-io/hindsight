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
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ...config import get_config
from ..llm_interface import LLM_TOOL_CHOICE_AUTO, LLMToolChoice
from .models import DirectiveInfo, LLMCall, ReflectAgentResult, StructuredOutputResult, TokenUsageSummary, ToolCall
from .prompts import (
    _SPLIT_SYNTHESIS_WARN_CHUNKS,
    CLAIMS_SYSTEM_PROMPT,
    _extract_directive_rules,
    build_agent_user_prompt,
    build_chunk_claims_prompt,
    build_final_prompt,
    build_final_system_prompt,
    build_reduce_prompt,
    build_system_prompt_for_tools,
    split_context_history,
)
from .structured_doc import (
    CanonicalDocument,
    StructuredDocument,
    document_from_sections,
    render_document,
    split_markdown,
)
from .tokenization import count_prompt_tokens
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


class ReflectNoAnswerError(RuntimeError):
    """The agent finished without producing an answer.

    Reflect used to substitute a human-readable placeholder here ("No answer
    provided.", or an iteration-limit sentence). Both are non-empty strings, so
    every downstream emptiness check let them through and callers could not tell
    a failed run from a real one -- a mental-model refresh persisted the
    placeholder over a document built across months of refreshes and reported
    ``completed`` with no error (#2959). A run that produced nothing is a
    failure, so it raises like any other, and callers that write what reflect
    returns simply never reach the write.
    """


class ReflectToolCallError(RuntimeError):
    """The model never produced a tool call reflect could understand.

    Reflect is driven entirely by structured tool calls (``recall``, ``expand``,
    ``done`` ...). Some provider transports do not actually support function
    calling and silently drop the tool definitions from the request (e.g. litellm's
    Vertex AI gpt-oss MaaS path strips ``tools``/``tool_choice`` when the model is
    flagged as not supporting them). The model then answers in free text that may
    mimic a ``done`` payload. Rather than salvage that untooled text -- and risk
    surfacing raw tool-call JSON as the answer -- we fail loudly so the caller can
    switch to a tool-calling-capable model/transport.
    """


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


async def _generate_structured_output(
    answer: str,
    response_schema: dict,
    llm_config: "LLMProvider",
    reflect_id: str,
    max_tokens: int | None = None,
) -> StructuredOutputResult:
    """Generate structured output from an answer using the provided JSON schema.

    Args:
        answer: The text answer to extract structured data from
        response_schema: JSON Schema for the expected output structure
        llm_config: LLM provider for making the extraction call
        reflect_id: Reflect ID for logging
        max_tokens: Output-token budget for the extraction call, mirroring the
            plain reflect calls (omitted when None); without it, reasoning /
            preamble models can exhaust the provider default before emitting any
            JSON (finish_reason=length, empty content -> issue #2431)

    Returns:
        A StructuredOutputResult carrying the structured output (None if
        generation fails) and the call's token usage.
    """
    try:
        from typing import Any as TypingAny

        from pydantic import create_model

        def _python_type_for(field_schema: dict, name: str) -> TypingAny:
            """Map a JSON-schema node to a Python type, recursing into nested
            objects/arrays. Nested objects become real Pydantic models (with
            declared properties) rather than a bare ``dict`` — a bare dict/list
            serialises with ``additionalProperties``, which Gemini's structured
            output rejects. Properly-typed nested models avoid that and also let
            the provider grammar-enforce the shape."""
            json_type = field_schema.get("type", "string")
            if json_type == "object":
                nested_props = field_schema.get("properties")
                if isinstance(nested_props, dict) and nested_props:
                    return _model_for(field_schema, name)
                return dict  # free-form object with no declared properties
            if json_type == "array":
                items = field_schema.get("items")
                if isinstance(items, dict):
                    item_type = _python_type_for(items, f"{name}Item")
                    return list[item_type]
                return list
            if json_type == "integer":
                return int
            if json_type == "number":
                return float
            if json_type == "boolean":
                return bool
            return str

        def _model_for(schema: dict, name: str) -> type:
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            model_fields: dict[str, TypingAny] = {}
            for fname, fschema in props.items():
                ftype = _python_type_for(fschema if isinstance(fschema, dict) else {}, f"{name}_{fname}")
                default = ... if fname in required else None
                model_fields[fname] = (ftype, default)
            return create_model(name, **model_fields)

        schema_props = response_schema.get("properties", {})
        required_fields = set(response_schema.get("required", []))

        if not schema_props:
            logger.warning(f"[REFLECT {reflect_id}] No fields found in response_schema, skipping structured output")
            return StructuredOutputResult()

        DynamicModel = _model_for(response_schema, "StructuredResponse")

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
            strict_schema=get_config().llm_strict_schema_reflect,
            # Schema extraction should be deterministic. The configured reflect
            # temperature applies to answer generation, not this parsing pass.
            temperature=0.0,
            max_completion_tokens=max_tokens,
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
    """Estimate the token count of the messages list using the configured encoding."""
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, str):
            total += count_prompt_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += count_prompt_tokens(part["text"])
        # Tool call arguments and results also count
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                total += count_prompt_tokens(func.get("arguments", ""))
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


# Detached cache-teardown tasks. asyncio holds only weak references to tasks, so
# a fire-and-forget task can be garbage-collected mid-flight — keep a strong
# reference here until it finishes.
_cache_cleanup_tasks: set[asyncio.Task] = set()


def _spawn_cache_cleanup(
    provider_impl: Any,
    session_id: str,
    cache_tasks: list[asyncio.Task],
    reflect_id: str,
) -> None:
    """Delete a reflect's ephemeral context caches in the background.

    The per-reflect caches are dead the moment the reflect returns — nothing ever
    reuses them — so the caller must not wait on teardown: draining the in-flight
    create plus the delete round-trips would add latency to every single answer.
    Detach it instead. The short cache TTL is the backstop if the process dies
    before the task runs.
    """

    async def _cleanup() -> None:
        try:
            # Let any overlapped create land first, so its cache is registered in
            # the session and actually gets deleted rather than lingering to TTL.
            if cache_tasks:
                await asyncio.gather(*cache_tasks, return_exceptions=True)
            await provider_impl.delete_cache_session(session_id)
        except Exception:
            logger.debug("[REFLECT %s] cache session teardown failed (will age out on TTL)", reflect_id)

    try:
        task = asyncio.create_task(_cleanup())
    except RuntimeError:
        # No running loop to detach onto (not expected in the server); TTL cleans up.
        return
    _cache_cleanup_tasks.add(task)
    task.add_done_callback(_cache_cleanup_tasks.discard)


async def run_reflect_agent(
    llm_config: "LLMProvider",
    bank_id: str,
    query: str,
    bank_profile: dict[str, Any],
    search_mental_models_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    search_observations_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    recall_fn: Callable[[str, int, int], Awaitable[dict[str, Any]]],
    expand_fn: Callable[[list[str], str], Awaitable[dict[str, Any]]],
    **kwargs: Any,
) -> ReflectAgentResult:
    """Public entrypoint: runs the agent loop and tears down any per-step context
    caches it created.

    The step-by-step caches (Gemini ``CachedContent``) are ephemeral — scoped to
    exactly one reflect and never reused after it — so teardown is scheduled on
    every exit path (answer, error, cancellation) but runs **detached**: the
    caller gets its answer without waiting on the delete round-trips. The short
    cache TTL is the backstop if the teardown never runs; the delete is
    best-effort and never allowed to fail a reflect.
    """
    reflect_id = f"{bank_id[:8]}-{int(time.time() * 1000) % 100000}"
    provider_impl = getattr(llm_config, "_provider_impl", None)
    # Reflect step-by-step caching needs the provider to support it AND the
    # dedicated reflect flag (on by default; distinct from the global prompt-cache
    # switch so it can be turned off for reflect alone).
    incremental_caching = (
        provider_impl is not None
        and provider_impl.supports_incremental_prompt_cache()
        and get_config().reflect_prompt_cache_enabled
    )
    cache_session_id = f"reflect:{reflect_id}"
    # In-flight cache-create tasks (scheduled to overlap tool execution). Awaited
    # before teardown so every created cache is tracked and deleted — no orphans.
    cache_tasks: list[asyncio.Task] = []
    try:
        return await _run_reflect_agent_inner(
            llm_config,
            bank_id,
            query,
            bank_profile,
            search_mental_models_fn,
            search_observations_fn,
            recall_fn,
            expand_fn,
            reflect_id=reflect_id,
            provider_impl=provider_impl,
            incremental_caching=incremental_caching,
            cache_session_id=cache_session_id,
            cache_tasks=cache_tasks,
            **kwargs,
        )
    finally:
        if incremental_caching and provider_impl is not None:
            _spawn_cache_cleanup(provider_impl, cache_session_id, cache_tasks, reflect_id)


async def _run_reflect_agent_inner(
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
    store_document_text: bool = True,
    answer_as_document: bool = False,
    *,
    reflect_id: str,
    provider_impl: Any,
    incremental_caching: bool,
    cache_session_id: str,
    cache_tasks: list[asyncio.Task],
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
        max_tokens: Desired *visible* length of the final answer. Communicated to
            the model as a soft directive and enforced by the post-hoc rewrite --
            NOT passed as the provider's ``max_completion_tokens``, which on
            thinking models is consumed by reasoning tokens and would truncate the
            answer mid-word (#3365). The transport-level cost cap is a separate,
            uncapped-by-default config (``reflect_max_completion_tokens``).
        response_schema: Optional JSON Schema for structured output in final response
        directives: Optional list of directive mental models to inject as hard rules

    Returns:
        ReflectAgentResult with final answer and metadata
    """
    start_time = time.time()

    # Transport-level output cap for the synthesis calls. Decoupled from
    # ``max_tokens`` (a page-length target enforced via prompt + rewrite): None by
    # default so reasoning models run to a natural stop instead of truncating the
    # visible page mid-word (#3365). An operator can set a hard cost ceiling via
    # HINDSIGHT_API_REFLECT_MAX_COMPLETION_TOKENS.
    synthesis_max_completion_tokens = get_config().reflect_max_completion_tokens

    # Build directives_applied for the trace
    directives_applied = _build_directives_applied(directives)

    # Extract directive rules for tool schema (if any)
    directive_rules = _extract_directive_rules(directives) if directives else None

    # Get tools for this agent (with directive compliance field if directives exist).
    # The expand tool only reads back raw source text (chunks/documents), so it is
    # useless and excluded when document text storage is disabled (per bank).
    include_expand = store_document_text
    tools = get_reflect_tools(
        directive_rules=directive_rules,
        include_mental_models=has_mental_models,
        include_observations=include_observations,
        include_recall=include_recall,
        include_expand=include_expand,
        answer_as_document=answer_as_document,
        llm_output_language=llm_output_language,
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
        answer_as_document=answer_as_document,
        llm_output_language=llm_output_language,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_agent_user_prompt(query, llm_output_language)},
    ]

    # Step-by-step context caching for the agentic tool loop.
    #
    # Caching only the static system+tools prefix wins little here: it's dwarfed
    # by the tool results (recall/observations) that get re-sent on every turn.
    # Instead we roll a cache forward one step at a time — after each turn the
    # cache is extended to cover that turn's FULL input, so the next ``auto`` turn
    # reuses the entire prior conversation at the cached rate and sends only its
    # own new tool results as the delta. Each new tool payload is therefore billed
    # at full price exactly once (the turn it's produced), then cached thereafter.
    #
    # The cache create for turn N+1 covers turn N's input, which is fully known the
    # moment turn N's LLM call returns — so we kick it off as a background task that
    # runs CONCURRENTLY with turn N's tool execution (``_schedule_cache``) and only
    # await it (``_resolve_pending_cache``) right before the next ``auto`` call,
    # hiding the create latency behind work we'd do anyway.
    #
    # ``rolling_cache_boundary`` is the number of leading ``messages`` baked into
    # the adopted ``rolling_cache_name``. ``incremental_caching`` is False for
    # providers/config without explicit caching, so every branch below is a no-op.
    rolling_cache_name: str | None = None
    rolling_cache_boundary = 0
    pending_cache_task: asyncio.Task | None = None
    pending_cache_boundary = 0

    async def _resolve_pending_cache() -> None:
        """Adopt the overlapped next-cache once it's ready as the rolling cache.

        Best-effort: a failed/``None`` create just leaves the previous (smaller)
        cache in place, so the next call sends a larger delta but stays correct.
        """
        nonlocal rolling_cache_name, rolling_cache_boundary, pending_cache_task
        if pending_cache_task is None:
            return
        task = pending_cache_task
        pending_cache_task = None
        try:
            new_name = await task
        except Exception:
            new_name = None
        if new_name is not None:
            rolling_cache_name = new_name
            rolling_cache_boundary = pending_cache_boundary

    def _schedule_cache(upto: int) -> None:
        """Start building the cache covering ``messages[:upto]`` in the background
        so it overlaps the tool execution that follows this turn."""
        nonlocal pending_cache_task, pending_cache_boundary
        # ``messages[:upto]`` is snapshotted now, so appends during tool execution
        # can't change what gets cached. ``ensure_future`` raises if the provider
        # didn't return a coroutine (e.g. a test double) — caching is a soft
        # optimisation and must never break a reflect, so swallow and skip.
        try:
            task = asyncio.ensure_future(
                provider_impl.create_incremental_cache(
                    session_id=cache_session_id, messages=messages[:upto], tools=tools
                )
            )
        except Exception:
            return
        pending_cache_boundary = upto
        pending_cache_task = task
        cache_tasks.append(task)

    # Tracking
    total_tools_called = 0
    # Whether the model has ever produced a tool call reflect could understand.
    # Stays False when a transport silently strips tool support (the model then
    # only ever returns free text) -- that case fails via ReflectToolCallError.
    saw_tool_call = False
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

    async def _tracked_llm_call(prompt: str, trace_scope: str, system_prompt: str, completion_cap: int | None) -> str:
        """One tool-less LLM call with usage/trace accounting folded in."""
        nonlocal total_input_tokens, total_output_tokens, total_cached_tokens, total_thoughts_tokens
        llm_start = time.time()
        response, usage = await llm_config.call(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            scope="reflect",
            temperature=get_config().llm_temperature_reflect,
            max_completion_tokens=completion_cap,
            return_usage=True,
        )
        llm_duration = int((time.time() - llm_start) * 1000)
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens
        total_cached_tokens += getattr(usage, "cached_tokens", 0) or 0
        total_thoughts_tokens += getattr(usage, "thoughts_tokens", 0) or 0
        llm_trace.append(
            {
                "scope": trace_scope,
                "duration_ms": llm_duration,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }
        )
        return response.strip()

    async def _forced_final_synthesis(iterations_completed: int) -> ReflectAgentResult:
        """Answer without tools from the accumulated tool results.

        When the accumulated results fit the prompt budget this is one LLM call,
        exactly as before. When they exceed it, they are SPLIT — not truncated:
        each budget-sized chunk is compressed in parallel into dated, cited
        claims, and one reduce call synthesizes the answer from every chunk's
        claims. The old behavior dropped any over-budget block whole (plus all
        older ones), which produced confident "no information" answers carrying
        hundreds of citations the synthesis model never saw (#3122).
        """
        nonlocal total_input_tokens, total_output_tokens, total_cached_tokens, total_thoughts_tokens
        final_system = build_final_system_prompt(bank_profile.get("mission"), llm_output_language, directives)
        chunks = split_context_history(context_history, max_context_tokens)
        # Every call below uses the transport-level cap, never the caller's
        # max_tokens: that is a visible-length target carried as a prompt
        # directive (#3365), and capping the transport with it would truncate
        # thinking models mid-word — or, on the map calls, starve the evidence
        # extraction.
        if len(chunks) <= 1:
            prompt = build_final_prompt(
                query,
                context_history,
                bank_profile,
                context,
                max_context_tokens=max_context_tokens,
                max_tokens=max_tokens,
                llm_output_language=llm_output_language,
            )
            answer = await _tracked_llm_call(prompt, "final", final_system, synthesis_max_completion_tokens)
        else:
            log = logger.warning if len(chunks) > _SPLIT_SYNTHESIS_WARN_CHUNKS else logger.info
            log(
                f"[REFLECT {reflect_id}] Retrieved data exceeds the context budget; "
                f"split synthesis over {len(chunks)} chunks."
            )
            # Map: each chunk in parallel.
            claim_sections = await asyncio.gather(
                *(
                    _tracked_llm_call(
                        build_chunk_claims_prompt(query, chunk),
                        f"final_map_{i}",
                        CLAIMS_SYSTEM_PROMPT,
                        synthesis_max_completion_tokens,
                    )
                    for i, chunk in enumerate(chunks, 1)
                )
            )
            # Reduce: one synthesis call over every chunk's claims.
            prompt = build_reduce_prompt(
                query,
                list(claim_sections),
                bank_profile,
                context,
                max_tokens=max_tokens,
                llm_output_language=llm_output_language,
            )
            answer = await _tracked_llm_call(prompt, "final", final_system, synthesis_max_completion_tokens)

        if not (answer or "").strip():
            # Tools were disabled for this call and the model still returned nothing.
            # There is no answer to hand back, so the run failed -- see #2959 for why
            # a placeholder here is worse than an exception.
            raise ReflectNoAnswerError(
                f"Reflect's final synthesis returned no text after {iterations_completed} iteration(s) "
                f"over {len(chunks)} context chunk(s)."
            )

        structured_output = None
        # ``answer`` is non-empty past the guard above, so only the schema gates this.
        if response_schema:
            struct = await _generate_structured_output(answer, response_schema, llm_config, reflect_id, max_tokens)
            structured_output = struct.structured_output
            total_input_tokens += struct.input_tokens
            total_output_tokens += struct.output_tokens
            total_cached_tokens += struct.cached_tokens
            total_thoughts_tokens += struct.thoughts_tokens

        _log_completion(answer, iterations_completed, forced=True)
        return ReflectAgentResult(
            text=answer,
            structured_output=structured_output,
            iterations=iterations_completed,
            tools_called=total_tools_called,
            tool_trace=tool_trace,
            llm_trace=_get_llm_trace(),
            usage=_get_usage(),
            directives_applied=directives_applied,
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
            return await _forced_final_synthesis(iteration + 1)

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
            return await _forced_final_synthesis(iteration + 1)

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
            iter_tool_choice = LLM_TOOL_CHOICE_AUTO
        elif iteration < len(forced_sequence):
            iter_tool_choice = LLMToolChoice.named(forced_sequence[iteration])
        else:
            iter_tool_choice = LLM_TOOL_CHOICE_AUTO

        # Will the NEXT turn be an ``auto`` turn (the only kind that references a
        # cache)? The cache we schedule this turn covers this turn's input and is
        # used by the next turn, so we only bother building it when the next turn
        # can use it — skipping the wasted creates between two forced turns.
        next_iter = iteration + 1
        if stop_forcing_from_iteration is not None and next_iter >= stop_forcing_from_iteration:
            next_is_auto = True
        elif next_iter < len(forced_sequence):
            next_is_auto = False
        else:
            next_is_auto = True

        # Before an ``auto`` turn, adopt the cache that was being built in the
        # background during the previous turn's tool execution. It covers that
        # turn's full input, so THIS call reuses the entire prior conversation at
        # the cached rate and sends only the turns appended since. Forced turns
        # can't use a cache (Gemini rejects ``cached_content`` + ``tool_config``),
        # but the cache still advances underneath them, so the first ``auto`` turn
        # inherits a cache covering all the forced results.
        if incremental_caching and iter_tool_choice is LLM_TOOL_CHOICE_AUTO:
            await _resolve_pending_cache()

        call_msg_count = len(messages)
        try:
            ct_kwargs: dict[str, Any] = dict(
                messages=messages,
                tools=tools,
                scope="reflect_tool_call",
                tool_choice=iter_tool_choice,
                temperature=get_config().llm_temperature_reflect,
            )
            if incremental_caching and iter_tool_choice is LLM_TOOL_CHOICE_AUTO and rolling_cache_name is not None:
                ct_kwargs["cached_prefix"] = rolling_cache_name
                ct_kwargs["cached_prefix_message_count"] = rolling_cache_boundary
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
            return await _forced_final_synthesis(iteration + 1)

        # No tool calls this turn.
        if not result.tool_calls:
            # Reflect is driven by structured tool calls. A turn with no tool call
            # means one of two things:
            #   * the model already gathered evidence via earlier tool calls and is
            #     now stopping -- fine, synthesize a clean final answer below;
            #   * the transport can't produce tool calls at all, so it only ever
            #     returns free text (e.g. litellm strips tools on the Vertex gpt-oss
            #     MaaS path). In that case ``saw_tool_call`` is still False.
            # We no longer salvage that free text as the answer -- it can be a raw
            # done()-payload with sibling id fields leaking into user-visible text.
            # Fail loudly instead so the caller picks a tool-calling-capable model.
            if not saw_tool_call:
                snippet = (result.content or "").strip()
                if len(snippet) > 500:
                    snippet = snippet[:500] + "..."
                detail = f" Response: {snippet!r}" if snippet else " The model returned no content."
                raise ReflectToolCallError(
                    f"Reflect requires a tool-calling model, but {llm_config.provider}/{llm_config.model} "
                    f"produced no usable tool call (the transport may not support function calling)." + detail
                )
            # Model tool-called earlier and is now stopping: fall through to a clean
            # forced final synthesis (tools disabled, prose expected).
            return await _forced_final_synthesis(iteration + 1)

        # The model produced at least one tool call reflect could parse: it can
        # drive the loop, so a later text-only turn is a legitimate stop, not a
        # broken transport.
        saw_tool_call = True

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
                    max_tokens=max_tokens,
                )

        # Execute other tools in parallel (exclude done tool in all its format variants)
        other_tools = [tc for tc in result.tool_calls if not _is_done_tool(tc.name)]
        if other_tools:
            # Partition into enabled vs hallucinated (not in enabled_tools set),
            # carrying each call's position in the model's original batch so the
            # results can be re-emitted in that order below.
            allowed_tools = []
            allowed_positions: list[int] = []
            hallucinated_tools = []
            hallucinated_positions: list[int] = []
            for position, tc in enumerate(other_tools):
                norm = _normalize_tool_name(tc.name)
                # "done" is always available. "expand" is governed by enabled_tools
                # (it is excluded when text storage is disabled), so it is not hardcoded here.
                if enabled_tools is not None and norm not in enabled_tools and norm != "done":
                    hallucinated_tools.append(tc)
                    hallucinated_positions.append(position)
                else:
                    allowed_tools.append(tc)
                    allowed_positions.append(position)

            # Build assistant message with all tool calls (LLM requires them for history)
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [_tool_call_to_dict(tc) for tc in other_tools],
                }
            )

            # Serialize tool results in the ORIGINAL tool_calls order. Anthropic
            # requires tool_result blocks to match the assistant tool_use order,
            # so collect every result first and emit them in the model's order
            # after execution (execution order may differ).
            #
            # Slots are indexed by POSITION, never by tool_call_id: a
            # non-conforming OpenAI-compatible gateway can hand back duplicate or
            # empty ids for a parallel batch, and an id-keyed map would silently
            # drop one tool's evidence and duplicate another's.
            ordered_tool_calls = list(other_tools)
            tool_outputs: list[str] = [""] * len(ordered_tool_calls)
            for position, tc in zip(hallucinated_positions, hallucinated_tools):
                tool_outputs[position] = json.dumps(
                    {
                        "error": f"Tool '{_normalize_tool_name(tc.name)}' is not available. Use only the tools provided to you."
                    },
                    ensure_ascii=False,
                )

            other_tools = allowed_tools

            # Kick off the next-turn cache (covering THIS call's input) so it
            # builds concurrently with the tool execution below — hiding the
            # create latency. Only schedule when the next turn is ``auto`` (the
            # only kind that references it); the next turn's pre-call resolve then
            # adopts it. Resolve any prior in-flight create first so we don't drop
            # its handle.
            if incremental_caching and next_is_auto:
                await _resolve_pending_cache()
                _schedule_cache(call_msg_count)

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
            for position, tc, result_data in zip(allowed_positions, other_tools, tool_results):
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

                # Record the serialized result; emitted in original order below.
                tool_outputs[position] = json.dumps(output, default=str, ensure_ascii=False)

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

            # Emit tool_result messages in the assistant tool_calls order so the
            # serialized history matches the tool_use blocks (Anthropic requires
            # tool_result blocks in the same order as the corresponding tool_use).
            for tc, tool_output in zip(ordered_tool_calls, tool_outputs):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_output,
                    }
                )

    # Unreachable in practice: the last iteration returns the forced synthesis
    # above, so the loop cannot fall out of the bottom. Kept as a hard failure
    # rather than a fallback sentence -- if it ever does fire, the run produced
    # no answer, and inventing one here is exactly the silent overwrite of #2959.
    raise ReflectNoAnswerError(
        f"Reflect exhausted its {max_iterations} iteration(s) without producing an answer "
        f"({total_tools_called} tool call(s) made)."
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


def _document_from_rewrite(rewritten: str, previous_answer: str) -> CanonicalDocument:
    """Read a shortened document back, falling back to the text if it is not JSON.

    The rewrite is asked for as sections, but it is still model output on a path
    where failing would throw away a whole reflect. A response that does not parse
    is treated as the prose it looks like and split, which is lossless — so the
    worst case is the old behaviour rather than a lost answer.
    """
    from hindsight_api.engine.llm_wrapper import parse_llm_json

    text = rewritten.strip()
    try:
        payload = parse_llm_json(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and payload.get("sections"):
        document = document_from_sections(payload)
        rendered = render_document(document).strip()
        if rendered:
            return CanonicalDocument(markdown=rendered, structure=document)
    if not text:
        # An empty rewrite must not empty the answer; keep what was there.
        return CanonicalDocument(markdown=previous_answer, structure=split_markdown(previous_answer))
    return CanonicalDocument(markdown=text, structure=split_markdown(text))


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
    max_tokens: int | None = None,
) -> ReflectAgentResult:
    """Process the done tool call and return the result."""
    args = done_call.arguments

    # ``done`` is a structured tool call: trust its ``answer`` field verbatim.
    # Sibling id fields (memory_ids, ...) live in their own arguments and are
    # validated separately below -- they can't bleed into a parsed answer string.
    #
    # In document mode the model states the document's structure instead, and the
    # markdown is rendered from it. The rendered text still flows on as ``text``
    # so every consumer (structured-output extraction, the length rewrite, the
    # HTTP response) is unchanged -- what changes is that nobody has to read the
    # model's markdown back to find out what it meant.
    document: StructuredDocument | None = None
    raw_document = args.get("document")
    if isinstance(raw_document, dict):
        document = document_from_sections(raw_document)
        answer = render_document(document).strip()
    else:
        answer = args.get("answer", "").strip()
    if not answer:
        # The model called ``done`` with nothing in it -- typically its output was
        # cut off mid-tool-call by the completion cap, so the answer field arrived
        # empty even though the evidence was gathered. Fail instead of standing in
        # a placeholder: it is non-empty, so every downstream emptiness guard reads
        # it as a real answer and stores it over working content (#2959).
        raise ReflectNoAnswerError(
            f"Reflect's done tool returned no answer (iteration {iterations}, "
            f"{total_tools_called} tool call(s) made). The model's output may have been "
            "truncated before the answer field was written."
        )

    final_usage = usage
    if llm_config and max_tokens is not None and count_prompt_tokens(answer) > max_tokens:
        rewrite_start = time.time()
        # In document mode the trim is asked for as a document too. Asking for
        # prose here would put the model back in the business of writing the
        # markdown that gets stored — on the one path where the answer is long
        # enough that its structure matters most.
        if document is not None:
            rewrite_system = (
                "Shorten the user's document so it fits within the requested token budget. "
                "Preserve the key facts and the document's structure; drop lower-priority detail. "
                'Respond ONLY with JSON: {"sections": [{"heading": "...", "level": 2, '
                '"blocks": ["...", "..."]}]}. A heading carries no "#", and each block is one '
                "paragraph, list, table or code fence."
            )
            rewrite_user = f"Target budget: {max_tokens} tokens.\n\nDocument to shorten:\n{answer}"
        else:
            # The token budget is enforced via the prompt, not a hard provider cap:
            # on thinking models a hard cap is eaten by reasoning tokens and would
            # truncate the rewrite mid-word (#3365). Cost is bounded by the separate
            # reflect_max_completion_tokens config (uncapped by default).
            rewrite_system = (
                "Rewrite the user's text so it fits within the requested token budget. "
                "Preserve the key facts and structure; drop lower-priority detail. "
                "Respond with the rewritten text only, no preamble."
            )
            rewrite_user = f"Target budget: {max_tokens} tokens.\n\nText to rewrite:\n{answer}"

        rewritten, rewrite_usage = await llm_config.call(
            messages=[
                {"role": "system", "content": rewrite_system},
                {"role": "user", "content": rewrite_user},
            ],
            scope="reflect",
            temperature=get_config().llm_temperature_reflect,
            max_completion_tokens=get_config().reflect_max_completion_tokens,
            return_usage=True,
        )
        if document is not None:
            trimmed = _document_from_rewrite(rewritten, answer)
            document, answer = trimmed.structure, trimmed.markdown
        else:
            answer = rewritten.strip()
        final_usage = TokenUsageSummary(
            input_tokens=usage.input_tokens + rewrite_usage.input_tokens,
            output_tokens=usage.output_tokens + rewrite_usage.output_tokens,
            total_tokens=usage.total_tokens + rewrite_usage.input_tokens + rewrite_usage.output_tokens,
            cached_tokens=usage.cached_tokens + (getattr(rewrite_usage, "cached_tokens", 0) or 0),
            thoughts_tokens=usage.thoughts_tokens + (getattr(rewrite_usage, "thoughts_tokens", 0) or 0),
        )
        llm_trace.append(
            LLMCall(
                scope="final_rewrite",
                duration_ms=int((time.time() - rewrite_start) * 1000),
                input_tokens=rewrite_usage.input_tokens,
                output_tokens=rewrite_usage.output_tokens,
            )
        )

    # Validate IDs (only include IDs that were actually retrieved)
    used_memory_ids = [mid for mid in (args.get("memory_ids") or []) if mid in available_memory_ids]
    used_mental_model_ids = [mid for mid in (args.get("mental_model_ids") or []) if mid in available_mental_model_ids]
    used_observation_ids = [oid for oid in (args.get("observation_ids") or []) if oid in available_observation_ids]

    # Generate structured output if schema provided
    structured_output = None
    if response_schema and llm_config and answer:
        struct = await _generate_structured_output(answer, response_schema, llm_config, reflect_id, max_tokens)
        structured_output = struct.structured_output
        # Add structured output tokens to usage
        final_usage = TokenUsageSummary(
            input_tokens=final_usage.input_tokens + struct.input_tokens,
            output_tokens=final_usage.output_tokens + struct.output_tokens,
            total_tokens=final_usage.total_tokens + struct.input_tokens + struct.output_tokens,
            cached_tokens=final_usage.cached_tokens + struct.cached_tokens,
            thoughts_tokens=final_usage.thoughts_tokens + struct.thoughts_tokens,
        )

    log_completion(answer, iterations)
    return ReflectAgentResult(
        text=answer,
        document=document,
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
        max_results, error = _parse_tool_int_arg_or_error(args, "max_results", default=5)
        if error:
            return {"error": error}
        return await search_mental_models_fn(query, max_results)

    elif tool_name == "search_observations":
        query = args.get("query")
        if not query:
            return {"error": "search_observations requires a query parameter"}
        max_tokens, error = _parse_tool_int_arg_or_error(args, "max_tokens", default=5000, minimum=1000)
        if error:
            return {"error": error}
        return await search_observations_fn(query, max_tokens)

    elif tool_name == "recall":
        query = args.get("query")
        if not query:
            return {"error": "recall requires a query parameter"}
        max_tokens, error = _parse_tool_int_arg_or_error(args, "max_tokens", default=2048, minimum=1000)
        if error:
            return {"error": error}
        max_chunk_tokens, error = _parse_tool_int_arg_or_error(
            args,
            "max_chunk_tokens",
            default=1000,
            minimum=1000,
        )
        if error:
            return {"error": error}
        return await recall_fn(query, max_tokens, max_chunk_tokens)

    elif tool_name == "expand":
        memory_ids = args.get("memory_ids", [])
        if not memory_ids:
            return {"error": "expand requires memory_ids"}
        depth = args.get("depth", "chunk")
        return await expand_fn(memory_ids, depth)

    else:
        return {"error": f"Unknown tool: {tool_name}"}


_NULLISH_TOOL_INT_STRINGS = {"", "none", "null"}


def _parse_tool_int_arg(args: dict[str, Any], key: str, *, default: int, minimum: int | None = None) -> int:
    raw_value = args.get(key)
    if not raw_value:
        value = default
    elif isinstance(raw_value, str) and raw_value.strip().lower() in _NULLISH_TOOL_INT_STRINGS:
        value = default
    else:
        value = int(raw_value)
    if minimum is None:
        return value
    return max(value, minimum)


def _parse_tool_int_arg_or_error(
    args: dict[str, Any],
    key: str,
    *,
    default: int,
    minimum: int | None = None,
) -> tuple[int, str | None]:
    try:
        return _parse_tool_int_arg(args, key, default=default, minimum=minimum), None
    except (OverflowError, TypeError, ValueError):
        return default, f"{key} must be an integer or null-like value"


def _summarize_tool_int_arg(args: dict[str, Any], key: str, *, default: int, minimum: int | None = None) -> str:
    try:
        return str(_parse_tool_int_arg(args, key, default=default, minimum=minimum))
    except (OverflowError, TypeError, ValueError):
        return f"invalid:{args.get(key)!r}"


def _summarize_tool_query(args: dict[str, Any]) -> str:
    query = args.get("query") or ""
    if not isinstance(query, str):
        query = str(query)
    return f"'{query[:30]}...'" if len(query) > 30 else f"'{query}'"


def _summarize_input(tool_name: str, args: dict[str, Any]) -> str:
    """Create a summary of tool input for logging, showing all params."""
    if tool_name == "search_mental_models":
        query_preview = _summarize_tool_query(args)
        max_results = _summarize_tool_int_arg(args, "max_results", default=5)
        return f"(query={query_preview}, max_results={max_results})"
    elif tool_name == "search_observations":
        query_preview = _summarize_tool_query(args)
        max_tokens = _summarize_tool_int_arg(args, "max_tokens", default=5000, minimum=1000)
        return f"(query={query_preview}, max_tokens={max_tokens})"
    elif tool_name == "recall":
        query_preview = _summarize_tool_query(args)
        max_tokens = _summarize_tool_int_arg(args, "max_tokens", default=2048, minimum=1000)
        max_chunk_tokens = _summarize_tool_int_arg(args, "max_chunk_tokens", default=1000, minimum=1000)
        return f"(query={query_preview}, max_tokens={max_tokens}, max_chunk_tokens={max_chunk_tokens})"
    elif tool_name == "expand":
        memory_ids = args.get("memory_ids", [])
        depth = args.get("depth", "chunk")
        return f"(memory_ids=[{len(memory_ids)} ids], depth={depth})"
    elif tool_name == "done":
        raw_document = args.get("document")
        if isinstance(raw_document, dict):
            sections = raw_document.get("sections") or []
            blocks = sum(len(s.get("blocks") or []) for s in sections if isinstance(s, dict))
            memory_ids = args.get("memory_ids", [])
            mental_model_ids = args.get("mental_model_ids", [])
            observation_ids = args.get("observation_ids", [])
            return (
                f"(document={len(sections)} sections/{blocks} blocks, mem={len(memory_ids)}, "
                f"mm={len(mental_model_ids)}, obs={len(observation_ids)})"
            )
        answer = args.get("answer", "")
        answer_preview = f"'{answer[:30]}...'" if len(answer) > 30 else f"'{answer}'"
        memory_ids = args.get("memory_ids", [])
        mental_model_ids = args.get("mental_model_ids", [])
        observation_ids = args.get("observation_ids", [])
        return (
            f"(answer={answer_preview}, mem={len(memory_ids)}, mm={len(mental_model_ids)}, obs={len(observation_ids)})"
        )
    return str(args)
