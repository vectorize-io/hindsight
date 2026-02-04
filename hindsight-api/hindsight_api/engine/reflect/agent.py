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

from .models import DirectiveInfo, LLMCall, ReflectAgentResult, TokenUsageSummary, ToolCall
from .prompts import FINAL_SYSTEM_PROMPT, _extract_directive_rules, build_final_prompt, build_system_prompt_for_tools
from .tools_schema import get_reflect_tools


def _build_directives_applied(directives: list[dict[str, Any]] | None) -> list[DirectiveInfo]:
    """Build list of DirectiveInfo from directive mental models.

    Handles multiple directive formats:
    1. New format: directives have direct 'content' field
    2. Fallback: directives have 'description' field
    """
    if not directives:
        return []

    result = []
    for directive in directives:
        directive_id = directive.get("id", "")
        directive_name = directive.get("name", "")

        # Get content from 'content' field or fallback to 'description'
        content = directive.get("content", "") or directive.get("description", "")

        result.append(DirectiveInfo(id=directive_id, name=directive_name, content=content))

    return result


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
_LEAKED_JSON_OBJECT = re.compile(
    r'\s*\{[^{]*"(?:observation_ids|memory_ids|mental_model_ids|answer)"[^}]*\}\s*$', re.DOTALL
)
_TRAILING_IDS_PATTERN = re.compile(
    r"\s*(?:observation_ids|memory_ids|mental_model_ids)\s*[=:]\s*\[.*?\]\s*$", re.DOTALL | re.IGNORECASE
)


def _clean_answer_text(text: str) -> str:
    """Clean up answer text by removing any done() tool call syntax.

    Some LLMs output the done() call as text instead of a proper tool call.
    This strips out patterns like: done({"answer": "...", ...})
    """
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

    cleaned = text

    # Remove leaked JSON in code blocks at the end
    cleaned = _LEAKED_JSON_SUFFIX.sub("", cleaned).strip()

    # Remove leaked raw JSON objects at the end
    cleaned = _LEAKED_JSON_OBJECT.sub("", cleaned).strip()

    # Remove trailing ID patterns
    cleaned = _TRAILING_IDS_PATTERN.sub("", cleaned).strip()

    return cleaned if cleaned else text


async def _generate_structured_output(
    answer: str,
    response_schema: dict,
    llm_config: "LLMProvider",
    reflect_id: str,
) -> tuple[dict[str, Any] | None, int, int]:
    """Generate structured output from an answer using the provided JSON schema.

    Args:
        answer: The text answer to extract structured data from
        response_schema: JSON Schema for the expected output structure
        llm_config: LLM provider for making the extraction call
        reflect_id: Reflect ID for logging

    Returns:
        Tuple of (structured_output, input_tokens, output_tokens).
        structured_output is None if generation fails.
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
            return None, 0, 0

        DynamicModel = create_model("StructuredResponse", **fields)

        # Include the full schema in the prompt for better LLM guidance
        schema_str = json.dumps(response_schema, indent=2)

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
        return structured_output, usage.input_tokens, usage.output_tokens

    except Exception as e:
        logger.warning(f"[REFLECT {reflect_id}] Failed to generate structured output: {e}")
        return None, 0, 0


async def run_reflect_agent(
    llm_config: "LLMProvider",
    bank_id: str,
    query: str,
    bank_profile: dict[str, Any],
    search_mental_models_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    search_observations_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    recall_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    expand_fn: Callable[[list[str], str], Awaitable[dict[str, Any]]],
    context: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tokens: int | None = None,
    response_schema: dict | None = None,
    directives: list[dict[str, Any]] | None = None,
    has_mental_models: bool = False,
    budget: str | None = None,
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

    # Get tools for this agent (with directive compliance field if directives exist)
    tools = get_reflect_tools(directive_rules=directive_rules)

    # Build initial messages (directives are injected into system prompt at START and END)
    system_prompt = build_system_prompt_for_tools(
        bank_profile, context, directives=directives, has_mental_models=has_mental_models, budget=budget
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    # Tracking
    total_tools_called = 0
    tool_trace: list[ToolCall] = []
    tool_trace_summary: list[dict[str, Any]] = []
    llm_trace: list[dict[str, Any]] = []
    context_history: list[dict[str, Any]] = []  # For final prompt fallback

    # Token usage tracking - accumulate across all LLM calls
    total_input_tokens = 0
    total_output_tokens = 0

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

    for iteration in range(max_iterations):
        is_last = iteration == max_iterations - 1

        if is_last:
            # Force text response on last iteration - no tools
            prompt = build_final_prompt(query, context_history, bank_profile, context)
            llm_start = time.time()
            response, usage = await llm_config.call(
                messages=[
                    {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                scope="reflect_agent_final",
                max_completion_tokens=max_tokens,
                return_usage=True,
            )
            llm_duration = int((time.time() - llm_start) * 1000)
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
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
                structured_output, struct_in, struct_out = await _generate_structured_output(
                    answer, response_schema, llm_config, reflect_id
                )
                total_input_tokens += struct_in
                total_output_tokens += struct_out

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

        try:
            result = await llm_config.call_with_tools(
                messages=messages,
                tools=tools,
                scope="reflect_agent",
                tool_choice="required" if iteration == 0 else "auto",  # Force tool use on first iteration
            )
            llm_duration = int((time.time() - llm_start) * 1000)
            total_input_tokens += result.input_tokens
            total_output_tokens += result.output_tokens
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
            logger.warning(f"[REFLECT {reflect_id}] LLM error on iteration {iteration + 1}: {e} ({err_duration}ms)")
            llm_trace.append({"scope": f"agent_{iteration + 1}_err", "duration_ms": err_duration})
            # Guardrail: If no evidence gathered yet, retry
            has_gathered_evidence = (
                bool(available_memory_ids) or bool(available_mental_model_ids) or bool(available_observation_ids)
            )
            if not has_gathered_evidence and iteration < max_iterations - 1:
                continue
            prompt = build_final_prompt(query, context_history, bank_profile, context)
            llm_start = time.time()
            response, usage = await llm_config.call(
                messages=[
                    {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                scope="reflect_agent_final",
                max_completion_tokens=max_tokens,
                return_usage=True,
            )
            llm_duration = int((time.time() - llm_start) * 1000)
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
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
                structured_output, struct_in, struct_out = await _generate_structured_output(
                    answer, response_schema, llm_config, reflect_id
                )
                total_input_tokens += struct_in
                total_output_tokens += struct_out

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
            if result.content:
                answer = _clean_answer_text(result.content.strip())

                # Generate structured output if schema provided
                structured_output = None
                if response_schema and answer:
                    structured_output, struct_in, struct_out = await _generate_structured_output(
                        answer, response_schema, llm_config, reflect_id
                    )
                    total_input_tokens += struct_in
                    total_output_tokens += struct_out

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
            prompt = build_final_prompt(query, context_history, bank_profile, context)
            llm_start = time.time()
            response, usage = await llm_config.call(
                messages=[
                    {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                scope="reflect_agent_final",
                max_completion_tokens=max_tokens,
                return_usage=True,
            )
            llm_duration = int((time.time() - llm_start) * 1000)
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
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
                structured_output, struct_in, struct_out = await _generate_structured_output(
                    answer, response_schema, llm_config, reflect_id
                )
                total_input_tokens += struct_in
                total_output_tokens += struct_out

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
                            }
                        ),
                    }
                )
                continue

            # Process done tool
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
            # Add assistant message with tool calls
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [_tool_call_to_dict(tc) for tc in other_tools],
                }
            )

            # Execute tools in parallel
            tool_tasks = [
                _execute_tool_with_timing(
                    tc,
                    search_mental_models_fn,
                    search_observations_fn,
                    recall_fn,
                    expand_fn,
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
                        "content": json.dumps(output, default=str),
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
                    output_chars = len(json.dumps(output))
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
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.name,
            "arguments": json.dumps(tc.arguments),
        },
    }


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
    used_memory_ids = [mid for mid in args.get("memory_ids", []) if mid in available_memory_ids]
    used_mental_model_ids = [mid for mid in args.get("mental_model_ids", []) if mid in available_mental_model_ids]
    used_observation_ids = [oid for oid in args.get("observation_ids", []) if oid in available_observation_ids]

    # Generate structured output if schema provided
    structured_output = None
    final_usage = usage
    if response_schema and llm_config and answer:
        structured_output, struct_in, struct_out = await _generate_structured_output(
            answer, response_schema, llm_config, reflect_id
        )
        # Add structured output tokens to usage
        final_usage = TokenUsageSummary(
            input_tokens=usage.input_tokens + struct_in,
            output_tokens=usage.output_tokens + struct_out,
            total_tokens=usage.total_tokens + struct_in + struct_out,
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
    recall_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    expand_fn: Callable[[list[str], str], Awaitable[dict[str, Any]]],
) -> tuple[dict[str, Any], int]:
    """Execute a tool call and return result with timing."""
    start = time.time()
    result = await _execute_tool(
        tc.name,
        tc.arguments,
        search_mental_models_fn,
        search_observations_fn,
        recall_fn,
        expand_fn,
    )
    duration_ms = int((time.time() - start) * 1000)
    return result, duration_ms


async def _execute_tool(
    tool_name: str,
    args: dict[str, Any],
    search_mental_models_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    search_observations_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    recall_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    expand_fn: Callable[[list[str], str], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    """Execute a single tool by name."""
    # Normalize tool name for various LLM output formats
    tool_name = _normalize_tool_name(tool_name)

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
        return await recall_fn(query, max_tokens)

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
        # Show actual value used (default 2048, min 1000)
        max_tokens = max(int(args.get("max_tokens") or 2048), 1000)
        return f"(query={query_preview}, max_tokens={max_tokens})"
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
