"""
Reflect agent - agentic loop for reflection with native tool calling.
"""

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .models import DirectiveInfo, LLMCall, MentalModelInput, ReflectAgentResult, ToolCall
from .prompts import FINAL_SYSTEM_PROMPT, _extract_directive_rules, build_final_prompt, build_system_prompt_for_tools
from .tools_schema import get_reflect_tools


def _build_directives_applied(directives: list[dict[str, Any]] | None) -> list[DirectiveInfo]:
    """Build list of DirectiveInfo from directive mental models."""
    if not directives:
        return []

    result = []
    for directive in directives:
        directive_id = directive.get("id", "")
        directive_name = directive.get("name", "")
        observations = directive.get("observations", [])

        rules = []
        for obs in observations:
            # Support both Pydantic Observation objects and dicts
            if hasattr(obs, "content"):
                rules.append(obs.content)
            elif isinstance(obs, dict) and obs.get("content"):
                rules.append(obs["content"])

        result.append(DirectiveInfo(id=directive_id, name=directive_name, rules=rules))

    return result


if TYPE_CHECKING:
    from ..llm_wrapper import LLMProvider
    from ..response_models import LLMToolCall

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 10


async def _generate_structured_output(
    answer: str,
    response_schema: dict,
    llm_config: "LLMProvider",
    reflect_id: str,
) -> dict[str, Any] | None:
    """Generate structured output from an answer using the provided JSON schema.

    Args:
        answer: The text answer to extract structured data from
        response_schema: JSON Schema for the expected output structure
        llm_config: LLM provider for making the extraction call
        reflect_id: Reflect ID for logging

    Returns:
        Structured output dict if successful, None otherwise
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
            return None

        DynamicModel = create_model("StructuredResponse", **fields)

        # Include the full schema in the prompt for better LLM guidance
        schema_str = json.dumps(response_schema, indent=2)

        # Call LLM with the answer to extract structured data
        structured_prompt = f"""Based on this answer, extract the information into the requested structured format.

Answer: {answer}

JSON Schema to follow:
```json
{schema_str}
```

Return ONLY a valid JSON object that matches this exact schema. Pay special attention to field types:
- "type": "array" means the value must be a JSON array/list, NOT a string
- "type": "string" means the value must be a string
- "type": "object" means the value must be a JSON object

Do not include any explanation, only the JSON object."""

        structured_result = await llm_config.call(
            messages=[
                {
                    "role": "system",
                    "content": "Extract structured data from the given answer. Return only valid JSON matching the provided schema exactly.",
                },
                {"role": "user", "content": structured_prompt},
            ],
            response_format=DynamicModel,
            scope="reflect_structured",
            skip_validation=True,  # We'll handle the dict ourselves
        )

        # Convert to dict
        if hasattr(structured_result, "model_dump"):
            structured_output = structured_result.model_dump()
        elif isinstance(structured_result, dict):
            structured_output = structured_result
        else:
            # Try to parse as JSON
            structured_output = json.loads(str(structured_result))

        logger.info(f"[REFLECT {reflect_id}] Generated structured output with {len(structured_output)} fields")
        return structured_output

    except Exception as e:
        logger.warning(f"[REFLECT {reflect_id}] Failed to generate structured output: {e}")
        return None


async def run_reflect_agent(
    llm_config: "LLMProvider",
    bank_id: str,
    query: str,
    bank_profile: dict[str, Any],
    lookup_fn: Callable[[str | None], Awaitable[dict[str, Any]]],
    recall_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    expand_fn: Callable[[list[str], str], Awaitable[dict[str, Any]]],
    learn_fn: Callable[[MentalModelInput], Awaitable[dict[str, Any]]] | None = None,
    context: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tokens: int | None = None,
    response_schema: dict | None = None,
    directives: list[dict[str, Any]] | None = None,
) -> ReflectAgentResult:
    """
    Execute the reflect agent loop using native tool calling.

    The agent iteratively calls tools to gather information and learn,
    then provides a final answer via the done() tool.

    Args:
        llm_config: LLM provider for agent calls
        bank_id: Bank identifier
        query: Question to answer
        bank_profile: Bank profile with name and mission
        lookup_fn: Tool callback for lookup (model_id) -> result
        recall_fn: Tool callback for recall (query, max_tokens) -> result
        expand_fn: Tool callback for expand (memory_id, depth) -> result
        learn_fn: Optional tool callback for learn (MentalModelInput) -> result.
                  If None, learn tool is disabled.
        context: Optional additional context
        max_iterations: Maximum number of iterations before forcing response
        max_tokens: Maximum tokens for the final response
        response_schema: Optional JSON Schema for structured output in final response
        directives: Optional list of directive mental models to inject as hard rules

    Returns:
        ReflectAgentResult with final answer and metadata
    """
    enable_learn = learn_fn is not None
    reflect_id = f"{bank_id[:8]}-{int(time.time() * 1000) % 100000}"
    start_time = time.time()

    # Build directives_applied for the trace
    directives_applied = _build_directives_applied(directives)

    # Extract directive rules for tool schema (if any)
    directive_rules = _extract_directive_rules(directives) if directives else None

    # Get tools for this agent (with directive compliance field if directives exist)
    tools = get_reflect_tools(enable_learn=enable_learn, directive_rules=directive_rules)

    # Build initial messages (directives are injected into system prompt at START and END)
    system_prompt = build_system_prompt_for_tools(bank_profile, context, directives=directives)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    # Tracking
    mental_models_created: list[str] = []
    total_tools_called = 0
    tool_trace: list[ToolCall] = []
    tool_trace_summary: list[dict[str, Any]] = []
    llm_trace: list[dict[str, Any]] = []
    context_history: list[dict[str, Any]] = []  # For final prompt fallback

    # Track available IDs for validation (prevents hallucinated citations)
    available_memory_ids: set[str] = set()
    available_model_ids: set[str] = set()

    # Pre-fetch mental models so the agent always starts with this knowledge
    prefetch_start = time.time()
    models_result = await lookup_fn(None)  # List all mental models
    prefetch_duration = int((time.time() - prefetch_start) * 1000)

    # Track available model IDs
    if isinstance(models_result, dict) and "models" in models_result:
        for model in models_result["models"]:
            if "id" in model:
                available_model_ids.add(model["id"])

    # Add to context history for the agent
    context_history.append({"tool": "list_mental_models", "output": models_result})

    # Add to tool trace
    tool_trace.append(
        ToolCall(
            tool="list_mental_models",
            input={"tool": "list_mental_models"},
            output=models_result,
            duration_ms=prefetch_duration,
            iteration=0,
        )
    )
    tool_trace_summary.append(
        {
            "tool": "list_mental_models",
            "input_summary": "(prefetch)",
            "duration_ms": prefetch_duration,
            "output_chars": len(json.dumps(models_result, default=str)),
        }
    )
    total_tools_called += 1

    # Include in the user message so the agent sees it
    models_info = json.dumps(models_result, indent=2, default=str)
    messages[1]["content"] = f"{query}\n\n## Available Mental Models (pre-fetched)\n```json\n{models_info}\n```"

    def _get_llm_trace() -> list[LLMCall]:
        return [LLMCall(scope=c["scope"], duration_ms=c["duration_ms"]) for c in llm_trace]

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
            response = await llm_config.call(
                messages=[
                    {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                scope="reflect_agent_final",
                max_completion_tokens=max_tokens,
            )
            llm_trace.append({"scope": "final", "duration_ms": int((time.time() - llm_start) * 1000)})
            answer = response.strip()

            # Generate structured output if schema provided
            structured_output = None
            if response_schema and answer:
                structured_output = await _generate_structured_output(answer, response_schema, llm_config, reflect_id)

            _log_completion(answer, iteration + 1, forced=True)
            return ReflectAgentResult(
                text=answer,
                structured_output=structured_output,
                iterations=iteration + 1,
                tools_called=total_tools_called,
                mental_models_created=mental_models_created,
                tool_trace=tool_trace,
                llm_trace=_get_llm_trace(),
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
            llm_trace.append({"scope": f"agent_{iteration + 1}", "duration_ms": llm_duration})

        except Exception:
            llm_trace.append(
                {"scope": f"agent_{iteration + 1}_err", "duration_ms": int((time.time() - llm_start) * 1000)}
            )
            # Guardrail: If no evidence gathered yet, retry
            has_gathered_evidence = bool(available_memory_ids) or bool(available_model_ids)
            if not has_gathered_evidence and iteration < max_iterations - 1:
                continue
            prompt = build_final_prompt(query, context_history, bank_profile, context)
            llm_start = time.time()
            response = await llm_config.call(
                messages=[
                    {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                scope="reflect_agent_final",
                max_completion_tokens=max_tokens,
            )
            llm_trace.append({"scope": "final", "duration_ms": int((time.time() - llm_start) * 1000)})
            answer = response.strip()

            # Generate structured output if schema provided
            structured_output = None
            if response_schema and answer:
                structured_output = await _generate_structured_output(answer, response_schema, llm_config, reflect_id)

            _log_completion(answer, iteration + 1, forced=True)
            return ReflectAgentResult(
                text=answer,
                structured_output=structured_output,
                iterations=iteration + 1,
                tools_called=total_tools_called,
                mental_models_created=mental_models_created,
                tool_trace=tool_trace,
                llm_trace=_get_llm_trace(),
                directives_applied=directives_applied,
            )

        # No tool calls - LLM wants to respond with text
        if not result.tool_calls:
            if result.content:
                answer = result.content.strip()

                # Generate structured output if schema provided
                structured_output = None
                if response_schema and answer:
                    structured_output = await _generate_structured_output(
                        answer, response_schema, llm_config, reflect_id
                    )

                _log_completion(answer, iteration + 1)
                return ReflectAgentResult(
                    text=answer,
                    structured_output=structured_output,
                    iterations=iteration + 1,
                    tools_called=total_tools_called,
                    mental_models_created=mental_models_created,
                    tool_trace=tool_trace,
                    llm_trace=_get_llm_trace(),
                    directives_applied=directives_applied,
                )
            # Empty response, force final
            prompt = build_final_prompt(query, context_history, bank_profile, context)
            llm_start = time.time()
            response = await llm_config.call(
                messages=[
                    {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                scope="reflect_agent_final",
                max_completion_tokens=max_tokens,
            )
            llm_trace.append({"scope": "final", "duration_ms": int((time.time() - llm_start) * 1000)})
            answer = response.strip()

            # Generate structured output if schema provided
            structured_output = None
            if response_schema and answer:
                structured_output = await _generate_structured_output(answer, response_schema, llm_config, reflect_id)

            _log_completion(answer, iteration + 1, forced=True)
            return ReflectAgentResult(
                text=answer,
                structured_output=structured_output,
                iterations=iteration + 1,
                tools_called=total_tools_called,
                mental_models_created=mental_models_created,
                tool_trace=tool_trace,
                llm_trace=_get_llm_trace(),
                directives_applied=directives_applied,
            )

        # Check for done tool call (handle both 'done' and 'functions.done')
        done_call = next((tc for tc in result.tool_calls if tc.name == "done" or tc.name == "functions.done"), None)
        if done_call:
            # Guardrail: Require evidence before done
            has_gathered_evidence = bool(available_memory_ids) or bool(available_model_ids)
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
                                "error": "You must call recall() or list_mental_models() to gather evidence before providing your final answer."
                            }
                        ),
                    }
                )
                continue

            # Process done tool
            return await _process_done_tool(
                done_call,
                available_memory_ids,
                available_model_ids,
                iteration + 1,
                total_tools_called,
                mental_models_created,
                tool_trace,
                _get_llm_trace(),
                _log_completion,
                reflect_id,
                directives_applied=directives_applied,
                llm_config=llm_config,
                response_schema=response_schema,
            )

        # Execute other tools in parallel (exclude done and functions.done)
        other_tools = [tc for tc in result.tool_calls if tc.name not in ("done", "functions.done")]
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
                _execute_tool_with_timing(tc, lookup_fn, recall_fn, expand_fn, learn_fn) for tc in other_tools
            ]
            tool_results = await asyncio.gather(*tool_tasks, return_exceptions=True)
            total_tools_called += len(other_tools)

            # Process results and add to messages
            for tc, result_data in zip(other_tools, tool_results):
                if isinstance(result_data, Exception):
                    # Tool execution failed - log and raise to fail the request
                    logger.error(f"[REFLECT {reflect_id}] Tool {tc.name} failed with exception: {result_data}")
                    raise RuntimeError(f"Reflect tool '{tc.name}' failed: {result_data}")

                output, duration_ms = result_data

                # Check if tool returned an error response
                if isinstance(output, dict) and "error" in output:
                    logger.error(f"[REFLECT {reflect_id}] Tool {tc.name} returned error: {output['error']}")
                    raise RuntimeError(f"Reflect tool '{tc.name}' error: {output['error']}")

                # Track created mental models
                if tc.name == "learn" and isinstance(output, dict) and "model_id" in output:
                    mental_models_created.append(output["model_id"])

                # Track available memory IDs from recall
                if tc.name == "recall" and isinstance(output, dict) and "memories" in output:
                    for memory in output["memories"]:
                        if "id" in memory:
                            available_memory_ids.add(memory["id"])

                # Track available model IDs
                if tc.name in ("list_mental_models", "get_mental_model") and isinstance(output, dict):
                    if output.get("found") and "model" in output:
                        model_id = output["model"].get("id")
                        if model_id:
                            available_model_ids.add(model_id)
                    elif "models" in output:
                        for model in output["models"]:
                            if "id" in model:
                                available_model_ids.add(model["id"])

                # Add tool result message
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(output, default=str),
                    }
                )

                # Track for logging and context history
                input_dict = {"tool": tc.name, **tc.arguments}
                input_summary = _summarize_input(tc.name, tc.arguments)

                tool_trace.append(
                    ToolCall(
                        tool=tc.name, input=input_dict, output=output, duration_ms=duration_ms, iteration=iteration + 1
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
        mental_models_created=mental_models_created,
        tool_trace=tool_trace,
        llm_trace=_get_llm_trace(),
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
    available_model_ids: set[str],
    iterations: int,
    total_tools_called: int,
    mental_models_created: list[str],
    tool_trace: list[ToolCall],
    llm_trace: list[LLMCall],
    log_completion: Callable,
    reflect_id: str,
    directives_applied: list[DirectiveInfo],
    llm_config: "LLMProvider | None" = None,
    response_schema: dict | None = None,
) -> ReflectAgentResult:
    """Process the done tool call and return the result."""
    args = done_call.arguments

    answer = args.get("answer", "").strip()
    if not answer:
        answer = "No answer provided."

    # Validate IDs
    used_memory_ids = [mid for mid in args.get("memory_ids", []) if mid in available_memory_ids]
    used_model_ids = [mid for mid in args.get("model_ids", []) if mid in available_model_ids]

    # Generate structured output if schema provided
    structured_output = None
    if response_schema and llm_config and answer:
        structured_output = await _generate_structured_output(answer, response_schema, llm_config, reflect_id)

    log_completion(answer, iterations)
    return ReflectAgentResult(
        text=answer,
        structured_output=structured_output,
        iterations=iterations,
        tools_called=total_tools_called,
        mental_models_created=mental_models_created,
        tool_trace=tool_trace,
        llm_trace=llm_trace,
        used_memory_ids=used_memory_ids,
        used_model_ids=used_model_ids,
        directives_applied=directives_applied,
    )


async def _execute_tool_with_timing(
    tc: "LLMToolCall",
    lookup_fn: Callable[[str | None], Awaitable[dict[str, Any]]],
    recall_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    expand_fn: Callable[[list[str], str], Awaitable[dict[str, Any]]],
    learn_fn: Callable[[MentalModelInput], Awaitable[dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], int]:
    """Execute a tool call and return result with timing."""
    start = time.time()
    result = await _execute_tool(tc.name, tc.arguments, lookup_fn, recall_fn, expand_fn, learn_fn)
    duration_ms = int((time.time() - start) * 1000)
    return result, duration_ms


async def _execute_tool(
    tool_name: str,
    args: dict[str, Any],
    lookup_fn: Callable[[str | None], Awaitable[dict[str, Any]]],
    recall_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    expand_fn: Callable[[list[str], str], Awaitable[dict[str, Any]]],
    learn_fn: Callable[[MentalModelInput], Awaitable[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Execute a single tool by name."""
    # Normalize tool name - some LLMs return 'functions.done' instead of 'done'
    if tool_name.startswith("functions."):
        tool_name = tool_name[len("functions.") :]

    if tool_name == "list_mental_models":
        return await lookup_fn(None)

    elif tool_name == "get_mental_model":
        model_id = args.get("model_id")
        if not model_id:
            return {"error": "get_mental_model requires model_id"}
        return await lookup_fn(model_id)

    elif tool_name == "recall":
        query = args.get("query")
        if not query:
            return {"error": "recall requires a query parameter"}
        max_tokens = max(args.get("max_tokens") or 2048, 1000)  # Default 2048, min 1000
        return await recall_fn(query, max_tokens)

    elif tool_name == "learn":
        if learn_fn is None:
            return {"error": "learn tool is not available"}
        name = args.get("name")
        description = args.get("description")
        if not name or not description:
            return {"error": "learn requires name and description"}
        return await learn_fn(MentalModelInput(name=name, description=description))

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
    if tool_name == "list_mental_models":
        return "()"
    elif tool_name == "get_mental_model":
        return f"(model_id={args.get('model_id', '?')})"
    elif tool_name == "recall":
        query = args.get("query", "")
        query_preview = f"'{query[:30]}...'" if len(query) > 30 else f"'{query}'"
        # Show actual value used (default 2048, min 1000)
        max_tokens = max(args.get("max_tokens") or 2048, 1000)
        return f"(query={query_preview}, max_tokens={max_tokens})"
    elif tool_name == "learn":
        name = args.get("name", "?")
        desc = args.get("description", "")
        desc_preview = f"'{desc[:20]}...'" if len(desc) > 20 else f"'{desc}'"
        return f"(name='{name}', description={desc_preview})"
    elif tool_name == "expand":
        memory_ids = args.get("memory_ids", [])
        depth = args.get("depth", "chunk")
        return f"(memory_ids=[{len(memory_ids)} ids], depth={depth})"
    elif tool_name == "done":
        answer = args.get("answer", "")
        answer_preview = f"'{answer[:30]}...'" if len(answer) > 30 else f"'{answer}'"
        memory_ids = args.get("memory_ids", [])
        model_ids = args.get("model_ids", [])
        return f"(answer={answer_preview}, memory_ids={len(memory_ids)}, model_ids={len(model_ids)})"
    return str(args)
