"""
Reflect agent - agentic loop for reflection with tools.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .models import MentalModelInput, ReflectAction, ReflectActionBatch, ReflectAgentResult
from .prompts import (
    FINAL_SYSTEM_PROMPT,
    build_agent_prompt,
    build_final_prompt,
    build_system_prompt,
)

if TYPE_CHECKING:
    from ..llm_wrapper import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 10


async def run_reflect_agent(
    llm_config: "LLMProvider",
    bank_id: str,
    query: str,
    bank_profile: dict[str, Any],
    lookup_fn: Callable[[str | None], Awaitable[dict[str, Any]]],
    recall_fn: Callable[[str], Awaitable[dict[str, Any]]],
    expand_fn: Callable[[str, str], Awaitable[dict[str, Any]]],
    learn_fn: Callable[[MentalModelInput], Awaitable[dict[str, Any]]] | None = None,
    context: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> ReflectAgentResult:
    """
    Execute the reflect agent loop.

    The agent iteratively calls tools to gather information and learn,
    then provides a final answer.

    Args:
        llm_config: LLM provider for agent calls
        bank_id: Bank identifier
        query: Question to answer
        bank_profile: Bank profile with name, background, goal
        lookup_fn: Tool callback for lookup (model_id) -> result
        recall_fn: Tool callback for recall (query) -> result
        expand_fn: Tool callback for expand (memory_id, depth) -> result
        learn_fn: Optional tool callback for learn (MentalModelInput) -> result.
                  If None, learn tool is disabled.
        context: Optional additional context
        max_iterations: Maximum number of iterations before forcing response

    Returns:
        ReflectAgentResult with final answer and metadata
    """
    enable_learn = learn_fn is not None
    reflect_id = f"{bank_id[:8]}-{int(time.time() * 1000) % 100000}"
    start_time = time.time()

    context_history: list[dict[str, Any]] = []
    mental_models_created: list[str] = []
    total_tools_called = 0
    tool_trace: list[dict[str, Any]] = []  # Track tool calls with timing
    llm_trace: list[dict[str, Any]] = []  # Track LLM calls with timing

    def _log_completion(answer: str, iterations: int, forced: bool = False):
        """Log final summary at INFO level."""
        elapsed_ms = int((time.time() - start_time) * 1000)
        # Format tool calls
        tools_summary = (
            ", ".join(f"{t['tool']}({t['input_summary']})={t['duration_ms']}ms" for t in tool_trace) or "none"
        )
        # Format LLM calls
        llm_summary = ", ".join(f"{c['scope']}={c['duration_ms']}ms" for c in llm_trace) or "none"
        total_llm_ms = sum(c["duration_ms"] for c in llm_trace)
        total_tools_ms = sum(t["duration_ms"] for t in tool_trace)

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
        logger.debug(f"[REFLECT {reflect_id}] Iteration {iteration + 1}/{max_iterations}")

        if is_last:
            # Force text response on last iteration - no tools available
            prompt = build_final_prompt(query, context_history, bank_profile, context)
            llm_start = time.time()
            response = await llm_config.call(
                messages=[
                    {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                scope="reflect_agent_final",
            )
            llm_trace.append({"scope": "final", "duration_ms": int((time.time() - llm_start) * 1000)})
            answer = response.strip()
            _log_completion(answer, iteration + 1, forced=True)
            return ReflectAgentResult(
                text=answer,
                iterations=iteration + 1,
                tools_called=total_tools_called,
                mental_models_created=mental_models_created,
            )

        # Build prompt with accumulated context
        prompt = build_agent_prompt(query, context_history, bank_profile, context)

        # Get action(s) from LLM
        system_prompt = build_system_prompt(enable_learn=enable_learn)
        llm_start = time.time()
        try:
            result = await llm_config.call(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format=ReflectActionBatch,
                scope="reflect_agent",
            )
            llm_trace.append({"scope": f"agent_{iteration + 1}", "duration_ms": int((time.time() - llm_start) * 1000)})
        except Exception as e:
            llm_trace.append({"scope": f"agent_{iteration + 1}_err", "duration_ms": int((time.time() - llm_start) * 1000)})
            logger.warning(f"[REFLECT {reflect_id}] LLM call failed: {e}, forcing final response")
            prompt = build_final_prompt(query, context_history, bank_profile, context)
            llm_start = time.time()
            response = await llm_config.call(
                messages=[
                    {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                scope="reflect_agent_final",
            )
            llm_trace.append({"scope": "final", "duration_ms": int((time.time() - llm_start) * 1000)})
            answer = response.strip()
            _log_completion(answer, iteration + 1, forced=True)
            return ReflectAgentResult(
                text=answer,
                iterations=iteration + 1,
                tools_called=total_tools_called,
                mental_models_created=mental_models_created,
            )

        actions = result.actions

        # Check for done action
        done_action = next((a for a in actions if a.tool == "done"), None)
        if done_action and done_action.answer:
            answer = done_action.answer.strip()
            _log_completion(answer, iteration + 1)
            return ReflectAgentResult(
                text=answer,
                iterations=iteration + 1,
                tools_called=total_tools_called,
                mental_models_created=mental_models_created,
            )

        # Filter out done actions and execute remaining tools
        tool_actions = [a for a in actions if a.tool != "done"]
        if not tool_actions:
            # No tools to call, force final response
            prompt = build_final_prompt(query, context_history, bank_profile, context)
            llm_start = time.time()
            response = await llm_config.call(
                messages=[
                    {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                scope="reflect_agent_final",
            )
            llm_trace.append({"scope": "final", "duration_ms": int((time.time() - llm_start) * 1000)})
            answer = response.strip()
            _log_completion(answer, iteration + 1, forced=True)
            return ReflectAgentResult(
                text=answer,
                iterations=iteration + 1,
                tools_called=total_tools_called,
                mental_models_created=mental_models_created,
            )

        # Execute tools in parallel with timing
        tool_tasks = []
        for action in tool_actions:
            task = _execute_tool_with_timing(action, lookup_fn, recall_fn, expand_fn, learn_fn)
            tool_tasks.append(task)

        tool_results = await asyncio.gather(*tool_tasks, return_exceptions=True)
        total_tools_called += len(tool_actions)

        # Process results and add to context
        for action, result in zip(tool_actions, tool_results):
            if isinstance(result, Exception):
                output = {"error": str(result)}
                duration_ms = 0
                logger.debug(f"[REFLECT {reflect_id}] Tool {action.tool} failed: {result}")
            else:
                output, duration_ms = result
                # Track created mental models
                if action.tool == "learn" and isinstance(output, dict) and "model_id" in output:
                    mental_models_created.append(output["model_id"])

            # Build input summary for logging
            input_dict = _action_to_input_dict(action)
            input_summary = _summarize_input(input_dict)
            tool_trace.append({"tool": action.tool, "input_summary": input_summary, "duration_ms": duration_ms})

            context_history.append({"tool": action.tool, "input": input_dict, "output": output})

    # Should not reach here, but safety fallback
    answer = "I was unable to formulate a complete answer within the iteration limit."
    _log_completion(answer, max_iterations, forced=True)
    return ReflectAgentResult(
        text=answer,
        iterations=max_iterations,
        tools_called=total_tools_called,
        mental_models_created=mental_models_created,
    )


async def _execute_tool_with_timing(
    action: ReflectAction,
    lookup_fn: Callable[[str | None], Awaitable[dict[str, Any]]],
    recall_fn: Callable[[str], Awaitable[dict[str, Any]]],
    expand_fn: Callable[[str, str], Awaitable[dict[str, Any]]],
    learn_fn: Callable[[MentalModelInput], Awaitable[dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], int]:
    """Execute a single tool action and return result with timing in ms."""
    start = time.time()
    result = await _execute_tool(action, lookup_fn, recall_fn, expand_fn, learn_fn)
    duration_ms = int((time.time() - start) * 1000)
    return result, duration_ms


async def _execute_tool(
    action: ReflectAction,
    lookup_fn: Callable[[str | None], Awaitable[dict[str, Any]]],
    recall_fn: Callable[[str], Awaitable[dict[str, Any]]],
    expand_fn: Callable[[str, str], Awaitable[dict[str, Any]]],
    learn_fn: Callable[[MentalModelInput], Awaitable[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Execute a single tool action."""
    if action.tool == "lookup":
        return await lookup_fn(action.model_id)

    elif action.tool == "recall":
        if not action.query:
            return {"error": "recall requires a query parameter"}
        return await recall_fn(action.query)

    elif action.tool == "learn":
        if learn_fn is None:
            return {"error": "learn tool is not available"}
        if not action.mental_model:
            return {"error": "learn requires a mental_model parameter"}
        return await learn_fn(action.mental_model)

    elif action.tool == "expand":
        if not action.memory_id:
            return {"error": "expand requires a memory_id parameter"}
        depth = action.depth or "chunk"
        return await expand_fn(action.memory_id, depth)

    else:
        return {"error": f"Unknown tool: {action.tool}"}


def _action_to_input_dict(action: ReflectAction) -> dict[str, Any]:
    """Convert action to a dict showing the input parameters."""
    result: dict[str, Any] = {"tool": action.tool}
    if action.model_id:
        result["model_id"] = action.model_id
    if action.query:
        result["query"] = action.query
    if action.mental_model:
        result["mental_model"] = {
            "name": action.mental_model.name,
            "type": action.mental_model.type.value,
        }
    if action.memory_id:
        result["memory_id"] = action.memory_id
    if action.depth:
        result["depth"] = action.depth
    if action.reasoning:
        result["reasoning"] = action.reasoning
    return result


def _summarize_input(input_dict: dict[str, Any]) -> str:
    """Create a brief summary of tool input for logging."""
    tool = input_dict.get("tool", "")
    if tool == "lookup":
        model_id = input_dict.get("model_id")
        return model_id if model_id else "all"
    elif tool == "recall":
        query = input_dict.get("query", "")
        return f"'{query[:20]}...'" if len(query) > 20 else f"'{query}'"
    elif tool == "learn":
        mm = input_dict.get("mental_model", {})
        return mm.get("name", "?")[:20]
    elif tool == "expand":
        mem_id = input_dict.get("memory_id", "?")[:8]
        depth = input_dict.get("depth", "chunk")
        return f"{mem_id}/{depth}"
    return ""
