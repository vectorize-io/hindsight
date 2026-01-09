"""
Research endpoint for querying mental models.

The research endpoint uses an agentic loop where the LLM can:
1. Get specific mental models by name
2. Recall facts using semantic search
3. Return a final answer when ready
"""

import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from pydantic import BaseModel, Field

from .models import ResearchResult

if TYPE_CHECKING:
    from ..llm_wrapper import LLMConfig

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10


class ResearchAction(BaseModel):
    """Action the LLM wants to take during research."""

    action: str = Field(description="One of: get_mental_model, recall, answer")
    name: str | None = Field(default=None, description="Mental model name (for get_mental_model)")
    query: str | None = Field(default=None, description="Search query (for recall)")
    answer: str | None = Field(default=None, description="Final answer (for answer action)")
    reasoning: str | None = Field(default=None, description="Brief reasoning for this action")


class ResearchActionResponse(BaseModel):
    """Response wrapper for research action."""

    action: ResearchAction


def build_research_system_message() -> str:
    """System message for research agent."""
    return """You are a research agent answering questions using mental models and facts.

You have these tools:
- get_mental_model(name): Get details of a specific mental model
- recall(query): Search for relevant facts
- answer(answer): Return your final answer

Process:
1. Look at available mental models
2. Get relevant ones using get_mental_model
3. If needed, use recall to find additional facts
4. When ready, use answer to respond

Be efficient - only fetch what you need. Usually 1-3 mental models is enough."""


def build_research_prompt(
    query: str,
    available_models: list[dict[str, Any]],
    context: list[dict[str, Any]],
) -> str:
    """Build the prompt for a research iteration."""
    # Format available models
    models_text = "AVAILABLE MENTAL MODELS:\n"
    if available_models:
        for mm in available_models:
            models_text += f"- {mm['name']}: {mm['description']}\n"
    else:
        models_text += "(none)\n"

    # Format context from previous actions
    context_text = ""
    if context:
        context_text = "\nCONTEXT FROM PREVIOUS ACTIONS:\n"
        for item in context:
            if item["type"] == "mental_model":
                context_text += f"\n## Mental Model: {item['name']}\n"
                context_text += f"Description: {item['description']}\n"
                if item.get("summary"):
                    context_text += f"Summary: {item['summary']}\n"
                else:
                    context_text += "(No summary available)\n"
            elif item["type"] == "recall":
                context_text += f"\n## Recall results for: {item['query']}\n"
                for fact in item["facts"][:10]:
                    context_text += f"- {fact}\n"

    return f"""QUESTION: {query}

{models_text}
{context_text}

What's your next action? Choose one:
- get_mental_model(name) - to get details of a mental model
- recall(query) - to search for facts
- answer(answer) - to provide your final answer

Respond with your action."""


async def research(
    llm_config: "LLMConfig",
    query: str,
    available_models: list[dict[str, Any]],
    get_mental_model_fn: Callable[[str], Awaitable[dict[str, Any] | None]],
    recall_fn: Callable[[str], Awaitable[list[str]]],
) -> ResearchResult:
    """
    Execute a research query using an agentic loop.

    The LLM can call tools to gather information before answering.

    Args:
        llm_config: LLM configuration
        query: The research query
        available_models: List of available mental models (name, description only)
        get_mental_model_fn: Function to get full mental model by name
        recall_fn: Function to recall facts by query

    Returns:
        ResearchResult with answer and sources
    """
    context: list[dict[str, Any]] = []
    mental_models_used: list[str] = []
    facts_used: list[str] = []

    for iteration in range(MAX_ITERATIONS):
        logger.debug(f"[RESEARCH] Iteration {iteration + 1}/{MAX_ITERATIONS}")

        prompt = build_research_prompt(query, available_models, context)

        try:
            result = await llm_config.call(
                messages=[
                    {"role": "system", "content": build_research_system_message()},
                    {"role": "user", "content": prompt},
                ],
                response_format=ResearchActionResponse,
                scope="mental_model_research",
            )

            action = result.action
            logger.debug(f"[RESEARCH] Action: {action.action}, reasoning: {action.reasoning}")

            if action.action == "answer":
                # Final answer
                return ResearchResult(
                    answer=action.answer or "I couldn't find an answer.",
                    mental_models_used=mental_models_used,
                    facts_used=facts_used,
                    question_type=None,
                )

            elif action.action == "get_mental_model" and action.name:
                # Get mental model
                mm = await get_mental_model_fn(action.name)
                if mm:
                    context.append(
                        {
                            "type": "mental_model",
                            "name": mm["name"],
                            "description": mm["description"],
                            "summary": mm.get("summary"),
                        }
                    )
                    mental_models_used.append(mm["id"])
                    logger.debug(f"[RESEARCH] Got mental model: {action.name}")
                else:
                    context.append(
                        {
                            "type": "mental_model",
                            "name": action.name,
                            "description": "(not found)",
                            "summary": None,
                        }
                    )
                    logger.debug(f"[RESEARCH] Mental model not found: {action.name}")

            elif action.action == "recall" and action.query:
                # Recall facts
                facts = await recall_fn(action.query)
                context.append(
                    {
                        "type": "recall",
                        "query": action.query,
                        "facts": facts,
                    }
                )
                facts_used.extend(facts[:10])
                logger.debug(f"[RESEARCH] Recalled {len(facts)} facts for: {action.query}")

            else:
                logger.warning(f"[RESEARCH] Unknown action: {action.action}")

        except Exception as e:
            logger.error(f"[RESEARCH] Error in iteration {iteration + 1}: {e}")
            break

    # Max iterations reached
    return ResearchResult(
        answer="I was unable to complete the research. Please try a more specific question.",
        mental_models_used=mental_models_used,
        facts_used=facts_used,
        question_type=None,
    )
