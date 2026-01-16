"""
Tool schema definitions for the reflect agent.

These are OpenAI-format tool definitions used with native tool calling.
"""

from typing import Literal

# Tool definitions in OpenAI format
TOOL_LIST_MENTAL_MODELS = {
    "type": "function",
    "function": {
        "name": "list_mental_models",
        "description": "List all available mental models - your synthesized knowledge about entities, concepts, and events. Returns an array of models with id, name, and description.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

TOOL_GET_MENTAL_MODEL = {
    "type": "function",
    "function": {
        "name": "get_mental_model",
        "description": "Get full details of a specific mental model including all observations and memory references.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "ID of the mental model (from list_mental_models results)",
                },
            },
            "required": ["model_id"],
        },
    },
}

TOOL_RECALL = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": "Search memories using semantic + temporal retrieval. Returns relevant memories from experience and world knowledge, each with an 'id' you can reference.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Optional limit on result size (default 2048). Use higher values for broader searches.",
                },
            },
            "required": ["query"],
        },
    },
}

TOOL_LEARN = {
    "type": "function",
    "function": {
        "name": "learn",
        "description": "Create a new mental model to track an important recurring topic. Use when you discover a person, project, concept, or pattern that appears frequently and would benefit from synthesized knowledge. The model content will be generated automatically.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Human-readable name (e.g., 'Project Alpha', 'John Smith', 'Product Strategy')",
                },
                "description": {
                    "type": "string",
                    "description": "What to track and synthesize (e.g., 'Track goals, milestones, blockers, and key decisions for Project Alpha')",
                },
            },
            "required": ["name", "description"],
        },
    },
}

TOOL_EXPAND = {
    "type": "function",
    "function": {
        "name": "expand",
        "description": "Get more context for one or more memories. Memory hierarchy: memory -> chunk -> document.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of memory IDs from recall results (batch multiple for efficiency)",
                },
                "depth": {
                    "type": "string",
                    "enum": ["chunk", "document"],
                    "description": "chunk: surrounding text chunk, document: full source document",
                },
            },
            "required": ["memory_ids", "depth"],
        },
    },
}

TOOL_DONE_ANSWER = {
    "type": "function",
    "function": {
        "name": "done",
        "description": "Signal completion with your final answer. Use this when you have gathered enough information to answer the question.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your response as plain text. Do NOT use markdown formatting. NEVER include memory IDs, UUIDs, or 'Memory references' in this text - put IDs only in memory_ids array.",
                },
                "memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of memory IDs that support your answer (put IDs here, NOT in answer text)",
                },
                "model_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of mental model IDs that support your answer",
                },
            },
            "required": ["answer"],
        },
    },
}

TOOL_DONE_OBSERVATIONS = {
    "type": "function",
    "function": {
        "name": "done",
        "description": "Signal completion with MULTIPLE structured observations. Each observation must be a SEPARATE item in the array covering ONE theme. Do NOT combine all content into a single observation.",
        "parameters": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "minItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Short header for this observation's theme (e.g., 'Work Style', 'Technical Skills')",
                            },
                            "text": {
                                "type": "string",
                                "description": "Observation content about ONE theme. End with 'Key evidence:' containing text citations (summaries of what memories say), NOT memory IDs.",
                            },
                            "memory_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Full UUIDs of memories supporting this observation (put IDs here, not in text)",
                            },
                        },
                        "required": ["title", "text", "memory_ids"],
                    },
                    "description": "Array of 3-8 observations, each covering a DIFFERENT aspect/theme. Do NOT put everything in one observation.",
                },
            },
            "required": ["observations"],
        },
    },
}


def get_reflect_tools(
    enable_learn: bool = True, output_mode: Literal["answer", "observations"] = "answer"
) -> list[dict]:
    """
    Get the list of tools for the reflect agent.

    Args:
        enable_learn: Whether to include the learn tool
        output_mode: "answer" or "observations" - determines done tool format
                     In observations mode, mental model tools are excluded to avoid
                     using potentially outdated models during regeneration.

    Returns:
        List of tool definitions in OpenAI format
    """
    tools = []

    # In answer mode, include mental model tools for lookup
    # In observations mode (mental model generation), exclude them to avoid circular references
    if output_mode == "answer":
        tools.append(TOOL_LIST_MENTAL_MODELS)
        tools.append(TOOL_GET_MENTAL_MODEL)

    tools.append(TOOL_RECALL)

    if enable_learn:
        tools.append(TOOL_LEARN)

    tools.append(TOOL_EXPAND)

    # Add appropriate done tool based on output mode
    if output_mode == "observations":
        tools.append(TOOL_DONE_OBSERVATIONS)
    else:
        tools.append(TOOL_DONE_ANSWER)

    return tools
