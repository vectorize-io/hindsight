"""
Tool schema definitions for the reflect agent.

These are OpenAI-format tool definitions used with native tool calling.
"""

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


def _build_done_tool_with_directives(directive_rules: list[str]) -> dict:
    """
    Build the done tool schema with directive compliance field.

    When directives are present, adds a required field that forces the agent
    to confirm compliance with each directive before submitting.

    Args:
        directive_rules: List of directive rule strings
    """
    from typing import Any, cast

    # Build rules list for description
    rules_list = "\n".join(f"  {i + 1}. {rule}" for i, rule in enumerate(directive_rules))

    # Build the tool with directive compliance field
    return {
        "type": "function",
        "function": {
            "name": "done",
            "description": (
                "Signal completion with your final answer. IMPORTANT: You must confirm directive compliance before submitting. "
                "Your answer will be REJECTED if it violates any directive."
            ),
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
                    "directive_compliance": {
                        "type": "string",
                        "description": f"REQUIRED: Confirm your answer complies with ALL directives. List each directive and how your answer follows it:\n{rules_list}\n\nFormat: 'Directive 1: [how answer complies]. Directive 2: [how answer complies]...'",
                    },
                },
                "required": ["answer", "directive_compliance"],
            },
        },
    }


def get_reflect_tools(enable_learn: bool = True, directive_rules: list[str] | None = None) -> list[dict]:
    """
    Get the list of tools for the reflect agent.

    Args:
        enable_learn: Whether to include the learn tool
        directive_rules: Optional list of directive rule strings. If provided,
                        the done() tool will require directive compliance confirmation.

    Returns:
        List of tool definitions in OpenAI format
    """
    tools = []

    # Include mental model tools for lookup
    tools.append(TOOL_LIST_MENTAL_MODELS)
    tools.append(TOOL_GET_MENTAL_MODEL)
    tools.append(TOOL_RECALL)

    if enable_learn:
        tools.append(TOOL_LEARN)

    tools.append(TOOL_EXPAND)

    # Use directive-aware done tool if directives are present
    if directive_rules:
        tools.append(_build_done_tool_with_directives(directive_rules))
    else:
        tools.append(TOOL_DONE_ANSWER)

    return tools
