"""
Tool schema definitions for the reflect agent.

These are OpenAI-format tool definitions used with native tool calling.
The reflect agent uses a hierarchical retrieval strategy:
1. search_mental_models - User-curated stored reflect responses (highest quality, if applicable)
2. search_observations - Consolidated knowledge with freshness awareness
3. recall - Raw facts (world/experience) as ground truth fallback
"""

# Tool definitions in OpenAI format

TOOL_SEARCH_MENTAL_MODELS = {
    "type": "function",
    "function": {
        "name": "search_mental_models",
        "description": (
            "Search user-curated mental models (stored reflect responses). These are high-quality, manually created "
            "summaries about specific topics. Use FIRST when the question might be covered by an "
            "existing mental model. Returns mental models with their content and last refresh time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant mental models",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of mental models to return (default 5)",
                },
            },
            "required": ["query"],
        },
    },
}

TOOL_SEARCH_OBSERVATIONS = {
    "type": "function",
    "function": {
        "name": "search_observations",
        "description": (
            "Search consolidated observations (auto-generated knowledge). These are automatically "
            "synthesized from memories. Returns observations with freshness info (updated_at, is_stale). "
            "If an observation is STALE, you should ALSO use recall() to verify with current facts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant observations",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens for results (default 5000). Use higher values for broader searches.",
                },
            },
            "required": ["query"],
        },
    },
}

TOOL_RECALL = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": (
            "Search raw memories (facts and experiences). This is the ground truth data. "
            "Use when: (1) no reflections/mental models exist, (2) mental models are stale, "
            "(3) you need specific details not in synthesized knowledge. "
            "Returns individual memory facts with their timestamps."
        ),
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
                "mental_model_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of mental model IDs that support your answer",
                },
                "observation_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of observation IDs that support your answer",
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
                    "mental_model_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of mental model IDs that support your answer",
                    },
                    "observation_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of observation IDs that support your answer",
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


def get_reflect_tools(directive_rules: list[str] | None = None) -> list[dict]:
    """
    Get the list of tools for the reflect agent.

    The tools support a hierarchical retrieval strategy:
    1. search_mental_models - User-curated stored reflect responses (try first)
    2. search_observations - Consolidated knowledge with freshness
    3. recall - Raw facts as ground truth

    Args:
        directive_rules: Optional list of directive rule strings. If provided,
                        the done() tool will require directive compliance confirmation.

    Returns:
        List of tool definitions in OpenAI format
    """
    tools = [
        TOOL_SEARCH_MENTAL_MODELS,
        TOOL_SEARCH_OBSERVATIONS,
        TOOL_RECALL,
        TOOL_EXPAND,
    ]

    # Use directive-aware done tool if directives are present
    if directive_rules:
        tools.append(_build_done_tool_with_directives(directive_rules))
    else:
        tools.append(TOOL_DONE_ANSWER)

    return tools
