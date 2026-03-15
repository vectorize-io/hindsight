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
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why you're making this search (for debugging)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant mental models",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of mental models to return (default 5)",
                },
            },
            "required": ["reason", "query"],
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
            "If an observation is STALE, you should ALSO use recall() to verify with current facts. "
            "IMPORTANT: If search_mental_models is available, you MUST call it FIRST before using this tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why you're making this search (for debugging)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant observations",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens for results (default 5000). Use higher values for broader searches.",
                },
            },
            "required": ["reason", "query"],
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
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why you're making this search (for debugging)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Optional limit on result size (default 2048). Use higher values for broader searches.",
                },
                "max_chunk_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens for raw source chunk text included alongside each memory fact (default 1000, min 1000). Chunks provide the surrounding context the fact was extracted from. Increase for broader context.",
                },
            },
            "required": ["reason", "query"],
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
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why you need more context (for debugging)",
                },
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
            "required": ["reason", "memory_ids", "depth"],
        },
    },
}

TOOL_DECOMPOSE = {
    "type": "function",
    "function": {
        "name": "decompose",
        "description": (
            "Break a complex query into 2-4 focused sub-questions that can each be answered "
            "independently. Use this when the query spans multiple topics, requires comparing "
            "different domains, or needs step-by-step reasoning.\n\n"
            "After decomposing, use recall/search tools to gather evidence for EACH sub-question "
            "before calling done with your synthesized answer.\n\n"
            "ONLY available for MID and HIGH budget queries. Do NOT use for simple factual lookups."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sub_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-4 sub-questions to investigate. Each should be self-contained and answerable via recall/search tools.",
                },
                "rationale": {
                    "type": "string",
                    "description": "Brief explanation of why this decomposition makes sense and what each sub-question contributes to answering the original query.",
                },
            },
            "required": ["sub_questions", "rationale"],
        },
    },
}

_REASONING_STEPS_PROPERTY = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "sub_question": {"type": "string"},
            "conclusion": {"type": "string"},
            "source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Memory IDs that support this step's conclusion",
            },
        },
        "required": ["sub_question", "conclusion"],
    },
    "description": (
        "Optional reasoning chain. Include when you used decompose() to break the query into "
        "sub-questions. Each step should summarize what you found for that sub-question."
    ),
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
                    "description": "Your response as well-formatted markdown. Use headers, lists, bold/italic, and code blocks for clarity. NEVER include memory IDs, UUIDs, or 'Memory references' in this text - put IDs only in memory_ids array. LANGUAGE: By default, write in the SAME language as the user's question. However, if a language directive in the system prompt specifies a different language, follow that directive instead.",
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
                "reasoning_steps": _REASONING_STEPS_PROPERTY,
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
                        "description": (
                            "Your response as well-formatted markdown. Use headers, lists, bold/italic, and code blocks for clarity. "
                            "NEVER include memory IDs, UUIDs, or 'Memory references' in this text - put IDs only in memory_ids array. "
                            f"MANDATORY: Your answer MUST comply with ALL directives:\n{rules_list}"
                        ),
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
                    "reasoning_steps": _REASONING_STEPS_PROPERTY,
                    "directive_compliance": {
                        "type": "string",
                        "description": f"REQUIRED: Confirm your answer complies with ALL directives. List each directive and how your answer follows it:\n{rules_list}\n\nFormat: 'Directive 1: [how answer complies]. Directive 2: [how answer complies]...'",
                    },
                },
                "required": ["answer", "directive_compliance"],
            },
        },
    }


def get_reflect_tools(
    directive_rules: list[str] | None = None,
    budget: str | None = None,
) -> list[dict]:
    """
    Get the list of tools for the reflect agent.

    The tools support a hierarchical retrieval strategy:
    1. search_mental_models - User-curated stored reflect responses (try first)
    2. search_observations - Consolidated knowledge with freshness
    3. recall - Raw facts as ground truth

    Args:
        directive_rules: Optional list of directive rule strings. If provided,
                        the done() tool will require directive compliance confirmation.
        budget: Optional budget level string ("low", "mid", "high"). When "mid" or "high",
                the decompose() tool is included to enable multi-step reasoning.

    Returns:
        List of tool definitions in OpenAI format
    """
    tools: list[dict] = [
        TOOL_SEARCH_MENTAL_MODELS,
        TOOL_SEARCH_OBSERVATIONS,
        TOOL_RECALL,
        TOOL_EXPAND,
    ]

    # Include decompose tool for MID and HIGH budgets only
    if budget is not None and budget.lower() in ("mid", "high"):
        tools.append(TOOL_DECOMPOSE)

    # Use directive-aware done tool if directives are present
    if directive_rules:
        tools.append(_build_done_tool_with_directives(directive_rules))
    else:
        tools.append(TOOL_DONE_ANSWER)

    return tools
