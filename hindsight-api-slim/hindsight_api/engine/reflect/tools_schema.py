"""
Tool schema definitions for the reflect agent.

These are OpenAI-format tool definitions used with native tool calling.
The reflect agent uses a hierarchical retrieval strategy:
1. search_mental_models - User-curated stored reflect responses (highest quality, if applicable)
2. search_observations - Consolidated knowledge with freshness awareness
3. recall - Raw facts (world/experience) as ground truth fallback
4. search_world_graph - C3 cross-agent shared graph (only when bank is federated)
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

TOOL_SEARCH_WORLD_GRAPH = {
    "type": "function",
    "function": {
        "name": "search_world_graph",
        "description": (
            "Search the cross-agent shared world graph (Graphiti) for facts shared "
            "across banks in this federation. Use as a fallback evidence source when "
            "private memory (search_mental_models / search_observations / recall) "
            "is silent or stale on a topic that other agents in the same group may "
            "have observed. Each returned fact carries a bi-temporal ledger (valid_at / "
            "invalid_at) — superseded facts are returned with a 'superseded' annotation, "
            "not filtered, so disposition reasoning can weigh the timeline explicitly. "
            "Graphiti outages and timeouts degrade gracefully: the tool returns an error "
            "and the reflect loop continues with private memory only."
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
                    "description": "Search query string. Graphiti runs a mixed retrieval (semantic + BM25 + graph traversal) on the federation's group_id.",
                },
                "max_facts": {
                    "type": "integer",
                    "description": "Maximum number of facts to return (default 10). Capped to the tool's token budget; excess is truncated.",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Token budget for the rendered output block (default 1024, min 100). Used both to cap the fact-count request and to truncate long result sets.",
                },
            },
            "required": ["reason", "query"],
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
                "world_fact_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of cross-agent world-graph fact UUIDs that support your answer. Only present when search_world_graph is available.",
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
                    "world_fact_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of cross-agent world-graph fact UUIDs that support your answer. Only present when search_world_graph is available.",
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


def get_reflect_tools(
    directive_rules: list[str] | None = None,
    include_mental_models: bool = True,
    include_observations: bool = True,
    include_recall: bool = True,
    include_expand: bool = True,
    include_world_graph: bool = False,
) -> list[dict]:
    """
    Get the list of tools for the reflect agent.

    The tools support a hierarchical retrieval strategy:
    1. search_mental_models - User-curated stored reflect responses (try first)
    2. search_observations - Consolidated knowledge with freshness
    3. recall - Raw facts as ground truth
    4. search_world_graph - Cross-agent shared graph (C3, only when federated)

    Args:
        directive_rules: Optional list of directive rule strings. If provided,
                        the done() tool will require directive compliance confirmation.
        include_mental_models: Whether to include the search_mental_models tool.
        include_observations: Whether to include the search_observations tool.
        include_recall: Whether to include the recall tool.
        include_expand: Whether to include the expand tool. Disabled when raw
            document/chunk text is not stored, since expand only reads back
            source text and would return empty results.
        include_world_graph: Whether to include the search_world_graph tool
            (C3). Only true when the bank is federated (graphiti_group_id set
            AND GRAPHITI_BASE_URL configured) — call sites must apply both
            gates, not just one.

    Returns:
        List of tool definitions in OpenAI format
    """
    tools = []

    if include_mental_models:
        tools.append(TOOL_SEARCH_MENTAL_MODELS)
    if include_observations:
        tools.append(TOOL_SEARCH_OBSERVATIONS)
    if include_recall:
        tools.append(TOOL_RECALL)

    if include_expand:
        tools.append(TOOL_EXPAND)

    if include_world_graph:
        tools.append(TOOL_SEARCH_WORLD_GRAPH)

    # Use directive-aware done tool if directives are present
    if directive_rules:
        tools.append(_build_done_tool_with_directives(directive_rules))
    else:
        tools.append(TOOL_DONE_ANSWER)

    return tools
