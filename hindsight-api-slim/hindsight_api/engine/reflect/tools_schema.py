"""
Tool schema definitions for the reflect agent.

These are OpenAI-format tool definitions used with native tool calling.
The reflect agent uses a hierarchical retrieval strategy:
1. search_mental_models - User-curated stored reflect responses (highest quality, if applicable)
2. search_observations - Consolidated knowledge with freshness awareness
3. recall - Raw facts (world/experience) as ground truth fallback
"""

import copy

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

# The done tool's own source-language rule, read at the moment the model writes the
# answer. Mutually exclusive with a configured output language, like the rule in every
# prompt (see prompt_utils.default_language_section): with a language set it is removed by
# _done_tool_for_output_language, and the directive takes its place on the tool. Left in,
# it out-ranked the directive from the very last position in the model's context — and it
# pointed at "a language directive in the system prompt", where the directive no longer
# lives (#3776).
_DONE_ANSWER_DEFAULT_LANGUAGE = (
    " LANGUAGE: By default, write in the SAME language as the user's question. However, if a "
    "language directive in the system prompt specifies a different language, follow that directive instead."
)

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
                    "description": (
                        "Your response as well-formatted markdown. Use headers, lists, bold/italic, and code blocks "
                        "for clarity. NEVER include memory IDs, UUIDs, or 'Memory references' in this text - put IDs "
                        f"only in memory_ids array.{_DONE_ANSWER_DEFAULT_LANGUAGE}"
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
            },
            "required": ["answer"],
        },
    },
}


# Document mode: the answer IS the stored document, so the model emits its
# structure and the markdown is rendered from it. Nothing the model writes is
# ever parsed back into structure, which is the whole point — a document that is
# generated as markdown and read back is a document that can be misread, and was
# (#3361). See ``structured_doc`` for the schema this mirrors.
#
# Deliberately flat: an array of sections, each an array of block strings. No
# union types, no ``oneOf`` — a tool schema goes to the provider verbatim and not
# every provider accepts a discriminated union (Gemini rejects ``oneOf``), and a
# shape the model can fill without thinking is a shape it fills correctly.
_DONE_DOCUMENT_PROPERTY = {
    "type": "object",
    "description": (
        "Your response as a structured document. The markdown the user reads is rendered "
        "from this — do NOT write a markdown document yourself."
    ),
    "properties": {
        "sections": {
            "type": "array",
            "description": "The document's sections, in order.",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {
                        "type": "string",
                        "description": (
                            "Heading text WITHOUT any leading '#'. Empty string for an "
                            "opening section that should render with no heading."
                        ),
                    },
                    "level": {
                        "type": "integer",
                        "description": "Heading level, 1-6. Use 2 unless the document needs deeper nesting.",
                    },
                    "blocks": {
                        "type": "array",
                        "description": (
                            "The section's content, one entry per block: a single paragraph, a single "
                            "list, a single table, or a single fenced code block. Never put two "
                            "paragraphs in one entry, and never put a heading in one — headings are the "
                            "'heading' field. Within an entry, write the content as you would in a "
                            "document: '- one\n- two' for a list, "
                            "'| a | b |\n| --- | --- |\n| 1 | 2 |' for a table."
                        ),
                        "items": {"type": "string"},
                    },
                },
                "required": ["heading", "level", "blocks"],
            },
        }
    },
    "required": ["sections"],
}


def _done_tool_for_document(base: dict) -> dict:
    """Swap a done tool's free-text ``answer`` for a structured ``document``."""
    tool = copy.deepcopy(base)
    params = tool["function"]["parameters"]
    params["properties"].pop("answer", None)
    params["properties"]["document"] = _DONE_DOCUMENT_PROPERTY
    params["required"] = ["document"] + [name for name in params.get("required", []) if name != "answer"]
    tool["function"]["description"] = (
        "Signal completion with your final answer, as a structured document. Use this when you have "
        "gathered enough information to answer the question."
    )
    return tool


TOOL_DONE_DOCUMENT = _done_tool_for_document(TOOL_DONE_ANSWER)


def _done_tool_for_output_language(base: dict, language: str) -> dict:
    """Replace a done tool's default source-language rule with the configured language.

    The directive goes on the tool's own description rather than the ``answer`` field so it
    reaches every variant the same way — the document variant has no ``answer`` field and
    the directive-aware variant's ``answer`` carries the compliance rules instead.
    """
    tool = copy.deepcopy(base)
    answer = tool["function"]["parameters"]["properties"].get("answer")
    if answer is not None:
        answer["description"] = answer["description"].replace(_DONE_ANSWER_DEFAULT_LANGUAGE, "")
    tool["function"]["description"] += (
        f" LANGUAGE: Write the answer exclusively in {language}, whatever language the question "
        "and the retrieved memories are in."
    )
    return tool


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
    answer_as_document: bool = False,
    llm_output_language: str | None = None,
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
        include_mental_models: Whether to include the search_mental_models tool.
        include_observations: Whether to include the search_observations tool.
        include_recall: Whether to include the recall tool.
        include_expand: Whether to include the expand tool. Disabled when raw
            document/chunk text is not stored, since expand only reads back
            source text and would return empty results.
        answer_as_document: Have ``done`` take a structured ``document`` instead
            of a markdown ``answer``. Used when the answer is stored as a
            document (mental-model refresh) so the model never writes the
            markdown that gets persisted.
        llm_output_language: Configured output language. Swaps ``done``'s default
            answer-in-the-question's-language rule for a directive to write in it.

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

    # Use directive-aware done tool if directives are present
    if directive_rules:
        done_tool = _build_done_tool_with_directives(directive_rules)
    else:
        done_tool = TOOL_DONE_ANSWER
    if answer_as_document:
        done_tool = _done_tool_for_document(done_tool)
    # After the document swap: that replaces the tool description this appends to.
    if llm_output_language:
        done_tool = _done_tool_for_output_language(done_tool, llm_output_language)
    tools.append(done_tool)

    return tools
