"""Hindsight-LangGraph: Persistent memory for LangGraph and LangChain agents.

Provides Hindsight-backed tools, nodes, and a memory instructions helper,
giving agents long-term memory across conversations.

The **tools** and **memory_instructions** patterns work with both LangChain
and LangGraph — only ``langchain-core`` is required. The **nodes** pattern
requires ``langgraph`` (install with ``pip install hindsight-langgraph[langgraph]``).

Basic usage with tools (LangChain or LangGraph)::

    from hindsight_langgraph import create_hindsight_tools

    # Uses the default API URL (set HINDSIGHT_API_KEY env var to authenticate)
    tools = create_hindsight_tools(bank_id="user-123")

    # Or point at a different instance:
    # tools = create_hindsight_tools(bank_id="user-123", hindsight_api_url="http://localhost:8888")

    # Bind tools to your model
    model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

Usage with memory nodes (requires langgraph)::

    from hindsight_langgraph import create_recall_node, create_retain_node

    recall = create_recall_node(client=client, bank_id="user-123")
    retain = create_retain_node(client=client, bank_id="user-123")

    builder.add_node("recall", recall)
    builder.add_node("agent", agent_node)
    builder.add_node("retain", retain)
    builder.add_edge("recall", "agent")
    builder.add_edge("agent", "retain")

Usage with memory_instructions (LangChain, no graph needed)::

    from hindsight_langgraph import memory_instructions

    get_instructions = memory_instructions(
        client=client, bank_id="user-123",
        base_instructions="You are a helpful assistant.",
    )
    instructions = await get_instructions()
"""

from .config import (
    HindsightLangGraphConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightError
from .tools import create_hindsight_tools, memory_instructions


def __getattr__(name: str):
    """Lazy-import LangGraph-specific modules so langgraph is optional."""
    if name == "create_recall_node" or name == "create_retain_node":
        try:
            from .nodes import create_recall_node, create_retain_node
        except ImportError:
            raise ImportError(
                f"'{name}' requires langgraph. Install with: pip install hindsight-langgraph[langgraph]"
            ) from None
        return create_recall_node if name == "create_recall_node" else create_retain_node

    raise AttributeError(f"module 'hindsight_langgraph' has no attribute {name!r}")


__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HindsightLangGraphConfig",
    "HindsightError",
    "create_hindsight_tools",
    "memory_instructions",
]

try:
    import langgraph  # noqa: F401

    __all__ += ["create_recall_node", "create_retain_node"]
except ImportError:
    pass
