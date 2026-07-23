"""The memories store: which one is installed, and how the engine reaches it.

Resolved through the ordinary extension loader — ``HINDSIGHT_API_MEMORIES_EXTENSION``
names a ``module:Class``, and ``HINDSIGHT_API_MEMORIES_*`` becomes its config — so
this behaves like every other extension point. Unset (the normal case) means
:class:`~hindsight_api.engine.memories.postgres.PostgresMemories`: rows in
`memory_units`, links in `memory_links` / `unit_entities`, retrieval as SQL.
"""

from __future__ import annotations

import logging

from .base import (
    FACT_TYPE_TO_MEMORY_TYPE,
    MEMORY_TYPE_TO_FACT_TYPE,
    CausalEdgeRecord,
    DeletePredicate,
    FactRecord,
    MemoriesExtension,
    MemoryPatch,
    ScanPage,
    StoredMemory,
    build_fact_records,
    build_text_signals,
    source_key,
)

logger = logging.getLogger(__name__)

_memories: MemoriesExtension | None = None


def create_memories(context=None) -> MemoriesExtension:
    """Build the configured memories store, or the Postgres default."""
    from ...extensions.loader import load_extension

    loaded = load_extension("MEMORIES", MemoriesExtension, context=context)
    if loaded is not None:
        logger.info("[memories] store=%s (memory rows do not go to postgres)", loaded.name)
        return loaded

    from .postgres import PostgresMemories

    return PostgresMemories({})


def get_memories() -> MemoriesExtension:
    """The process-wide memories store, built on first use.

    Retrieval and the retain pipeline reach it through call chains that do not
    carry the engine, so it is resolved here rather than threaded through every
    signature.
    """
    global _memories
    if _memories is None:
        _memories = create_memories()
    return _memories


def set_memories(memories: MemoriesExtension | None) -> None:
    """Override the store (tests, and engine startup after initialize())."""
    global _memories
    _memories = memories
    # The graph arm's retriever is chosen from the store and then cached, so it
    # has to be re-resolved whenever the store changes.
    from ..search.retrieval import set_default_graph_retriever

    set_default_graph_retriever(None)


__all__ = [
    "FACT_TYPE_TO_MEMORY_TYPE",
    "MEMORY_TYPE_TO_FACT_TYPE",
    "CausalEdgeRecord",
    "DeletePredicate",
    "FactRecord",
    "MemoriesExtension",
    "MemoryPatch",
    "ScanPage",
    "StoredMemory",
    "build_fact_records",
    "build_text_signals",
    "create_memories",
    "get_memories",
    "set_memories",
    "source_key",
]
