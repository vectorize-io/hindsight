"""The memories store: which one is installed, and how the engine reaches it.

Resolved through the ordinary extension loader — ``HINDSIGHT_API_MEMORIES_EXTENSION``
names a ``module:Class``, and ``HINDSIGHT_API_MEMORIES_*`` becomes its config — so
this behaves like every other extension point. Unset (the normal case) means
:class:`~hindsight_api.engine.memories.postgres.PostgresMemories`: rows in
`memory_units`, links in `memory_links` / `unit_entities`, retrieval as SQL.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Awaitable, Callable

from .base import (
    DOC_META_ATTACHMENT_FILENAMES,
    META_ATTACHMENT_IDS,
    META_CHUNK_ID,
    CausalEdgeRecord,
    DeletePredicate,
    FactRecord,
    FullRecallRequest,
    KnowledgePageEntry,
    KnowledgePageMatch,
    KnowledgePageRef,
    MemoriesExtension,
    MemoryPatch,
    MemoryScopeWatermark,
    RecallArms,
    ScanPage,
    StoredMemory,
    WriteBatch,
    build_fact_records,
    build_text_signals,
    document_attachment_filenames,
    document_record_metadata,
    source_key,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .postgres import PostgresMemories

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


_bank_gone_check: Callable[[str], Awaitable[None]] | None = None


def set_bank_gone_check(check: Callable[[str], Awaitable[None]] | None) -> None:
    """Install the engine's uncached "is this bank gone?" check. Set by the engine at startup."""
    global _bank_gone_check
    _bank_gone_check = check


async def raise_if_bank_gone(bank_id: str) -> None:
    """For a store that owns its storage: call this after a call found no storage for `bank_id`.

    Raises the 404 the engine's existence guard gives when the bank no longer exists, and returns
    when it does -- the store then re-raises its own error, because a live bank with no storage is
    a real fault, not a missing bank.

    The guard reads a per-process cache, and a delete invalidates only the process that served it,
    so another process lets a request for a deleted bank through to the store until its entry
    expires. A store whose storage the delete already dropped then fails, on whichever call the
    request makes first. Asking HERE, from the one place that sees every such failure, is what
    covers every caller at once -- routes, tools and background work alike -- and costs nothing on
    a call that succeeds. With no engine installed (a bare store in a test) this is a no-op.
    """
    if _bank_gone_check is not None:
        await _bank_gone_check(bank_id)


_sql_memories: PostgresMemories | None = None


def sql_memories() -> PostgresMemories:
    """The Postgres store, for rows Postgres owns whatever store is configured.

    A caller reaches for this only after it has already established that the rows in question are
    SQL-backed — a bank the configured store says it does not own, or one it could not answer for
    (``bank_indexes_are_store_owned``) — or when it walks the Postgres schema's own tables (the
    admin backup / restore / bank rename). Those operations are Postgres's alone, so they live only
    on :class:`PostgresMemories`, not on the interface: asking :func:`get_memories` for them would
    reach a store that has no such rows (for a non-Postgres store, no such method).

    When the configured store IS the Postgres store (every Postgres-only deployment) this returns
    that same instance, so that path is unchanged; otherwise one is built once, lazily.

    Not the configured store, and never a substitute for it: everything that follows the BANK's
    owner still goes through :func:`get_memories`.
    """
    from .postgres import PostgresMemories

    configured = get_memories()
    if isinstance(configured, PostgresMemories):
        return configured
    global _sql_memories
    if _sql_memories is None:
        _sql_memories = PostgresMemories({})
    return _sql_memories


def set_memories(memories: MemoriesExtension | None) -> None:
    """Override the store (tests, and engine startup after initialize())."""
    global _memories
    _memories = memories
    # The graph arm's retriever is chosen from the store and then cached, so it
    # has to be re-resolved whenever the store changes.
    from ..search.retrieval import set_default_graph_retriever

    set_default_graph_retriever(None)


__all__ = [
    "DOC_META_ATTACHMENT_FILENAMES",
    "META_ATTACHMENT_IDS",
    "META_CHUNK_ID",
    "CausalEdgeRecord",
    "DeletePredicate",
    "FactRecord",
    "KnowledgePageEntry",
    "KnowledgePageMatch",
    "KnowledgePageRef",
    "MemoriesExtension",
    "MemoryPatch",
    "MemoryScopeWatermark",
    "FullRecallRequest",
    "RecallArms",
    "ScanPage",
    "StoredMemory",
    "WriteBatch",
    "build_fact_records",
    "build_text_signals",
    "create_memories",
    "document_attachment_filenames",
    "document_record_metadata",
    "get_memories",
    "set_memories",
    "source_key",
    "sql_memories",
]
