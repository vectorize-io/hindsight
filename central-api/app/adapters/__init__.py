"""Adapter registry — the single place engines are wired into the control plane."""

from __future__ import annotations

from app.adapters.base import BaseMemoryAdapter
from app.adapters.coderag import CodeRAGAdapter
from app.adapters.hindsight import HindsightAdapter
from app.adapters.internal import InternalAdapter
from app.adapters.memlord import MemlordAdapter
from app.adapters.openmemory import OpenMemoryAdapter

_internal = InternalAdapter()
_openmemory = OpenMemoryAdapter()
_hindsight = HindsightAdapter()
_memlord = MemlordAdapter()
_coderag = CodeRAGAdapter()


def get_write_adapter(memory_type: str | None = None) -> BaseMemoryAdapter:
    """Writes go to the governed native store (Hindsight for persistence)."""
    return _hindsight if _hindsight.configured else _internal


def get_memory_search_adapters() -> list[BaseMemoryAdapter]:
    """Distinct backends joined in unified memory search.

    hindsight and memlord join only when configured so unconfigured/dev
    environments don't attempt network calls on every search.
    """
    adapters: list[BaseMemoryAdapter] = [_internal, _openmemory]
    if _hindsight.configured:
        adapters.append(_hindsight)
    if _memlord.configured:
        adapters.append(_memlord)
    return adapters


def get_primary_adapter() -> HindsightAdapter:
    """Get the primary read-write adapter (Hindsight)."""
    return _hindsight


def get_code_adapter() -> BaseMemoryAdapter:
    return _coderag


def get_docs_adapter() -> BaseMemoryAdapter:
    return _coderag


def get_all_adapters() -> list[BaseMemoryAdapter]:
    return [_internal, _openmemory, _hindsight, _memlord, _coderag]
