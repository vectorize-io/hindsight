"""
Memory Engine - Core implementation of the memory system.

This package contains all the implementation details of the memory engine:
- MemoryEngine: Main class for memory operations
- Utility modules: embedding_utils, link_utils, think_utils, bank_utils
- Supporting modules: embeddings, cross_encoder, entity_resolver, etc.
"""

from .cross_encoder import CrossEncoderModel, LocalSTCrossEncoder, RemoteTEICrossEncoder
from .db_utils import acquire_with_retry
from .embeddings import Embeddings, LocalSTEmbeddings, RemoteTEIEmbeddings
from .llm_wrapper import LLMConfig
from .memory_engine import (
    MemoryEngine,
    UnqualifiedTableError,
    fq_table,
    get_current_schema,
    validate_sql_schema,
)
from .response_models import MemoryFact, RecallResult, ReflectResult
from .search.trace import (
    EntryPoint,
    LinkInfo,
    NodeVisit,
    PruningDecision,
    QueryInfo,
    SearchPhaseMetrics,
    SearchSummary,
    SearchTrace,
    WeightComponents,
)
from .search.tracer import SearchTracer

__all__ = [
    "MemoryEngine",
    "acquire_with_retry",
    "Embeddings",
    "LocalSTEmbeddings",
    "RemoteTEIEmbeddings",
    "CrossEncoderModel",
    "LocalSTCrossEncoder",
    "RemoteTEICrossEncoder",
    "SearchTrace",
    "SearchTracer",
    "QueryInfo",
    "EntryPoint",
    "NodeVisit",
    "WeightComponents",
    "LinkInfo",
    "PruningDecision",
    "SearchSummary",
    "SearchPhaseMetrics",
    "LLMConfig",
    "RecallResult",
    "ReflectResult",
    "MemoryFact",
    # Schema safety utilities
    "fq_table",
    "get_current_schema",
    "validate_sql_schema",
    "UnqualifiedTableError",
]
