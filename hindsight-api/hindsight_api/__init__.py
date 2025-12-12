"""
Memory System for AI Agents.

Temporal + Semantic Memory Architecture using PostgreSQL with pgvector.
"""
from .engine.memory_engine import MemoryEngine
from .engine.search.trace import (
    SearchTrace,
    QueryInfo,
    EntryPoint,
    NodeVisit,
    WeightComponents,
    LinkInfo,
    PruningDecision,
    SearchSummary,
    SearchPhaseMetrics,
)
from .engine.search.tracer import SearchTracer
from .engine.embeddings import Embeddings, LocalSTEmbeddings, RemoteTEIEmbeddings
from .engine.cross_encoder import CrossEncoderModel, LocalSTCrossEncoder, RemoteTEICrossEncoder
from .engine.llm_wrapper import LLMConfig
from .config import HindsightConfig, get_config

__all__ = [
    "MemoryEngine",
    "HindsightConfig",
    "get_config",
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
    "Embeddings",
    "LocalSTEmbeddings",
    "RemoteTEIEmbeddings",
    "CrossEncoderModel",
    "LocalSTCrossEncoder",
    "RemoteTEICrossEncoder",
    "LLMConfig",
]
__version__ = "0.1.0"
