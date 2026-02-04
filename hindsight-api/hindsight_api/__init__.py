"""
Memory System for AI Agents.

Temporal + Semantic Memory Architecture using PostgreSQL with pgvector.
"""

from .config import HindsightConfig, get_config
from .engine.cross_encoder import CrossEncoderModel, LocalSTCrossEncoder, RemoteTEICrossEncoder
from .engine.embeddings import Embeddings, LocalSTEmbeddings, RemoteTEIEmbeddings
from .engine.llm_wrapper import LLMConfig
from .engine.memory_engine import MemoryEngine
from .engine.search.trace import (
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
from .engine.search.tracer import SearchTracer
from .models import RequestContext

__all__ = [
    "MemoryEngine",
    "RequestContext",
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
__version__ = "0.4.9"
