"""
Memory System for AI Agents.

Temporal + Semantic Memory Architecture using PostgreSQL with pgvector.
"""

# Cap native ML thread pools (OpenBLAS/OpenMP/MKL) before any import pulls in
# numpy/torch/onnxruntime — they read these env vars only at load time. See
# hindsight_api/_thread_limits.py for the rationale.
from ._thread_limits import apply_default_thread_limits

apply_default_thread_limits()

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
__version__ = "0.8.3"
