"""
Search module for memory retrieval.

Provides modular search architecture:
- Retrieval: 4-way parallel (semantic + BM25 + graph + temporal)
- Graph retrieval: Link expansion strategy
- Reranking: Pluggable strategies (heuristic, cross-encoder)
"""

from .graph_retrieval import GraphRetriever
from .reranking import CrossEncoderReranker
from .retrieval import (
    ParallelRetrievalResult,
    get_default_graph_retriever,
    set_default_graph_retriever,
)

__all__ = [
    "get_default_graph_retriever",
    "set_default_graph_retriever",
    "ParallelRetrievalResult",
    "GraphRetriever",
    "CrossEncoderReranker",
]
