"""
Embedding generation utilities for memory units.
"""

import asyncio
import contextvars
import logging
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

EmbeddingInputType = Literal["document", "query"]


class EmbeddingsBackend(Protocol):
    """Minimal duck-typed surface used by retain/recall — the concrete `Embeddings`
    ABC supplies default implementations that delegate to `encode()`."""

    @property
    def dimension(self) -> int: ...

    def encode_query(self, texts: list[str]) -> list[list[float]]: ...

    def encode_documents(self, texts: list[str]) -> list[list[float]]: ...


def _validate_embedding_vector(vector: list[float], *, index: int, expected_dimension: int) -> list[float]:
    actual_dimension = len(vector)
    if actual_dimension == 0:
        raise RuntimeError(f"embedding {index} has dimension 0; expected {expected_dimension}")
    if actual_dimension != expected_dimension:
        raise RuntimeError(f"embedding {index} has dimension {actual_dimension}; expected {expected_dimension}")
    return vector


def generate_embedding(
    embeddings_backend: EmbeddingsBackend, text: str, input_type: EmbeddingInputType = "document"
) -> list[float]:
    """
    Generate embedding for text using the provided embeddings backend.

    Args:
        embeddings_backend: Embeddings instance to use for encoding
        text: Text to embed
        input_type: Whether text is retained document text or recall/search query text.

    Returns:
        Embedding vector (dimension depends on embeddings backend)
    """
    try:
        embeddings = _encode_with_input_type(embeddings_backend, [text], input_type)
    except Exception as e:
        raise Exception(f"Failed to generate embedding: {str(e)}")

    if len(embeddings) != 1:
        raise RuntimeError(
            f"Embeddings backend returned {len(embeddings)} vectors for 1 input text; expected exact 1:1 alignment"
        )
    return _validate_embedding_vector(
        embeddings[0],
        index=0,
        expected_dimension=embeddings_backend.dimension,
    )


def _encode_with_input_type(
    embeddings_backend: EmbeddingsBackend, texts: list[str], input_type: EmbeddingInputType
) -> list[list[float]]:
    if input_type == "query":
        return embeddings_backend.encode_query(texts)
    return embeddings_backend.encode_documents(texts)


async def generate_embeddings_batch(
    embeddings_backend: EmbeddingsBackend, texts: list[str], input_type: EmbeddingInputType = "document"
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts using the provided embeddings backend.

    Runs the embedding generation in a thread pool to avoid blocking the event loop
    for CPU-bound operations.

    Args:
        embeddings_backend: Embeddings instance to use for encoding
        texts: List of texts to embed
        input_type: Whether texts are retained documents or recall/search queries.

    Returns:
        List of embeddings in same order as input texts
    """
    try:
        loop = asyncio.get_event_loop()
        # run_in_executor runs the encode in a worker thread, which does NOT inherit
        # the caller's contextvars. Capture the current context and run the encode
        # inside it so context-dependent behavior (e.g. per-bank `user` attribution
        # read via get_current_bank_id()) survives the thread hop.
        ctx = contextvars.copy_context()
        embeddings = await loop.run_in_executor(
            None, lambda: ctx.run(_encode_with_input_type, embeddings_backend, texts, input_type)
        )
    except Exception as e:
        raise Exception(f"Failed to generate batch embeddings: {str(e)}")

    # Guarantee 1:1 alignment with input texts. A silent length mismatch here
    # propagates downstream as zip() drops items, eventually surfacing as an
    # IndexError in retain mapping (see issue #1037).
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embeddings backend returned {len(embeddings)} vectors for {len(texts)} input texts; "
            "expected exact 1:1 alignment"
        )

    return [
        _validate_embedding_vector(embedding, index=index, expected_dimension=embeddings_backend.dimension)
        for index, embedding in enumerate(embeddings)
    ]
