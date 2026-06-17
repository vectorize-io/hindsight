"""Cosine-similarity duplicate detection over observation embeddings.

The pipeline is deliberately split so each stage can be tested or swapped
independently:

1. ``embed_observations`` turns observation text into unit-normalised vectors
   using the same local embedding model Hindsight uses by default.
2. ``find_similar_pairs`` does a block-wise cosine scan and returns every pair
   at or above the threshold.
3. ``cluster_pairs`` merges those pairs into transitive clusters (union-find).

``find_duplicate_clusters`` wires the three together. An agentic verifier can
later be inserted between steps 2 and 3 (or applied to the resulting clusters)
to confirm that a candidate pair is a true semantic duplicate.
"""

import asyncio
from dataclasses import dataclass

import numpy as np
from hindsight_api.config import DEFAULT_EMBEDDINGS_LOCAL_MODEL
from hindsight_api.engine.embeddings import LocalSTEmbeddings

from .models import DuplicateCluster, Observation


@dataclass(frozen=True)
class _SimilarPair:
    i: int
    j: int
    similarity: float


def embed_observations(
    observations: list[Observation],
    *,
    model_name: str = DEFAULT_EMBEDDINGS_LOCAL_MODEL,
    force_cpu: bool = False,
) -> np.ndarray:
    """Embed observation text and return L2-normalised float32 vectors.

    Normalising up front turns the cosine similarity of two vectors into a
    plain dot product, which keeps ``find_similar_pairs`` a single matmul.
    """
    if not observations:
        return np.zeros((0, 0), dtype=np.float32)

    embeddings = LocalSTEmbeddings(model_name=model_name, force_cpu=force_cpu)
    asyncio.run(embeddings.initialize())
    # encode_documents matches how Hindsight embeds stored facts (no query prefix).
    vectors = embeddings.encode_documents([obs.text for obs in observations])
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def find_similar_pairs(
    matrix: np.ndarray,
    *,
    threshold: float,
    block_size: int = 512,
) -> list[_SimilarPair]:
    """Return all (i, j) pairs with i < j and cosine similarity >= threshold.

    Computed block-wise so peak memory is ``block_size * n`` floats rather than
    the full ``n * n`` similarity matrix — this keeps banks with tens of
    thousands of observations tractable.
    """
    n = matrix.shape[0]
    pairs: list[_SimilarPair] = []
    if n < 2:
        return pairs

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        sims = matrix[start:end] @ matrix.T  # shape: (end - start, n)
        for local_row in range(end - start):
            i = start + local_row
            # Only look at j > i to avoid self-matches and double-counting.
            tail = sims[local_row, i + 1 :]
            hits = np.nonzero(tail >= threshold)[0]
            for offset in hits:
                j = i + 1 + int(offset)
                pairs.append(_SimilarPair(i=i, j=j, similarity=float(tail[offset])))
    return pairs


def cluster_pairs(
    observations: list[Observation],
    pairs: list[_SimilarPair],
    *,
    min_cluster_size: int = 2,
) -> list[DuplicateCluster]:
    """Merge similar pairs into transitive clusters via union-find."""
    parent = list(range(len(observations)))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression.
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for pair in pairs:
        union(pair.i, pair.j)

    members: dict[int, list[int]] = {}
    for idx in range(len(observations)):
        members.setdefault(find(idx), []).append(idx)

    # Similarity range per cluster, keyed by root.
    sims_by_root: dict[int, list[float]] = {}
    for pair in pairs:
        sims_by_root.setdefault(find(pair.i), []).append(pair.similarity)

    clusters: list[DuplicateCluster] = []
    for root, idxs in members.items():
        if len(idxs) < min_cluster_size:
            continue
        sims = sims_by_root.get(root, [])
        clusters.append(
            DuplicateCluster(
                observations=tuple(observations[i] for i in sorted(idxs)),
                max_similarity=max(sims) if sims else 1.0,
                min_similarity=min(sims) if sims else 1.0,
            )
        )

    # Biggest, most-similar clusters first.
    clusters.sort(key=lambda c: (c.size, c.max_similarity), reverse=True)
    return clusters


def find_duplicate_clusters(
    observations: list[Observation],
    *,
    threshold: float,
    min_cluster_size: int = 2,
    model_name: str = DEFAULT_EMBEDDINGS_LOCAL_MODEL,
    force_cpu: bool = False,
) -> list[DuplicateCluster]:
    """End-to-end: embed observations and return near-duplicate clusters."""
    matrix = embed_observations(observations, model_name=model_name, force_cpu=force_cpu)
    pairs = find_similar_pairs(matrix, threshold=threshold)
    return cluster_pairs(observations, pairs, min_cluster_size=min_cluster_size)
