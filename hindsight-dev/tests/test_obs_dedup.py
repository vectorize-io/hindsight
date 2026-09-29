"""Unit tests for the observation-dedup core (no network, no embedding model)."""

import numpy as np

from hindsight_dev.obs_dedup.dedup import cluster_pairs, find_similar_pairs
from hindsight_dev.obs_dedup.models import Observation
from hindsight_dev.obs_dedup.report import DedupReport


def _unit(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


def _matrix(rows: list[list[float]]) -> np.ndarray:
    return np.asarray([_unit(r) for r in rows], dtype=np.float32)


def _obs(n: int) -> list[Observation]:
    return [Observation(id=f"obs-{i}", text=f"text {i}") for i in range(n)]


def test_find_similar_pairs_thresholds() -> None:
    # rows 0 and 1 are identical; row 2 is orthogonal.
    matrix = _matrix([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    pairs = find_similar_pairs(matrix, threshold=0.9)
    assert [(p.i, p.j) for p in pairs] == [(0, 1)]
    assert pairs[0].similarity == 1.0


def test_find_similar_pairs_respects_block_size() -> None:
    matrix = _matrix([[1.0, 0.0]] * 5)
    # Every pair is identical; block_size smaller than n must still find them all.
    pairs = find_similar_pairs(matrix, threshold=0.99, block_size=2)
    assert len(pairs) == 5 * 4 // 2  # all unordered pairs of 5 items


def test_cluster_pairs_is_transitive() -> None:
    observations = _obs(4)
    matrix = _matrix([[1.0, 0.0], [1.0, 0.01], [1.0, 0.02], [0.0, 1.0]])
    pairs = find_similar_pairs(matrix, threshold=0.9)
    clusters = cluster_pairs(observations, pairs)
    assert len(clusters) == 1
    cluster = clusters[0]
    assert {o.id for o in cluster.observations} == {"obs-0", "obs-1", "obs-2"}
    assert cluster.size == 3
    assert cluster.redundant_count == 2
    assert cluster.min_similarity <= cluster.max_similarity


def test_cluster_pairs_min_size_filters_singletons() -> None:
    observations = _obs(3)
    matrix = _matrix([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    pairs = find_similar_pairs(matrix, threshold=0.9)
    # Default min_cluster_size=2 drops the lone third observation.
    clusters = cluster_pairs(observations, pairs)
    assert len(clusters) == 1
    assert clusters[0].size == 2
    # Raising the floor above the only cluster's size yields nothing.
    assert cluster_pairs(observations, pairs, min_cluster_size=3) == []


def test_report_to_dict_counts() -> None:
    observations = _obs(3)
    matrix = _matrix([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    pairs = find_similar_pairs(matrix, threshold=0.9)
    clusters = cluster_pairs(observations, pairs)
    report = DedupReport(bank_id="b", total_observations=3, threshold=0.9, clusters=clusters)
    payload = report.to_dict()
    assert payload["bank_id"] == "b"
    assert payload["duplicate_clusters"] == 1
    assert payload["redundant_observations"] == 2
    assert len(payload["clusters"][0]["observations"]) == 3
