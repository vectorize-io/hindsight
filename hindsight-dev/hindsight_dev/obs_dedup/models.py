"""Typed data structures shared across the observation-dedup pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    """A single observation memory unit read from the Hindsight API."""

    id: str
    text: str
    entities: str = ""
    tags: tuple[str, ...] = ()
    mentioned_at: str | None = None


@dataclass(frozen=True)
class DuplicateCluster:
    """A group of observations that are near-duplicates of one another.

    Membership is transitive: every observation is linked to at least one
    other member with cosine similarity >= the run threshold, but not every
    pair within the cluster necessarily clears it (that is what
    ``min_similarity`` exposes).
    """

    observations: tuple[Observation, ...]
    max_similarity: float
    min_similarity: float

    @property
    def size(self) -> int:
        return len(self.observations)

    @property
    def redundant_count(self) -> int:
        """Observations that could be removed if the cluster collapsed to one."""
        return max(0, self.size - 1)
