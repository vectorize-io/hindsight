"""
Models and utilities for evidence-grounded observations with computed trends.

Observations are part of mental models and represent patterns/beliefs derived
from memories. Each observation must be grounded in specific evidence (quotes)
from memories, and trends are computed algorithmically from evidence timestamps.
"""

from datetime import datetime, timedelta, timezone
from enum import Enum

from pydantic import BaseModel, Field, computed_field, field_validator


class Trend(str, Enum):
    """Computed trend for an observation based on evidence timestamps.

    Trends indicate how an observation's evidence is distributed over time:
    - STABLE: Evidence spread across time, continues to present
    - STRENGTHENING: More/denser evidence recently than before
    - WEAKENING: Evidence mostly old, sparse recently
    - NEW: All evidence within recent window
    - STALE: No evidence in recent window (may no longer apply)
    """

    STABLE = "stable"
    STRENGTHENING = "strengthening"
    WEAKENING = "weakening"
    NEW = "new"
    STALE = "stale"


class ObservationEvidence(BaseModel):
    """A single piece of evidence supporting an observation.

    Each evidence item must include an exact quote from the source memory
    to ensure observations are grounded and verifiable.
    """

    memory_id: str = Field(description="ID of the memory unit this evidence comes from")
    quote: str = Field(description="Exact quote from the memory supporting the observation")
    relevance: str = Field(default="", description="Brief explanation of how this quote supports the observation")
    timestamp: datetime = Field(description="When the source memory was created")

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_timezone_aware(cls, v: datetime | str | None) -> datetime:
        """Ensure timestamp is always timezone-aware UTC."""
        if v is None:
            return datetime.now(timezone.utc)
        if isinstance(v, str):
            # Parse ISO format string, handling 'Z' suffix
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v
        raise ValueError(f"Invalid timestamp type: {type(v)}")


class Observation(BaseModel):
    """A single observation within a mental model.

    Observations represent patterns, preferences, beliefs, or other insights
    derived from memories. Each observation must be grounded in evidence
    with exact quotes from source memories.
    """

    title: str = Field(description="Short summary title for the observation (5-10 words)")
    content: str = Field(description="The observation content - detailed explanation of what we believe to be true")
    evidence: list[ObservationEvidence] = Field(default_factory=list, description="Supporting evidence with quotes")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When this observation was first created"
    )

    @field_validator("created_at", mode="before")
    @classmethod
    def ensure_created_at_timezone_aware(cls, v: datetime | str | None) -> datetime:
        """Ensure created_at is always timezone-aware UTC."""
        if v is None:
            return datetime.now(timezone.utc)
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v
        raise ValueError(f"Invalid created_at type: {type(v)}")

    @computed_field
    @property
    def trend(self) -> Trend:
        """Compute trend from evidence timestamps."""
        return compute_trend(self.evidence)

    @computed_field
    @property
    def evidence_span(self) -> dict[str, str | None]:
        """Get the time span covered by evidence."""
        if not self.evidence:
            return {"from": None, "to": None}
        timestamps = [e.timestamp for e in self.evidence]
        return {
            "from": min(timestamps).isoformat(),
            "to": max(timestamps).isoformat(),
        }

    @computed_field
    @property
    def evidence_count(self) -> int:
        """Number of evidence items supporting this observation."""
        return len(self.evidence)


def compute_trend(
    evidence: list[ObservationEvidence],
    now: datetime | None = None,
    recent_days: int = 30,
    old_days: int = 90,
) -> Trend:
    """Compute the trend for an observation based on evidence timestamps.

    The trend indicates how the evidence is distributed over time:
    - STABLE: Evidence spread across time, continues to present
    - STRENGTHENING: More evidence recently than historically
    - WEAKENING: Evidence mostly old, sparse recently
    - NEW: All evidence is recent (within recent_days)
    - STALE: No evidence in recent window

    Args:
        evidence: List of evidence items with timestamps
        now: Reference time for calculations (defaults to current UTC time)
        recent_days: Number of days to consider "recent" (default 30)
        old_days: Number of days to consider "old" (default 90)

    Returns:
        Computed Trend enum value
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Ensure now is timezone-aware
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    if not evidence:
        return Trend.STALE

    recent_cutoff = now - timedelta(days=recent_days)
    old_cutoff = now - timedelta(days=old_days)

    # Normalize timestamps to UTC for comparison
    def normalize_ts(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts

    recent = [e for e in evidence if normalize_ts(e.timestamp) > recent_cutoff]
    old = [e for e in evidence if normalize_ts(e.timestamp) < old_cutoff]
    middle = [e for e in evidence if old_cutoff <= normalize_ts(e.timestamp) <= recent_cutoff]

    # No recent evidence = stale
    if not recent:
        return Trend.STALE

    # All evidence is recent = new
    if not old and not middle:
        return Trend.NEW

    # Compare density (evidence per day)
    recent_density = len(recent) / recent_days if recent_days > 0 else 0
    older_period = old_days - recent_days
    older_density = (len(old) + len(middle)) / older_period if older_period > 0 else 0

    # Avoid division by zero
    if older_density == 0:
        return Trend.NEW

    ratio = recent_density / older_density

    if ratio > 1.5:
        return Trend.STRENGTHENING
    elif ratio < 0.5:
        return Trend.WEAKENING
    else:
        return Trend.STABLE
