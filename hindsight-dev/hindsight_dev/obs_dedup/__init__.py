"""Observation deduplication tooling.

Reads every observation from a Hindsight bank, embeds the text locally, and
surfaces near-duplicate clusters via cosine similarity. The cosine pass is a
cheap first filter; an agentic verification step can be layered on top of the
candidate clusters later (see ``dedup.find_duplicate_clusters``).
"""

from .models import DuplicateCluster, Observation
from .report import DedupReport

__all__ = ["DuplicateCluster", "Observation", "DedupReport"]
