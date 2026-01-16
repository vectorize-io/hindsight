"""
Mental models module for Hindsight.

Mental models are synthesized summaries that represent understanding. They come
in different subtypes based on how they were created:

- Structural: Derived from the bank's mission (e.g., "Be a PM for engineering team")
  These are created upfront based on what any agent with this role would need.

- Emergent: Discovered from data patterns (named entities, temporal clusters, etc.)
  These surface organically as facts are retained.

- Pinned: User-defined models that persist across refreshes.
"""

from .models import MentalModel, MentalModelSubtype

__all__ = ["MentalModel", "MentalModelSubtype"]
