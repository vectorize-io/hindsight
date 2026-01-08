"""
Mental models module for Hindsight.

Mental models are synthesized summaries that represent understanding about
entities, concepts, or events. They come in two types:

- Structural: Derived from the bank's goal (e.g., "Be a PM for engineering team")
  These are created upfront based on what any agent with this role would need.

- Emergent: Discovered from data patterns (named entities, temporal clusters, etc.)
  These surface organically as facts are retained.
"""

from .models import MentalModel, MentalModelSubtype, MentalModelType

__all__ = ["MentalModel", "MentalModelType", "MentalModelSubtype"]
