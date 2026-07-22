"""Shared causal-link taxonomy.

Retain writes only the canonical relationship.  Transfer import/export also
preserves historical relationship types so existing banks keep their graph
semantics without allowing new retain output to create those types.
"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

CausalLinkType = Literal["caused_by", "causes", "enables", "prevents"]

CANONICAL_CAUSAL_LINK_TYPE: CausalLinkType = "caused_by"
LEGACY_CAUSAL_LINK_TYPE_NAMES: tuple[CausalLinkType, ...] = ("causes", "enables", "prevents")

CANONICAL_CAUSAL_LINK_TYPES = frozenset({CANONICAL_CAUSAL_LINK_TYPE})
LEGACY_CAUSAL_LINK_TYPES = frozenset(LEGACY_CAUSAL_LINK_TYPE_NAMES)
CAUSAL_LINK_TYPES = (CANONICAL_CAUSAL_LINK_TYPE, *LEGACY_CAUSAL_LINK_TYPE_NAMES)


class CausalLinkDescriptor(BaseModel):
    """Durable causal provenance carried by both endpoint memory units."""

    model_config = ConfigDict(frozen=True)

    from_unit_id: UUID
    to_unit_id: UUID
    link_type: CausalLinkType
    weight: float = Field(ge=0.0, le=1.0)

    @property
    def identity(self) -> str:
        """Stable key used to deduplicate materialized and persisted copies."""
        return f"{self.from_unit_id}:{self.to_unit_id}:{self.link_type}"
