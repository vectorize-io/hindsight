"""Shared causal-link taxonomy.

Retain writes only the canonical relationship.  Transfer import/export also
preserves historical relationship types so existing banks keep their graph
semantics without allowing new retain output to create those types.
"""

from dataclasses import dataclass
from typing import Any

CANONICAL_CAUSAL_LINK_TYPE = "caused_by"
LEGACY_CAUSAL_LINK_TYPE_NAMES = ("causes", "enables", "prevents")

CANONICAL_CAUSAL_LINK_TYPES = frozenset({CANONICAL_CAUSAL_LINK_TYPE})
LEGACY_CAUSAL_LINK_TYPES = frozenset(LEGACY_CAUSAL_LINK_TYPE_NAMES)
CAUSAL_LINK_TYPES = (CANONICAL_CAUSAL_LINK_TYPE, *LEGACY_CAUSAL_LINK_TYPE_NAMES)

DEFAULT_CAUSAL_LINK_WEIGHT = 1.0


@dataclass(frozen=True)
class CausalLinkDescriptor:
    """One causal edge, parked on the curation archive while an endpoint is invalidated.

    Invalidation moves a fact out of ``memory_units``, so the FK cascade deletes
    its ``memory_links`` rows — and nothing could recreate a causal edge, which
    is extraction output rather than derived data. The descriptor is what the
    archive row stores so revert can rematerialize the edge (#2864).
    """

    from_unit_id: str
    to_unit_id: str
    link_type: str
    weight: float = DEFAULT_CAUSAL_LINK_WEIGHT

    def as_json_dict(self) -> dict[str, Any]:
        """Serializable form written to ``invalidated_memory_units.causal_links``.

        The key names double as the column list of the ``jsonb_to_recordset``
        read in ``snapshot_causal_links`` — keep them in sync.
        """
        return {
            "from_unit_id": self.from_unit_id,
            "to_unit_id": self.to_unit_id,
            "link_type": self.link_type,
            "weight": self.weight,
        }

    @classmethod
    def from_json_dict(cls, raw: Any) -> "CausalLinkDescriptor | None":
        """Parse one stored descriptor, or None when it isn't a usable causal edge.

        The archive column is plain JSON with no schema enforcement (a restore
        from an older backup, or a hand-edited row, can put anything there), and
        ``memory_links`` has a ``link_type`` CHECK constraint — so an unusable
        entry is skipped rather than allowed to abort the whole revert.
        """
        if not isinstance(raw, dict):
            return None
        from_unit_id = raw.get("from_unit_id")
        to_unit_id = raw.get("to_unit_id")
        link_type = raw.get("link_type")
        if not from_unit_id or not to_unit_id or link_type not in CAUSAL_LINK_TYPES:
            return None
        return cls(
            from_unit_id=str(from_unit_id),
            to_unit_id=str(to_unit_id),
            link_type=str(link_type),
            weight=float(raw.get("weight") or DEFAULT_CAUSAL_LINK_WEIGHT),
        )
