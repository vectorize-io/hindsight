"""Shared causal-link taxonomy.

Retain writes only the canonical relationship.  Transfer import/export also
preserves historical relationship types so existing banks keep their graph
semantics without allowing new retain output to create those types.
"""

CANONICAL_CAUSAL_LINK_TYPE = "caused_by"
LEGACY_CAUSAL_LINK_TYPE_NAMES = ("causes", "enables", "prevents")

CANONICAL_CAUSAL_LINK_TYPES = frozenset({CANONICAL_CAUSAL_LINK_TYPE})
LEGACY_CAUSAL_LINK_TYPES = frozenset(LEGACY_CAUSAL_LINK_TYPE_NAMES)
CAUSAL_LINK_TYPES = (CANONICAL_CAUSAL_LINK_TYPE, *LEGACY_CAUSAL_LINK_TYPE_NAMES)
