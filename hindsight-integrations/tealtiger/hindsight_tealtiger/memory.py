"""Hindsight governance memory — importance-weighted storage and contextual recall.

Stores TealTiger governance decisions in Hindsight with importance-based retention:
- Critical DENYs (PII, secrets) → high importance → retained for months
- Notable MONITORs → medium importance → retained for weeks
- Routine ALLOWs → low importance → natural decay within days

Enables contextual recall: "what governance decisions were made for this agent
in similar situations?" — informing (but never overriding) future policy evaluation.

Design principle: Storage = evidence/continuity, NOT authority.
A stored ALLOW from yesterday cannot authorize today's action.
Every new request gets a fresh deterministic evaluation.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional


def default_importance(decision: Dict[str, Any]) -> float:
    """Default importance mapping: decision action drives base importance.

    DENY → 0.90 (retain for compliance audits)
    REQUIRE_APPROVAL → 0.85 (retain for approval tracking)
    MONITOR → 0.70 (retain for pattern detection)
    ALLOW → 0.55 (natural decay — routine)

    Args:
        decision: TealTiger governance decision dict.

    Returns:
        Importance score between 0.0 and 1.0.
    """
    action = decision.get("action", "ALLOW").upper()
    base_map = {
        "DENY": 0.90,
        "REQUIRE_APPROVAL": 0.85,
        "REFER": 0.85,
        "MONITOR": 0.70,
        "ALLOW": 0.55,
    }
    return base_map.get(action, 0.60)


class HindsightGovernanceMemory:
    """Governance decision memory backed by Hindsight.

    Stores governance decisions with importance-weighted retention and enables
    contextual recall of past decisions. Critical security events persist for
    compliance; routine approvals naturally decay.

    Args:
        client: A Hindsight client instance (from hindsight_client).
        bank_id: Memory bank for governance decisions (default: "governance").
        importance_fn: Function mapping a decision dict to an importance score
            (0.0-1.0). Default uses action type (DENY=0.90, ALLOW=0.55).
        tags: Additional tags to attach to all stored memories.
        metadata_fields: Which decision fields to include as Hindsight metadata
            for filtering during recall.
    """

    def __init__(
        self,
        client: Any,
        bank_id: str = "governance",
        importance_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
        tags: Optional[List[str]] = None,
        metadata_fields: Optional[List[str]] = None,
    ):
        self._client = client
        self._bank_id = bank_id
        self._importance_fn = importance_fn or default_importance
        self._tags = tags or ["tealtiger", "governance"]
        self._metadata_fields = metadata_fields or [
            "agent_id", "mode", "action", "tool_name", "risk_score",
        ]
        self._store_count: int = 0
        self._recall_count: int = 0

    def store(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Store a governance decision in Hindsight memory.

        The decision is stored with importance-weighted retention:
        - High importance (DENY) → retained for months
        - Low importance (ALLOW) → natural decay

        Args:
            decision: TealTiger governance decision dict containing at minimum:
                - action: "ALLOW", "DENY", "MONITOR", or "REFER"
                - correlation_id: UUID v4

        Returns:
            Dict with store confirmation (Hindsight response).
        """
        importance = self._importance_fn(decision)
        content = self._format_content(decision)
        metadata = self._extract_metadata(decision)
        tags = self._build_tags(decision)

        result = self._client.retain(
            bank_id=self._bank_id,
            content=content,
            metadata=metadata,
            tags=tags,
            importance=importance,
        )

        self._store_count += 1
        return result

    def recall(
        self,
        agent_id: Optional[str] = None,
        context: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 5,
        tags: Optional[List[str]] = None,
        min_importance: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Recall past governance decisions from Hindsight memory.

        Retrieves contextually relevant past decisions.

        Args:
            agent_id: Filter by agent.
            context: Contextual query (e.g., "tool:web_search").
            query: Free-text semantic query.
            limit: Maximum number of memories to return.
            tags: Additional tags to filter by.
            min_importance: Minimum importance threshold.

        Returns:
            List of recalled governance memory entries.
        """
        recall_query = query or self._build_recall_query(agent_id, context)
        recall_tags = tags or self._tags

        results = self._client.recall(
            bank_id=self._bank_id,
            query=recall_query,
            tags=recall_tags,
            max_results=limit,
        )

        self._recall_count += 1
        return results

    def reflect(
        self,
        agent_id: Optional[str] = None,
        query: Optional[str] = None,
    ) -> str:
        """Reflect on governance history — synthesize patterns and insights.

        Args:
            agent_id: Filter reflection to a specific agent.
            query: Reflection query.

        Returns:
            Synthesized reflection text from Hindsight.
        """
        reflect_query = query or (
            f"Summarize governance patterns for agent {agent_id or 'all agents'}"
        )

        result = self._client.reflect(
            bank_id=self._bank_id,
            query=reflect_query,
        )

        return result

    @property
    def store_count(self) -> int:
        """Number of decisions stored."""
        return self._store_count

    @property
    def recall_count(self) -> int:
        """Number of recall operations performed."""
        return self._recall_count

    def _format_content(self, decision: Dict[str, Any]) -> str:
        """Format a governance decision as content for Hindsight."""
        action = decision.get("action", "ALLOW")
        tool = decision.get("tool_name", decision.get("tool_slug", "unknown"))
        agent = decision.get("agent_id", "unknown")
        reasons = decision.get("reason_codes", [])
        risk = decision.get("risk_score", 0)
        mode = decision.get("mode", "OBSERVE")
        correlation_id = decision.get("correlation_id", "")

        parts = [
            f"Governance decision: {action}",
            f"Agent: {agent}",
            f"Tool: {tool}",
            f"Mode: {mode}",
            f"Risk score: {risk}",
        ]

        if reasons:
            parts.append(f"Reasons: {', '.join(reasons)}")
        if correlation_id:
            parts.append(f"Decision ID: {correlation_id}")
        if "cost_tracked" in decision:
            parts.append(f"Cost: ${decision['cost_tracked']:.4f}")
        if "cumulative_cost" in decision:
            parts.append(f"Cumulative cost: ${decision['cumulative_cost']:.4f}")

        return " | ".join(parts)

    def _extract_metadata(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata fields for Hindsight storage."""
        metadata = {}
        for field in self._metadata_fields:
            if field in decision:
                value = decision[field]
                if isinstance(value, (str, int, float, bool)):
                    metadata[field] = value
                elif isinstance(value, list):
                    metadata[field] = json.dumps(value)
        return metadata

    def _build_tags(self, decision: Dict[str, Any]) -> List[str]:
        """Build tags for the stored memory."""
        tags = list(self._tags)
        action = decision.get("action", "ALLOW").lower()
        tags.append(f"action:{action}")

        if "agent_id" in decision:
            tags.append(f"agent:{decision['agent_id']}")

        tool = decision.get("tool_name", decision.get("tool_slug"))
        if tool:
            tags.append(f"tool:{tool}")

        return tags

    def _build_recall_query(
        self, agent_id: Optional[str], context: Optional[str]
    ) -> str:
        """Build a recall query from agent_id and context."""
        parts = []
        if agent_id:
            parts.append(f"governance decisions for agent {agent_id}")
        if context:
            parts.append(context)
        return " ".join(parts) if parts else "recent governance decisions"
