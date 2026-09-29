"""
Bank ID resolution for AgentCore Runtime agents.

Maps AgentCore invocation identity (tenant, user, agent) to a Hindsight bank ID
that persists across Runtime session churn.

Critical rule: Never use runtimeSessionId as the bank ID.
Runtime sessions expire. Long-term memory must survive session churn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .errors import BankResolutionError


@dataclass
class TurnContext:
    """Identity context extracted from an AgentCore Runtime invocation.

    This should be populated from trusted sources (validated JWT claims,
    AgentCore auth context) — never from arbitrary client-supplied payload.

    Attributes:
        runtime_session_id: AgentCore Runtime session ID. Used only as metadata,
            NOT as the primary bank ID — sessions expire and are reprovisioned.
        user_id: Stable user identity. Primary key for long-term memory.
        agent_name: Name/slug of this agent deployment.
        tenant_id: Optional tenant/organization ID for multi-tenant deployments.
        request_id: Optional per-request ID for tracing/deduplication.
    """

    runtime_session_id: str
    user_id: str
    agent_name: str
    tenant_id: str | None = None
    request_id: str | None = None

    def as_metadata(self) -> dict[str, str]:
        """Return a metadata dict safe to attach to retained memories."""
        meta: dict[str, str] = {
            "channel": "agentcore-runtime",
            "runtime_session_id": self.runtime_session_id,
            "agent_name": self.agent_name,
            "user_id": self.user_id,
        }
        if self.tenant_id:
            meta["tenant_id"] = self.tenant_id
        if self.request_id:
            meta["request_id"] = self.request_id
        return meta

    def as_tags(self) -> list[str]:
        """Return tags to attach to retained memories."""
        tags = [
            f"user:{self.user_id}",
            f"agent:{self.agent_name}",
            f"session:{self.runtime_session_id}",
        ]
        if self.tenant_id:
            tags.insert(0, f"tenant:{self.tenant_id}")
        return tags


class BankResolver(Protocol):
    """Protocol for bank ID resolution.

    Implement this to customize how invocation identity maps to Hindsight bank IDs.
    """

    def __call__(self, context: TurnContext) -> str:
        """Resolve and return a Hindsight bank ID for the given context."""
        ...


def default_bank_resolver(context: TurnContext) -> str:
    """Default bank resolver: one bank per (tenant, user, agent) tuple.

    Output format:
        With tenant:    tenant:{tenant_id}:user:{user_id}:agent:{agent_name}
        Without tenant: user:{user_id}:agent:{agent_name}

    This bank ID is stable across Runtime session churn. The runtimeSessionId
    is stored only as a tag/metadata for tracing — never as the bank ID.

    Args:
        context: Turn identity context.

    Returns:
        A stable Hindsight bank ID string.

    Raises:
        BankResolutionError: If user_id or agent_name is missing. Fails closed
            to prevent cross-user memory leakage.
    """
    if not context.user_id or not context.user_id.strip():
        raise BankResolutionError(
            "Cannot resolve bank ID: user_id is required. "
            "Ensure a trusted user identity is extracted from the AgentCore auth context."
        )
    if not context.agent_name or not context.agent_name.strip():
        raise BankResolutionError("Cannot resolve bank ID: agent_name is required.")

    parts: list[str] = []
    if context.tenant_id:
        parts.append(f"tenant:{context.tenant_id}")
    parts.append(f"user:{context.user_id}")
    parts.append(f"agent:{context.agent_name}")

    return ":".join(parts)
