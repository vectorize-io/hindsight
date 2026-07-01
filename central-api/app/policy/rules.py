"""Governance schema enums and the write-gate.

Status lifecycle and sensitivity levels match the control-plane spec
(`docs/architecture/memory-control-plane.md`, Phase 5).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from app.policy.redactor import redact_secrets


class Sensitivity(StrEnum):
    public = "public"
    internal = "internal"
    private = "private"
    sensitive = "sensitive"
    secret_blocked = "secret_blocked"


class PolicyDecision(StrEnum):
    allow = "allow"
    quarantine = "quarantine"
    reject = "reject"


class Status(StrEnum):
    draft = "draft"
    active = "active"
    verified = "verified"
    superseded = "superseded"
    quarantined = "quarantined"
    deleted = "deleted"


class GovernanceResult(BaseModel):
    content: str  # redacted
    sensitivity: Sensitivity
    decision: PolicyDecision
    redactions: int
    reasons: list[str]


def policy_decision(sensitivity: Sensitivity) -> PolicyDecision:
    """Map a sensitivity level to a write decision.

    - secret_blocked -> reject (never stored, even redacted)
    - sensitive/private -> quarantine (stored, excluded from retrieval pending review)
    - else -> allow
    """
    if sensitivity is Sensitivity.secret_blocked:
        return PolicyDecision.reject
    if sensitivity in (Sensitivity.sensitive, Sensitivity.private):
        return PolicyDecision.quarantine
    return PolicyDecision.allow


def run_write_gate(content: str) -> GovernanceResult:
    """redact -> classify -> decide. The hard rule: secrets are blocked here."""
    # Lazy import avoids a module-load cycle (classifier imports Sensitivity).
    from app.policy.classifier import classify_sensitivity

    redacted, count, kinds = redact_secrets(content)
    sensitivity = classify_sensitivity(content, kinds)
    decision = policy_decision(sensitivity)

    reasons: list[str] = []
    if count:
        reasons.append(f"redacted {count} secret(s): {', '.join(kinds)}")
    if decision is not PolicyDecision.allow:
        reasons.append(f"policy={decision.value} (sensitivity={sensitivity.value})")

    return GovernanceResult(
        content=redacted,
        sensitivity=sensitivity,
        decision=decision,
        redactions=count,
        reasons=reasons,
    )
