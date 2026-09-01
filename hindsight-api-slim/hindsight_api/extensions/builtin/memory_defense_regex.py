"""Memory Defense (regex) — the default extension shipping with hindsight-api-slim.

Scrubs known secret/PII patterns from retained content via the
``sensitive_data`` detector. Matching is pure regex (see ``apply_redaction``):
no LLM call, no external dependency. A ``sensitive_data`` rule may either
``redact`` matches in place or ``block`` the item entirely.
"""

from __future__ import annotations

import logging

from hindsight_api.extensions.memory_defense import (
    REGEX_DETECTORS,
    SENSITIVE_DATA_DETECTOR,
    DefenseAction,
    DefenseDecision,
    DefensePolicy,
    MemoryDefenseExtension,
    apply_redaction,
    warn_unimplemented_detectors,
)

logger = logging.getLogger(__name__)


class MemoryDefenseRegexExtension(MemoryDefenseExtension):
    """Default Memory Defense — regex-based secret/PII redaction."""

    implemented_detectors = REGEX_DETECTORS

    async def screen(
        self,
        *,
        policy: DefensePolicy,
        bank_id: str,
        document_id: str | None,
        content: str,
        tags: list[str],
    ) -> DefenseDecision:
        # Say once, in the log, that a configured rule names a detector this
        # build cannot run. No decision depends on this call.
        warn_unimplemented_detectors(policy, self.implemented_detectors)

        if not policy.enabled:
            return DefenseDecision(action=DefenseAction.ALLOW)

        # The regex extension only runs the sensitive_data detector. If the
        # policy doesn't include a rule for it, there's nothing to do.
        rule = next((r for r in policy.rules if r.on == SENSITIVE_DATA_DETECTOR), None)
        if rule is None or rule.action is DefenseAction.ALLOW:
            return DefenseDecision(action=DefenseAction.ALLOW)

        result = apply_redaction(content)
        if not result.matched_types:
            return DefenseDecision(action=DefenseAction.ALLOW)

        return DefenseDecision(
            action=rule.action,
            detector=SENSITIVE_DATA_DETECTOR,
            message=f"Sensitive data pattern matched: {', '.join(result.matched_types)}",
            redacted_content=result.content if rule.action is DefenseAction.REDACT else None,
            matched_types=result.matched_types,
            hits=result.hits,
        )
