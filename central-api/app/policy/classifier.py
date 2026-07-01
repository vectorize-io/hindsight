"""Sensitivity classification — heuristic for the scaffold.

If a secret was detected during redaction the content is `secret_blocked`.
Otherwise content cues map to a sensitivity level. A model-based classifier can
replace this without changing the interface.
"""

from __future__ import annotations

import re

from app.policy.rules import Sensitivity

_SENSITIVE_RE = re.compile(
    r"\b(wallet|seed phrase|private key|ssn|credit card|bank account|passport)\b",
    re.IGNORECASE,
)
_INTERNAL_RE = re.compile(
    r"\b(internal|confidential|do not share|production)\b",
    re.IGNORECASE,
)


def classify_sensitivity(original: str, secret_kinds: list[str]) -> Sensitivity:
    if secret_kinds:
        return Sensitivity.secret_blocked
    if _SENSITIVE_RE.search(original):
        return Sensitivity.sensitive
    if _INTERNAL_RE.search(original):
        return Sensitivity.internal
    return Sensitivity.public
