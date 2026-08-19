"""Write-time identifier-retention gate for mental-model refreshes.

WHY
---
A refresh can silently drop anchored identifiers -- dates, file paths, commit
shas, registry ids, env vars -- from a mental model while the document still
GROWS, so no length or emptiness check can see it. Measured over 364 real
refresh events on one production bank:

    lost 0 identifiers: 337 events (92.6%)
    lost 1:              17
    lost 2:               3
    lost 3 or more:       7   <-- the class worth refusing

One of those seven dropped ``TRIAL-CLOSE-RUNBOOK.md``,
``TRIAL-EVIDENCE-PLAN.md``, ``build_spec_a_overlay.py`` and a commit sha from a
single model in one write. Losing one or two is frequently LEGITIMATE churn --
a superseded date, a path that genuinely stopped being relevant -- which is why
this gate is GRADED rather than absolute: refusing on any loss at all would
fail roughly one refresh in fourteen and quickly be switched off.

This is a graded sibling of the existing placeholder/empty-retrieval guard on
the same write path: same trigger condition (only when there is existing real
content to clobber), same preserve-and-fail shape, under its own outcome value
(``refresh_failed_identifier_retention``) so the refusal is never mistaken for
an empty LLM answer. It runs AFTER the #3112 delta-window guard: a failed
delta keeps its own, more precise refusal instead of being relabelled as
identifier loss.

The identifier taxonomy is deliberately IDENTICAL to the offline retention
probe's. Two instruments that disagree about what counts as an identifier
would produce contradictory evidence about the same event.
"""

from __future__ import annotations

import os
import re

#: The audit taxonomy, one compiled alternation. Order matters only for
#: overlap (longer, more specific first). Kept byte-identical to the offline
#: probe's pattern on purpose -- see the module docstring.
_IDENTIFIER_RE = re.compile(
    r"(?:[A-Za-z]:\\[\w\\.\-]+"  # windows paths
    r"|(?:~|/)[\w/.\-]*/[\w.\-]+"  # posix-ish paths
    r"|[\w-]+\.(?:py|md|json|jsonl|yml|yaml|toml|rs|ts|tsx|cmd|ps1|vbs|sh|exe)\b"
    r"|https?://[^\s)\"'\]]+"  # urls
    r"|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"  # uuids
    r"|\b[0-9a-f]{7,40}\b(?<![0-9]{7})"  # hex ids (not pure digits)
    r"|\b(?:G|INV|PROP|HQ|BUG|PKT|REF|WO)-?\d+[\w-]*"  # registry ids
    r"|\bHINDSIGHT_[A-Z0-9_]+|\bHQ_[A-Z0-9_]+"  # env vars
    r"|\b\d{4}-\d{2}-\d{2}\b"  # dates
    r"|\bv\d+\.\d+(?:\.\d+)?\b"  # versions
    r"|\b:\d{4,5}\b|\b\d{4,5}/(?:tcp|udp)\b"  # ports
    r")"
)

ENV_IDENTIFIER_LOSS_REFUSE = "HINDSIGHT_API_MENTAL_MODEL_IDENTIFIER_LOSS_REFUSE"

#: Refuse a refresh that drops this many DISTINCT identifiers. Default 3 from
#: the distribution above: it blocks the 7 catastrophic events (1.9% of
#: refreshes) and lets the 20 one-or-two-identifier events (5.5%) through with
#: a warning. 0 disables refusal entirely (warn-only).
DEFAULT_IDENTIFIER_LOSS_REFUSE = 3

#: Cap on how many lost identifiers are named in a warning before it is
#: summarised. The warning exists to be read.
_MAX_NAMED = 10


def refuse_threshold() -> int:
    """Read the refusal threshold from the environment, clamped at >= 0.

    Read per call rather than at import so a running worker picks up a change
    without a restart, and so tests can set it without reloading the module.
    A non-numeric value falls back to the default rather than raising: this
    sits on a write path, and a typo'd env var must not break refreshes.
    """
    raw = os.getenv(ENV_IDENTIFIER_LOSS_REFUSE)
    if raw is None or not raw.strip():
        return DEFAULT_IDENTIFIER_LOSS_REFUSE
    try:
        return max(0, int(raw.strip()))
    except ValueError:
        return DEFAULT_IDENTIFIER_LOSS_REFUSE


def extract_identifiers(text: str | None) -> set[str]:
    """Every distinct identifier in ``text``. Set semantics are the point:
    an identifier that MOVES within the document counts as kept."""
    if not text:
        return set()
    return set(_IDENTIFIER_RE.findall(text))


def lost_identifiers(before: str | None, after: str | None) -> set[str]:
    """Identifiers present in ``before`` and absent from ``after``."""
    return extract_identifiers(before) - extract_identifiers(after)


def format_warning(lost: set[str]) -> str:
    """A warning that names the lost identifiers verbatim.

    Verbatim matters: "3 identifiers lost" sends a human digging through two
    document versions, whereas the names usually make the cause obvious at a
    glance.
    """
    names = sorted(lost)
    shown = ", ".join(names[:_MAX_NAMED])
    if len(names) > _MAX_NAMED:
        shown += f", +{len(names) - _MAX_NAMED} more"
    return f"identifier-retention: refresh dropped {len(names)} identifier(s) present in the previous content: {shown}."


def evaluate(
    previous_content: str | None,
    candidate_content: str | None,
    has_delta_baseline: bool,
    threshold: int | None = None,
) -> tuple[bool, str | None]:
    """Grade one candidate write.

    Returns ``(should_refuse, warning_or_None)``.

    ``has_delta_baseline`` false means there is no existing real content to
    clobber -- a bootstrap write over an empty or PENDING model cannot "lose"
    anything, and blocking it would stop a model ever being populated. That is
    the same condition the sibling placeholder guard uses.

    A threshold of 0 never refuses but still warns, so the signal stays
    visible while the refusal is disabled.
    """
    if not has_delta_baseline:
        return False, None
    lost = lost_identifiers(previous_content, candidate_content)
    if not lost:
        return False, None
    limit = refuse_threshold() if threshold is None else max(0, threshold)
    warning = format_warning(lost)
    should_refuse = limit > 0 and len(lost) >= limit
    return should_refuse, warning
