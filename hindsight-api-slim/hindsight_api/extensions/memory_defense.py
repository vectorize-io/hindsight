"""Memory Defense extension contract and shared policy types.

Lives in extensions/ (not engine/) because it defines the public contract
between the retain orchestrator and any installed Memory Defense extension —
the same shape as TenantExtension and OperationValidatorExtension.

api-slim ships the :class:`MemoryDefenseExtension` protocol and a regex default
that scrubs known secret/PII patterns from retained content.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from hindsight_api.extensions.base import Extension

logger = logging.getLogger(__name__)


class DefenseAction(str, Enum):
    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


_VALID_ACTIONS = {a.value for a in DefenseAction}

# ``policy.rules[*].on`` names a detector. The OSS extension only screens for
# ``sensitive_data``; any other name is a silent no-op here and is dispatched
# by whichever extension is loaded (e.g. hindsight-cloud screens cloud-only
# detectors). The parser therefore does NOT validate ``on`` against a fixed
# list — pinning the OSS roster to cloud's would force an OSS bump for every
# new cloud detector just to avoid 422-ing a write it never interprets. We
# only require ``on`` to be a non-empty string; entitlement and dispatch are
# the loaded extension's ``screen()`` job.


@dataclass(frozen=True)
class PolicyRule:
    on: str
    action: DefenseAction


@dataclass(frozen=True)
class DefensePolicy:
    enabled: bool = False
    rules: tuple[PolicyRule, ...] = ()


@dataclass
class DefenseDecision:
    action: DefenseAction
    detector: str | None = None
    message: str = ""
    redacted_content: str | None = None
    matched_types: list[str] = field(default_factory=list)
    # Per-match fingerprinted previews. Each entry is
    # ``{"detector": <pattern label>, "preview": <fingerprinted value>}``.
    # The preview is *never* the raw value — see :func:`_fingerprint_value`.
    # OSS populates this from ``apply_redaction``; downstream extensions
    # populate it from their own detectors. Optional: empty when the
    # match path didn't capture per-hit values.
    hits: list[dict] = field(default_factory=list)


@dataclass
class RedactionResult:
    content: str
    matched_types: list[str]
    # Same shape as ``DefenseDecision.hits`` — one entry per matched value
    # (so a single content with two GitHub tokens produces two entries).
    hits: list[dict] = field(default_factory=list)


def _fingerprint_value(value: str) -> str:
    """Return a redaction-identifiable preview of a matched value.

    The preview keeps the prefix and a short suffix so a SIEM operator can
    correlate against their credential inventory (the prefix names the
    provider; the suffix disambiguates specific instances) without the raw
    secret crossing the wire. Length-aware so short values don't accidentally
    leak material:

    - Length < 6:  redact entirely (return a fixed-length mask). Catches
      noise like a single ``-----BEGIN...`` marker line.
    - Length 6-15: keep the first 2 + last 2 around an ellipsis.
    - Length > 15: keep the first 4 + last 4 around an ellipsis.

    Examples::

        _fingerprint_value("ghp_AAAA...AAAA" + "A" * 36)  -> "ghp_...AAAA"
        _fingerprint_value("AKIA" + "B" * 16)              -> "AKIA...BBBB"
        _fingerprint_value("123-45-6789")                  -> "12...89"
        _fingerprint_value("abc")                          -> "[redacted]"
    """
    n = len(value)
    if n < 6:
        return "[redacted]"
    if n <= 15:
        return f"{value[:2]}...{value[-2:]}"
    return f"{value[:4]}...{value[-4:]}"


def parse_policy(raw: dict | None) -> DefensePolicy:
    """Parse a raw bank-config dict into a frozen DefensePolicy.

    Raises ValueError for a missing/empty ``on`` or an unknown action; the
    HTTP layer converts those into a 422 response.
    """
    if raw is None:
        return DefensePolicy()

    rules: list[PolicyRule] = []
    for item in raw.get("rules", []) or []:
        on_raw = item.get("on")
        if not isinstance(on_raw, str) or not on_raw:
            raise ValueError(f"invalid on {on_raw!r}; must be a non-empty string")
        action_raw = item.get("action")
        if action_raw not in _VALID_ACTIONS:
            raise ValueError(f"invalid action {action_raw!r}; must be one of {sorted(_VALID_ACTIONS)}")
        rules.append(PolicyRule(on=on_raw, action=DefenseAction(action_raw)))

    return DefensePolicy(
        enabled=bool(raw.get("enabled", False)),
        rules=tuple(rules),
    )


# Secret/PII redaction patterns.
#
# Scope: high-confidence patterns with unambiguous prefixes (low false-positive
# rate). Context-dependent matches (e.g. Cohere/Mistral keys that only stand
# out near surrounding "cohere"/"mistral" tokens) are NOT covered by pure
# regex — operators who need that should layer a context-aware secret
# scanner (detect-secrets, trufflehog) on top.
#
# Order matters: more-specific patterns first so broader ones don't consume
# substrings partially. Example: `sk-ant-...` and `sk-proj-...` must run
# before the generic `sk-...` pattern.
_REDACTION_PATTERNS: list[tuple[str, str]] = [
    # --- AI / LLM providers ---
    ("anthropic_key", r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    ("openai_project_key", r"\bsk-proj-[A-Za-z0-9_-]{48,}\b"),
    ("openai_admin_key", r"\bsk-admin-[A-Za-z0-9_-]{40,}\b"),
    ("openai_key", r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    ("google_api_key", r"\bAIza[0-9A-Za-z_-]{35}\b"),
    ("google_oauth_token", r"\bya29\.[0-9A-Za-z_-]{20,}\b"),
    ("xai_key", r"\bxai-[A-Za-z0-9]{40,}\b"),
    ("groq_key", r"\bgsk_[A-Za-z0-9]{20,}\b"),
    ("huggingface_token", r"\bhf_[A-Za-z0-9]{30,}\b"),
    ("replicate_token", r"\br8_[A-Za-z0-9]{30,}\b"),
    ("perplexity_key", r"\bpplx-[A-Za-z0-9]{40,}\b"),
    ("databricks_token", r"\bdapi[A-Za-z0-9]{32}\b"),
    # --- Cloud providers ---
    ("aws_access_key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("aws_session_token", r"\bASIA[0-9A-Z]{16}\b"),
    (
        "aws_secret_key",
        r"(?i)aws(.{0,20})?(secret|private)?[\s_-]?access[\s_-]?key[\s_-]?[:=][\s\"']*([A-Za-z0-9/+=]{40})",
    ),
    ("digitalocean_token", r"\bdop_v1_[a-f0-9]{64}\b"),
    # --- Source control & CI ---
    ("github_fg_pat", r"\bgithub_pat_[A-Za-z0-9_]{60,}\b"),
    ("github_token", r"\bghp_[A-Za-z0-9]{36}\b"),
    ("github_app_token", r"\bghs_[A-Za-z0-9]{36}\b"),
    ("github_user_token", r"\bghu_[A-Za-z0-9]{36}\b"),
    ("github_refresh", r"\bghr_[A-Za-z0-9]{36}\b"),
    ("github_oauth", r"\bgho_[A-Za-z0-9]{36}\b"),
    ("gitlab_pat", r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    ("npm_token", r"\bnpm_[A-Za-z0-9]{30,}\b"),
    ("pypi_token", r"\bpypi-AgEIcHlwaS5vcmc[A-Za-z0-9_-]{20,}\b"),
    # --- Payment processors ---
    ("stripe_secret", r"\bsk_(?:live|test)_[A-Za-z0-9]{20,}\b"),
    ("stripe_restricted", r"\brk_(?:live|test)_[A-Za-z0-9]{20,}\b"),
    ("square_token", r"\bsq0[a-z]{3}-[A-Za-z0-9_-]{22,}\b"),
    ("braintree_token", r"\baccess_token\$production\$[a-z0-9]{16}\$[a-f0-9]{32}\b"),
    # --- Communication / email ---
    ("slack_token", r"\bxox[abpr]-[0-9A-Za-z-]{10,}\b"),
    ("slack_webhook", r"https://hooks\.slack\.com/services/T[A-Za-z0-9_]{8,}/B[A-Za-z0-9_]{8,}/[A-Za-z0-9_]{20,}"),
    ("twilio_api_key", r"\bSK[0-9a-fA-F]{32}\b"),
    ("twilio_account_sid", r"\bAC[0-9a-fA-F]{32}\b"),
    ("sendgrid_key", r"\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}\b"),
    ("mailgun_key", r"\bkey-[A-Za-z0-9]{32}\b"),
    ("discord_bot", r"\b[MNO][A-Za-z0-9]{23}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27}\b"),
    ("telegram_bot", r"\b[0-9]{8,10}:[A-Za-z0-9_-]{35}\b"),
    # --- Commerce ---
    ("shopify_token", r"\bshpat_[a-fA-F0-9]{32}\b"),
    # --- Database connection strings (creds embedded in URL) ---
    ("db_url_postgres", r"postgres(?:ql)?://[^\s:/@]+:[^\s/@]+@[^\s]+"),
    ("db_url_mysql", r"mysql://[^\s:/@]+:[^\s/@]+@[^\s]+"),
    ("db_url_mongodb", r"mongodb(?:\+srv)?://[^\s:/@]+:[^\s/@]+@[^\s]+"),
    # --- Private keys & generic credentials ---
    ("private_key_pem", r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY( BLOCK)?-----"),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    # --- PII (US-centric defaults; can be tuned per deployment) ---
    # NOTE: credit_card regex is intentionally narrowed to 13-19 digits with
    # exact separators to reduce false positives on long product IDs.
    ("credit_card", r"\b(?:\d{4}[ -]?){3}\d{1,4}\b"),
    ("ssn_us", r"\b\d{3}-\d{2}-\d{4}\b"),
]
_COMPILED_REDACTIONS: list[tuple[str, re.Pattern]] = [
    (label, re.compile(pattern)) for label, pattern in _REDACTION_PATTERNS
]


def apply_redaction(content: str) -> RedactionResult:
    """Scrub known secret/PII patterns from content with [REDACTED:type] markers.

    Returns the (possibly unchanged) content alongside:
      - ``matched_types``: pattern labels that matched (deduplicated, in
        first-occurrence order). Empty when nothing matched.
      - ``hits``: per-match fingerprinted previews — one entry per matched
        substring (so two GitHub tokens in the same content produce two
        entries). Each entry is ``{"detector": label, "preview": fingerprint}``
        where ``preview`` is a length-aware redaction of the original value.
        The raw secret never appears in ``hits``.

    The two-pass shape (find matches first, then substitute) lets us capture
    raw values for fingerprinting before they're replaced by ``[REDACTED:type]``
    markers. A single-pass approach would lose the originals.
    """
    matched: list[str] = []
    hits: list[dict] = []
    for label, pattern in _COMPILED_REDACTIONS:
        raw_hits = pattern.findall(content)
        if not raw_hits:
            continue
        if label not in matched:
            matched.append(label)
        for raw in raw_hits:
            # findall returns either a string or a tuple of capture groups
            # depending on the pattern. The redaction-pattern catalog uses a
            # mix; coerce to the matched substring as best we can.
            if isinstance(raw, tuple):
                # Pick the longest non-empty group as the canonical match.
                non_empty = [g for g in raw if g]
                raw_str = max(non_empty, key=len) if non_empty else ""
            else:
                raw_str = raw
            if not raw_str:
                continue
            hits.append({"detector": label, "preview": _fingerprint_value(raw_str)})
        content = pattern.sub(f"[REDACTED:{label}]", content)
    return RedactionResult(content=content, matched_types=matched, hits=hits)


class MemoryDefenseExtension(Extension, ABC):
    """Abstract base for Memory Defense extensions.

    Implementations decide whether to allow, redact, or block a given retain
    item by inspecting its content against a per-bank policy. The orchestrator
    applies the returned decision (redacts content / drops blocked items) and
    fires a webhook for non-allow decisions when one is configured.
    """

    @abstractmethod
    async def screen(
        self,
        *,
        policy: DefensePolicy,
        bank_id: str,
        document_id: str | None,
        content: str,
        tags: list[str],
    ) -> DefenseDecision:
        """Inspect content under the given policy and return a decision."""
        ...
