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
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
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


# ASCII token boundaries.
#
# ``re`` compiles ``\b`` (and ``\w``) with Unicode semantics, so a CJK
# character counts as a word character: there is no word boundary between
# 为 and s, and ``\bsk_test_...\b`` silently fails to match in
# 凭证为sk_test_ABC... . Secrets embedded in Chinese/Japanese/Korean prose
# therefore reached memory units unredacted.
#
# These lookarounds anchor on the ASCII token alphabet instead. They are
# strictly more permissive than ``\b`` (``[A-Za-z0-9_]`` is a subset of
# ``\w``), so nothing that matched before stops matching, while a secret
# butted up against non-ASCII text is now detected and a partial ASCII token
# still isn't. New boundary-based patterns must use this helper, not ``\b``
# — test_redaction_patterns_do_not_use_unicode_word_classes enforces it.
_ASCII_TOKEN_START = r"(?<![A-Za-z0-9_])"
_ASCII_TOKEN_END = r"(?![A-Za-z0-9_])"


def _ascii_token_pattern(body: str) -> str:
    """Wrap an ASCII token pattern without treating CJK letters as token chars."""
    return f"{_ASCII_TOKEN_START}{body}{_ASCII_TOKEN_END}"


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
    ("anthropic_key", _ascii_token_pattern(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("openai_project_key", _ascii_token_pattern(r"sk-proj-[A-Za-z0-9_-]{48,}")),
    ("openai_admin_key", _ascii_token_pattern(r"sk-admin-[A-Za-z0-9_-]{40,}")),
    ("openai_key", _ascii_token_pattern(r"sk-[A-Za-z0-9_-]{20,}")),
    ("google_api_key", _ascii_token_pattern(r"AIza[0-9A-Za-z_-]{35}")),
    ("google_oauth_token", _ascii_token_pattern(r"ya29\.[0-9A-Za-z_-]{20,}")),
    ("xai_key", _ascii_token_pattern(r"xai-[A-Za-z0-9]{40,}")),
    ("groq_key", _ascii_token_pattern(r"gsk_[A-Za-z0-9]{20,}")),
    ("huggingface_token", _ascii_token_pattern(r"hf_[A-Za-z0-9]{30,}")),
    ("replicate_token", _ascii_token_pattern(r"r8_[A-Za-z0-9]{30,}")),
    ("perplexity_key", _ascii_token_pattern(r"pplx-[A-Za-z0-9]{40,}")),
    ("databricks_token", _ascii_token_pattern(r"dapi[A-Za-z0-9]{32}")),
    # --- Cloud providers ---
    ("aws_access_key", _ascii_token_pattern(r"AKIA[0-9A-Z]{16}")),
    ("aws_session_token", _ascii_token_pattern(r"ASIA[0-9A-Z]{16}")),
    (
        "aws_secret_key",
        r"(?i)aws(.{0,20})?(secret|private)?[\s_-]?access[\s_-]?key[\s_-]?[:=][\s\"']*([A-Za-z0-9/+=]{40})",
    ),
    ("digitalocean_token", _ascii_token_pattern(r"dop_v1_[a-f0-9]{64}")),
    # --- Source control & CI ---
    ("github_fg_pat", _ascii_token_pattern(r"github_pat_[A-Za-z0-9_]{60,}")),
    ("github_token", _ascii_token_pattern(r"ghp_[A-Za-z0-9]{36}")),
    ("github_app_token", _ascii_token_pattern(r"ghs_[A-Za-z0-9]{36}")),
    ("github_user_token", _ascii_token_pattern(r"ghu_[A-Za-z0-9]{36}")),
    ("github_refresh", _ascii_token_pattern(r"ghr_[A-Za-z0-9]{36}")),
    ("github_oauth", _ascii_token_pattern(r"gho_[A-Za-z0-9]{36}")),
    ("gitlab_pat", _ascii_token_pattern(r"glpat-[A-Za-z0-9_-]{20,}")),
    ("npm_token", _ascii_token_pattern(r"npm_[A-Za-z0-9]{30,}")),
    ("pypi_token", _ascii_token_pattern(r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9_-]{20,}")),
    # --- Payment processors ---
    ("stripe_secret", _ascii_token_pattern(r"sk_(?:live|test)_[A-Za-z0-9]{20,}")),
    ("stripe_restricted", _ascii_token_pattern(r"rk_(?:live|test)_[A-Za-z0-9]{20,}")),
    ("square_token", _ascii_token_pattern(r"sq0[a-z]{3}-[A-Za-z0-9_-]{22,}")),
    ("braintree_token", _ascii_token_pattern(r"access_token\$production\$[a-z0-9]{16}\$[a-f0-9]{32}")),
    # --- Communication / email ---
    ("slack_token", _ascii_token_pattern(r"xox[abpr]-[0-9A-Za-z-]{10,}")),
    ("slack_webhook", r"https://hooks\.slack\.com/services/T[A-Za-z0-9_]{8,}/B[A-Za-z0-9_]{8,}/[A-Za-z0-9_]{20,}"),
    ("twilio_api_key", _ascii_token_pattern(r"SK[0-9a-fA-F]{32}")),
    ("twilio_account_sid", _ascii_token_pattern(r"AC[0-9a-fA-F]{32}")),
    ("sendgrid_key", _ascii_token_pattern(r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}")),
    ("mailgun_key", _ascii_token_pattern(r"key-[A-Za-z0-9]{32}")),
    ("discord_bot", _ascii_token_pattern(r"[MNO][A-Za-z0-9]{23}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27}")),
    ("telegram_bot", _ascii_token_pattern(r"[0-9]{8,10}:[A-Za-z0-9_-]{35}")),
    # --- Commerce ---
    ("shopify_token", _ascii_token_pattern(r"shpat_[a-fA-F0-9]{32}")),
    # --- Database connection strings (creds embedded in URL) ---
    ("db_url_postgres", r"postgres(?:ql)?://[^\s:/@]+:[^\s/@]+@[^\s]+"),
    ("db_url_mysql", r"mysql://[^\s:/@]+:[^\s/@]+@[^\s]+"),
    ("db_url_mongodb", r"mongodb(?:\+srv)?://[^\s:/@]+:[^\s/@]+@[^\s]+"),
    # --- Private keys & generic credentials ---
    ("private_key_pem", r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY( BLOCK)?-----"),
    ("jwt", _ascii_token_pattern(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    # --- PII ---
    # These patterns use explicit ASCII lookarounds instead of ``\b`` so a
    # value immediately next to CJK text is still detected. Name and address
    # matching is intentionally context-bound to reduce false positives.
    (
        "email",
        r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
        r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
        r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
        r"(?![A-Za-z0-9._%+-])",
    ),
    (
        "phone_cn",
        _ascii_token_pattern(r"(?:(?:\+|00)86[ -]?)?1[3-9](?:[ -]?\d){9}"),
    ),
    (
        "phone_international",
        _ascii_token_pattern(r"(?:\+[1-9]\d{0,2}|00[1-9]\d{0,2})(?:[ .()-]*\d){6,14}"),
    ),
    (
        "id_cn",
        _ascii_token_pattern(
            r"[1-9]\d{5}(?:18|19|20)\d{2}"
            r"(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"
        ),
    ),
    (
        "bank_card_cn",
        _ascii_token_pattern(r"62(?:[ -]?\d){14,17}"),
    ),
    (
        "person_name",
        _ascii_token_pattern(
            r"(?i:(?:full[ -]?name|name|contact|recipient|"
            r"客户姓名|姓名|联系人|收件人)[ \t:：]+"
            r"(?:[\u4e00-\u9fff]{2,4}|[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){0,2}))"
        ),
    ),
    (
        "address",
        _ascii_token_pattern(
            r"(?im:(?:(?:^|(?<=[\r\n]))[ \t]*address[ \t:：]+|"
            r"(?:home[ -]?address|shipping[ -]?address|postal[ -]?address|"
            r"收件地址|收货地址|联系地址|(?<![\u4e00-\u9fff])(?:地址|住址))[ \t:：]+)"
            r"[^\r\n;；。.!！？]{4,160}(?<![ \t,，]))"
        ),
    ),
    # NOTE: credit_card regex is intentionally narrowed to 13-19 digits with
    # exact separators to reduce false positives on long product IDs.
    ("credit_card", _ascii_token_pattern(r"(?:\d{4}[ -]?){3}\d{1,4}")),
    ("ssn_us", _ascii_token_pattern(r"\d{3}-\d{2}-\d{4}")),
]


_COMPILED_REDACTIONS: list[tuple[str, re.Pattern]] = [
    (label, re.compile(pattern)) for label, pattern in _REDACTION_PATTERNS
]

_CN_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_CN_ID_CHECK_CODES = "10X98765432"


def _is_valid_cn_id(value: str) -> bool:
    """Validate the birth date and ISO 7064 checksum of a PRC ID number."""
    normalized = value.upper()
    try:
        birth_date = date(int(normalized[6:10]), int(normalized[10:12]), int(normalized[12:14]))
    except ValueError:
        return False
    if birth_date > date.today():
        return False

    checksum_index = sum(int(digit) * weight for digit, weight in zip(normalized[:17], _CN_ID_WEIGHTS)) % 11
    return normalized[-1] == _CN_ID_CHECK_CODES[checksum_index]


def _is_luhn_valid(value: str) -> bool:
    """Validate a separated or contiguous payment-card number with Luhn."""
    digits = [int(char) for char in value if char.isdigit()]
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _is_generic_credit_card_candidate(value: str) -> bool:
    """Do not let invalid 16-digit UnionPay candidates fall through as credit cards."""
    digits = "".join(char for char in value if char.isdigit())
    return not (len(digits) == 16 and digits.startswith("62"))


_REDACTION_VALIDATORS: dict[str, Callable[[str], bool]] = {
    "id_cn": _is_valid_cn_id,
    "bank_card_cn": _is_luhn_valid,
    "credit_card": _is_generic_credit_card_candidate,
}
_FULLY_MASKED_REDACTION_TYPES = {
    "email",
    "phone_cn",
    "phone_international",
    "id_cn",
    "bank_card_cn",
    "person_name",
    "address",
}


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
        validator = _REDACTION_VALIDATORS.get(label)
        if validator is None:
            raw_hits = pattern.findall(content)
        else:
            raw_hits = [match.group(0) for match in pattern.finditer(content) if validator(match.group(0))]
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
            preview = "[redacted]" if label in _FULLY_MASKED_REDACTION_TYPES else _fingerprint_value(raw_str)
            hits.append({"detector": label, "preview": preview})
        if validator is None:
            content = pattern.sub(f"[REDACTED:{label}]", content)
        else:
            content = pattern.sub(
                lambda match: f"[REDACTED:{label}]" if validator(match.group(0)) else match.group(0),
                content,
            )
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
