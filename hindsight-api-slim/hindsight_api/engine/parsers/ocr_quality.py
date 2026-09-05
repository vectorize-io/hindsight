"""Deterministic OCR output quality admission checks.

Runs on the parser fallback path after a parser returns text and before that
result is treated as a successful conversion. Hard-rejects refusals, empty-of-
meaning output, and transcriptions dominated by uncertainty markers so they
cannot become memory evidence. Sparse legitimate OCR (codes, amounts, signs)
is not rejected for short length alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Match MarkitdownParser._is_image_file — OCR quality gating applies here.
OCR_QUALITY_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})

_UNCLEAR_TOKEN = "[unclear]"
_REPLACEMENT_CHAR = "\ufffd"

# Phrases that mean the model refused or found no text. Matched case-insensitively
# against normalized whitespace; a hit is a hard reject (strong signal).
_REFUSAL_OR_NO_TEXT_PHRASES: tuple[str, ...] = (
    "i can't read this image",
    "i cannot read this image",
    "i can't read this",
    "i cannot read this",
    "unable to read this image",
    "unable to read the image",
    "i'm unable to read",
    "i am unable to read",
    "can't extract text",
    "cannot extract text",
    "unable to extract text",
    "no visible text",
    "no text visible",
    "no text found",
    "no text detected",
    "there is no text",
    "there is no visible text",
    "no readable text",
    "nothing to transcribe",
    "i can't see any text",
    "i cannot see any text",
    "i don't see any text",
    "i do not see any text",
    "sorry, i can't help",
    "sorry, i cannot help",
    "as an ai language model",
    "as an ai, i cannot",
    "i'm not able to transcribe",
    "i am not able to transcribe",
    "unable to transcribe",
    "cannot transcribe",
)

_UI_CHROME_WORDS = frozenset(
    {
        "ok",
        "cancel",
        "back",
        "next",
        "home",
        "settings",
        "menu",
        "search",
        "close",
        "done",
        "save",
        "edit",
        "delete",
        "share",
        "more",
        "options",
        "profile",
        "login",
        "logout",
        "sign",
        "submit",
        "continue",
        "skip",
        "retry",
        "refresh",
        "download",
        "upload",
        "help",
        "about",
        "privacy",
        "terms",
        "cookie",
        "cookies",
        "accept",
        "decline",
        "allow",
        "deny",
        "yes",
        "no",
    }
)

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\s\w]", re.UNICODE)
_ALNUM_RE = re.compile(r"[A-Za-z0-9]", re.UNICODE)
_TIMESTAMP_RE = re.compile(
    r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\s*(?:am|pm)?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OcrQualityResult:
    """Admission decision plus observability fields (never includes OCR body)."""

    accepted: bool
    reasons: tuple[str, ...]
    measurements: dict[str, float | int]


def is_ocr_quality_candidate(filename: str) -> bool:
    """Return whether OCR quality gating applies to this filename."""
    return Path(filename).suffix.lower() in OCR_QUALITY_EXTENSIONS


def assess_ocr_output_quality(text: str) -> OcrQualityResult:
    """Evaluate whether OCR text is admissible as memory evidence.

    Hard rejects use a single strong signal. Soft/compound rejects require
    multiple independent bad signals so sparse legitimate text (verification
    codes, amounts, signs) is not dropped for length alone.
    """
    raw = text or ""
    normalized = " ".join(raw.split()).strip()
    lower = normalized.lower()

    char_count = len(normalized)
    unclear_count = lower.count(_UNCLEAR_TOKEN)
    replacement_count = normalized.count(_REPLACEMENT_CHAR)

    # Strip uncertainty markers before tokenizing "meaningful" content.
    without_markers = lower.replace(_UNCLEAR_TOKEN, " ").replace(_REPLACEMENT_CHAR, " ")
    words = [w for w in re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", without_markers) if w]
    word_count = len(words)
    unique_word_count = len(set(words))

    alnum_chars = len(_ALNUM_RE.findall(normalized))
    alpha_chars = sum(1 for c in normalized if c.isalpha())
    digit_chars = sum(1 for c in normalized if c.isdigit())
    alnum_ratio = (alnum_chars / char_count) if char_count else 0.0
    unclear_char_span = unclear_count * len(_UNCLEAR_TOKEN)
    unclear_ratio = (unclear_char_span / char_count) if char_count else 0.0
    replacement_ratio = (replacement_count / char_count) if char_count else 0.0
    repetition_ratio = 1.0 - (unique_word_count / word_count) if word_count else 0.0
    timestamp_hits = len(_TIMESTAMP_RE.findall(normalized))
    chrome_hits = sum(1 for w in words if w in _UI_CHROME_WORDS)
    chrome_ratio = (chrome_hits / word_count) if word_count else 0.0

    measurements: dict[str, float | int] = {
        "char_count": char_count,
        "word_count": word_count,
        "unique_word_count": unique_word_count,
        "alnum_ratio": round(alnum_ratio, 4),
        "alpha_chars": alpha_chars,
        "digit_chars": digit_chars,
        "unclear_count": unclear_count,
        "unclear_ratio": round(unclear_ratio, 4),
        "replacement_count": replacement_count,
        "replacement_ratio": round(replacement_ratio, 4),
        "repetition_ratio": round(repetition_ratio, 4),
        "chrome_ratio": round(chrome_ratio, 4),
        "timestamp_hits": timestamp_hits,
    }

    hard_reasons: list[str] = []

    if not normalized:
        hard_reasons.append("empty_output")
    elif _matches_refusal_or_no_text(lower):
        hard_reasons.append("explicit_refusal_or_no_text")
    elif alnum_chars == 0 and unclear_count == 0 and replacement_count == 0:
        hard_reasons.append("zero_meaningful_tokens")
    elif alnum_chars == 0 and (unclear_count > 0 or replacement_count > 0):
        # Only uncertainty markers / replacement chars — nothing usable.
        hard_reasons.append("zero_meaningful_tokens")
    elif unclear_ratio >= 0.5 or (unclear_count >= 2 and unclear_ratio >= 0.4):
        hard_reasons.append("dominated_by_unclear")
    elif replacement_ratio >= 0.5:
        hard_reasons.append("dominated_by_replacement_chars")
    elif chrome_ratio >= 0.8 and chrome_hits >= 4:
        # Navigation-only OCR is a strong signal even without length heuristics.
        hard_reasons.append("dominated_by_ui_chrome")

    if hard_reasons:
        return OcrQualityResult(
            accepted=False,
            reasons=tuple(hard_reasons),
            measurements=measurements,
        )

    soft_reasons: list[str] = []
    if word_count > 0 and chrome_ratio >= 0.6 and chrome_hits >= 3:
        soft_reasons.append("ui_chrome_dense")
    if word_count >= 3 and repetition_ratio >= 0.5:
        soft_reasons.append("high_repetition")
    # Timestamps often tokenize into many short alnum pieces ("12","34","pm").
    # Measure residual content after stripping timestamp spans instead.
    without_timestamps = _TIMESTAMP_RE.sub(" ", normalized)
    residual_alnum = len(_ALNUM_RE.findall(without_timestamps))
    if timestamp_hits >= 2 and residual_alnum <= 4:
        soft_reasons.append("timestamp_dense")
    if char_count > 0 and alnum_ratio < 0.25 and word_count <= 6:
        soft_reasons.append("low_alnum_ratio")
    # Sparse is a supporting signal only — never sufficient alone.
    if word_count <= 3 and char_count <= 24 and soft_reasons:
        soft_reasons.append("sparse_with_other_signals")
    # Timestamp-only / chrome-heavy short dumps: pair with low residual content.
    if "timestamp_dense" in soft_reasons and residual_alnum <= 2:
        soft_reasons.append("low_residual_content")

    # Compound reject: need at least two soft signals.
    if len(soft_reasons) >= 2:
        return OcrQualityResult(
            accepted=False,
            reasons=tuple(soft_reasons),
            measurements=measurements,
        )

    return OcrQualityResult(accepted=True, reasons=(), measurements=measurements)


def _matches_refusal_or_no_text(lower_normalized: str) -> bool:
    """True when the OCR body is (or is dominated by) a refusal / no-text phrase."""
    if not lower_normalized:
        return False
    # Strip trailing punctuation the model sometimes adds.
    compact = lower_normalized.strip(" .!?,;:\"'")
    for phrase in _REFUSAL_OR_NO_TEXT_PHRASES:
        if compact == phrase or compact.startswith(phrase):
            return True
        if phrase in compact and len(phrase) >= 0.6 * len(compact):
            return True
    return False
