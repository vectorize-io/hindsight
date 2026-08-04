"""Small, dependency-free language heuristics used by consolidation.

Consolidation runs on installations that do not necessarily have local ML
packages available.  The detector therefore deliberately uses Unicode script
evidence and a small set of common Latin-language words.  It is conservative:
text with too little evidence is reported as unknown and is not used to reject
an LLM response.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "latin": "Latin-script text",
    "nl": "Dutch",
    "pt": "Portuguese",
    "ru": "Russian",
    "th": "Thai",
    "tr": "Turkish",
    "zh": "Chinese",
}

# These are intentionally short, high-signal words.  They are only used when
# the text is otherwise Latin-script, so names and technical terms do not by
# themselves make a source fact look like a different language.
_LATIN_STOPWORDS = {
    "de": frozenset(
        "der die das und ist im in ein eine nicht mit auf für den dem des ich bin lebt wohnt arbeitet".split()
    ),
    "en": frozenset(
        "the a an and is are was were in on of to for with from not this that user has have i am my live lives works working owns currently".split()
    ),
    "es": frozenset(
        "el la los las y es son en un una de del para con por que no hola yo vivo vive está esta tiene tengo".split()
    ),
    "fr": frozenset(
        "le la les et est sont en un une de des du pour avec dans que pas bonjour je j suis à habite vit".split()
    ),
    "it": frozenset("il lo la gli le e è sono in un una di del per con che non ciao io vivo abita vive ha".split()),
    "nl": frozenset("de het een en is zijn in van voor met dat niet ik woon werkt".split()),
    "pt": frozenset("o a os as e é são em um uma de do da para com que não olá eu moro vive tenho está".split()),
    "tr": frozenset("bir ve bu için ile olan olanı değil merhaba ben yaşıyor".split()),
}


@dataclass(frozen=True)
class LanguageDetection:
    """Language evidence for one piece of text.

    ``language`` is an ISO-639-1-style code, or ``latin`` when only the script
    can be identified safely. ``None`` means that there is not enough evidence
    to make a safe decision. ``evidence`` is an approximate count of meaningful
    characters/word hits and is useful when combining several source facts into
    a dominant-language decision.
    """

    language: str | None
    confidence: float
    evidence: int
    script: str | None = None

    @property
    def reliable(self) -> bool:
        """Whether this result has enough signal for a persistence guard."""

        return self.language is not None and self.evidence >= 2 and self.confidence >= 0.35


def language_name(language: str | None) -> str:
    """Return a human-readable language name for prompts and diagnostics."""

    if not language:
        return "unknown"
    return _LANGUAGE_NAMES.get(language, language)


def _script_for_character(character: str) -> str | None:
    """Classify a letter by the Unicode block that carries language signal."""

    codepoint = ord(character)
    if 0x3040 <= codepoint <= 0x30FF or 0xFF66 <= codepoint <= 0xFF9D:
        return "kana"
    if 0xAC00 <= codepoint <= 0xD7AF or 0x1100 <= codepoint <= 0x11FF or 0x3130 <= codepoint <= 0x318F:
        return "hangul"
    if 0x3400 <= codepoint <= 0x4DBF or 0x4E00 <= codepoint <= 0x9FFF or 0xF900 <= codepoint <= 0xFAFF:
        return "han"
    if 0x0400 <= codepoint <= 0x052F:
        return "cyrillic"
    if 0x0370 <= codepoint <= 0x03FF:
        return "greek"
    if 0x0590 <= codepoint <= 0x05FF:
        return "hebrew"
    if 0x0600 <= codepoint <= 0x06FF or 0x0750 <= codepoint <= 0x077F:
        return "arabic"
    if 0x0900 <= codepoint <= 0x097F:
        return "devanagari"
    if 0x0E00 <= codepoint <= 0x0E7F:
        return "thai"
    if 0x0530 <= codepoint <= 0x058F:
        return "armenian"
    if 0x10A0 <= codepoint <= 0x10FF:
        return "georgian"
    if character.isalpha():
        # ``isalpha`` includes a few non-script letters.  The name check keeps
        # accented Latin words in the Latin bucket without importing a locale
        # or language-identification package.
        if unicodedata.name(character, "").startswith("LATIN"):
            return "latin"
    return None


def _script_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for character in text:
        script = _script_for_character(character)
        if script is not None:
            counts[script] += 1
    return counts


def _latin_language_scores(text: str) -> dict[str, float]:
    """Score Latin languages using high-signal function-word matches."""

    tokens = {token.casefold() for token in _WORD_RE.findall(text)}
    owners: defaultdict[str, list[str]] = defaultdict(list)
    for token in tokens:
        for language, words in _LATIN_STOPWORDS.items():
            if token in words:
                owners[token].append(language)

    scores = {language: 0.0 for language in _LATIN_STOPWORDS}
    for languages in owners.values():
        # Words shared by several languages (for example ``in`` or ``a``) are
        # weak evidence. Distinctive words carry the full weight and prevent a
        # common token from making an English sentence look German or Spanish.
        weight = 3.0 if len(languages) == 1 else 0.25
        for language in languages:
            scores[language] += weight
    return scores


def detect_language(text: str | None) -> LanguageDetection:
    """Detect the strongest language signal in ``text``.

    Script-heavy languages are identified from Unicode ranges.  Latin-script
    languages use common-word evidence and otherwise fall back conservatively
    to a generic Latin-script result. The result is intentionally heuristic;
    callers should honor ``reliable`` before treating a mismatch as actionable.
    """

    if not text:
        return LanguageDetection(language=None, confidence=0.0, evidence=0)

    counts = _script_counts(text)
    latin_scores = _latin_language_scores(text)

    # Weight kana more heavily than Han characters so Japanese text containing
    # kanji is not mistaken for Chinese.  The other script scores are direct
    # character evidence and are intentionally deterministic.
    candidates: dict[str, float] = {
        "ja": (counts.get("kana", 0) * 2.0 + counts.get("han", 0)) if counts.get("kana", 0) else 0,
        "ko": counts.get("hangul", 0) * 2.0,
        "zh": counts.get("han", 0),
        "ru": counts.get("cyrillic", 0),
        "el": counts.get("greek", 0),
        "he": counts.get("hebrew", 0),
        "ar": counts.get("arabic", 0),
        "hi": counts.get("devanagari", 0),
        "th": counts.get("thai", 0),
    }
    candidates.update(latin_scores)
    candidates = {language: score for language, score in candidates.items() if score > 0}
    if counts.get("latin", 0) and not any(language in candidates for language in _LATIN_STOPWORDS):
        # Names and technical strings often have no function words. Treat them
        # as generic Latin-script evidence instead of guessing English: a false
        # language guess could reject a valid French/Spanish/etc. observation.
        # Omit this generic candidate when another script is present so product
        # names such as "Google" do not overpower the surrounding source text.
        if not any(script != "latin" and count for script, count in counts.items()):
            candidates["latin"] = math.ceil(counts["latin"] / 4)
    if not candidates:
        return LanguageDetection(language=None, confidence=0.0, evidence=0)

    # ``max`` preserves insertion order for exact ties.  The order above makes
    # mixed-script ties deterministic, while source-batch ties are resolved by
    # ``detect_dominant_language`` below.
    language = max(candidates, key=lambda item: candidates[item])
    strongest = candidates[language]
    total = sum(candidates.values())
    script = {
        "ja": "kana",
        "ko": "hangul",
        "zh": "han",
        "ru": "cyrillic",
        "el": "greek",
        "he": "hebrew",
        "ar": "arabic",
        "hi": "devanagari",
        "th": "thai",
    }.get(language, "latin")
    return LanguageDetection(
        language=language,
        confidence=strongest / total if total else 0.0,
        evidence=max(1, math.ceil(strongest)),
        script=script,
    )


def detect_dominant_language(texts: Iterable[str]) -> LanguageDetection:
    """Choose a deterministic dominant language across source facts.

    Evidence is summed by language.  When two languages have equal evidence,
    the language detected in the earliest non-empty source fact wins; this
    makes mixed-language batches stable across retries and processes.
    """

    scores: defaultdict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    scripts: dict[str, str | None] = {}
    for index, text in enumerate(texts):
        detection = detect_language(text)
        if not detection.reliable:
            continue
        assert detection.language is not None
        scores[detection.language] += detection.evidence
        first_seen.setdefault(detection.language, index)
        scripts.setdefault(detection.language, detection.script)

    if not scores:
        return LanguageDetection(language=None, confidence=0.0, evidence=0)

    language = max(scores, key=lambda item: (scores[item], -first_seen[item]))
    strongest = scores[language]
    total = sum(scores.values())
    return LanguageDetection(
        language=language,
        confidence=strongest / total if total else 0.0,
        evidence=max(1, math.ceil(strongest)),
        script=scripts.get(language),
    )


def languages_match(expected: LanguageDetection, actual: LanguageDetection) -> bool:
    """Return whether two reliable detections describe the same language."""

    if not expected.reliable or not actual.reliable:
        return True
    if expected.script == actual.script == "latin" and "latin" in {expected.language, actual.language}:
        # A generic Latin-script result means the text lacked enough lexical
        # evidence to distinguish English, French, Spanish, and related
        # languages. Do not turn that uncertainty into a false rejection.
        return True
    return expected.language == actual.language
