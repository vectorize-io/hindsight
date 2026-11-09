"""Behaviour-identical, allocation-free replacement for dateparser's search-time
language detection.

``dateparser.search.search_dates`` spends ~98% of its time in
``FullTextLanguageDetector._best_language`` deciding which of 205 locales the
text is in. Three costs in that function are pure waste, and none of them are
fixed upstream. This mirrors the 1.5.0 detector (kana drops Chinese candidates,
native-letter counts break ties), which is why ``dateparser>=1.5.0`` is the floor:

1. ``get_unique_characters()`` does an O(locales²) set-difference sweep — 42,025
   set operations — to work out which characters are unique to each locale. The
   result depends only on the locale set, which never changes, but a fresh
   ``FullTextLanguageDetector`` is constructed per call so it is recomputed every
   time. **Cached here.**

2. Each locale that scores zero is retried with ``strip_timezone=True``, and that
   retry calls ``pop_tz_offset_from_string`` on *the same string* — up to 199
   times per query, each one a linear scan of 773 compiled timezone regexes
   (~154k regex searches per query). The result cannot differ between locales.
   **Hoisted to one call.**

3. When stripping the timezone does not change the string, the retry re-runs a
   computation whose result is already known to be ``[0, 0]``. **Skipped.**

**Thread safety.** ``Locale.count_applicability`` reaches
``Locale.clean_dictionary``, which deletes the sub-threshold keys from the locale's
*cached* ``_split_dictionary`` in place. Two threads reaching a given locale for the
first time therefore iterate and delete from the same dict at once:

    RuntimeError: dictionary changed size during iteration

and whichever caller loses dies. This is reachable today:
``MemoryEngine.initialize()`` warms the analyzer with
``await loop.run_in_executor(None, self.query_analyzer.load)``, so the warmup already
runs on a thread pool — two engines initialising in one process race here.

The mutation is idempotent -- once the short keys are gone the delete list is empty
and nothing is written -- so the race window is first use per locale, and
``_ensure_dictionary_warm`` closes it by doing that first use once under a lock.
After warming, the hot path takes no lock: it is a concurrent read of a dict nobody
mutates. Fixing it here rather than forking dateparser is enough because this module
is the only route into locale applicability counting (``query_analyzer._find_dates``
calls ``best_language``, never dateparser's own detector).

Every transformation here is equivalence-preserving by construction, and
``tests/test_temporal_extraction.py`` asserts that directly: it runs this
implementation and dateparser's side by side over the corpus plus thousands of
randomly generated strings and requires identical answers.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

from dateparser.conf import Settings
from dateparser.conf import settings as default_settings
from dateparser.languages.locale import Locale
from dateparser.timezone_parser import pop_tz_offset_from_string
from dateparser.utils import normalize_unicode

# Mirrors FullTextLanguageDetector.character_check. A string made only of these
# characters carries no language signal, so detection short-circuits to the
# first candidate locale.
_SYMBOL_SET = frozenset("0123456789 /-)(.:\\,'")
# Hiragana and katakana, except the middle dot, which Chinese also uses.
_KANA = re.compile(r"[\u3041-\u3096\u30a1-\u30fa\u30fc-\u30ff]")


@dataclass(frozen=True)
class LocaleCharTables:
    """Character evidence per locale, in locale order.

    ``language_chars[i]`` is every character locale *i* uses in its date
    vocabulary; ``unique_chars[i]`` is the subset no other candidate locale uses,
    which lets a single character decide the language outright.
    """

    language_chars: list[set[str]]
    unique_chars: list[set[str]]


_char_table_lock = threading.Lock()
# id(locale tuple) is not stable, so key on the locale shortnames.
_char_table_cache: dict[tuple[str, ...], LocaleCharTables] = {}


def _char_tables(languages: list[Locale], settings: Settings) -> LocaleCharTables:
    """Per-locale character sets, and the characters unique to each locale.

    This is ``FullTextLanguageDetector.get_unique_characters`` with its result
    memoised. The *result* is a deterministic function of the locale set:
    ``get_wordchars_for_detection`` caches on the (singleton) ``Locale``, so the
    O(n²) difference sweep over those sets always produces the same answer.

    It is not, however, side-effect free — this docstring used to call it "a pure
    function", and that is what made computing it outside the lock look safe. On a
    cache miss it *builds* the locale dictionaries it then reads. See the lock
    comment below.
    """
    key = tuple(lang.shortname for lang in languages)
    cached = _char_table_cache.get(key)
    if cached is not None:
        return cached

    # The *computation* has to be inside the lock, not just the assignment.
    # ``get_wordchars_for_detection`` is not a read: on a miss it builds each
    # locale's dictionary under a fresh ``Settings`` (hence a fresh
    # ``registry_key``) and writes it into ``locale.dictionaries`` and
    # dateparser's class-level regex caches — for all 200+ locales. Two threads
    # missing together therefore mutate the same dicts concurrently, which can
    # segfault inside the ``regex`` extension rather than merely raising
    # "dictionary changed size during iteration". Missing at the same moment is
    # the *normal* case, not a rare one: it is what concurrent warmups on the
    # startup executor do. Same shape as
    # ``_ensure_dictionary_warm`` below — unlocked pre-check, then
    # double-checked under the lock, so the steady state is still lock-free.
    with _char_table_lock:
        cached = _char_table_cache.get(key)
        if cached is not None:
            return cached

        detection_settings = settings.replace(NORMALIZE=False)
        language_chars = [lang.get_wordchars_for_detection(settings=detection_settings) for lang in languages]

        unique_chars = []
        for char_set in language_chars:
            remaining = char_set
            for other in language_chars:
                if other != char_set:
                    remaining = remaining - other
            unique_chars.append(remaining)

        tables = LocaleCharTables(language_chars=language_chars, unique_chars=unique_chars)
        _char_table_cache[key] = tables
        return tables


def _character_check(text: str, languages: list[Locale], settings: Settings) -> list[Locale]:
    """Narrow the candidate locales by character evidence.

    Faithful to ``FullTextLanguageDetector.character_check``, including its
    quirks: it inspects the *original-cased* string, and a single unique
    character decides the language outright.
    """
    lowered = text.lower()
    text_chars = set(lowered)
    if text_chars & _SYMBOL_SET == text_chars:
        return languages[:1]

    if _KANA.search(text):
        # Chinese words are also Japanese ones, e.g. 後天 ("acquired").
        languages = [lang for lang in languages if lang.shortname.split("-")[0] not in {"zh", "yue"}] or languages

    tables = _char_tables(languages, settings)

    for i, chars in enumerate(tables.unique_chars):
        for char in chars:
            if char.lower() in lowered:
                return [languages[i]]

    return [lang for i, lang in enumerate(languages) if text_chars & tables.language_chars[i]]


_WARM_ATTR = "_hindsight_split_dictionary_warm"
_warm_lock = threading.Lock()


def _ensure_dictionary_warm(locale: Locale, settings: Settings) -> None:
    """Populate and clean ``locale``'s split dictionary exactly once, under a lock.

    Marked on the Locale instance rather than in a set keyed by ``shortname``: the
    same shortname can be a different object if dateparser's loader is rebuilt, and
    warming is per-instance state.

    The unlocked pre-check is the point -- after the first call this costs one
    ``getattr`` and no synchronisation, so the per-query path is unaffected.
    """
    if getattr(locale, _WARM_ATTR, False):
        return
    with _warm_lock:
        if getattr(locale, _WARM_ATTR, False):
            return
        # Any text works: this is called for the mutation it performs, not the count.
        locale.count_applicability("a", strip_timezone=False, settings=settings)
        setattr(locale, _WARM_ATTR, True)


def best_language(text: str, languages: list[Locale], settings: Settings | None = None) -> str | None:
    """Return the detected locale shortname, or None.

    Equivalent to ``FullTextLanguageDetector(languages)._best_language(text)``.

    Note the argument order dateparser uses and that we must preserve: the
    character check runs on the raw string, and only afterwards is the text
    lowercased and unicode-normalised for the applicability counts.
    """
    settings = settings or default_settings

    candidates = _character_check(text, languages, settings)
    text = normalize_unicode(text.lower())
    if len(candidates) == 1:
        return candidates[0].shortname

    # Hoisted out of the per-locale loop: the timezone strip depends only on the
    # text, so all 199 retries would compute the same thing.
    stripped, _ = pop_tz_offset_from_string(text, as_offset=False)
    tz_changed = stripped != text

    applicable: list[tuple[str, tuple[int, ...]]] = []
    for language in candidates:
        # The retry on the timezone-stripped text is skipped when there was no
        # timezone: it would re-run the identical computation and get [0, 0] again.
        for candidate_text in (text, stripped) if tz_changed else (text,):
            counts = _counts(language, candidate_text, settings)
            if counts[0] > 0 or counts[1] > 0:
                # On a tie, prefer the language that matches more words without
                # letter variants, so that e.g. Arabic text is not taken for Persian.
                # ``_native`` is a dateparser 1.5.0 private, like the rest of what
                # this module mirrors; the differential tests catch it changing.
                native = language._native
                native_counts = counts if native is language else _counts(native, candidate_text, settings)
                applicable.append((language.shortname, (*counts, *native_counts)))
                break

    if not applicable:
        return None
    return max(applicable, key=lambda pair: pair[1])[0]


def _counts(language: Locale, text: str, settings: Settings) -> list[int]:
    """``count_applicability`` on an already timezone-stripped text, after the one-time warmup."""
    _ensure_dictionary_warm(language, settings)
    return language.count_applicability(text, strip_timezone=False, settings=settings)
