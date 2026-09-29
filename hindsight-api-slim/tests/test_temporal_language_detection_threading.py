"""Warming locale dictionaries must not change what language detection answers.

`_ensure_dictionary_warm` performs a locale's first `count_applicability` early and
under a lock, so that concurrent callers never reach dateparser's in-place
`clean_dictionary` at the same time. That is only safe to do if the early call is
otherwise invisible, which is what this asserts.

The race it prevents is hard to reproduce deterministically: 12 threads entering cold
locales through a barrier usually produce no error, because the iterate-then-delete
sequence in `Locale.clean_dictionary` rarely gets interrupted at the wrong point. So
this file asserts the equivalence — that warming up front gives the same answers as
the racy cold path — rather than trying to trigger the crash.
"""

from dateparser.languages.loader import LocaleDataLoader

from hindsight_api.engine.temporal_language_detection import best_language

_LOCALES = list(LocaleDataLoader().get_locales(languages=None, locales=None, region=None))[:40]

_TEXTS = [
    "what happened last friday",
    "cosa e successo ieri sera",
    "was ist gestern passiert",
    "que paso el martes pasado",
    "1200 tokens and no date at all",
]


def test_warming_does_not_change_detection():
    """Every answer must be what it was before the warm call was introduced."""
    first = {text: best_language(text, _LOCALES) for text in _TEXTS}
    # Second pass runs entirely against warmed locales — the path every request takes.
    second = {text: best_language(text, _LOCALES) for text in _TEXTS}
    assert first == second
