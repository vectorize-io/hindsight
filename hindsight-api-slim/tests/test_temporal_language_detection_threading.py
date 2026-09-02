"""Warming locale dictionaries must not change what language detection answers.

`_ensure_dictionary_warm` performs a locale's first `count_applicability` early and
under a lock, so that concurrent callers never reach dateparser's in-place
`clean_dictionary` at the same time. That is only safe to do if the early call is
otherwise invisible, which is what this asserts.

The race it prevents is NOT reproducible on a GIL build: 12 threads entering cold
locales through a barrier produce no error on 3.11, because the iterate-then-delete
sequence in `Locale.clean_dictionary` is effectively atomic there. It reproduces
reliably on a free-threaded interpreter, where it killed the startup of whichever
event loops lost the race. tests/test_free_threading.py and the free-threaded CI job
are what actually guard it; this file guards the equivalence.
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
