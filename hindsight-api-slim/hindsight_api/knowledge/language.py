"""Which language a text is in, from its stopwords — enough to pick a stemmer and to gate keyword search.

Keyword scoring helps a query only against documents in the query's own language (PDF_GEMMA.md: gated
on the PDF's majority language it adds ~0.4 nDCG@10 on ViDoRe V3; applied to every query it hurts,
since 5 of 6 query languages miss English pages entirely).
"""

from __future__ import annotations

import re

# The most frequent function words per language: unambiguous enough at document scale.
_STOPWORDS: dict[str, frozenset[str]] = {
    "english": frozenset(
        "the of and to in is that for it with as was on be by this are or from at an have not which".split()
    ),
    "french": frozenset(
        "le la les de des du et est en un une que qui dans pour pas sur au aux par avec ce sont".split()
    ),
    "german": frozenset(
        "der die das und ist nicht ein eine zu den von mit sich des auf für im dem auch es wird".split()
    ),
    "italian": frozenset("il lo la gli le di del della e è che un una per non con sono nel alla dei delle".split()),
    "spanish": frozenset("el la los las de del y que en un una es por con para no se su al lo como más".split()),
    "portuguese": frozenset("o a os as de do da dos das e que em um uma para com não por se no na é ao".split()),
}
_WORD = re.compile(r"[^\W\d_]+")


def detect(text: str | None, *, max_words: int = 2000) -> str | None:
    """The language whose stopwords cover the most words, or None when nothing matches."""
    words = _WORD.findall((text or "").lower())[:max_words]
    best, hits = None, 0
    for language, stop in _STOPWORDS.items():
        n = sum(w in stop for w in words)
        if n > hits:
            best, hits = language, n
    return best


def text_search_config(language: str | None) -> str:
    """The Postgres text search configuration for a detected language ('simple' when unknown)."""
    return language if language in _STOPWORDS else "simple"
