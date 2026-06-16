"""Explicit period extraction helpers for DateparserQueryAnalyzer.

This module keeps the public period-extraction API and the non-Chinese period
rules. Chinese rules live in chinese_temporal_periods.py because that rule set is
substantially larger and has different boundary behavior from whitespace-based
languages.
"""

import calendar
import re
import unicodedata
from datetime import datetime, timedelta

DateRange = tuple[datetime, datetime]


class NoTemporalConstraintSentinel:
    pass


NO_TEMPORAL_CONSTRAINT = NoTemporalConstraintSentinel()

__all__ = [
    "NO_TEMPORAL_CONSTRAINT",
    "extract_period",
    "is_embedded_cjk_dateparser_match",
]


def _is_cjk_character(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def is_embedded_cjk_dateparser_match(query: str, matched_text: str) -> bool:
    from hindsight_api.engine.chinese_temporal_periods import (
        is_embedded_cjk_dateparser_match as chinese_is_embedded_cjk_dateparser_match,
    )

    return chinese_is_embedded_cjk_dateparser_match(query, matched_text)


def _constraint(start: datetime, end: datetime) -> DateRange:
    return (
        start.replace(hour=0, minute=0, second=0, microsecond=0),
        end.replace(hour=23, minute=59, second=59, microsecond=999999),
    )


def _month_end(year: int, month: int) -> datetime:
    return datetime(year, month, calendar.monthrange(year, month)[1])


def _extract_non_chinese_period(query: str, reference_date: datetime) -> DateRange | None:
    if re.search(r"\b(yesterday|ayer|ieri|hier|gestern)\b", query, re.IGNORECASE):
        d = reference_date - timedelta(days=1)
        return _constraint(d, d)

    if re.search(r"\b(today|hoy|oggi|aujourd\'?hui|heute)\b", query, re.IGNORECASE):
        return _constraint(reference_date, reference_date)

    if re.search(r"\b(a\s+)?couple\s+(of\s+)?days?\s+ago\b", query, re.IGNORECASE):
        return _constraint(reference_date - timedelta(days=3), reference_date - timedelta(days=1))

    if re.search(r"\b(a\s+)?few\s+days?\s+ago\b", query, re.IGNORECASE):
        return _constraint(reference_date - timedelta(days=5), reference_date - timedelta(days=2))

    if re.search(r"\b(a\s+)?couple\s+(of\s+)?weeks?\s+ago\b", query, re.IGNORECASE):
        return _constraint(reference_date - timedelta(weeks=3), reference_date - timedelta(weeks=1))

    if re.search(r"\b(a\s+)?few\s+weeks?\s+ago\b", query, re.IGNORECASE):
        return _constraint(reference_date - timedelta(weeks=5), reference_date - timedelta(weeks=2))

    if re.search(r"\b(a\s+)?couple\s+(of\s+)?months?\s+ago\b", query, re.IGNORECASE):
        return _constraint(reference_date - timedelta(days=90), reference_date - timedelta(days=30))

    if re.search(r"\b(a\s+)?few\s+months?\s+ago\b", query, re.IGNORECASE):
        return _constraint(reference_date - timedelta(days=150), reference_date - timedelta(days=60))

    if re.search(
        r"\b(last\s+week|la\s+semana\s+pasada|la\s+settimana\s+scorsa|la\s+semaine\s+derni[eè]re|letzte\s+woche)\b",
        query,
        re.IGNORECASE,
    ):
        start = reference_date - timedelta(days=reference_date.weekday() + 7)
        return _constraint(start, start + timedelta(days=6))

    if re.search(
        r"\b(last\s+month|el\s+mes\s+pasado|il\s+mese\s+scorso|le\s+mois\s+dernier|letzten?\s+monat)\b",
        query,
        re.IGNORECASE,
    ):
        first = reference_date.replace(day=1)
        end = first - timedelta(days=1)
        start = end.replace(day=1)
        return _constraint(start, end)

    if re.search(
        r"\b(last\s+year|el\s+a[ñn]o\s+pasado|l\'anno\s+scorso|l\'ann[ée]e\s+derni[eè]re|letztes?\s+jahr)\b",
        query,
        re.IGNORECASE,
    ):
        year = reference_date.year - 1
        return _constraint(datetime(year, 1, 1), datetime(year, 12, 31))

    if re.search(
        r"\b(last\s+weekend|el\s+fin\s+de\s+semana\s+pasado|lo\s+scorso\s+fine\s+settimana|le\s+week-?end\s+dernier|letztes?\s+wochenende)\b",
        query,
        re.IGNORECASE,
    ):
        days_since_sat = (reference_date.weekday() + 2) % 7
        if days_since_sat == 0:
            days_since_sat = 7
        sat = reference_date - timedelta(days=days_since_sat)
        return _constraint(sat, sat + timedelta(days=1))

    month_patterns = {
        "january|enero|gennaio|janvier|januar": 1,
        "february|febrero|febbraio|f[ée]vrier|februar": 2,
        "march|marzo|mars|m[äa]rz": 3,
        "april|abril|aprile|avril": 4,
        "may|mayo|maggio|mai": 5,
        "june|junio|giugno|juin|juni": 6,
        "july|julio|luglio|juillet|juli": 7,
        "august|agosto|ao[uû]t": 8,
        "september|septiembre|settembre|septembre": 9,
        "october|octubre|ottobre|octobre|oktober": 10,
        "november|noviembre|novembre": 11,
        "december|diciembre|dicembre|d[ée]cembre|dezember": 12,
    }
    for pattern, month_num in month_patterns.items():
        match = re.search(rf"\b({pattern})\s+(\d{{4}})\b", query, re.IGNORECASE)
        if match:
            year = int(match.group(2))
            start = datetime(year, month_num, 1)
            return _constraint(start, _month_end(year, month_num))

    return None


def extract_period(query: str, reference_date: datetime) -> DateRange | NoTemporalConstraintSentinel | None:
    """Extract explicit period-based temporal expressions.

    Non-Chinese rules are kept here. Chinese rules are delegated to
    chinese_temporal_periods.py and are skipped entirely for non-CJK queries.
    """
    query = unicodedata.normalize("NFKC", query)

    if any(_is_cjk_character(char) for char in query):
        from hindsight_api.engine.chinese_temporal_periods import extract_chinese_period

        chinese_result = extract_chinese_period(query, reference_date)
        if chinese_result is not None:
            return chinese_result

    return _extract_non_chinese_period(query, reference_date)
