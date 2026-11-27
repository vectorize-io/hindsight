"""Recall's default date finder, text2date, and the HINDSIGHT_API_QUERY_ANALYZER switch."""

from datetime import UTC, datetime, time, timedelta, timezone

import pytest

from hindsight_api.config import HindsightConfig, _validate_query_analyzer
from hindsight_api.engine.query_analyzer import (
    DateparserQueryAnalyzer,
    Text2DateQueryAnalyzer,
    create_query_analyzer,
)
from hindsight_api.engine.search import temporal_extraction

# Wednesday 2026-11-18
REF = datetime(2026, 11, 18, 12, 0, 0)


@pytest.fixture(scope="module")
def analyzer() -> Text2DateQueryAnalyzer:
    a = Text2DateQueryAnalyzer()
    a.load()
    return a


def _days(analyzer: Text2DateQueryAnalyzer, query: str, ref: datetime = REF) -> tuple[str, str] | None:
    c = analyzer.analyze(query, ref).temporal_constraint
    return None if c is None else (c.start_date.date().isoformat(), c.end_date.date().isoformat())


@pytest.mark.parametrize(
    ("query", "start", "end"),
    [
        ("what did I decide last week", "2026-11-09", "2026-11-15"),
        ("what happened yesterday", "2026-11-17", "2026-11-17"),
        ("expenses this month", "2026-11-01", "2026-11-30"),
        ("trips 2 years ago", "2024-01-01", "2024-12-31"),
        ("dogs in June 2023", "2023-06-01", "2023-06-30"),
        ("what did I do last Friday", "2026-11-13", "2026-11-13"),
        ("meetings in the last 6 months", "2026-05-18", "2026-11-18"),
        ("what did I read since May", "2026-05-01", "2026-11-18"),
        ("¿qué hice la semana pasada?", "2026-11-09", "2026-11-15"),
        ("was habe ich letzten Monat gemacht", "2026-10-01", "2026-10-31"),
        ("上个月的会议", "2026-10-01", "2026-10-31"),
        ("上上周做了什么", "2026-11-02", "2026-11-08"),
        ("先週の会議のメモ", "2026-11-09", "2026-11-15"),
        ("что я делал в прошлом году", "2025-01-01", "2025-12-31"),
    ],
)
def test_finds_the_dates(analyzer, query, start, end):
    assert _days(analyzer, query) == (start, end)


@pytest.mark.parametrize(
    "query",
    ["what is my wifi password", "install Office 2019", "remind me what June said", "show the last slide", ""],
)
def test_no_date(analyzer, query):
    assert analyzer.analyze(query, REF).temporal_constraint is None


def test_whole_days_naive(analyzer):
    c = analyzer.analyze("what happened yesterday", REF).temporal_constraint
    assert c is not None
    assert c.start_date == datetime(2026, 11, 17, 0, 0)
    assert c.end_date == datetime.combine(c.start_date.date(), time.max)
    assert c.start_date.tzinfo is None and c.end_date.tzinfo is None


@pytest.mark.parametrize("query", ["notes from before 2020", "what happened after 2020"])
def test_one_sided_range_is_no_constraint(analyzer, query):
    assert analyzer.analyze(query, REF).temporal_constraint is None


def test_several_ranges_take_the_first(analyzer):
    assert _days(analyzer, "trips in May 2025 and June 2026") == ("2025-05-01", "2025-05-31")


def test_aware_reference_date_keeps_the_callers_day(analyzer):
    assert _days(analyzer, "what happened yesterday", REF.replace(tzinfo=UTC)) == ("2026-11-17", "2026-11-17")
    # 22:00 in New York is already the next day in UTC; yesterday is still the caller's yesterday.
    late = datetime(2024, 4, 5, 22, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert _days(analyzer, "what did I do yesterday", late) == ("2024-04-04", "2024-04-04")


def test_default_reference_date_is_today(analyzer):
    c = analyzer.analyze("today").temporal_constraint
    assert c is not None
    assert c.start_date.date() == datetime.now(UTC).date()


def test_factory(caplog):
    assert isinstance(create_query_analyzer("text2date"), Text2DateQueryAnalyzer)
    assert "QUERY_ANALYZER_LANGUAGES" not in caplog.text
    assert isinstance(create_query_analyzer("text2date", ["en"]), Text2DateQueryAnalyzer)
    assert "only applies to HINDSIGHT_API_QUERY_ANALYZER=dateparser" in caplog.text
    dp = create_query_analyzer("dateparser", ["en"])
    assert isinstance(dp, DateparserQueryAnalyzer)


@pytest.mark.parametrize(
    ("raw", "want"),
    [(None, "text2date"), ("dateparser", "dateparser"), (" Text2Date ", "text2date"), ("nope", "text2date")],
)
def test_config(monkeypatch, raw, want):
    if raw is None:
        monkeypatch.delenv("HINDSIGHT_API_QUERY_ANALYZER", raising=False)
    else:
        monkeypatch.setenv("HINDSIGHT_API_QUERY_ANALYZER", raw)
    assert HindsightConfig.from_env().query_analyzer == want


def test_invalid_name_warns(caplog):
    assert _validate_query_analyzer("nope") == "text2date"
    assert "Invalid query analyzer" in caplog.text


def test_default_analyzer_is_text2date(monkeypatch):
    monkeypatch.setattr(temporal_extraction, "_default_analyzer", None)
    assert isinstance(temporal_extraction.get_default_analyzer(), Text2DateQueryAnalyzer)
    assert temporal_extraction.extract_temporal_constraint("last week", REF) == (
        datetime(2026, 11, 9),
        datetime.combine(datetime(2026, 11, 15).date(), time.max),
    )


def test_model_runs_in_a_child_process(analyzer):
    """Off the GIL: the temporal worker thread only waits on a pipe, so recall's event loop never stalls."""
    from text2date import ProcessParser

    assert isinstance(analyzer._parser, ProcessParser)
    assert analyzer._parser._proc is not None and analyzer._parser._proc.poll() is None


def test_falls_back_in_process_when_the_child_cannot_start(monkeypatch, caplog):
    import text2date
    from text2date import Parser

    def fail(self):
        raise RuntimeError("no subprocesses here")

    monkeypatch.setattr(text2date.ProcessParser, "load", fail)
    a = Text2DateQueryAnalyzer()
    a.load()
    assert isinstance(a._parser, Parser)
    assert "running the model in-process" in caplog.text
    assert _days(a, "what happened yesterday") == ("2026-11-17", "2026-11-17")
