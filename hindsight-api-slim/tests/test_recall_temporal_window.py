"""Tests for the caller-supplied `temporal_window` on recall.

`retrieve_all_fact_types_parallel` normally derives the temporal arm's window by
parsing the query text. A caller that already knows the range it means can pass
it instead, which both removes the guesswork and skips the extraction — pure CPU
serialised through a single worker.

These are pure mechanics (which window reaches the store, and whether the
analyzer ran at all), so they assert directly against a stub store: no LLM, no
database.
"""

from datetime import UTC, datetime

import pytest

from hindsight_api.engine.memories import RecallArms
from hindsight_api.engine.query_analyzer import QueryAnalysis, QueryAnalyzer
from hindsight_api.engine.response_models import TemporalWindow
from hindsight_api.engine.search.retrieval import retrieve_all_fact_types_parallel

WINDOW = TemporalWindow(start=datetime(2023, 4, 1, tzinfo=UTC), end=datetime(2023, 6, 30, tzinfo=UTC))

# The same window as JSON, and as the model the handler should build from it.
WINDOW_JSON = {"start": "2023-04-01T00:00:00Z", "end": "2023-06-30T23:59:59Z"}
WINDOW_FROM_JSON = TemporalWindow(
    start=datetime(2023, 4, 1, tzinfo=UTC), end=datetime(2023, 6, 30, 23, 59, 59, tzinfo=UTC)
)


class _RecordingStore:
    """Stands in for the memories store, capturing the window it was handed."""

    def __init__(self) -> None:
        self.temporal_window = "not-called"

    async def recall_unified(self, *, fact_types, temporal_window=None, **kwargs):
        self.temporal_window = temporal_window
        return {ft: RecallArms() for ft in fact_types}


class _ExplodingAnalyzer(QueryAnalyzer):
    """Fails the test if recall parses the query text for dates."""

    def load(self) -> None:
        pass

    def analyze(self, query: str, reference_date: datetime | None = None) -> QueryAnalysis:
        raise AssertionError("query text must not be analysed when a temporal_window is supplied")


class _FixedAnalyzer(QueryAnalyzer):
    """Extracts a window unrelated to WINDOW, so the two are never confused."""

    def __init__(self) -> None:
        self.calls = 0

    def load(self) -> None:
        pass

    def analyze(self, query: str, reference_date: datetime | None = None) -> QueryAnalysis:
        self.calls += 1
        from hindsight_api.engine.query_analyzer import TemporalConstraint

        return QueryAnalysis(
            temporal_constraint=TemporalConstraint(
                start_date=datetime(1999, 1, 1),
                end_date=datetime(1999, 12, 31),
            )
        )


async def _run(monkeypatch, *, analyzer, temporal_window, enable_temporal_retrieval=True):
    store = _RecordingStore()
    monkeypatch.setattr("hindsight_api.engine.memories.get_memories", lambda: store)
    result = await retrieve_all_fact_types_parallel(
        None,  # pool: only ever handed to the store, which is stubbed
        "what did we ship",
        "[0.0]",
        "bank-1",
        ["world"],
        100,
        None,
        analyzer,
        temporal_window=temporal_window,
        enable_temporal_retrieval=enable_temporal_retrieval,
        enable_graph_retrieval=False,
    )
    return store, result


@pytest.mark.asyncio
async def test_supplied_window_reaches_the_store_verbatim(monkeypatch):
    """The caller's bounds are used as-is, and the query text is never parsed."""
    store, result = await _run(monkeypatch, analyzer=_ExplodingAnalyzer(), temporal_window=WINDOW)

    assert store.temporal_window == (WINDOW.start, WINDOW.end)
    assert result.results_by_fact_type["world"].temporal_constraint == (WINDOW.start, WINDOW.end)


@pytest.mark.asyncio
async def test_supplied_window_beats_what_the_query_text_says(monkeypatch):
    """A query that would extract a date does not override an explicit window."""
    analyzer = _FixedAnalyzer()

    store, _ = await _run(monkeypatch, analyzer=analyzer, temporal_window=WINDOW)

    assert store.temporal_window == (WINDOW.start, WINDOW.end)
    assert analyzer.calls == 0


@pytest.mark.asyncio
async def test_without_a_window_the_query_text_is_still_analysed(monkeypatch):
    """The extraction path is unchanged when no window is supplied."""
    analyzer = _FixedAnalyzer()

    store, _ = await _run(monkeypatch, analyzer=analyzer, temporal_window=None)

    assert store.temporal_window == (datetime(1999, 1, 1), datetime(1999, 12, 31))
    assert analyzer.calls == 1


@pytest.mark.asyncio
async def test_disabled_temporal_retrieval_still_wins(monkeypatch):
    """enable_temporal_retrieval stays the single switch for the arm.

    A supplied window does not re-enable an arm the bank turned off, so the
    store gets no window and nothing is analysed.
    """
    store, _ = await _run(
        monkeypatch,
        analyzer=_ExplodingAnalyzer(),
        temporal_window=WINDOW,
        enable_temporal_retrieval=False,
    )

    assert store.temporal_window is None


def test_window_rejects_reversed_bounds():
    with pytest.raises(ValueError, match="must not be earlier"):
        TemporalWindow(start=datetime(2023, 6, 30, tzinfo=UTC), end=datetime(2023, 4, 1, tzinfo=UTC))


def test_window_accepts_equal_bounds():
    """A single instant is a degenerate but legitimate window."""
    instant = datetime(2023, 4, 1, tzinfo=UTC)

    assert TemporalWindow(start=instant, end=instant).start == instant


def test_naive_bounds_are_read_as_utc():
    """The arm coerces naive datetimes to UTC; do it at parse time so both
    bounds are unambiguous before they reach any query."""
    window = TemporalWindow(start=datetime(2023, 4, 1), end=datetime(2023, 6, 30))

    assert window.start.tzinfo == UTC
    assert window.end.tzinfo == UTC


# ---------------------------------------------------------------------------
# HTTP surface: the field has to survive the handler, and bad input must 400.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_recall_forwards_the_window(api_client, memory, monkeypatch):
    """The handler passes temporal_window through to the engine untouched."""
    from hindsight_api.engine.response_models import RecallResult

    captured = {}

    async def _spy(**kwargs):
        captured.update(kwargs)
        return RecallResult(results=[])

    monkeypatch.setattr(memory, "recall_async", _spy)

    response = await api_client.post(
        "/v1/default/banks/temporal_window_bank/memories/recall",
        json={
            "query": "what did we decide",
            "temporal_window": WINDOW_JSON,
        },
    )

    assert response.status_code == 200
    assert captured["temporal_window"] == WINDOW_FROM_JSON


@pytest.mark.asyncio
async def test_http_rejects_reversed_window(api_client):
    """A window that ends before it starts is rejected at the boundary."""
    response = await api_client.post(
        "/v1/default/banks/temporal_window_bank/memories/recall",
        json={
            "query": "what did we decide",
            "temporal_window": {"start": WINDOW_JSON["end"], "end": WINDOW_JSON["start"]},
        },
    )

    assert response.status_code == 422
