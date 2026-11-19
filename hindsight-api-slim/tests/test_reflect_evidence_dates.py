"""Reflect shows every piece of evidence with its dates, raw chunks included.

A chunk is source text ("I signed up for a pottery class yesterday"), and before this it reached
both the agent and the decision model with no date at all. On LoComo the answer then read
"yesterday" against the question date: a sign-up from July came back as October 21. Facts carried
their dates; the chunk they came from did not.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.reflect.agent import _evidence_line
from hindsight_api.engine.reflect.tools import tool_recall
from hindsight_api.engine.response_models import ChunkInfo, MemoryFact, RecallResult
from hindsight_api.models import RequestContext


def test_evidence_line_shows_when_it_happened_and_when_it_was_said() -> None:
    line = _evidence_line(
        {
            "text": "Melanie went camping with her kids",
            "occurred_start": "2023-06-17",
            "occurred_end": "2023-06-23",
            "mentioned_at": "2023-06-27",
        }
    )
    assert line == "[occurred 2023-06-17 to 2023-06-23; mentioned 2023-06-27] Melanie went camping with her kids"


def test_evidence_line_names_a_single_day_once() -> None:
    line = _evidence_line({"text": "x", "occurred_start": "2023-07-02", "occurred_end": "2023-07-02"})
    assert line == "[occurred 2023-07-02] x"


def test_evidence_line_keeps_timestamps_short() -> None:
    """A full ISO timestamp is ~19 tokens; once candidates are truncated it crowded out the text (#6659)."""
    line = _evidence_line(
        {
            "text": "x",
            "occurred_start": "2026-09-18T01:41:00.020000+00:00",
            "occurred_end": "2026-09-19T23:59:59+00:00",
            "mentioned_at": "2026-09-24T12:00:00+00:00",
        }
    )
    assert line == "[occurred 2026-09-18 01:41 to 2026-09-19; mentioned 2026-09-24 12:00] x"


def test_evidence_line_shows_times_in_utc_and_a_whole_day_once() -> None:
    assert (
        _evidence_line({"text": "x", "mentioned_at": "2026-09-24T14:30:00+02:00"}) == "[mentioned 2026-09-24 12:30] x"
    )
    line = _evidence_line(
        {"text": "x", "occurred_start": "2026-09-19T00:00:00+00:00", "occurred_end": "2026-09-19T23:59:59.999999+00:00"}
    )
    assert line == "[occurred 2026-09-19] x"


def test_evidence_line_for_a_chunk_shows_when_it_was_said() -> None:
    assert _evidence_line({"text": "I signed up yesterday", "mentioned_at": "2023-07-03"}) == (
        "[mentioned 2023-07-03] I signed up yesterday"
    )


def test_evidence_line_without_dates_is_the_text() -> None:
    assert _evidence_line({"text": "undated"}) == "undated"


@pytest.mark.asyncio
async def test_recall_tool_dates_each_chunk_by_its_facts() -> None:
    """A chunk is said when its facts were mentioned; a chunk no fact points at stays undated."""
    engine = MagicMock()
    engine.recall_async = AsyncMock(
        return_value=RecallResult(
            results=[
                MemoryFact(
                    id="f1",
                    text="Melanie signed up for a pottery class",
                    fact_type="world",
                    chunk_id="c-july",
                    mentioned_at="2023-07-03T13:36:00Z",
                ),
                MemoryFact(id="f2", text="Caroline went to a parade", fact_type="world", chunk_id="c-other"),
            ],
            chunks={
                "c-july": ChunkInfo(chunk_text="I just signed up for a pottery class yesterday", chunk_index=0),
                "c-other": ChunkInfo(chunk_text="Last week I went to a pride parade", chunk_index=1),
            },
        )
    )

    out = await tool_recall(engine, "bank", "pottery class", request_context=RequestContext())

    assert out["chunks"]["c-july"]["mentioned_at"] == "2023-07-03T13:36:00Z"
    assert "mentioned_at" not in out["chunks"]["c-other"]
