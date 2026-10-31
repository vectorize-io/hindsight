"""Retain times with no UTC offset are read in the content's zone, not UTC (issue #6208)."""

from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.engine.retain.fact_extraction import _content_timezone, _parse_datetime
from hindsight_api.engine.retain.orchestrator import _build_contents, bank_timezone
from hindsight_api.engine.retain.types import RetainContent

SHANGHAI = timezone(timedelta(hours=8))


def test_llm_time_without_offset_is_read_in_event_date_offset():
    """The issue's repro: event date +08:00, content says 09:44:23 local."""
    content = RetainContent(content="x", event_date=datetime(2026, 1, 5, 10, 0, tzinfo=SHANGHAI))
    parsed = _parse_datetime("2026-01-05T09:44:23", _content_timezone(content, SimpleNamespace()))
    assert parsed.astimezone(UTC) == datetime(2026, 1, 5, 1, 44, 23, tzinfo=UTC)


def test_llm_time_with_offset_is_kept():
    parsed = _parse_datetime("2026-01-05T09:44:23Z", SHANGHAI)
    assert parsed == datetime(2026, 1, 5, 9, 44, 23, tzinfo=UTC)


def test_content_without_event_date_falls_back_to_bank_zone():
    content = RetainContent(content="x", event_date=None)
    assert _content_timezone(content, SimpleNamespace(default_timezone="Europe/Rome")) == ZoneInfo("Europe/Rome")
    assert _content_timezone(content, SimpleNamespace()) == UTC


def test_build_contents_reads_naive_timestamp_in_bank_zone():
    tz = bank_timezone(SimpleNamespace(default_timezone="Asia/Shanghai"))
    naive, missing = _build_contents(
        [{"content": "a", "event_date": "2026-01-05T10:00:00"}, {"content": "b"}], None, tz
    )
    assert naive.event_date.astimezone(UTC) == datetime(2026, 1, 5, 2, 0, tzinfo=UTC)
    # A missing timestamp is "now" carrying the bank's zone, so the prompt's Event
    # Date shows the offset the LLM's own times are read in.
    assert missing.event_date.tzinfo == ZoneInfo("Asia/Shanghai")


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_bank_config_rejects_unknown_zone(api_client):
    bank_id = f"tz_{datetime.now().timestamp()}"
    bad = await api_client.patch(
        f"/v1/default/banks/{bank_id}/config", json={"updates": {"default_timezone": "Mars/Base"}}
    )
    assert bad.status_code == 400, bad.text
    good = await api_client.patch(
        f"/v1/default/banks/{bank_id}/config", json={"updates": {"default_timezone": "Asia/Shanghai"}}
    )
    assert good.status_code == 200, good.text
    assert good.json()["config"]["default_timezone"] == "Asia/Shanghai"
