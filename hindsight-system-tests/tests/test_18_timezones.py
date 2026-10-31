"""A clock time with no UTC offset is local time, not UTC.

The extraction LLM copies a clock time out of the content as written —
"the job started at 09:44:23" becomes `occurred_start="2026-01-05T09:44:23"`,
with no offset. That time is local to whoever wrote the content, and it used to
be stored as UTC, so every such fact landed shifted by the writer's offset: +8h
for a client in Shanghai, which put it after the moment it was learned (#6208).

The zone it is read in is the retain item's own timestamp offset when it has
one, and otherwise the bank's `default_timezone` — which also covers a timestamp
sent with no offset of its own.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

SHANGHAI = timezone(timedelta(hours=8))
# 09:44:23 in Shanghai.
JOB_STARTED_UTC = datetime(2026, 1, 5, 1, 44, 23, tzinfo=timezone.utc)


def _as_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


async def _retain_job_start(client, llm, settled, bank_id: str, timestamp: datetime) -> None:
    llm.on_step("extract_facts").returns(
        extracted(
            fact(
                "The nightly job started",
                entities=["nightly job"],
                fact_kind="event",
                occurred_start="2026-01-05T09:44:23",
                occurred_end="2026-01-05T09:44:23",
            )
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="At 2026-01-05 09:44:23 the nightly job started.", timestamp=timestamp)
    await settled(bank_id)


async def _only_fact(client, bank_id: str):
    response = await client.arecall(bank_id=bank_id, query="When did the nightly job start?")
    assert len(response.results) == 1
    return response.results[0]


async def test_an_extracted_time_takes_the_offset_of_its_items_timestamp(client, llm, bank_id, settled):
    """The issue's case: the client sent +08:00, the bank sets nothing."""
    await _retain_job_start(client, llm, settled, bank_id, datetime(2026, 1, 5, 10, 0, tzinfo=SHANGHAI))

    result = await _only_fact(client, bank_id)
    assert _as_datetime(result.occurred_start) == JOB_STARTED_UTC
    assert _as_datetime(result.mentioned_at) == datetime(2026, 1, 5, 2, 0, tzinfo=timezone.utc)


async def test_the_bank_timezone_reads_a_timestamp_sent_without_an_offset(client, llm, bank_id, settled):
    """No offset anywhere: both the item's timestamp and the extracted time are
    read in the bank's zone, so the two clocks still agree with each other."""
    await client.banks.update_bank_config(bank_id, {"updates": {"default_timezone": "Asia/Shanghai"}})

    await _retain_job_start(client, llm, settled, bank_id, datetime(2026, 1, 5, 10, 0))

    result = await _only_fact(client, bank_id)
    assert _as_datetime(result.occurred_start) == JOB_STARTED_UTC
    assert _as_datetime(result.mentioned_at) == datetime(2026, 1, 5, 2, 0, tzinfo=timezone.utc)
