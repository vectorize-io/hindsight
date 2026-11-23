"""A clock time with no UTC offset is local time, not UTC.

The extraction LLM copies a clock time out of the content as written —
"the job started at 09:44:23" becomes `occurred_start="2026-01-05T09:44:23"`,
with no offset. That time is local to whoever wrote the content, and it used to
be stored as UTC, so every such fact landed shifted by the writer's offset: +8h
for a client in Shanghai, which put it after the moment it was learned (#6208).

The zone it is read in is the retain item's own timestamp offset when it has
one, and otherwise the bank's `default_timezone` — which also covers a timestamp
sent with no offset of its own.

That zone is also the one every model reads the memories in afterwards — consolidation,
the observation tree and reflect — so their text keeps the local day and hour (#6441).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hindsight_system_tests.payloads import Consolidation, Observation, batch_fact_id, consolidation, extracted, fact
from hindsight_system_tests.reflect import reflect_loop
from hindsight_system_tests.steps import anchor_for

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
    await client.aretain(
        bank_id=bank_id, content="At 2026-01-05 09:44:23 the nightly job started.", timestamp=timestamp
    )
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


# 07:10 on 8 November in Shanghai: still 7 November in UTC.
DOWNLOAD_CHECKED = datetime(2026, 11, 8, 7, 10, tzinfo=SHANGHAI)
DOWNLOAD = "The model download reached 57%"


async def test_a_bank_in_shanghai_reads_its_memories_in_shanghai_time(client, llm, bank_id, settled):
    """Every model that reads the memories reads the bank's local time, not UTC (#6441).

    The stored time is right either way; what went wrong was the text written from it.
    Shown UTC with nothing saying so, consolidation wrote "on 7 November" into an
    observation of an 8 November morning, and the observation tree filed it the same way.
    So retain, consolidation, the tree and reflect must all agree on the local day.
    """
    await client.banks.update_bank_config(bank_id, {"updates": {"default_timezone": "Asia/Shanghai"}})
    llm.on_step("extract_facts").returns(extracted(fact(DOWNLOAD, entities=["model download"])))

    def one_observation(request) -> Consolidation:
        fid = batch_fact_id(request, DOWNLOAD)
        return Consolidation(creates=[Observation(text=DOWNLOAD, source_fact_ids=[fid], reason="system test")])

    llm.on_step("consolidate").answers_with(one_observation)
    await client.aretain(bank_id=bank_id, content=f"{DOWNLOAD}.", timestamp=DOWNLOAD_CHECKED)
    await settled(bank_id)

    [prompt] = [c.user_text for c in llm.calls if anchor_for("consolidate") in c.all_text]
    assert "local time in Asia/Shanghai" in prompt
    assert "2026-11-08T07:10+08:00" in prompt and "2026-11-07T23:10" not in prompt

    tree = await client.memory.get_observation_tree(bank_id)
    [topic] = [f for f in list(tree.files) + [f for sub in tree.folders for f in sub.files] if f.kind == "topic"]
    assert "Times are Asia/Shanghai." in topic.content
    assert "2026-11-08 07:10" in topic.content and "2026-11-07" not in topic.content

    reflect_loop(llm, answer="It reached 57% on 8 November.", query="model download")
    await client.areflect(bank_id=bank_id, query="When did the model download reach 57%?")
    reads = [c.all_text for c in llm.calls if anchor_for("reflect") in c.all_text and DOWNLOAD in c.all_text]
    assert reads, "reflect never read the memory"
    assert all("Asia/Shanghai (UTC+08:00)" in text for text in reads)
    assert all("2026-11-08 07:10" in text and "2026-11-07 23:10" not in text for text in reads)
