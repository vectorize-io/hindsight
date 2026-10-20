"""The audit list stays light; one entry carries its bodies (#5822).

A retain's audit row stores everything that was submitted. The list used to return
those bodies for every row on the page, so a few pages of big retains could take the
API process down. The story pins the split: an audited retain shows up in the list
with no request or response, and fetching that entry by id gives the full content back.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from hindsight_system_tests.payloads import extracted, fact, observes

pytestmark = pytest.mark.asyncio


async def test_the_list_leaves_bodies_out_and_the_entry_has_them(client, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice"])))
    llm.on_step("consolidate").answers_with(observes("Alice lives in Berlin"))
    await client.banks.update_bank_config(bank_id, {"updates": {"audit_log_enabled": True}})

    marker = "Alice moved to Berlin. " + "padding " * 5_000
    await client.aretain(bank_id=bank_id, content=marker)
    await settled(bank_id)

    # The audit row is written after the response is sent, so give it a moment.
    for _ in range(50):
        listed = await client.audit.list_audit_logs(bank_id, action="retain")
        if listed.total:
            break
        await asyncio.sleep(0.1)
    assert listed.total == 1
    [summary] = listed.items
    assert summary.action == "retain"
    assert summary.transport == "http"
    assert summary.bank_id == bank_id
    assert summary.duration_ms is not None
    assert "request" not in summary.to_dict() and "response" not in summary.to_dict()

    entry = await client.audit.get_audit_log(bank_id, summary.id)
    assert entry.id == summary.id
    assert marker in json.dumps(entry.request)
    assert entry.response is not None
