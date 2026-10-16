"""#2550: the later *message* wins, even when the document carrying it started earlier.

Session 1 opens on 2026-07-01, session 2 says "React is my favorite" on 2026-07-05, then session 1
is re-retained (same document_id) with a last message from 2026-07-10: "svelte is my favorite".
Ingested as the conversation happens, consolidation must end on Svelte, whether the update keeps
the session-start event_date or carries the time of the last message.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.memory_engine import MemoryEngine
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

SESSION_1_START = """\
[2026-07-01 09:00] user: I'm starting a new side project, a small dashboard for my workouts.
[2026-07-01 09:02] assistant: Nice. Which frontend framework are you leaning towards?
[2026-07-01 09:05] user: Not sure yet, I'll try a few and see.
"""
SESSION_1_FULL = (
    SESSION_1_START
    + "[2026-07-10 18:30] user: OK, after trying them all week: I've decided svelte is my favorite frontend framework.\n"
)

SESSION_2 = """\
[2026-07-05 14:00] user: React is my favorite frontend framework.
"""


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


SESSION_1_START_AT = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "update_event_date",
    [
        pytest.param(SESSION_1_START_AT, id="update-keeps-session-start"),
        pytest.param(datetime(2026, 7, 10, 18, 30, tzinfo=timezone.utc), id="update-at-last-message"),
    ],
)
async def test_later_message_wins_across_overlapping_sessions(
    memory_real_llm: MemoryEngine, request_context: Any, update_event_date: datetime
) -> None:
    """Ingest in real time: session 1 opens, session 2 happens, session 1 is re-retained with its last message."""
    memory = memory_real_llm
    bank_id = f"test-2550-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)

    async def retain(content: str, event_date: datetime, document_id: str) -> None:
        await memory.retain_async(
            bank_id=bank_id,
            content=content,
            event_date=event_date,
            document_id=document_id,
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

    await retain(SESSION_1_START, SESSION_1_START_AT, "session-1")  # 07-01
    await retain(SESSION_2, datetime(2026, 7, 5, 14, 0, tzinfo=timezone.utc), "session-2")  # 07-05
    await retain(SESSION_1_FULL, update_event_date, "session-1")  # 07-10: same doc, updated

    units = (await memory.list_memory_units(bank_id, limit=1000, request_context=request_context))["items"]
    print("\n" + "\n".join(f"- [{u['fact_type']}] (mentioned_at={u['mentioned_at']}) {u['text']}" for u in units))

    observations = "\n".join(f"- {u['text']}" for u in units if u["fact_type"] == "observation")
    await assert_meets_criteria(
        response=observations,
        criteria=(
            "The observations say the user's CURRENT favorite frontend framework is Svelte. "
            "Mentioning React as a past favorite is fine; saying React is the current favorite, "
            "or that the user moved from Svelte to React, fails."
        ),
        context=(
            "Svelte was declared favorite on 2026-07-10 (last message of a session that started 2026-07-01). "
            "React was declared favorite on 2026-07-05. Svelte is the later statement."
        ),
    )

    await memory.delete_bank(bank_id, request_context=request_context)
