"""A live session learns a rule from the agent's verdict on a note, and the next session hears it.

Sessions, the agent's note feedback, mental models and the session check meet here: a check gives
a note from an old decision, the agent answers it is wrong and says what holds instead, the server
writes that as a rule (a ``rule-`` mental model), and a later session's check in the same bank
reads the rule. Before, checks excluded mental models, and a learned rule never reached a later
session.

Checks and learning both need reflect's decision model, so the server is ``learning_server``; the
stub answers the decision model's "none" with ``stubs.rerank.pick_none``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator

import pytest
from hindsight_client_api.models import SessionAppendRequest, SessionMessage, SessionOptions

from hindsight_system_tests import wait_until_settled
from hindsight_system_tests.payloads import extracted, fact, observes
from hindsight_system_tests.steps import anchor_for

pytestmark = pytest.mark.asyncio

OLD_NOTE = "Before you change shell.tsx: Files, Knowledge and Settings each stay their own floating panel."
RULE_NAME = "Shell: new panels are System panel tabs"
RULE = "In shell.tsx, Files, Knowledge and Settings are tabs of one System panel; add a new panel there as a tab."
NEW_NOTE = "Before you change shell.tsx: add the Logs panel as a tab of the System panel."


@pytest.fixture
async def shell_bank(learning_client, llm) -> AsyncIterator[str]:
    """A bank holding the old decision: three floating panels."""
    bank = f"systest-{uuid.uuid4().hex[:12]}"
    llm.on_step("extract_facts").returns(
        extracted(fact("shell.tsx keeps Files, Knowledge and Settings as floating panels", entities=["shell.tsx"]))
    )
    llm.on_step("consolidate").answers_with(
        observes("shell.tsx keeps Files, Knowledge and Settings as floating panels")
    )
    await learning_client.aretain(bank_id=bank, content="We decided shell.tsx keeps three floating panels.")
    await wait_until_settled(learning_client, bank)
    yield bank
    with contextlib.suppress(Exception):
        await learning_client.banks.delete_bank(bank)


async def _until(client, bank: str, session: str, kind: str, timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    while True:
        items = (await client.sessions.list_session_events(bank, session)).items
        found = [e for e in items if e.kind == kind]
        if found:
            return found
        assert time.monotonic() < deadline, f"no {kind} in the log: {[(e.kind, e.payload) for e in items]}"
        await asyncio.sleep(0.25)


async def _ask(client, bank: str, session: str, request: str) -> None:
    await client.sessions.append_session_events(
        bank,
        session,
        SessionAppendRequest(
            session=SessionOptions(live=True, retain="off"),
            messages=[SessionMessage.from_dict({"id": "u1", "role": "user", "content": request})],
        ),
    )


async def test_a_rule_learned_from_a_wrong_note_reaches_the_next_session(learning_client, llm, stubs, shell_bank):
    stubs.rerank.cut_level = 5
    stubs.rerank.pick_none = 0.1
    llm.on_step("reflect_abstain_answer", contains="Merge the three").returns_text(
        json.dumps({"applies": True, "answer": OLD_NOTE})
    )
    llm.on_step("session_learn").returns_text(
        json.dumps({"create": True, "name": RULE_NAME, "query": "How are shell panels laid out?", "content": RULE})
    )
    llm.on_step("reflect_abstain_answer", contains="Logs panel").returns_text(
        json.dumps({"applies": True, "answer": NEW_NOTE})
    )

    # 1. The check speaks from the old decision; the agent says it is wrong, and why.
    await _ask(learning_client, shell_bank, "s1", "Merge the three floating panels in shell.tsx into one System panel")
    (note,) = await _until(learning_client, shell_bank, "s1", "note")
    assert note.payload["text"] == OLD_NOTE
    reply = (
        f"Merged them.\n> 🧠 Hindsight thought #{note.payload['number']} — they stay floating panels; wrong: "
        "the user asked to merge Files, Knowledge and Settings into one System panel in shell.tsx."
    )
    await learning_client.sessions.append_session_events(
        shell_bank,
        "s1",
        SessionAppendRequest(messages=[SessionMessage.from_dict({"id": "a1", "role": "assistant", "content": reply})]),
    )

    # 2. The verdict becomes a rule. The writer read the agent's reason, never the note it rejected.
    (learned,) = await _until(learning_client, shell_bank, "s1", "learn")
    assert learned.payload["outcome"] == "created" and learned.payload["name"] == RULE_NAME
    rule_id = learned.payload["mental_model_id"]
    assert rule_id.startswith("rule-")
    (writer,) = [c for c in llm.calls if anchor_for("session_learn") in c.all_text]
    assert "merge Files, Knowledge and Settings" in writer.all_text and OLD_NOTE not in writer.all_text
    models = (await learning_client.mental_models.list_mental_models(shell_bank, detail="content")).items
    assert [(m.id, m.content.rstrip("\n")) for m in models] == [(rule_id, RULE)]

    # 3. A later session's check reads the rule, and its note cites it.
    await _ask(learning_client, shell_bank, "s2", "Add a floating Logs panel to shell.tsx for the server logs")
    (check,) = await _until(learning_client, shell_bank, "s2", "check")
    assert check.payload["outcome"] == "note" and rule_id in check.payload["memory_ids"]
    (answer,) = [
        c for c in llm.calls if anchor_for("reflect_abstain_answer") in c.all_text and "Logs panel" in c.all_text
    ]
    assert RULE in answer.all_text, "the rule reached the answer"


PRICE_RULE = "Typesafe costs $0.04 per 1M tokens: use this price in margin and COGS tables."


async def test_a_rule_learned_from_the_users_correction_reaches_the_next_session(learning_client, llm, stubs):
    """The user corrects the agent ("it's 4 cent, not 40"); the next session's check reads the rule.

    Live, memory already held the lesson for 28 of 164 corrections in a week, and only note verdicts
    were ever learned from."""
    bank = f"systest-{uuid.uuid4().hex[:12]}"
    stubs.rerank.cut_level = 5
    stubs.rerank.pick_none = 0.1
    llm.on_step("session_learn", contains="it's 4 cent").returns_text(
        json.dumps(
            {
                "corrects": True,
                "create": True,
                "name": "Typesafe price",
                "query": "What does typesafe cost per token?",
                "content": PRICE_RULE,
            }
        )
    )
    llm.on_step("reflect_abstain_answer", contains="margin").returns_text(
        json.dumps({"applies": True, "answer": "Before you price typesafe: it costs $0.04 per 1M tokens."})
    )
    try:
        # 1. A request, the agent's wrong price, and the user's correction.
        await _ask(learning_client, bank, "s1", "Add the typesafe cost to the COGS table")
        for mid, role, text in (
            ("a1", "assistant", "I priced typesafe at $0.40 per 1M tokens."),
            ("u2", "user", "it's 4 cent, not 40"),
        ):
            await learning_client.sessions.append_session_events(
                bank,
                "s1",
                SessionAppendRequest(messages=[SessionMessage.from_dict({"id": mid, "role": role, "content": text})]),
            )

        # 2. The correction becomes a rule; the writer read the reply and the price it corrects.
        (learned,) = await _until(learning_client, bank, "s1", "learn")
        assert learned.payload["outcome"] == "created" and learned.payload["source"] == "correction"
        rule_id = learned.payload["mental_model_id"]
        (writer,) = [c for c in llm.calls if anchor_for("session_learn") in c.all_text]
        assert "it's 4 cent, not 40" in writer.all_text and "$0.40 per 1M" in writer.all_text

        # 3. A later session's check reads it.
        await _ask(learning_client, bank, "s2", "Recompute the margin table with the typesafe calls")
        (check,) = await _until(learning_client, bank, "s2", "check")
        assert check.payload["outcome"] == "note" and rule_id in check.payload["memory_ids"]
    finally:
        with contextlib.suppress(Exception):
            await learning_client.banks.delete_bank(bank)
