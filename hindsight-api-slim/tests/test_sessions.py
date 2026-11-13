"""Sessions: the append protocol, the watcher's checks and notes, and saving into the document.

The task backend is synchronous here, so a check or save an append queues runs before the append
returns. Reflect and retain are stubbed: what they do with a moment is tested elsewhere; this is
about what the session does around them.
"""

import asyncio
import json
import uuid

import pytest

from hindsight_api.engine.response_models import LLMCallTrace, MemoryFact, ReflectResult, TokenUsage
from hindsight_api.engine.sessions.models import SessionAppendRequest, SessionMessage, SessionOptions, SessionState


def _bank() -> str:
    return f"sess-{uuid.uuid4().hex[:10]}"


def _req(*steps: dict, ack: int | None = None, **session) -> SessionAppendRequest:
    """One step per dict, for brevity (`text`; `tool`+`call_id`+`input`; `kind: tool_result`;
    `kind: state`; `kind: turn_end`), sent the way a client sends them: as OpenAI-style messages."""
    messages: list[SessionMessage] = []
    state, turn_end = None, False
    for e in steps:
        kind = e.get("kind", "message")
        if kind == "message":
            messages.append(SessionMessage(id=e["id"], role=e["role"], content=e["text"]))
        elif kind == "tool_call":
            call = {"id": e["call_id"], "function": {"name": e["tool"], "arguments": json.dumps(e["input"])}}
            messages.append(SessionMessage(id=e["id"], role="assistant", tool_calls=[call]))
        elif kind == "tool_result":
            messages.append(
                SessionMessage(
                    id=e["id"],
                    role="tool",
                    tool_call_id=e["call_id"],
                    content=e["text"],
                    is_error=e.get("is_error", False),
                )
            )
        elif kind == "state":
            state = SessionState(**e["data"])
        elif kind == "turn_end":
            turn_end = True
    return SessionAppendRequest(
        messages=messages,
        state=state,
        turn_end=turn_end,
        ack=ack,
        session=SessionOptions(**session) if session else None,
    )


async def _watch_state(memory, bank: str, session: str) -> dict:
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT watch_state FROM sessions WHERE bank_id = $1 AND session_id = $2", bank, session
        )
    return conn.parse_json(row["watch_state"]) if hasattr(conn, "parse_json") else json.loads(row["watch_state"])


def _reflect_stub(answers: list[tuple[str, list[str]]], calls: list[dict]):
    """reflect_async that answers in order; each answer is (text, memory ids)."""

    async def reflect_async(**kwargs):
        calls.append(kwargs)
        text, ids = answers.pop(0) if answers else ("", [])
        return ReflectResult(
            text=text,
            abstained=not text,
            based_on={"observation": [MemoryFact(id=i, text="rule", fact_type="observation") for i in ids]},
            usage=TokenUsage(input_tokens=100, output_tokens=10, total_tokens=110),
            llm_trace=[LLMCallTrace(scope="decision_pick", duration_ms=5, detail="none=0.90")],
        )

    return reflect_async


@pytest.fixture(autouse=True)
def _host_never_busy(monkeypatch):
    """Checks skip on a loaded host (watch.host_busy); the tests must not depend on this machine's load."""
    from hindsight_api.engine.sessions import watch

    monkeypatch.setattr(watch, "BUSY_LOAD_PER_CPU", float("inf"))


async def test_append_is_ordered_and_a_retry_stores_nothing_twice(memory, request_context):
    bank = _bank()
    s = memory.sessions
    r = await s.append(
        bank,
        "s1",
        _req({"id": "a", "role": "user", "text": "hi"}, {"id": "b", "role": "assistant", "text": "hello"}),
        request_context=request_context,
    )
    assert r.notes == [] and not r.checking
    # The response was lost: the client sends the same events again, plus a new one.
    r = await s.append(
        bank,
        "s1",
        _req(
            {"id": "a", "role": "user", "text": "hi"},
            {"id": "b", "role": "assistant", "text": "hello"},
            {"id": "c", "role": "user", "text": "next"},
            {"id": "c", "role": "user", "text": "next"},
        ),
        request_context=request_context,
    )
    log = await s.events(bank, "s1", request_context=request_context)
    assert [(e.seq, e.event_id, e.source) for e in log.items] == [
        (1, "a", "client"),
        (2, "b", "client"),
        (3, "c", "client"),
    ]
    assert [e.seq for e in (await s.events(bank, "s1", after=2, request_context=request_context)).items] == [3]
    summary = await s.get(bank, "s1", request_context=request_context)
    assert summary.event_count == 3 and summary.document_id == "session:s1" and not summary.live


async def test_the_agent_telling_the_user_about_notes_is_counted_once(memory, request_context, monkeypatch):
    """The agent names each note by number: one it used, one it did not need. Each note counts once
    however often it is named or the turn is resent; a number that is not a note of this session,
    and a user quoting the line, do not count."""
    bank = _bank()
    s = memory.sessions
    calls: list[dict] = []
    answers = [
        ("Before you change units.py: K is 1024.", [str(uuid.uuid4())]),
        ("Email is never masked.", [str(uuid.uuid4())]),
    ]
    monkeypatch.setattr(memory, "reflect_async", _reflect_stub(answers, calls))
    ask = {"id": "u", "role": "user", "text": "Fix the unit sizes in units.py"}
    await s.append(bank, "s1", _req(ask, live=True, retain="off"), request_context=request_context)
    edit = (
        {
            "id": "e",
            "role": "assistant",
            "kind": "tool_call",
            "tool": "Edit",
            "call_id": "e",
            "input": {"file_path": "p"},
        },
        {"id": "r", "role": "tool", "kind": "tool_result", "call_id": "e", "text": "ok"},
        {"id": "d", "role": "system", "kind": "state", "data": {"diff": "+++ b/p\n+mask(email)"}},
    )
    await s.append(bank, "s1", _req(*edit), request_context=request_context)
    notes = (await s.events(bank, "s1", request_context=request_context)).items
    used, unneeded = [e.payload["number"] for e in notes if e.kind == "note"][:2]
    reply = (
        f"> 🧠 **From Hindsight memory (live session #{used})** — K is 1024, so I kept it.\n"
        f"> 🧠 **Hindsight note (live session #{unneeded})** — email masking; not needed here.\n"
        f"Again: **From Hindsight memory (live session #{used})** — and #999 is not a note: "
        "**From Hindsight memory (live session #999)**"
    )
    steps = (
        {"id": "q", "role": "user", "text": f"why **From Hindsight memory (live session #{unneeded})**?"},
        {"id": "a", "role": "assistant", "text": reply},
    )
    await s.append(bank, "s1", _req(*steps), request_context=request_context)
    await s.append(bank, "s1", _req(*steps), request_context=request_context)
    summary = await s.get(bank, "s1", request_context=request_context)
    assert (summary.credit_count, summary.dismissed_count) == (1, 1)

    # One line naming two notes credits both; a note first called not needed, then used, counts as used.
    later = {
        "id": "a2",
        "role": "assistant",
        "text": f"> 🧠 **From Hindsight memory (live session #{used} / #{unneeded})** — I masked nothing after all.",
    }
    await s.append(bank, "s1", _req(later), request_context=request_context)
    summary = await s.get(bank, "s1", request_context=request_context)
    assert (summary.credit_count, summary.dismissed_count) == (2, 0)


async def test_thought_lines_are_counted_like_note_lines(memory, request_context, monkeypatch):
    """The coding-agents plugin has the agent name notes as Hindsight thoughts: those count the same."""
    bank = _bank()
    s = memory.sessions
    answers = [
        ("Before you change units.py: K is 1024.", [str(uuid.uuid4())]),
        ("Email is never masked.", [str(uuid.uuid4())]),
    ]
    monkeypatch.setattr(memory, "reflect_async", _reflect_stub(answers, []))
    await s.append(
        bank,
        "s1",
        _req({"id": "u", "role": "user", "text": "Fix the unit sizes in units.py"}, live=True, retain="off"),
        request_context=request_context,
    )
    edit = (
        {
            "id": "e",
            "role": "assistant",
            "kind": "tool_call",
            "tool": "Edit",
            "call_id": "e",
            "input": {"file_path": "p"},
        },
        {"id": "r", "role": "tool", "kind": "tool_result", "call_id": "e", "text": "ok"},
        {"id": "d", "role": "system", "kind": "state", "data": {"diff": "+++ b/p\n+mask(email)"}},
    )
    await s.append(bank, "s1", _req(*edit), request_context=request_context)
    used, unneeded = [
        e.payload["number"]
        for e in (await s.events(bank, "s1", request_context=request_context)).items
        if e.kind == "note"
    ][:2]
    reply = (
        f"> 🧠 **Thanks to Hindsight thought #{used}**, I kept K at 1024.\n"
        f"> 🧠 Hindsight thought #{unneeded} — email masking; no change: no email here."
    )
    await s.append(bank, "s1", _req({"id": "a", "role": "assistant", "text": reply}), request_context=request_context)
    summary = await s.get(bank, "s1", request_context=request_context)
    assert (summary.credit_count, summary.dismissed_count) == (1, 1)
    # The note it set aside is kept so later checks are told not to raise it again.
    from hindsight_api.engine.sessions.watch import WatchState

    state = WatchState.model_validate(await _watch_state(memory, bank, "s1"))
    assert state.set_aside == ["Email is never masked."]
    # Each note carries what the agent made of it, in the agent's own words.
    notes = {
        e.payload["number"]: e.payload
        for e in (await s.events(bank, "s1", request_context=request_context)).items
        if e.kind == "note"
    }
    assert notes[used]["feedback"] == {"verdict": "used", "reason": "I kept K at 1024."}
    assert notes[unneeded]["feedback"] == {"verdict": "not_needed", "reason": "no email here."}


async def test_an_event_carrying_a_nul_is_stored_without_it(memory, request_context):
    """jsonb rejects U+0000: a tool result with one used to fail the whole append (a 500)."""
    bank = _bank()
    s = memory.sessions
    await s.append(
        bank,
        "s1",
        _req({"id": "r", "role": "tool", "kind": "tool_result", "call_id": "x", "text": "bin\x00ary"}),
        request_context=request_context,
    )
    log = (await s.events(bank, "s1", request_context=request_context)).items
    assert [e.payload["text"] for e in log] == ["binary"]


@pytest.mark.parametrize(
    ("sources", "also_gone", "noted"),
    [
        (["session:s1"], False, False),  # only this session's own saved turns: the agent already has them
        (["session:s1", "chat:past"], False, True),  # a real past decision restated in this session: kept
        (["session:s1"], True, False),  # another memory the session's own save just rewrote: still its own
    ],
)
async def test_a_note_built_only_from_this_session_is_not_sent(
    memory, request_context, monkeypatch, sources, also_gone, noted
):
    mem = str(uuid.uuid4())
    used = [mem, str(uuid.uuid4())] if also_gone else [mem]
    monkeypatch.setattr(memory, "reflect_async", _reflect_stub([("Before you change bar.js: total points.", used)], []))
    units = {mem: {"document_id": sources[0], "source_memory_ids": [f"src{i}" for i in range(len(sources))]}}
    units.update({f"src{i}": {"document_id": d} for i, d in enumerate(sources)})

    async def get_memory_unit(bank_id, memory_id, request_context):
        return units.get(memory_id)

    monkeypatch.setattr(memory, "get_memory_unit", get_memory_unit)
    bank, s = _bank(), memory.sessions
    await s.append(
        bank,
        "s1",
        _req({"id": "1", "role": "user", "text": "Fix the points total in bar.js"}, live=True, retain="off"),
        request_context=request_context,
    )
    log = (await s.events(bank, "s1", request_context=request_context)).items
    check = next(e for e in log if e.kind == "check").payload
    assert any(e.kind == "note" for e in log) is noted
    assert check["outcome"] == ("note" if noted else "repeat")
    assert (check.get("reason") == "this session") is (not noted)


async def test_a_check_that_runs_too_long_gives_up_and_the_session_goes_on(memory, request_context, monkeypatch):
    """A note after the agent has moved on is no use: a check past CHECK_TIMEOUT_S is logged as timed out."""
    from hindsight_api.engine.sessions import service as svc

    monkeypatch.setattr(svc, "CHECK_TIMEOUT_S", 0.2)

    async def slow_reflect(**kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(memory, "reflect_async", slow_reflect)
    bank, s = _bank(), memory.sessions
    await s.append(
        bank,
        "s1",
        _req({"id": "1", "role": "user", "text": "Fix the rounding in pay/rounding.py"}, live=True, retain="off"),
        request_context=request_context,
    )
    log = (await s.events(bank, "s1", request_context=request_context)).items
    check = next(e for e in log if e.kind == "check").payload
    assert check["outcome"] == "error" and "timed out" in check["error"]
    assert not any(e.kind == "note" for e in log)


async def test_without_watch_no_check_runs(memory, request_context, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(memory, "reflect_async", _reflect_stub([("note", [])], calls))
    bank = _bank()
    r = await memory.sessions.append(
        bank,
        "s1",
        _req({"id": "a", "role": "user", "text": "Fix the rounding bug in it"}),
        request_context=request_context,
    )
    assert calls == [] and r.notes == [] and not r.checking


async def test_a_note_is_logged_delivered_until_acked_and_never_repeated(memory, request_context, monkeypatch):
    calls: list[dict] = []
    m1 = str(uuid.uuid4())
    answers = [
        ("Before you change pay/rounding.py: half cents round down.", [m1]),  # the heads-up
        ("You used HALF_UP; the rule is HALF_DOWN.", [m1]),  # an edit against it: said once
        ("Still HALF_UP.", [m1]),  # the same memory again: not said twice
    ]
    monkeypatch.setattr(memory, "reflect_async", _reflect_stub(answers, calls))
    bank, s = _bank(), memory.sessions
    r = await s.append(
        bank,
        "s1",
        _req({"id": "1", "role": "user", "text": "Fix the rounding in pay/rounding.py"}, live=True, retain="off"),
        request_context=request_context,
    )
    # The check ran during the append (synchronous backend); the note comes back on the next append.
    assert calls and calls[0]["abstain"] is True and calls[0]["query"].startswith("Before I start")
    assert calls[0]["abstain_at"] == 0.5  # a heads-up abstains sooner than a conflict check
    log = (await s.events(bank, "s1", request_context=request_context)).items
    assert [(e.source, e.kind) for e in log] == [("client", "message"), ("hindsight", "check"), ("hindsight", "note")]
    check = log[1].payload
    assert check["trigger"] == "task" and check["outcome"] == "note" and check["question"] == "headsup"
    assert check["usage"]["input_tokens"] == 100 and check["llm_calls"][0]["scope"] == "decision_pick"

    r = await s.append(bank, "s1", _req(), request_context=request_context)
    assert [n.seq for n in r.notes] == [3] and r.notes[0].text.startswith("Before you change")
    r = await s.append(bank, "s1", _req(ack=3), request_context=request_context)
    assert r.notes == []

    edit = (
        {
            "id": "2",
            "role": "assistant",
            "kind": "tool_call",
            "tool": "Edit",
            "call_id": "e",
            "input": {"file_path": "p"},
        },
        {"id": "3", "role": "tool", "kind": "tool_result", "call_id": "e", "text": "ok"},
        {"id": "4", "role": "system", "kind": "state", "data": {"diff": "+++ b/p\n+HALF_UP"}},
    )
    r = await s.append(bank, "s1", _req(*edit, ack=3), request_context=request_context)
    r = await s.append(bank, "s1", _req(ack=3), request_context=request_context)
    assert [n.text for n in r.notes] == ["You used HALF_UP; the rule is HALF_DOWN."]
    assert "ALREADY TOLD ME:\nBefore you change pay/rounding.py" in calls[1]["context"]
    assert calls[1]["search_queries"] == ["Change to p ():\n+HALF_UP"]
    assert calls[1]["abstain_at"] is None

    edit2 = (
        {
            "id": "5",
            "role": "assistant",
            "kind": "tool_call",
            "tool": "Edit",
            "call_id": "f",
            "input": {"file_path": "p"},
        },
        {"id": "6", "role": "tool", "kind": "tool_result", "call_id": "f", "text": "ok"},
        {"id": "7", "role": "system", "kind": "state", "data": {"diff": "+++ b/p\n+HALF_UP  # still"}},
    )
    await s.append(bank, "s1", _req(*edit2), request_context=request_context)
    outcomes = [
        e.payload["outcome"]
        for e in (await s.events(bank, "s1", request_context=request_context)).items
        if e.kind == "check"
    ]
    assert outcomes == ["note", "note", "repeat"]
    summary = await s.get(bank, "s1", request_context=request_context)
    assert (summary.check_count, summary.note_count) == (3, 2) and not summary.checking
    assert summary.acked_seq == 3  # the client acknowledged the heads-up only


async def test_the_log_records_when_a_note_first_goes_back_to_the_client(memory, request_context, monkeypatch):
    monkeypatch.setattr(memory, "reflect_async", _reflect_stub([("Half cents round down.", [str(uuid.uuid4())])], []))
    bank, s = _bank(), memory.sessions
    await s.append(
        bank,
        "s1",
        _req(
            {"id": "1", "role": "user", "text": "Fix the half cent rounding in pay/rounding.py"},
            live=True,
            retain="off",
        ),
        request_context=request_context,
    )
    # The note (seq 3) goes back with the next append's events, then again on a poll until acked.
    await s.append(
        bank, "s1", _req({"id": "2", "role": "assistant", "text": "Looking."}), request_context=request_context
    )
    r = await s.append(bank, "s1", _req(), request_context=request_context)
    assert [n.seq for n in r.notes] == [3]
    log = (await s.events(bank, "s1", request_context=request_context)).items
    handoffs = [e for e in log if e.kind == "handoff"]
    # Logged once, at the first send, naming the client event whose reply carried it.
    assert len(handoffs) == 1 and handoffs[0].source == "hindsight"
    assert handoffs[0].payload == {"note_seq": 3, "with_seqs": [4]}
    assert handoffs[0].seq == 5


async def test_an_abstained_check_is_logged_silent(memory, request_context, monkeypatch):
    monkeypatch.setattr(memory, "reflect_async", _reflect_stub([("", [])], []))
    bank = _bank()
    r = await memory.sessions.append(
        bank,
        "s1",
        _req({"id": "1", "role": "user", "text": "Fix the rounding bug in it"}, live=True),
        request_context=request_context,
    )
    log = (await memory.sessions.events(bank, "s1", request_context=request_context)).items
    assert [e.kind for e in log] == ["message", "check"] and log[1].payload["outcome"] == "silent"
    assert r.notes == []


async def test_a_failed_check_is_logged_and_the_session_goes_on(memory, request_context, monkeypatch):
    async def broken(**_):
        raise RuntimeError("provider down")

    monkeypatch.setattr(memory, "reflect_async", broken)
    bank = _bank()
    r = await memory.sessions.append(
        bank,
        "s1",
        _req({"id": "1", "role": "user", "text": "Fix the rounding bug in it"}, live=True),
        request_context=request_context,
    )
    log = (await memory.sessions.events(bank, "s1", request_context=request_context)).items
    assert log[-1].payload["outcome"] == "error" and "provider down" in log[-1].payload["error"]
    assert r.notes == [] and log[0].event_id == "1"


async def test_a_turn_end_saves_the_turn_into_the_document_once(memory, request_context, monkeypatch):
    submitted: list[dict] = []

    async def submit_async_retain(bank_id, contents, **kwargs):
        submitted.append({"bank_id": bank_id, "contents": contents, **kwargs})
        return {"operation_id": kwargs.get("operation_id")}

    monkeypatch.setattr(memory, "submit_async_retain", submit_async_retain)
    bank, s = _bank(), memory.sessions
    turn = (
        {"id": "1", "role": "user", "text": "Keep the retries at 7 please"},
        {
            "id": "2",
            "role": "assistant",
            "kind": "tool_call",
            "tool": "Edit",
            "call_id": "e",
            "input": {"file_path": "a.py"},
        },
        {"id": "3", "role": "tool", "kind": "tool_result", "call_id": "e", "text": "done"},
        {"id": "4", "role": "assistant", "text": "Set to 7."},
    )
    await s.append(bank, "s1", _req(*turn, tags=["team:a"], metadata={"repo": "x"}), request_context=request_context)
    assert submitted == []  # no turn end yet: nothing is saved
    await s.append(
        bank, "s1", _req({"id": "5", "role": "assistant", "kind": "turn_end"}), request_context=request_context
    )
    assert len(submitted) == 1
    item = submitted[0]["contents"][0]
    lines = [json.loads(line) for line in item["content"].split("\n")]
    assert [(t["role"], t["content"]) for t in lines] == [
        ("system", "REF-ID: session:s1"),
        ("user", "Keep the retries at 7 please"),
        ("action", "Edit a.py"),
        ("assistant", "Set to 7."),
    ]
    assert item["document_id"] == "session:s1" and item["update_mode"] == "append"
    assert item["tags"] == ["team:a"] and item["metadata"]["repo"] == "x" and item["metadata"]["session_id"] == "s1"
    summary = await s.get(bank, "s1", request_context=request_context)
    assert summary.saved_seq == 5

    # The next turn appends only what is new, under a new operation id.
    await s.append(
        bank,
        "s1",
        _req(
            {"id": "6", "role": "user", "text": "And what about the request timeout?"},
            {"id": "7", "role": "assistant", "kind": "turn_end"},
        ),
        request_context=request_context,
    )
    assert len(submitted) == 2
    second = [json.loads(line) for line in submitted[1]["contents"][0]["content"].split("\n")]
    assert [t["content"] for t in second] == ["And what about the request timeout?"]
    assert submitted[1]["operation_id"] != submitted[0]["operation_id"]


async def test_a_long_turn_is_saved_without_waiting_for_its_end(memory, request_context, monkeypatch):
    """A turn that keeps going past SAVE_EVERY_EVENTS events is saved anyway, once per stretch."""
    from hindsight_api.engine.sessions import service as svc

    monkeypatch.setattr(svc, "SAVE_EVERY_EVENTS", 4)
    submitted: list[dict] = []

    async def submit_async_retain(bank_id, contents, **kwargs):
        submitted.append(kwargs)
        return {"operation_id": kwargs.get("operation_id")}

    monkeypatch.setattr(memory, "submit_async_retain", submit_async_retain)
    bank, s = _bank(), memory.sessions
    say = lambda i: {"id": str(i), "role": "assistant", "text": f"step {i}"}  # noqa: E731
    await s.append(bank, "s1", _req(*(say(i) for i in range(3))), request_context=request_context)
    assert submitted == []  # 3 events: under the threshold, and no turn end
    await s.append(bank, "s1", _req(say(3)), request_context=request_context)
    assert len(submitted) == 1  # 4 unsaved: saved mid-turn
    await s.append(bank, "s1", _req(say(4)), request_context=request_context)
    assert len(submitted) == 1  # 1 unsaved since: waits for the next stretch or the turn end


async def test_a_save_that_died_after_submitting_resubmits_the_same_operation(memory, request_context, monkeypatch):
    submitted: list[str | None] = []
    fail = [True]

    async def submit_async_retain(bank_id, contents, **kwargs):
        submitted.append(kwargs.get("operation_id"))
        if fail:
            fail.pop()
            raise RuntimeError("worker died")
        return {}

    monkeypatch.setattr(memory, "submit_async_retain", submit_async_retain)
    bank, s = _bank(), memory.sessions
    await s.append(
        bank, "s1", _req({"id": "1", "role": "user", "text": "hello"}, retain="idle"), request_context=request_context
    )
    with pytest.raises(RuntimeError):
        await s.run_save({"bank_id": bank, "session_id": "s1"})
    # More arrived before the retry: the retried save still covers exactly the planned range.
    await s.append(bank, "s1", _req({"id": "2", "role": "user", "text": "later"}), request_context=request_context)
    await s.run_save({"bank_id": bank, "session_id": "s1"})
    assert len(submitted) == 2 and submitted[0] == submitted[1]
    assert (await s.get(bank, "s1", request_context=request_context)).saved_seq == 1
    # The next save picks up from there.
    await s.run_save({"bank_id": bank, "session_id": "s1"})
    assert len(submitted) == 3 and submitted[2] != submitted[0]


async def test_an_idle_session_saves_what_is_unsaved_and_retain_off_never_does(memory, request_context, monkeypatch):
    """There is no close: a session that went quiet (finished, or its client died mid-turn) is idle,
    and its turns not saved yet are saved then. `idle` saves only then; `off` never."""
    from types import SimpleNamespace

    from hindsight_api.engine.sessions import service
    from hindsight_api.worker.exceptions import DeferOperation

    submitted: list[dict] = []

    async def submit_async_retain(bank_id, contents, **kwargs):
        submitted.append(contents[0])
        return {}

    monkeypatch.setattr(memory, "submit_async_retain", submit_async_retain)
    bank, s = _bank(), memory.sessions
    await s.append(
        bank, "a", _req({"id": "1", "role": "user", "text": "hello"}, retain="idle"), request_context=request_context
    )
    await s.append(
        bank, "a", _req({"id": "2", "role": "assistant", "kind": "turn_end"}), request_context=request_context
    )
    assert submitted == []  # `idle`: a turn end does not save
    # On a real queue the idle timer goes back for later: the session only just moved.
    monkeypatch.setattr(memory._task_backend, "can_defer", True)
    with pytest.raises(DeferOperation) as deferred:
        await s.run_idle({"bank_id": bank, "session_id": "a"})
    summary = await s.get(bank, "a", request_context=request_context)
    wait = (deferred.value.exec_date - summary.last_event_at).total_seconds()
    assert wait == 600 and submitted == []  # the default idle window, from the last event
    # Once idle, it queues the save (it runs at once on this test backend).
    monkeypatch.setattr(service, "get_config", lambda: SimpleNamespace(session_idle_ms=0))
    await s.run_idle({"bank_id": bank, "session_id": "a"})
    assert len(submitted) == 1 and "hello" in submitted[0]["content"]
    # Saved: a later timer finds nothing to do.
    await s.run_idle({"bank_id": bank, "session_id": "a"})
    assert len(submitted) == 1
    await s.append(
        bank, "b", _req({"id": "1", "role": "user", "text": "hello"}, retain="off"), request_context=request_context
    )
    await s.append(
        bank, "b", _req({"id": "2", "role": "assistant", "kind": "turn_end"}), request_context=request_context
    )
    assert len(submitted) == 1


async def test_sessions_are_bank_scoped_and_listed_newest_first(memory, request_context):
    a, b, s = _bank(), _bank(), memory.sessions
    await s.append(a, "same", _req({"id": "1", "role": "user", "text": "in a"}), request_context=request_context)
    await s.append(a, "other", _req({"id": "1", "role": "user", "text": "in a"}), request_context=request_context)
    await s.append(b, "same", _req({"id": "1", "role": "user", "text": "in b"}), request_context=request_context)
    listing = await s.list(a, request_context=request_context)
    assert listing.total == 2 and [x.session_id for x in listing.items] == ["other", "same"]
    assert [x.session_id for x in (await s.list(a, q="SAM", request_context=request_context)).items] == ["same"]
    in_b = (await s.events(b, "same", request_context=request_context)).items
    assert [e.payload["text"] for e in in_b] == ["in b"]


async def test_http_routes(api_client):
    bank = _bank()
    r = await api_client.post(
        f"/v1/default/banks/{bank}/sessions/s1/events",
        json={"messages": [{"id": "1", "role": "user", "content": "hi"}], "session": {"metadata": {"harness": "cli"}}},
    )
    assert r.status_code == 200 and r.json() == {"notes": [], "checking": False}
    r = await api_client.get(f"/v1/default/banks/{bank}/sessions/s1/events", params={"after": 0})
    assert r.status_code == 200 and r.json()["items"][0]["payload"]["text"] == "hi"
    assert (await api_client.get(f"/v1/default/banks/{bank}/sessions")).json()["total"] == 1
    assert (await api_client.get(f"/v1/default/banks/{bank}/sessions/s1")).json()["metadata"] == {"harness": "cli"}
    assert (await api_client.get(f"/v1/default/banks/{bank}/sessions/nope/events")).status_code == 404
    assert (await api_client.get(f"/v1/default/banks/{bank}/sessions/nope")).status_code == 404
    bad = await api_client.post(f"/v1/default/banks/{bank}/sessions/s1/events", json={"messages": [{"role": "user"}]})
    assert bad.status_code == 422


async def test_a_check_waits_for_the_session_to_settle_on_a_queue(memory, request_context, monkeypatch):
    from hindsight_api.worker.exceptions import DeferOperation

    calls: list[dict] = []
    monkeypatch.setattr(memory, "reflect_async", _reflect_stub([("", [])], calls))
    queued: list[str] = []

    async def queue_check(bank_id, session_id, rc):
        queued.append(session_id)

    monkeypatch.setattr(memory.sessions, "_queue_check", queue_check)
    bank = _bank()
    await memory.sessions.append(
        bank,
        "s1",
        _req({"id": "1", "role": "user", "text": "Fix the rounding bug in it"}, live=True),
        request_context=request_context,
    )
    assert queued == ["s1"] and calls == []
    # On a real queue the check goes back for later: the session only just moved.
    monkeypatch.setattr(memory._task_backend, "can_defer", True)
    with pytest.raises(DeferOperation) as deferred:
        await memory.sessions.run_check({"bank_id": bank, "session_id": "s1"})
    summary = await memory.sessions.get(bank, "s1", request_context=request_context)
    wait = (deferred.value.exec_date - summary.last_event_at).total_seconds()
    assert 0.5 < wait <= 1.0 and calls == []  # quiet window, by the database's clock
    # Once it has settled, it runs.
    monkeypatch.setenv("HINDSIGHT_API_SESSION_CHECK_QUIET_MS", "0")
    from hindsight_api.config import clear_config_cache

    clear_config_cache()
    await memory.sessions.run_check({"bank_id": bank, "session_id": "s1"})
    assert len(calls) == 1


async def test_a_check_waits_out_a_busy_host_instead_of_being_dropped(memory, request_context, monkeypatch):
    """Skipped as "host busy", most of an hour's checks were lost on a loaded laptop (beta)."""
    from hindsight_api.engine.sessions import watch
    from hindsight_api.worker.exceptions import DeferOperation

    calls: list[dict] = []
    monkeypatch.setattr(memory, "reflect_async", _reflect_stub([("", [])], calls))

    async def queue_check(bank_id, session_id, rc):
        pass

    monkeypatch.setattr(memory.sessions, "_queue_check", queue_check)
    monkeypatch.setenv("HINDSIGHT_API_SESSION_CHECK_QUIET_MS", "0")
    from hindsight_api.config import clear_config_cache

    clear_config_cache()
    bank = _bank()
    ask = {"id": "1", "role": "user", "text": "Fix the rounding bug in money.py"}
    await memory.sessions.append(bank, "s1", _req(ask, live=True), request_context=request_context)
    monkeypatch.setattr(memory._task_backend, "can_defer", True)
    monkeypatch.setattr(watch, "host_busy", lambda: True)
    with pytest.raises(DeferOperation) as deferred:
        await memory.sessions.run_check({"bank_id": bank, "session_id": "s1"})
    assert "settle" in deferred.value.reason and calls == []
    # The load never drops: once the moment has waited the cap, it is checked anyway (beta: two hours
    # of a load that stayed high ran no check at all).
    monkeypatch.setattr(watch, "BUSY_HOST_MAX_WAIT_S", 0)
    await memory.sessions.run_check({"bank_id": bank, "session_id": "s1"})
    assert len(calls) == 1
    log = (await memory.sessions.events(bank, "s1", request_context=request_context)).items
    assert not [e for e in log if e.kind == "check" and e.payload.get("reason") == "host busy"]


async def test_delete_removes_the_session_and_its_log(memory, request_context, api_client):
    bank = _bank()
    await memory.sessions.append(
        bank, "s1", _req({"id": "1", "role": "user", "text": "hi"}), request_context=request_context
    )
    r = await api_client.delete(f"/v1/default/banks/{bank}/sessions/s1")
    assert r.status_code == 200
    assert (await api_client.get(f"/v1/default/banks/{bank}/sessions/s1/events")).status_code == 404
    assert (await api_client.delete(f"/v1/default/banks/{bank}/sessions/s1")).status_code == 404


async def test_every_session_endpoint_is_validated_and_a_tag_scoped_caller_is_refused(memory, api_client, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from hindsight_api.engine.search.tags import TagGroupLeaf
    from hindsight_api.extensions import ValidationResult

    seen: list[str] = []

    async def record(ctx):
        seen.append(str(ctx.operation))
        return ValidationResult.accept()

    validator = MagicMock()
    validator.validate_bank_read = AsyncMock(side_effect=record)
    validator.validate_bank_write = AsyncMock(side_effect=record)
    validator.validate_create_bank = AsyncMock(return_value=ValidationResult.accept())
    validator.resolve_tag_scope = AsyncMock(return_value=None)
    validator.resolve_write_tag_scope = AsyncMock(return_value=None)
    monkeypatch.setattr(memory, "_operation_validator", validator)

    async def no_retain(*_, **__):
        return {}

    monkeypatch.setattr(memory, "submit_async_retain", no_retain)
    bank, base = _bank(), None
    base = f"/v1/default/banks/{bank}/sessions"
    assert (
        await api_client.post(f"{base}/s1/events", json={"messages": [{"id": "1", "role": "user", "content": "hi"}]})
    ).status_code == 200
    assert (await api_client.get(f"{base}/s1/events")).status_code == 200
    assert (await api_client.get(base)).status_code == 200
    assert (await api_client.get(f"{base}/s1")).status_code == 200
    assert seen == [
        "append_session_events",
        "list_session_events",
        "list_sessions",
        "get_session",
    ]
    # A caller confined to some tags would see the whole bank through the watcher's notes.
    validator.resolve_tag_scope = AsyncMock(return_value=[TagGroupLeaf(tags=["user:dan"], match="any_strict")])
    refused = await api_client.post(
        f"{base}/s1/events", json={"messages": [{"id": "2", "role": "user", "content": "x"}]}
    )
    assert refused.status_code == 403 and "tag-scoped caller" in refused.json()["detail"]
    assert (await api_client.get(f"{base}/s1/events")).status_code == 403


async def test_openai_style_messages_become_the_log_and_a_turn_end_is_stored_once(api_client):
    """A client sends the messages it keeps as they are: content parts, tool calls with their arguments
    as a JSON string, tool messages, extra keys it does not know about."""
    bank = _bank()
    url = f"/v1/default/banks/{bank}/sessions/s1/events"
    body = {
        "messages": [
            {
                "id": "m1",
                "role": "user",
                "content": [{"type": "text", "text": "Fix the rounding of half cents"}],
                "name": "dan",
            },
            {
                "id": "m2",
                "role": "assistant",
                "content": "Looking at it.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"command": "ls"}'},
                    },
                    {"id": "c2", "type": "function", "function": {"name": "write_file", "arguments": "not json"}},
                ],
            },
            {"id": "m3", "role": "tool", "tool_call_id": "c1", "content": "pay.py"},
        ],
        "state": {"diff": "+x"},
        "turn_end": True,
    }
    r = await api_client.post(url, json=body)
    assert r.status_code == 200
    items = [e for e in (await api_client.get(url)).json()["items"] if e["source"] == "client"]
    assert [(e["event_id"], e["kind"]) for e in items] == [
        ("m1", "message"),
        ("m2", "message"),
        ("m2:c1", "tool_call"),
        ("m2:c2", "tool_call"),
        ("m3", "tool_result"),
        ("state:" + items[5]["event_id"][6:], "state"),
        ("turn_end:m3", "turn_end"),
    ]
    assert items[0]["payload"] == {"role": "user", "text": "Fix the rounding of half cents"}
    assert items[2]["payload"]["input"] == {"command": "ls"}
    assert items[3]["payload"]["input"] == {"arguments": "not json"}
    # Retried whole, then a turn end alone: nothing new happened, so nothing is stored.
    await api_client.post(url, json=body)
    await api_client.post(url, json={"turn_end": True})
    assert len([e for e in (await api_client.get(url)).json()["items"] if e["source"] == "client"]) == 7


async def test_a_note_is_quoted_from_the_source_that_states_its_rule(memory, request_context, monkeypatch):
    """The newest source document is not always the one stating the rule: quote the best one."""
    units = {
        "obs": {"source_memory_ids": ["new", "old"]},
        "new": {"document_id": "doc-new", "mentioned_at": "2026-10-23T10:00:00Z"},
        "old": {"document_id": "doc-old", "mentioned_at": "2026-10-20T10:00:00Z"},
    }
    docs = {
        "doc-new": "The bucket is not created by CI, so each environment needs a manual step.",
        "doc-old": "Each query creates a fresh query session, with one query handled at a time.",
    }

    async def get_memory_unit(bank_id, memory_id, request_context):
        return units.get(memory_id)

    async def get_document(document_id, bank_id, request_context):
        return {"original_text": docs[document_id]}

    monkeypatch.setattr(memory, "get_memory_unit", get_memory_unit)
    monkeypatch.setattr(memory, "get_document", get_document)
    src = await memory.sessions._source_of(
        _bank(), ["obs"], "Revert the shared session: each query creates a fresh query session.", request_context
    )
    assert src is not None and src.document_id == "doc-old" and src.quote.startswith("Each query creates")


async def test_a_note_whose_memories_are_all_gone_is_judged_by_its_quoted_source(memory, request_context, monkeypatch):
    """The session's own save rewrote the note's only memory: its quote, the session's own, decides."""
    bank, s = _bank(), memory.sessions
    await s.append(
        bank,
        "s1",
        _req({"id": "1", "role": "user", "text": "Pin the new release build"}, retain="off"),
        request_context=request_context,
    )

    async def gone(bank_id, memory_id, request_context):
        return None

    monkeypatch.setattr(memory, "get_memory_unit", gone)
    own_doc = (await s.get(bank, "s1", request_context=request_context)).document_id
    assert await s._from_this_session_only(bank, "s1", ["m1"], request_context, own_doc)
    assert not await s._from_this_session_only(bank, "s1", ["m1"], request_context, "chat:past")
    assert not await s._from_this_session_only(bank, "s1", ["m1"], request_context, None)


async def test_a_note_from_a_mixed_memory_is_own_when_this_session_states_it_best(memory, request_context, monkeypatch):
    """An observation merged an older session with this one: the note's quoted source decides."""
    bank, s = _bank(), memory.sessions
    await s.append(
        bank,
        "s1",
        _req({"id": "1", "role": "user", "text": "Create the prod analytics bucket"}, retain="off"),
        request_context=request_context,
    )
    own_doc = (await s.get(bank, "s1", request_context=request_context)).document_id
    units = {
        "obs": {"document_id": "chat:past", "source_memory_ids": ["old", "mine"]},
        "old": {"document_id": "chat:past"},
        "mine": {"document_id": own_doc},
    }

    async def get_memory_unit(bank_id, memory_id, request_context):
        return units.get(memory_id)

    monkeypatch.setattr(memory, "get_memory_unit", get_memory_unit)
    assert await s._from_this_session_only(bank, "s1", ["obs"], request_context, own_doc)
    assert not await s._from_this_session_only(bank, "s1", ["obs"], request_context, "chat:past")


async def test_a_rerun_of_the_same_request_inherits_what_earlier_runs_set_aside(memory, request_context):
    """A scheduled job's runs are separate sessions with one request: what one run set aside, the next is told."""
    from hindsight_api.engine.sessions.watch import WatchState

    bank, s = _bank(), memory.sessions
    # Each run's request names its own run folder: only the numbers differ.
    job = "You are running the scheduled job 'disk-guard'. Read JOB.md and do what it says. Write runs/{}/summary.md."
    await s.append(
        bank,
        "run1",
        _req({"id": "1", "role": "user", "text": job.format("20261031-025859.367")}, live=True, retain="off"),
        request_context=request_context,
    )
    backend = await memory._get_backend()
    async with backend.acquire() as conn:  # an earlier run that set a note aside
        await conn.execute(
            "UPDATE sessions SET watch_state = jsonb_set(watch_state, '{set_aside}', '[\"The worktrees folder is outside the rules.\"]') WHERE bank_id = $1 AND session_id = 'run1'",
            bank,
        )
    await s.append(
        bank,
        "run2",
        _req({"id": "1", "role": "user", "text": job.format("20261031-040756.139")}, live=True, retain="off"),
        request_context=request_context,
    )
    state = WatchState.model_validate(await _watch_state(memory, bank, "run2"))
    assert state.set_aside == ["The worktrees folder is outside the rules."]
    await s.append(
        bank,
        "run3",
        _req({"id": "1", "role": "user", "text": "go on"}, live=True, retain="off"),
        request_context=request_context,
    )
    assert WatchState.model_validate(await _watch_state(memory, bank, "run3")).set_aside == []


@pytest.mark.parametrize(
    ("line", "verdict", "reason"),
    [
        ("> 🧠 **Thanks to Hindsight thought #7**, I kept K at 1024.", "used", "I kept K at 1024."),
        (
            "> 🧠 Hindsight thought #7 — K is 1024; no change: I already used 1024.",
            "not_needed",
            "I already used 1024.",
        ),
        (
            "> 🧠 Hindsight thought #7 — K is 1000; wrong: the commit it cites sets 1024.",
            "wrong",
            "the commit it cites sets 1024.",
        ),
        ("> 🧠 Hindsight thought #7 — K is 1024", "not_needed", "K is 1024"),
        # The shape an agent wrote in beta: a colon, and a sentence before "No change:".
        (
            "> 🧠 Hindsight thought #7: K is 1024. No change: the fix already uses it.",
            "not_needed",
            "the fix already uses it.",
        ),
    ],
)
def test_the_line_an_agent_writes_about_a_note_gives_its_verdict(line, verdict, reason):
    from hindsight_api.engine.sessions.service import note_feedback

    fb = note_feedback(line)[7]
    assert (fb.verdict, fb.reason) == (verdict, reason)


def test_used_wins_when_a_message_names_a_note_twice():
    from hindsight_api.engine.sessions.service import note_feedback

    text = "> 🧠 Hindsight thought #3 — x; no change: done\n> 🧠 **Thanks to Hindsight thought #3**, fixed y."
    assert note_feedback(text)[3].verdict == "used"


async def test_notes_are_numbered_across_the_bank_and_found_by_number(memory, request_context, monkeypatch):
    """A note's number names one note in the bank, and "#N" finds the session that got it."""
    bank = _bank()
    s = memory.sessions
    monkeypatch.setattr(
        memory,
        "reflect_async",
        _reflect_stub([("Before you change units.py: K is 1024.", [str(uuid.uuid4())])] * 2, []),
    )
    for sid in ("s1", "s2"):
        await s.append(
            bank,
            sid,
            _req({"id": "u", "role": "user", "text": "Fix the unit sizes in units.py"}, live=True, retain="off"),
            request_context=request_context,
        )
    numbers = {
        sid: [
            e.payload["number"]
            for e in (await s.events(bank, sid, request_context=request_context)).items
            if e.kind == "note"
        ]
        for sid in ("s1", "s2")
    }
    # Unique in the bank, and above every note's seq, so no older note shares a number.
    first, second = sorted(numbers["s1"] + numbers["s2"])
    assert second == first + 1
    newest = "s1" if numbers["s1"] == [second] else "s2"
    found = await s.list(bank, q=f"#{second}", request_context=request_context)
    assert [x.session_id for x in found.items] == [newest]


class _Pick:
    def __init__(self, score: float):
        self.none = 1.0 - score
        self.candidates = [score]


class _LearnStub:
    """The decision model and the writer, as the learn step calls them."""

    def __init__(self, score: float):
        self.score = score
        self.questions: list[str] = []

    async def pick_answering(self, question, context, texts):
        self.questions.append(question)
        return _Pick(self.score)


async def _session_with_a_verdict(memory, request_context, monkeypatch, bank: str, reply_line: str):
    """A live session whose one note the agent then answers with ``reply_line`` (``{n}`` = its number)."""
    s = memory.sessions
    monkeypatch.setattr(
        memory,
        "reflect_async",
        _reflect_stub([("Before you change panels.tsx: keep three panels.", [str(uuid.uuid4())])], []),
    )
    await s.append(
        bank,
        "s1",
        _req(
            {"id": "u", "role": "user", "text": "Merge the three panels in panels.tsx into one"},
            live=True,
            retain="off",
        ),
        request_context=request_context,
    )
    number = [
        e.payload["number"]
        for e in (await s.events(bank, "s1", request_context=request_context)).items
        if e.kind == "note"
    ][0]
    await s.append(
        bank,
        "s1",
        _req({"id": "a", "role": "assistant", "text": reply_line.format(n=number)}),
        request_context=request_context,
    )
    return [e for e in (await s.events(bank, "s1", request_context=request_context)).items if e.kind == "learn"]


def _learning_on(memory, monkeypatch, on: bool = True):
    real = memory._config_resolver.get_bank_config

    async def get_bank_config(bank_id, context=None, **kw):
        cfg = dict(await real(bank_id, context, **kw))
        cfg["session_learn_rules"] = on
        return cfg

    monkeypatch.setattr(memory._config_resolver, "get_bank_config", get_bank_config)


async def test_a_verdict_named_again_in_other_words_is_weighed_once(memory, request_context, monkeypatch):
    bank = _bank()
    _learning_on(memory, monkeypatch)
    stub = _LearnStub(0.1)
    monkeypatch.setattr(memory, "_reflect_decision_model", lambda: stub)
    await _session_with_a_verdict(
        memory, request_context, monkeypatch, bank, "> 🧠 Hindsight thought #{n} — keep three panels; no change: done."
    )
    number = [
        e.payload["number"]
        for e in (await memory.sessions.events(bank, "s1", request_context=request_context)).items
        if e.kind == "note"
    ][0]
    again = f"> 🧠 Hindsight thought #{number} — keep three panels; no change: I had already merged them."
    await memory.sessions.append(
        bank, "s1", _req({"id": "b", "role": "assistant", "text": again}), request_context=request_context
    )
    log = (await memory.sessions.events(bank, "s1", request_context=request_context)).items
    assert len([e for e in log if e.kind == "learn"]) == 1 and len(stub.questions) == 1
    # The note keeps the latest wording.
    (note,) = [e for e in log if e.kind == "note"]
    assert note.payload["feedback"]["reason"] == "I had already merged them."


async def test_a_verdict_the_decision_model_failed_on_is_weighed_again_later(memory, request_context, monkeypatch):
    """A dev gateway answering 429 skipped a verdict for good; verdicts are rare (beta)."""
    bank = _bank()
    _learning_on(memory, monkeypatch)

    class _Busy:
        async def pick_answering(self, question, context, texts):
            raise RuntimeError("HTTP 429: the decision model is busy or still loading")

    monkeypatch.setattr(memory, "_reflect_decision_model", lambda: _Busy())
    learned = await _session_with_a_verdict(
        memory, request_context, monkeypatch, bank, "> 🧠 Hindsight thought #{n} — keep three panels; no change: done."
    )
    assert learned == [], "nothing logged as skipped: the verdict waits"
    # The model is back: the same verdict is weighed.
    stub = _LearnStub(0.1)
    monkeypatch.setattr(memory, "_reflect_decision_model", lambda: stub)
    await memory.sessions.run_learn({"bank_id": bank, "session_id": "s1"})
    log = (await memory.sessions.events(bank, "s1", request_context=request_context)).items
    assert [e.payload["outcome"] for e in log if e.kind == "learn"] == ["no_rule"] and len(stub.questions) == 1


async def test_a_lasting_verdict_becomes_a_rule_every_later_check_reads(memory, request_context, monkeypatch):
    from hindsight_api.engine.response_models import LLMCallResult
    from hindsight_api.engine.sessions.learn import RuleDraft

    bank = _bank()
    _learning_on(memory, monkeypatch)
    stub = _LearnStub(0.9)
    monkeypatch.setattr(memory, "_reflect_decision_model", lambda: stub)
    prompts: list[str] = []

    async def call(messages, **kw):
        prompts.append(messages[0]["content"])
        draft = RuleDraft(
            create=True,
            name="Panels: one panel, not three",
            query="How many panels does the shell show?",
        )
        return LLMCallResult(content=draft)

    monkeypatch.setattr(memory._llm_config, "call", call)
    learned = await _session_with_a_verdict(
        memory,
        request_context,
        monkeypatch,
        bank,
        "> 🧠 Hindsight thought #{n} — keep three panels; wrong: you asked to merge them into one.",
    )
    assert [e.payload["outcome"] for e in learned] == ["created"]
    rule_id = learned[0].payload["mental_model_id"]
    assert rule_id.startswith("rule-")
    # The writer saw the agent's reason but not the note it rejected (it would copy the note back),
    # and the rule is a mental model reflect searches first.
    assert "you asked to merge them into one" in prompts[0]
    assert "keep three panels" not in prompts[0]
    models = (await memory.list_mental_models(bank, detail="content", request_context=request_context)).items
    assert [m["id"] for m in models if m["id"] == rule_id] == [rule_id]
    # A later session's check searches the rule, and only the rule: the bank's other mental models
    # (long pages written for people) stay out of checks. Before, checks excluded every mental
    # model, and a learned rule never reached a later session.
    await memory.create_mental_model(
        bank, "Shell overview", "What is the shell?", "A page.", request_context=request_context
    )
    other = [
        m["id"]
        for m in (await memory.list_mental_models(bank, detail="metadata", request_context=request_context)).items
        if m["id"] != rule_id
    ]
    calls: list[dict] = []
    monkeypatch.setattr(memory, "reflect_async", _reflect_stub([], calls))
    ask = {"id": "u", "role": "user", "text": "Add a Logs page to the shell in shell.tsx"}
    await memory.sessions.append(bank, "s2", _req(ask, live=True, retain="off"), request_context=request_context)
    assert calls and calls[0]["exclude_mental_models"] is False
    assert calls[0]["exclude_mental_model_ids"] == other


async def test_a_bank_without_rules_keeps_mental_models_out_of_checks(memory, request_context, monkeypatch):
    bank = _bank()
    await memory.create_mental_model(
        bank, "Shell overview", "What is the shell?", "A page.", request_context=request_context
    )
    calls: list[dict] = []
    monkeypatch.setattr(memory, "reflect_async", _reflect_stub([], calls))
    ask = {"id": "u", "role": "user", "text": "Add a Logs page to the shell in shell.tsx"}
    await memory.sessions.append(bank, "s1", _req(ask, live=True, retain="off"), request_context=request_context)
    assert calls and calls[0]["exclude_mental_models"] is True


@pytest.mark.parametrize(
    "verdict, retired",
    [
        ("wrong: tests use whatever port is free.", True),
        # Live, a correct page was retired by two checks that picked it for unrelated work (2026-11-13).
        ("no change: I used my own free port.", False),
    ],
)
async def test_a_rule_whose_notes_are_called_wrong_twice_is_retired_not_deleted(
    memory, request_context, monkeypatch, verdict, retired
):
    """Live, a rule learned from one run's free port came back to every test run and was rejected each time."""
    from hindsight_api.engine.sessions.learn import RETIRED_TAG

    bank = _bank()
    await memory.ensure_bank_profile(bank, request_context=request_context)
    await memory.create_mental_model(
        bank,
        "Tests: port 5617",
        "Which port?",
        "Tests must use port 5617.",
        mental_model_id="rule-port",
        request_context=request_context,
    )
    _learning_on(memory, monkeypatch)
    monkeypatch.setattr(memory, "_reflect_decision_model", lambda: _LearnStub(0.05))
    s = memory.sessions
    # Two different requests: a rerun of the same request inherits what the first run set aside.
    for session, request in (
        ("s1", "Run the database tests in conftest.py"),
        ("s2", "Fix the flaky queue test in conftest.py"),
    ):
        monkeypatch.setattr(
            memory,
            "reflect_async",
            _reflect_stub([("Before you change conftest.py: use port 5617.", ["rule-port"])], []),
        )
        ask = {"id": "u", "role": "user", "text": request}
        await s.append(bank, session, _req(ask, live=True, retain="off"), request_context=request_context)
        log = (await s.events(bank, session, request_context=request_context)).items
        notes = [e for e in log if e.kind == "note"]
        assert len(notes) == 1, (session, [(e.kind, e.payload) for e in log])
        (note,) = notes
        reply = f"> 🧠 Hindsight thought #{note.payload['number']} — use 5617; {verdict}"
        await s.append(
            bank, session, _req({"id": "a", "role": "assistant", "text": reply}), request_context=request_context
        )
    learned = [
        e.payload for e in (await s.events(bank, "s2", request_context=request_context)).items if e.kind == "learn"
    ]
    assert [x["outcome"] for x in learned if x["outcome"] == "retired"] == (["retired"] if retired else [])
    rule = await memory.get_mental_model(bank, "rule-port", request_context=request_context)
    assert rule is not None and (RETIRED_TAG in rule["tags"]) is retired, "tagged, not deleted"
    if not retired:
        return
    # The next check reads no rule: the bank's only rule is retired.
    calls: list[dict] = []
    monkeypatch.setattr(memory, "reflect_async", _reflect_stub([], calls))
    ask = {"id": "u", "role": "user", "text": "Run the database tests in conftest.py again"}
    await s.append(bank, "s3", _req(ask, live=True, retain="off"), request_context=request_context)
    assert calls and calls[0]["exclude_mental_models"] is True


async def test_a_verdict_that_settles_nothing_writes_no_rule(memory, request_context, monkeypatch):
    bank = _bank()
    _learning_on(memory, monkeypatch)
    monkeypatch.setattr(memory, "_reflect_decision_model", lambda: _LearnStub(0.1))

    async def call(messages, **kw):  # never reached: the decision model said no rule
        raise AssertionError("the writer must not run")

    monkeypatch.setattr(memory._llm_config, "call", call)
    learned = await _session_with_a_verdict(
        memory,
        request_context,
        monkeypatch,
        bank,
        "> 🧠 Hindsight thought #{n} — keep three panels; no change: I was already doing it.",
    )
    assert [e.payload["outcome"] for e in learned] == ["no_rule"]
    models = (await memory.list_mental_models(bank, detail="metadata", request_context=request_context)).items
    assert not [m for m in models if str(m["id"]).startswith("rule-")]


async def test_learning_off_weighs_nothing(memory, request_context, monkeypatch):
    bank = _bank()
    _learning_on(memory, monkeypatch, on=False)
    stub = _LearnStub(0.9)
    monkeypatch.setattr(memory, "_reflect_decision_model", lambda: stub)
    learned = await _session_with_a_verdict(
        memory,
        request_context,
        monkeypatch,
        bank,
        "> 🧠 Hindsight thought #{n} — keep three panels; wrong: you asked to merge them into one.",
    )
    assert learned == [] and stub.questions == []


async def test_a_rule_that_replaces_an_older_one_removes_it(memory, request_context, monkeypatch):
    from hindsight_api.engine.response_models import LLMCallResult
    from hindsight_api.engine.sessions.learn import RuleDraft

    bank = _bank()
    _learning_on(memory, monkeypatch)
    monkeypatch.setattr(memory, "_reflect_decision_model", lambda: _LearnStub(0.9))
    await memory.create_mental_model(
        bank,
        "Decision limit per pod",
        "What is the decision limit?",
        "2 decision calls per pod.",
        mental_model_id="rule-old",
        request_context=request_context,
    )

    async def call(messages, **kw):
        return LLMCallResult(
            content=RuleDraft(
                create=True,
                name="Decision limit: 8 API, 4 workers",
                query="What is the decision limit?",
                replaces=["Decision limit per pod"],
            )
        )

    monkeypatch.setattr(memory._llm_config, "call", call)
    learned = await _session_with_a_verdict(
        memory,
        request_context,
        monkeypatch,
        bank,
        "> 🧠 Hindsight thought #{n} — 2 per pod; wrong: #1643 made it 8 for API pods and 4 for workers.",
    )
    assert learned[0].payload["replaced"] == ["rule-old"]
    ids = [
        m["id"]
        for m in (await memory.list_mental_models(bank, detail="metadata", request_context=request_context)).items
    ]
    assert "rule-old" not in ids and learned[0].payload["mental_model_id"] in ids


async def test_a_rule_carrying_a_credential_is_not_written(memory, request_context, monkeypatch):
    from hindsight_api.engine.response_models import LLMCallResult
    from hindsight_api.engine.sessions.learn import RuleDraft

    bank = _bank()
    _learning_on(memory, monkeypatch)
    monkeypatch.setattr(memory, "_reflect_decision_model", lambda: _LearnStub(0.9))

    async def call(messages, **kw):
        return LLMCallResult(
            content=RuleDraft(
                create=True, name="Sheepit login", query="Sheepit sign-in: the password is hunter2x9, still?"
            )
        )

    monkeypatch.setattr(memory._llm_config, "call", call)
    learned = await _session_with_a_verdict(
        memory,
        request_context,
        monkeypatch,
        bank,
        "> 🧠 **Thanks to Hindsight thought #{n}**, I signed in with it.",
    )
    assert [(e.payload["outcome"], e.payload.get("reason")) for e in learned] == [("skipped", "carried a credential")]


async def _session_with_a_reply(memory, request_context, monkeypatch, bank: str, reply: str, draft):
    """A live session: a request, the agent's answer, then the user's ``reply``. The writer returns
    ``draft``; returns the learn log and the prompts the writer was sent."""
    from hindsight_api.engine.response_models import LLMCallResult

    prompts: list[str] = []

    async def call(messages, **kw):
        prompts.append(messages[0]["content"])
        return LLMCallResult(content=draft)

    monkeypatch.setattr(memory._llm_config, "call", call)
    monkeypatch.setattr(memory, "reflect_async", _reflect_stub([], []))
    refreshed: list[str] = []

    async def refresh(bank_id, mental_model_id, **kw):
        refreshed.append(mental_model_id)
        return {"operation_id": "op"}

    monkeypatch.setattr(memory, "submit_async_refresh_mental_model", refresh)
    monkeypatch.setattr(memory, "refreshed_rules", refreshed, raising=False)  # read back by the tests
    s = memory.sessions
    steps = [
        {"id": "u1", "role": "user", "text": "Add the typesafe cost to the margin table"},
        {"id": "a1", "role": "assistant", "text": "I priced typesafe at $0.40 per 1M tokens."},
        {"id": "u2", "role": "user", "text": reply},
    ]
    for step in steps:
        await s.append(bank, "s1", _req(step, live=True, retain="off"), request_context=request_context)
    log = [e for e in (await s.events(bank, "s1", request_context=request_context)).items if e.kind == "learn"]
    return log, prompts


async def test_a_users_correction_becomes_a_rule(memory, request_context, monkeypatch):
    """A week of live replies: memory already held the lesson for 28 of 164 corrections ("it's 4 cent,
    not 40"), and only note verdicts were ever learned from."""
    from hindsight_api.engine.sessions.learn import CorrectionDraft

    bank = _bank()
    _learning_on(memory, monkeypatch)
    draft = CorrectionDraft(corrects=True, create=True, name="Typesafe price", query="What does typesafe cost?")
    log, prompts = await _session_with_a_reply(memory, request_context, monkeypatch, bank, "it's 4 cent, not 40", draft)
    assert [(e.payload["outcome"], e.payload["source"]) for e in log] == [("created", "correction")]
    # The writer read the reply and what it answers; the first request is never weighed.
    assert len(prompts) == 1 and "it's 4 cent, not 40" in prompts[0] and "$0.40 per 1M" in prompts[0]
    # What the user had asked comes from the message before, not the session's request (now the reply).
    assert "Add the typesafe cost to the margin table" in prompts[0]
    # The rule is a knowledge page: its question is the writer's, its content reflect's, from the
    # bank's observations.
    rule_id = log[0].payload["mental_model_id"]
    pages = [
        n
        for n in await memory.list_knowledge_nodes(bank, request_context=request_context)
        if n["mental_model_id"] == rule_id
    ]
    assert [(p["kind"], p["name"], p["managed"]) for p in pages] == [("page", "Typesafe price", True)]
    model = await memory.get_mental_model(bank, rule_id, request_context=request_context)
    assert (model["source_query"], model["content"]) == ("What does typesafe cost?", "")
    assert model["trigger"]["refresh_after_consolidation"] is True
    assert model["max_tokens"] == 1024, "capped below the default page size (learn.PAGE_MAX_TOKENS)"
    assert memory.refreshed_rules == [rule_id]


async def test_a_reply_that_corrects_nothing_writes_no_rule(memory, request_context, monkeypatch):
    from hindsight_api.engine.sessions.learn import CorrectionDraft

    bank = _bank()
    _learning_on(memory, monkeypatch)
    # Asked only "create?", the writer made rules of the user's next requests: the draft says create,
    # but corrects is false, and that decides.
    draft = CorrectionDraft(corrects=False, create=True, name="n", query="What is the Knowledge tab called?")
    log, _ = await _session_with_a_reply(
        memory, request_context, monkeypatch, bank, "now rename Knowledge to Documents", draft
    )
    assert [e.payload["outcome"] for e in log] == ["no_rule"]
    assert (await memory.list_mental_models(bank, request_context=request_context)).items == []


async def test_a_notification_in_the_users_turn_is_not_weighed(memory, request_context, monkeypatch):
    from hindsight_api.engine.sessions.learn import CorrectionDraft

    bank = _bank()
    _learning_on(memory, monkeypatch)
    draft = CorrectionDraft(corrects=True, create=True, name="n", query="q")
    log, prompts = await _session_with_a_reply(
        memory,
        request_context,
        monkeypatch,
        bank,
        "<task-notification><task-id>b1</task-id></task-notification>",
        draft,
    )
    assert log == [] and prompts == []


async def test_a_correction_is_not_weighed_when_learning_is_off(memory, request_context, monkeypatch):
    from hindsight_api.engine.sessions.learn import CorrectionDraft

    bank = _bank()
    _learning_on(memory, monkeypatch, on=False)
    draft = CorrectionDraft(corrects=True, create=True, name="n", query="q")
    log, prompts = await _session_with_a_reply(memory, request_context, monkeypatch, bank, "it's 4 cent, not 40", draft)
    assert log == [] and prompts == []
