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
    used, unneeded = [e.seq for e in notes if e.kind == "note"][:2]
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
        e.seq for e in (await s.events(bank, "s1", request_context=request_context)).items if e.kind == "note"
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
    job = "You are running the scheduled job 'disk-guard'. Read JOB.md and do what it says."
    await s.append(
        bank,
        "run1",
        _req({"id": "1", "role": "user", "text": job}, live=True, retain="off"),
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
        _req({"id": "1", "role": "user", "text": job}, live=True, retain="off"),
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
