"""Sessions: a live conversation a client streams in, and the memory watcher's side of it.

The protocol is plain request/response, so any server instance can take any request:

- **append** (``POST .../sessions/{id}/events``): the client sends what happened since its last
  append, each event with its own id. One transaction stores the new ones (a retried event is
  found by its id and skipped), reads them into the watcher's state, and returns the notes the
  client has not acknowledged yet. When the events make a moment worth checking, a ``session_check``
  operation is queued; the append never waits for it.
- **check** (worker): takes the session's pending moment, asks reflect with ``abstain``, and writes
  what it did to the session log, plus a note when there is something to say. One check runs per
  session at a time (the operation's serialization key); a check that finds new moments when it
  finishes takes them too, the latest and heaviest first.
- **save** (worker): writes the conversation's new turns into the session's document, as an append
  to it, at the end of a turn (``retain: turns``) and when the session goes idle. Memories are extracted
  from the document as from any retained one. The session decides when to pay for extraction; the
  log itself is written on every append and costs no model call.
- **idle** (worker): a timer, not a save. Queued by an append that leaves turns unsaved, it waits in
  the queue until the session has been quiet for ``session_idle_ms`` and then queues a save. There is
  no "close": a client that stops sending (finished, or crashed mid-turn) is the same thing, and its
  last turns are saved either way.
- **read** (``GET .../sessions/{id}/events?after=``): the log from a position, for a client that
  waits for a late note and for the UI that follows the session live.

Everything a request needs is in the database: no server remembers a session between requests,
and a check that dies mid-way is retried by the next worker that claims its operation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ...config import get_config
from ...extensions import BankReadOperation, BankWriteOperation
from ...worker.exceptions import DeferOperation
from ..db_utils import acquire_with_retry
from ..llm_wrapper import sanitize_value
from ..schema import fq_table
from . import watch as W
from .models import (
    MAX_EVENT_TEXT,
    CheckLog,
    ContentPart,
    NoteLog,
    SavedTurn,
    SessionAppendRequest,
    SessionAppendResponse,
    SessionEvent,
    SessionEventsResponse,
    SessionListResponse,
    SessionNote,
    SessionNoteSource,
    SessionOptions,
    SessionSettings,
    SessionSummary,
)

if TYPE_CHECKING:
    from ...models import RequestContext
    from ..memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

CHECK_TASK = "session_check"
SAVE_TASK = "session_save"
IDLE_TASK = "session_idle"
# Moments one check takes before it hands over: a session that keeps producing moments still gets
# its notes written, and the next queued check carries on.
MAX_ROUNDS = 4
# A check that started longer ago than this is taken for dead (its worker crashed): the session no
# longer reports itself as checking. The operation itself is retried by the worker poller.
CHECKING_STALE_S = 300
REFLECT_MAX_TOKENS = 600
# A check that has not answered by then gives up: its note would reach an agent that has long moved
# on. Reflect's own limit is 300 s; in beta checks on a loaded machine took p90 97 s, p99 209 s, and
# four ran the full 300 s.
CHECK_TIMEOUT_S = 60
# A turn that runs this many events without ending is saved anyway: in beta an agent working for hours
# in one turn saved nothing until it ended (and the idle timer never fires while events keep coming),
# so a crash lost it all and the eventual save was one huge retain. Saves append only what is new.
SAVE_EVERY_EVENTS = 300
# How the agent tells the user about a note (the clients ask for these exact lines, naming the note
# by its seq): one it used, and one it did not need. Read from the assistant messages the client
# streams in, so every client that asks for the same lines is counted the same way.
# One line can name several notes ("live session #10 / #21"): in beta reading only the first left
# the other counted as not needed.
CREDIT_RE = re.compile(r"From Hindsight memory \(live session (#\d+[^)]*)\)")
DISMISS_RE = re.compile(r"Hindsight note \(live session (#\d+[^)]*)\)")
NOTE_NUMBER_RE = re.compile(r"#(\d+)")
# How the saved conversation reads to extraction when the client does not say: who is speaking.
DEFAULT_SAVE_CONTEXT = (
    "conversation between a user and an assistant: user turns are the user's words and decisions, "
    "assistant turns are the assistant's. When an assistant turn says something is done or works, "
    "record it as the assistant's claim, not as an established fact."
)
# A tool call saved as one compact line (the tool and its main target), as the coding-agents plugin
# has always saved its transcripts: tool output is mechanical noise for extraction.
_ACTION_TARGET_KEYS = ("file_path", "path", "notebook_path", "command", "cmd", "pattern", "query", "url", "name", "id")
_ACTION_TARGET_CAP = 100


class ClientEvent(BaseModel):
    """One entry of the session log, as the client's messages become it: what was said, each tool
    call, each tool result, a state snapshot, a turn end. The log, the watcher and the UI read these;
    the client only ever sends messages."""

    id: str
    kind: str
    payload: dict[str, Any]


def _text_of(content: str | list[ContentPart] | None) -> str:
    if isinstance(content, list):
        content = "\n".join(p.text for p in content if p.type == "text" and p.text)
    return (content or "")[:MAX_EVENT_TEXT]


def _arguments(raw: str | dict[str, Any]) -> dict[str, Any]:
    """OpenAI sends a tool's arguments as a JSON string; anything that is not an object is kept whole."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    return parsed if isinstance(parsed, dict) else {"arguments": raw[:MAX_EVENT_TEXT]}


def client_events(request: SessionAppendRequest) -> list[ClientEvent]:
    """The request's messages (and state) as log entries, each with an id derived from the client's:
    a message's text keeps the message id, each of its tool calls gets ``<message id>:<call id>``. The
    turn end's id needs the log, so the append adds it."""
    out: list[ClientEvent] = []
    for m in request.messages:
        text = _text_of(m.content)
        if m.role == "tool":
            payload: dict[str, Any] = {"role": "tool", "call_id": m.tool_call_id or "", "text": text}
            if m.is_error:
                payload["is_error"] = True
            out.append(ClientEvent(id=m.id, kind="tool_result", payload=payload))
            continue
        if text.strip():
            out.append(ClientEvent(id=m.id, kind="message", payload={"role": m.role, "text": text}))
        for c in m.tool_calls or []:
            out.append(
                ClientEvent(
                    id=f"{m.id}:{c.id}",
                    kind="tool_call",
                    payload={
                        "role": "assistant",
                        "tool": c.function.name,
                        "call_id": c.id,
                        "input": _arguments(c.function.arguments),
                    },
                )
            )
    if request.state is not None:
        # The same snapshot sent twice is one entry: nothing changed in between.
        data = request.state.model_dump(exclude_none=True)
        digest = hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
        out.append(ClientEvent(id=f"state:{digest}", kind="state", payload={"role": "system", "data": data}))
    return out


def default_document_id(session_id: str) -> str:
    return f"session:{session_id}"


def _settings_of(opts: SessionOptions | None, stored: Any = None) -> SessionSettings:
    """The session's settings: what is stored (the `settings` column, parsed), with the options an
    append carries merged over it; the defaults where neither sets a field."""
    current = SessionSettings.model_validate(stored or {})
    if opts is None:
        return current
    update = {
        k: v
        for k in ("live", "retain", "tags", "context", "strategy", "observation_scopes")
        if (v := getattr(opts, k)) is not None
    }
    return current.model_copy(update=update)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _checking(state: W.WatchState, checking_since: datetime | None) -> bool:
    if state.pending is not None:
        return True
    return checking_since is not None and (_now() - checking_since).total_seconds() < CHECKING_STALE_S


def _action_line(tool: str, data: dict[str, Any]) -> str:
    target = ""
    for key in _ACTION_TARGET_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            target = value.strip().split("\n")[0]
            break
    if len(target) > _ACTION_TARGET_CAP:
        target = f"{target[:_ACTION_TARGET_CAP]}…"
    return f"{tool} {target}".strip()


def _turn(kind: str, payload: dict[str, Any], at: datetime) -> SavedTurn | None:
    """One logged client event as a saved transcript line, or None when it is not saved: what was
    said, and one compact line per tool call. Tool output, state snapshots and markers are left out."""
    role = payload.get("role")
    if kind == "message" and role in ("user", "assistant") and (payload.get("text") or "").strip():
        return SavedTurn(role=role, content=payload["text"], timestamp=at.isoformat())
    if kind == "tool_call" and payload.get("tool"):
        inp = payload.get("input")
        return SavedTurn(
            role="action",
            content=_action_line(payload["tool"], inp if isinstance(inp, dict) else {}),
            timestamp=at.isoformat(),
        )
    return None


class SavePlan(BaseModel):
    """A save in flight: the log range it covers and the retain operation it submits."""

    from_seq: int
    upto_seq: int
    operation_id: str


class SessionNotFoundError(LookupError):
    pass


class SessionService:
    def __init__(self, engine: MemoryEngine) -> None:
        self._engine = engine

    async def _authorize(
        self,
        bank_id: str,
        request_context: RequestContext,
        *,
        read: BankReadOperation | None = None,
        write: BankWriteOperation | None = None,
    ) -> None:
        """Tenant, then the operation validator's read or write hook, then refuse a tag-scoped caller.

        Refused rather than narrowed: the watcher's reflect searches the whole bank, so a note could
        carry a memory outside the caller's tags, and the log holds whatever the client sent.
        """
        engine = self._engine
        await engine._authenticate_tenant(request_context)
        validator = engine._operation_validator
        if validator is not None and read is not None:
            from ...extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation=read, request_context=request_context)
            await engine._validate_operation(validator.validate_bank_read(ctx))
        if validator is not None and write is not None:
            from ...extensions import BankWriteContext

            wctx = BankWriteContext(bank_id=bank_id, operation=write, request_context=request_context)
            await engine._validate_operation(validator.validate_bank_write(wctx))
        await engine._refuse_bank_wide_if_scoped(bank_id, request_context, "use sessions")

    # ------------------------------------------------------------------ append

    async def append(
        self,
        bank_id: str,
        session_id: str,
        request: SessionAppendRequest,
        *,
        request_context: RequestContext,
    ) -> SessionAppendResponse:
        engine = self._engine
        await self._authorize(bank_id, request_context, write=BankWriteOperation.APPEND_SESSION_EVENTS)
        backend = await engine._get_backend()
        opts = request.session
        incoming = client_events(request)
        if not incoming and not request.turn_end and opts is None:
            return await self._poll(bank_id, session_id, request.ack)
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                created_bank = await engine._ensure_bank_exists(bank_id, request_context, conn=conn)
                # Create on first sight, then lock: concurrent first appends both insert-or-skip,
                # and the row lock serializes everything after (seq allocation, the watcher state).
                await conn.execute(
                    f"""
                    INSERT INTO {fq_table("sessions")}
                        (bank_id, session_id, document_id, settings, metadata)
                    VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)
                    ON CONFLICT DO NOTHING
                    """,
                    bank_id,
                    session_id,
                    (opts.document_id if opts and opts.document_id else default_document_id(session_id)),
                    _settings_of(opts).model_dump_json(),
                    json.dumps(opts.metadata if opts and opts.metadata else {}),
                )
                row = await conn.fetchrow(
                    f"""
                    SELECT next_seq, saved_seq, acked_seq, settings, metadata, watch_state, checking_since
                    FROM {fq_table("sessions")}
                    WHERE bank_id = $1 AND session_id = $2
                    FOR UPDATE
                    """,
                    bank_id,
                    session_id,
                )
                settings = _settings_of(opts, conn.parse_json(row["settings"]))
                metadata = {
                    **(conn.parse_json(row["metadata"]) or {}),
                    **(opts.metadata if opts and opts.metadata else {}),
                }
                state = W.WatchState.model_validate(conn.parse_json(row["watch_state"]) or {})

                if request.turn_end:
                    # A turn end is named after what it ends, so one sent again with nothing new in
                    # between (a retry, two hooks at the end of one turn) is stored once.
                    last = next((e.id for e in reversed(incoming) if e.kind != "state"), None)
                    if last is None:
                        last = await conn.fetchval(
                            f"""
                            SELECT event_id FROM {fq_table("session_events")}
                            WHERE bank_id = $1 AND session_id = $2 AND source = 'client' AND kind NOT IN ('turn_end', 'state')
                            ORDER BY seq DESC LIMIT 1
                            """,
                            bank_id,
                            session_id,
                        )
                    incoming.append(
                        ClientEvent(id=f"turn_end:{last or ''}", kind="turn_end", payload={"role": "assistant"})
                    )
                # A retried append resends events already stored: find them by id.
                ids = list(dict.fromkeys(e.id for e in incoming))
                known: set[str] = set()
                if ids:
                    known = {
                        r["event_id"]
                        for r in await conn.fetch(
                            f"""
                            SELECT event_id FROM {fq_table("session_events")}
                            WHERE bank_id = $1 AND session_id = $2 AND event_id = ANY($3)
                            """,
                            bank_id,
                            session_id,
                            ids,
                        )
                    }
                seq = int(row["next_seq"])
                rows: list[tuple[int, str, str, str]] = []
                seen: set[str] = set()
                cited: set[int] = set()
                said_unneeded: set[int] = set()
                for event in incoming:
                    if event.id in known or event.id in seen:
                        continue
                    seen.add(event.id)
                    # Tool output can carry a NUL (or a lone surrogate), which jsonb rejects: one such
                    # event failed the whole append, and the client resent it on every append after.
                    payload = sanitize_value(event.payload)
                    if settings.live:
                        step = W.step_of(event.kind, payload["role"], payload)
                        if step is not None:
                            W.read_step(state, step, seq)
                    if event.kind == "message" and payload["role"] == "assistant":
                        for named in CREDIT_RE.findall(payload["text"]):
                            cited.update(int(n) for n in NOTE_NUMBER_RE.findall(named))
                        for named in DISMISS_RE.findall(payload["text"]):
                            said_unneeded.update(int(n) for n in NOTE_NUMBER_RE.findall(named))
                    rows.append((seq, event.id, event.kind, json.dumps(payload)))
                    seq += 1
                turn_ended = any(r[2] == "turn_end" for r in rows)
                credits = dismissed = 0
                if cited or said_unneeded:
                    # Only this session's notes count, each once however often the agent names it.
                    notes_here = {
                        int(r["seq"])
                        for r in await conn.fetch(
                            f"""
                            SELECT seq FROM {fq_table("session_events")}
                            WHERE bank_id = $1 AND session_id = $2 AND kind = 'note' AND seq = ANY($3)
                            """,
                            bank_id,
                            session_id,
                            list(cited | said_unneeded),
                        )
                    }
                    # Used wins: in beta an agent first called a note not needed, then worked by it
                    # and credited it. A note once used is never counted as not needed.
                    new_credits = sorted((cited & notes_here) - set(state.credited))
                    state.credited += new_credits
                    unused_then_used = [n for n in state.dismissed if n in new_credits]
                    state.dismissed = [n for n in state.dismissed if n not in new_credits]
                    new_dismissed = sorted((said_unneeded & notes_here) - set(state.dismissed) - set(state.credited))
                    state.dismissed += new_dismissed
                    credits, dismissed = len(new_credits), len(new_dismissed) - len(unused_then_used)
                if rows:
                    await conn.bulk_insert_from_arrays(
                        fq_table("session_events"),
                        ["bank_id", "session_id", "seq", "event_id", "source", "kind", "payload"],
                        [
                            [bank_id] * len(rows),
                            [session_id] * len(rows),
                            [r[0] for r in rows],
                            [r[1] for r in rows],
                            ["client"] * len(rows),
                            [r[2] for r in rows],
                            [r[3] for r in rows],
                        ],
                        column_types=[
                            "text[]",
                            "text[]",
                            "int8[]",
                            "text[]",
                            "text[]",
                            "text[]",
                            "jsonb[]",
                        ],
                    )
                acked = max(int(row["acked_seq"]), request.ack or 0)
                # An append with nothing new (a client asking for notes) writes nothing: it is the
                # common case for a hook that runs on every tool call.
                if rows or opts is not None or acked != int(row["acked_seq"]):
                    await conn.execute(
                        f"""
                        UPDATE {fq_table("sessions")}
                        SET next_seq = $3, event_count = event_count + $4, acked_seq = $5,
                            credit_count = credit_count + $11, dismissed_count = dismissed_count + $12,
                            watch_state = $6::jsonb, settings = $7::jsonb, metadata = $8::jsonb,
                            last_event_at = CASE WHEN $9 > 0 THEN now() ELSE last_event_at END,
                            pending_since = CASE WHEN $10 = 1 THEN COALESCE(pending_since, now()) ELSE NULL END,
                            updated_at = now()
                        WHERE bank_id = $1 AND session_id = $2
                        """,
                        bank_id,
                        session_id,
                        seq,
                        len(rows),
                        acked,
                        state.model_dump_json(),
                        settings.model_dump_json(),
                        json.dumps(metadata),
                        len(rows),
                        1 if state.pending is not None else 0,
                        credits,
                        dismissed,
                    )
                notes = await self._notes_after(
                    conn, bank_id, session_id, request.ack if request.ack is not None else acked
                )
                if notes:
                    await self._hand_off(conn, bank_id, session_id, notes, [r[0] for r in rows])
                checking = settings.live and _checking(state, row["checking_since"])
        if created_bank:
            await engine._apply_default_bank_template(bank_id, request_context)
        # Only when this append brought something: the moment was queued by the append that made it.
        if rows and settings.live and state.pending is not None:
            await self._queue_check(bank_id, session_id, request_context)
        long_turn = rows and seq - 1 - int(row["saved_seq"]) >= SAVE_EVERY_EVENTS
        if settings.retain == "turns" and (turn_ended or long_turn):
            await self._queue_save(bank_id, session_id, request_context)
        elif rows and settings.retain != "off":
            # At most one idle timer waits per session (dedupe): it covers every append after it.
            await self._queue(IDLE_TASK, "session-idle", bank_id, session_id, request_context)
        return SessionAppendResponse(notes=notes, checking=checking)

    async def _poll(self, bank_id: str, session_id: str, ack: int | None) -> SessionAppendResponse:
        """An append with no events: what a client sends to fetch notes. No lock and no write but the
        acknowledgement, and a session that does not exist yet is not created."""
        backend = await self._engine._get_backend()
        async with acquire_with_retry(backend) as conn:
            row = await conn.fetchrow(
                f"""
                SELECT next_seq, acked_seq, settings, watch_state, checking_since FROM {fq_table("sessions")}
                WHERE bank_id = $1 AND session_id = $2
                """,
                bank_id,
                session_id,
            )
            if row is None:
                return SessionAppendResponse(notes=[], checking=False)
            acked = int(row["acked_seq"])
            if ack is not None and ack > acked:
                await conn.execute(
                    f"""
                    UPDATE {fq_table("sessions")} SET acked_seq = $3
                    WHERE bank_id = $1 AND session_id = $2 AND acked_seq < $3
                    """,
                    bank_id,
                    session_id,
                    ack,
                )
                acked = ack
            settings = _settings_of(None, conn.parse_json(row["settings"]))
            state = W.WatchState.model_validate(conn.parse_json(row["watch_state"]) or {})
            notes = await self._notes_after(conn, bank_id, session_id, ack if ack is not None else acked)
            # Locks only when a note goes out for the first time: polls stay lock-free otherwise.
            if notes and len(await self._handed(conn, bank_id, session_id, notes)) < len(notes):
                async with conn.transaction():
                    await conn.fetchval(
                        f"SELECT 1 FROM {fq_table('sessions')} WHERE bank_id = $1 AND session_id = $2 FOR UPDATE",
                        bank_id,
                        session_id,
                    )
                    await self._hand_off(conn, bank_id, session_id, notes, [])
            return SessionAppendResponse(
                notes=notes,
                checking=settings.live and _checking(state, row["checking_since"]),
            )

    async def _handed(self, conn: Any, bank_id: str, session_id: str, notes: list[SessionNote]) -> set[str]:
        return {
            r["event_id"]
            for r in await conn.fetch(
                f"""
                SELECT event_id FROM {fq_table("session_events")}
                WHERE bank_id = $1 AND session_id = $2 AND event_id = ANY($3)
                """,
                bank_id,
                session_id,
                [f"handoff:{n.seq}" for n in notes],
            )
        }

    async def _hand_off(
        self, conn: Any, bank_id: str, session_id: str, notes: list[SessionNote], with_seqs: list[int]
    ) -> None:
        """Logs the moment each note first goes back to the client, as a ``handoff`` entry naming the
        events whose reply carried it (none for a poll), so any reader of the log sees when the agent
        got it, whatever the client. A note is sent again until acknowledged; its handoff id is fixed
        (``handoff:<note seq>``), so only the first send is logged. Needs the session row locked."""
        handed = await self._handed(conn, bank_id, session_id, notes)
        for n in notes:
            if f"handoff:{n.seq}" not in handed:
                await self._log(
                    conn,
                    bank_id,
                    session_id,
                    "handoff",
                    {"note_seq": n.seq, "with_seqs": with_seqs},
                    event_id=f"handoff:{n.seq}",
                )

    async def _notes_after(self, conn: Any, bank_id: str, session_id: str, after: int) -> list[SessionNote]:
        rows = await conn.fetch(
            f"""
            SELECT seq, payload FROM {fq_table("session_events")}
            WHERE bank_id = $1 AND session_id = $2 AND source = 'hindsight' AND kind = 'note' AND seq > $3
            ORDER BY seq
            """,
            bank_id,
            session_id,
            after,
        )
        notes = []
        for r in rows:
            p = conn.parse_json(r["payload"]) or {}
            notes.append(
                SessionNote(
                    seq=int(r["seq"]),
                    text=p.get("text", ""),
                    source=SessionNoteSource.model_validate(p["source"]) if p.get("source") else None,
                )
            )
        return notes

    async def _queue(self, task: str, key: str, bank_id: str, session_id: str, request_context: RequestContext) -> None:
        payload: dict[str, Any] = {"session_id": session_id}
        if request_context.tenant_id:
            payload["_tenant_id"] = request_context.tenant_id
        if request_context.api_key_id:
            payload["_api_key_id"] = request_context.api_key_id
        # At most one waits per session (a waiting one reads the session when it runs, so it covers
        # everything since it was queued), and one runs at a time.
        #
        # A failure here does not fail the request: what the client sent is already stored, and the
        # work is queued again by the session's next append (a pending moment, or a turn end not yet
        # saved, stays in the session until a check or save takes it).
        try:
            await self._submit(task, key, bank_id, session_id, payload)
        except Exception as e:
            logger.warning(f"could not queue {task} for {bank_id}/{session_id}: {e}")

    async def _submit(self, task: str, key: str, bank_id: str, session_id: str, payload: dict[str, Any]) -> None:
        await self._engine._submit_async_operation(
            bank_id=bank_id,
            operation_type=task,
            task_type=task,
            task_payload=payload,
            dedupe_in_flight_payload_key="session_id",
            dedupe_in_flight_includes_processing=False,
            serialization_key=f"{key}:{session_id}",
        )

    async def _queue_check(self, bank_id: str, session_id: str, request_context: RequestContext) -> None:
        await self._queue(CHECK_TASK, "session", bank_id, session_id, request_context)

    async def _queue_save(self, bank_id: str, session_id: str, request_context: RequestContext) -> None:
        # Its own serialization key: a save does not wait behind a check's reflect, nor a check
        # behind a save.
        await self._queue(SAVE_TASK, "session-save", bank_id, session_id, request_context)

    async def run_idle(self, task: dict[str, Any]) -> None:
        """Queue a save once the session has gone quiet with turns not yet saved.

        Not a save itself: a save queued at a turn end must not wait behind this timer, so the timer
        is its own operation and hands over to the save job. Measured by the database's clock, like
        the check's settle wait."""
        bank_id: str = task["bank_id"]
        session_id: str = task["session_id"]
        backend = await self._engine._get_backend()
        async with acquire_with_retry(backend) as conn:
            row = await conn.fetchrow(
                f"""
                SELECT last_event_at, saved_seq, now() AS db_now
                FROM {fq_table("sessions")} WHERE bank_id = $1 AND session_id = $2
                """,
                bank_id,
                session_id,
            )
            unsaved = row is not None and await conn.fetchval(
                f"""
                SELECT 1 FROM {fq_table("session_events")}
                WHERE bank_id = $1 AND session_id = $2 AND source = 'client' AND seq > $3
                LIMIT 1
                """,
                bank_id,
                session_id,
                int(row["saved_seq"]),
            )
        if not unsaved or row["last_event_at"] is None:
            return
        idle_at = row["last_event_at"] + timedelta(milliseconds=get_config().session_idle_ms)
        if idle_at > row["db_now"]:
            if not self._engine._task_backend.can_defer:
                return  # the in-process test backend: turn ends still save
            raise DeferOperation(exec_date=idle_at, reason="waiting for the session to go idle")
        await self._queue_save(bank_id, session_id, self._task_context(task))

    # ------------------------------------------------------------------ check (worker)

    @staticmethod
    def _task_context(task: dict[str, Any]) -> RequestContext:
        from ...models import RequestContext

        return RequestContext(
            internal=True,
            tenant_id=task.get("_tenant_id"),
            api_key_id=task.get("_api_key_id"),
            retry_count=task.get("_retry_count", 0),
        )

    async def run_check(self, task: dict[str, Any]) -> None:
        bank_id: str = task["bank_id"]
        session_id: str = task["session_id"]
        rc = self._task_context(task)
        backend = await self._engine._get_backend()
        for _ in range(MAX_ROUNDS):
            # Take the moment under the session's lock; the slow part runs without it.
            async with acquire_with_retry(backend) as conn:
                async with conn.transaction():
                    row = await conn.fetchrow(
                        f"""
                        SELECT watch_state, pending_since, last_event_at, now() AS db_now FROM {fq_table("sessions")}
                        WHERE bank_id = $1 AND session_id = $2 FOR UPDATE
                        """,
                        bank_id,
                        session_id,
                    )
                    if row is None:
                        return
                    state = W.WatchState.model_validate(conn.parse_json(row["watch_state"]) or {})
                    settle = self._settle_at(row)
                    if state.pending is not None and settle is not None and self._engine._task_backend.can_defer:
                        # Not yet: back in the queue until the session settles. A deferral is not a
                        # failed attempt, and appends meanwhile fold into this same operation.
                        raise DeferOperation(exec_date=settle, reason="waiting for the session to settle")
                    moment = W.take_moment(state)
                    skip = W.skip_reason(state, moment) if moment else None
                    await conn.execute(
                        f"""
                        UPDATE {fq_table("sessions")}
                        SET watch_state = $3::jsonb, checking_since = $4, pending_since = NULL
                        WHERE bank_id = $1 AND session_id = $2
                        """,
                        bank_id,
                        session_id,
                        state.model_dump_json(),
                        _now() if moment and not skip else None,
                    )
                    if moment and skip:
                        await self._log(
                            conn,
                            bank_id,
                            session_id,
                            "check",
                            self._check_payload(moment, outcome="skipped", reason=skip),
                        )
            if moment is None:
                return
            if skip:
                continue
            await self._check(bank_id, session_id, moment, state.goal or moment.text, list(state.sent), rc)

    @staticmethod
    def _settle_at(row: Any) -> datetime | None:
        """When the pending moment may be checked, by the database's clock, or None when it may be now:
        once the session has been quiet for a while, and at the latest a while after the moment
        appeared. Database time because the append and the check may run on different machines."""
        cfg = get_config()
        if row["pending_since"] is None or row["last_event_at"] is None:
            return None
        at = min(
            row["last_event_at"] + timedelta(milliseconds=cfg.session_check_quiet_ms),
            row["pending_since"] + timedelta(milliseconds=cfg.session_check_max_wait_ms),
        )
        return at if at > row["db_now"] else None

    def _check_payload(self, m: W.Moment, **extra: Any) -> CheckLog:
        return CheckLog(trigger=m.trigger, moment_seq=m.seq, moment=m.text[:2000], **extra)

    async def _check(
        self, bank_id: str, session_id: str, m: W.Moment, goal: str, sent: list[str], rc: RequestContext
    ) -> None:
        from ..memory_engine import Budget

        context = W.situation(goal, m, sent)
        queries = W.search_queries(m)
        t0 = time.monotonic()
        error = None
        text, memory_ids, usage, calls, abstained = "", [], None, [], False
        try:
            result = await asyncio.wait_for(
                self._engine.reflect_async(
                    bank_id=bank_id,
                    query=W.question(m),
                    context=context,
                    budget=Budget.LOW,
                    max_tokens=REFLECT_MAX_TOKENS,
                    fact_types=["observation"],
                    exclude_mental_models=True,
                    abstain=True,
                    search_queries=queries,
                    request_context=rc,
                ),
                timeout=CHECK_TIMEOUT_S,
            )
            text = (result.text or "").strip()
            abstained = result.abstained
            memory_ids = [
                str(f.id)
                for kind, facts in result.based_on.items()
                if kind not in ("directives", "mental-models")
                for f in facts
            ]
            usage = result.usage.model_dump() if result.usage else None
            calls = [c.model_dump() for c in result.llm_trace]
        except TimeoutError:
            logger.warning(f"session check timed out for {bank_id}/{session_id} after {CHECK_TIMEOUT_S}s")
            error = f"the check timed out after {CHECK_TIMEOUT_S}s"
        except Exception as e:  # a failed check is logged and the session goes on: notes are best-effort
            logger.warning(f"session check failed for {bank_id}/{session_id}: {e}")
            error = str(e)[:500]
        ms = int((time.monotonic() - t0) * 1000)
        # Looked up before taking the lock, and only when there is a note to cite: a few reads.
        source = await self._source_of(bank_id, memory_ids, text, rc) if text and not error else None
        # A note built only from this session's own saved turns tells the agent what is already in its
        # context: in beta it came back as a "past decision" (the agent's own earlier change, then the
        # user's new request) turn after turn, each time under a new memory id, and the agent dismissed
        # every one. A memory with any source outside this session is a real past decision and stays.
        own = bool(text) and not error and await self._from_this_session_only(bank_id, session_id, memory_ids, rc)

        backend = await self._engine._get_backend()
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    f"""
                    SELECT watch_state FROM {fq_table("sessions")}
                    WHERE bank_id = $1 AND session_id = $2 FOR UPDATE
                    """,
                    bank_id,
                    session_id,
                )
                if row is None:
                    return
                state = W.WatchState.model_validate(conn.parse_json(row["watch_state"]) or {})
                spoke = not error and not own and W.accept_note(state, m, text, memory_ids)
                outcome = "error" if error else "note" if spoke else "repeat" if text else "silent"
                check_seq = await self._log(
                    conn,
                    bank_id,
                    session_id,
                    "check",
                    self._check_payload(
                        m,
                        outcome=outcome,
                        error=error,
                        ms=ms,
                        question="headsup" if m.trigger == "task" else "conflict",
                        context=context,
                        search_queries=queries,
                        answer=text,
                        abstained=abstained,
                        memory_ids=memory_ids,
                        usage=usage,
                        llm_calls=calls,
                        **({"reason": "this session"} if own else {}),
                    ),
                )
                if spoke:
                    await self._log(
                        conn,
                        bank_id,
                        session_id,
                        "note",
                        NoteLog(
                            text=text, trigger=m.trigger, check_seq=check_seq, memory_ids=memory_ids, source=source
                        ),
                    )
                await conn.execute(
                    f"""
                    UPDATE {fq_table("sessions")}
                    SET watch_state = $3::jsonb, checking_since = NULL,
                        check_count = check_count + 1, note_count = note_count + $4, updated_at = now()
                    WHERE bank_id = $1 AND session_id = $2
                    """,
                    bank_id,
                    session_id,
                    state.model_dump_json(),
                    1 if spoke else 0,
                )

    async def _log(
        self,
        conn: Any,
        bank_id: str,
        session_id: str,
        kind: str,
        payload: CheckLog | NoteLog | dict[str, Any],
        event_id: str = "",
    ) -> int:
        """Append one of Hindsight's own entries (a check, a note, a save) to the session log, at the
        next position. Needs the session row locked."""
        seq = int(
            await conn.fetchval(
                f"SELECT next_seq FROM {fq_table('sessions')} WHERE bank_id = $1 AND session_id = $2",
                bank_id,
                session_id,
            )
        )
        await conn.execute(
            f"""
            INSERT INTO {fq_table("session_events")} (bank_id, session_id, seq, event_id, source, kind, payload)
            VALUES ($1, $2, $3, $4, 'hindsight', $5, $6::jsonb)
            """,
            bank_id,
            session_id,
            seq,
            event_id or f"hindsight:{seq}",
            kind,
            # A typed entry leaves out its unset fields, so the log keeps the shape it always had.
            payload.model_dump_json(exclude_none=True)
            if isinstance(payload, BaseModel)
            else json.dumps(payload, default=str),
        )
        await conn.execute(
            f"UPDATE {fq_table('sessions')} SET next_seq = $3 WHERE bank_id = $1 AND session_id = $2",
            bank_id,
            session_id,
            seq + 1,
        )
        return seq

    async def _from_this_session_only(
        self, bank_id: str, session_id: str, memory_ids: list[str], rc: RequestContext
    ) -> bool:
        """Whether every memory the note used comes only from this session's own document (each memory
        and the memories it was built from). Unknown -> False: the note stands. A memory gone by now is
        skipped: in beta the session's own save rewrote it between the check and this lookup, and the
        note, quoting only the session itself, went out. At least one memory must still be found."""
        if not memory_ids:
            return False
        try:
            backend = await self._engine._get_backend()
            async with acquire_with_retry(backend) as conn:
                own = await conn.fetchval(
                    f"SELECT document_id FROM {fq_table('sessions')} WHERE bank_id = $1 AND session_id = $2",
                    bank_id,
                    session_id,
                )
            if not own:
                return False
            found = False
            for mid in memory_ids:
                unit = await self._engine.get_memory_unit(bank_id, mid, request_context=rc)
                if not unit:
                    continue
                found = True
                docs = {unit.get("document_id")} if unit.get("document_id") else set()
                for sid in unit.get("source_memory_ids") or []:
                    src = await self._engine.get_memory_unit(bank_id, sid, request_context=rc)
                    docs.add((src or {}).get("document_id"))
                if docs != {own}:
                    return False
            return found
        except Exception as e:  # unknown: the note stands
            logger.warning(f"session own-source check failed for {bank_id}: {e}")
            return False

    async def _source_of(
        self, bank_id: str, memory_ids: list[str], note: str, rc: RequestContext
    ) -> SessionNoteSource | None:
        """Where the note comes from: the newest document behind the first memory it used, and that
        document's closest sentence to the note (verbatim), so the reader can check it."""
        if not memory_ids:
            return None
        try:
            memory = await self._engine.get_memory_unit(bank_id, memory_ids[0], request_context=rc)
            sources = []
            for sid in ((memory or {}).get("source_memory_ids") or [])[:6]:
                unit = await self._engine.get_memory_unit(bank_id, sid, request_context=rc)
                if unit and unit.get("document_id"):
                    sources.append(unit)
            if memory and memory.get("document_id"):
                sources.append(memory)
            sources.sort(key=lambda u: str(u.get("mentioned_at") or ""), reverse=True)
            for unit in sources:
                doc = await self._engine.get_document(unit["document_id"], bank_id, request_context=rc)
                if not doc:
                    continue
                quote = W.best_sentence(W.readable(doc.get("original_text") or ""), note)
                if quote:
                    return SessionNoteSource(
                        document_id=unit["document_id"],
                        quote=quote,
                        mentioned_at=unit.get("mentioned_at"),
                    )
            if sources:
                return SessionNoteSource(
                    document_id=sources[0]["document_id"], mentioned_at=sources[0].get("mentioned_at")
                )
        except Exception as e:  # the note stands without its source
            logger.warning(f"session note source lookup failed for {bank_id}: {e}")
        return None

    # ------------------------------------------------------------------ save (worker)

    async def run_save(self, task: dict[str, Any]) -> None:
        """Write the conversation's turns since the last save into the session's document.

        Three steps, each safe to repeat. (1) Under the session lock, plan the save: the range of
        the log it covers and the retain operation id it will use, stored in ``save_plan``. A plan
        already there (an earlier attempt that died) is reused as is. (2) Submit the retain as an
        append to the document, with that operation id: the server turns a resubmission of a known
        id into a no-op, so a retried save never writes a turn twice. (3) Move ``saved_seq`` to the
        end of the range and clear the plan.
        """
        bank_id: str = task["bank_id"]
        session_id: str = task["session_id"]
        rc = self._task_context(task)
        backend = await self._engine._get_backend()
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    f"""
                    SELECT document_id, settings, metadata, created_at, next_seq, saved_seq, save_plan
                    FROM {fq_table("sessions")} WHERE bank_id = $1 AND session_id = $2 FOR UPDATE
                    """,
                    bank_id,
                    session_id,
                )
                if row is None:
                    return
                stored = conn.parse_json(row["save_plan"])
                if stored:
                    plan = SavePlan.model_validate(stored)
                else:
                    upto = int(row["next_seq"]) - 1
                    if upto <= int(row["saved_seq"]):
                        return
                    plan = SavePlan(from_seq=int(row["saved_seq"]), upto_seq=upto, operation_id=str(uuid.uuid4()))
                    await conn.execute(
                        f"UPDATE {fq_table('sessions')} SET save_plan = $3::jsonb WHERE bank_id = $1 AND session_id = $2",
                        bank_id,
                        session_id,
                        plan.model_dump_json(),
                    )
            events = await conn.fetch(
                f"""
                SELECT seq, kind, created_at, payload FROM {fq_table("session_events")}
                WHERE bank_id = $1 AND session_id = $2 AND source = 'client' AND seq > $3 AND seq <= $4
                ORDER BY seq
                """,
                bank_id,
                session_id,
                plan.from_seq,
                plan.upto_seq,
            )
            turns = [t for e in events if (t := _turn(e["kind"], conn.parse_json(e["payload"]) or {}, e["created_at"]))]
            settings = _settings_of(None, conn.parse_json(row["settings"]))
            metadata = conn.parse_json(row["metadata"]) or {}
        document_id: str = row["document_id"]
        if turns:
            if plan.from_seq == 0:
                # The document names itself, as the coding-agents transcripts always have: a reader of
                # a memory's source can find the session it came from.
                turns.insert(
                    0, SavedTurn(role="system", content=f"REF-ID: {document_id}", timestamp=turns[0].timestamp)
                )
            item: dict[str, Any] = {
                "content": "\n".join(json.dumps(t.model_dump(), ensure_ascii=False) for t in turns),
                "context": settings.context or DEFAULT_SAVE_CONTEXT,
                "document_id": document_id,
                "timestamp": row["created_at"].isoformat(),
                "update_mode": "append",
                # The session's own metadata, as the client set it, plus which session wrote it.
                "metadata": {**{k: str(v) for k, v in metadata.items()}, "session_id": session_id},
            }
            if settings.tags:
                item["tags"] = settings.tags
            if settings.observation_scopes is not None:
                item["observation_scopes"] = settings.observation_scopes
            await self._engine.submit_async_retain(
                bank_id,
                [item],
                request_context=rc,
                strategy=settings.strategy,
                operation_id=plan.operation_id,
            )
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                current = await conn.fetchval(
                    f"SELECT save_plan FROM {fq_table('sessions')} WHERE bank_id = $1 AND session_id = $2 FOR UPDATE",
                    bank_id,
                    session_id,
                )
                # The document was deleted (or the bank cleared) while this save ran: forget_saves
                # dropped the plan so the next save rewrites the whole conversation. Moving saved_seq
                # now would undo that. ponytail: the turns this retain wrote may land twice in the
                # rebuilt document; a replace-mode save would fix it if it ever matters.
                stored = conn.parse_json(current)
                if not stored or SavePlan.model_validate(stored).operation_id != plan.operation_id:
                    return
                await self._log(
                    conn,
                    bank_id,
                    session_id,
                    "save",
                    {
                        "from_seq": plan.from_seq,
                        "to_seq": plan.upto_seq,
                        "turns": len(turns),
                        "document_id": document_id,
                        "operation_id": plan.operation_id if turns else None,
                    },
                )
                await conn.execute(
                    f"""
                    UPDATE {fq_table("sessions")} SET saved_seq = $3, save_plan = NULL
                    WHERE bank_id = $1 AND session_id = $2
                    """,
                    bank_id,
                    session_id,
                    plan.upto_seq,
                )
                # All saved: the idle timer waiting on this session has nothing left to do. Left in the
                # queue it reads as unfinished work for ten minutes after every turn (a bank that never
                # settles), so it goes now. Under the session lock: an append that brings new turns
                # either committed before (they are unsaved, the timer stays) or queues a new timer after.
                if not await conn.fetchval(
                    f"""
                    SELECT 1 FROM {fq_table("session_events")}
                    WHERE bank_id = $1 AND session_id = $2 AND source = 'client' AND seq > $3
                    LIMIT 1
                    """,
                    bank_id,
                    session_id,
                    plan.upto_seq,
                ):
                    await conn.execute(
                        f"""
                        DELETE FROM {fq_table("async_operations")}
                        WHERE bank_id = $1 AND operation_type = '{IDLE_TASK}' AND status = 'pending'
                          AND task_payload->>'session_id' = $2
                        """,
                        bank_id,
                        session_id,
                    )

    # ------------------------------------------------------------------ reads

    async def events(
        self,
        bank_id: str,
        session_id: str,
        *,
        after: int = 0,
        limit: int = 200,
        request_context: RequestContext,
    ) -> SessionEventsResponse:
        engine = self._engine
        await self._authorize(bank_id, request_context, read=BankReadOperation.LIST_SESSION_EVENTS)
        backend = await engine._get_backend()
        async with acquire_with_retry(backend) as conn:
            head = await conn.fetchrow(
                f"""
                SELECT next_seq, watch_state, checking_since, settings FROM {fq_table("sessions")}
                WHERE bank_id = $1 AND session_id = $2
                """,
                bank_id,
                session_id,
            )
            if head is None:
                raise SessionNotFoundError(session_id)
            rows = await conn.fetch(
                f"""
                SELECT seq, event_id, source, kind, created_at, payload FROM {fq_table("session_events")}
                WHERE bank_id = $1 AND session_id = $2 AND seq > $3
                ORDER BY seq
                LIMIT $4
                """,
                bank_id,
                session_id,
                after,
                limit,
            )
            state = W.WatchState.model_validate(conn.parse_json(head["watch_state"]) or {})
            settings = _settings_of(None, conn.parse_json(head["settings"]))
            return SessionEventsResponse(
                items=[
                    SessionEvent(
                        seq=int(r["seq"]),
                        event_id=r["event_id"],
                        source=r["source"],
                        kind=r["kind"],
                        created_at=r["created_at"],
                        payload=conn.parse_json(r["payload"]) or {},
                    )
                    for r in rows
                ],
                checking=settings.live and _checking(state, head["checking_since"]),
            )

    def _summary(self, conn: Any, r: Any) -> SessionSummary:
        settings = _settings_of(None, conn.parse_json(r["settings"]))
        state = W.WatchState.model_validate(conn.parse_json(r["watch_state"]) or {})
        return SessionSummary(
            session_id=r["session_id"],
            document_id=r["document_id"],
            live=settings.live,
            retain=settings.retain,
            event_count=int(r["event_count"]),
            check_count=int(r["check_count"]),
            note_count=int(r["note_count"]),
            credit_count=int(r["credit_count"]),
            dismissed_count=int(r["dismissed_count"]),
            last_seq=int(r["next_seq"]) - 1,
            saved_seq=int(r["saved_seq"]),
            acked_seq=int(r["acked_seq"]),
            checking=settings.live and _checking(state, r["checking_since"]),
            created_at=r["created_at"],
            last_event_at=r["last_event_at"],
            metadata=conn.parse_json(r["metadata"]) or {},
        )

    @staticmethod
    def _summary_columns() -> str:
        return (
            "s.session_id, s.document_id, s.settings, s.metadata, s.watch_state, "
            "s.next_seq, s.saved_seq, s.acked_seq, s.event_count, s.check_count, s.note_count, "
            "s.credit_count, s.dismissed_count, s.checking_since, "
            "s.created_at, s.last_event_at"
        )

    async def get(self, bank_id: str, session_id: str, *, request_context: RequestContext) -> SessionSummary:
        await self._authorize(bank_id, request_context, read=BankReadOperation.GET_SESSION)
        return await self._fetch(bank_id, session_id)

    async def _fetch(self, bank_id: str, session_id: str) -> SessionSummary:
        backend = await self._engine._get_backend()
        async with acquire_with_retry(backend) as conn:
            r = await conn.fetchrow(
                f"SELECT {self._summary_columns()} FROM {fq_table('sessions')} s WHERE s.bank_id = $1 AND s.session_id = $2",
                bank_id,
                session_id,
            )
            if r is None:
                raise SessionNotFoundError(session_id)
            return self._summary(conn, r)

    async def list(
        self, bank_id: str, *, limit: int = 50, offset: int = 0, q: str | None = None, request_context: RequestContext
    ) -> SessionListResponse:
        engine = self._engine
        await self._authorize(bank_id, request_context, read=BankReadOperation.LIST_SESSIONS)
        backend = await engine._get_backend()
        where = "bank_id = $1"
        args: list[Any] = [bank_id]
        if q:
            args.append(f"%{q.lower()}%")
            where += f" AND LOWER(session_id) LIKE ${len(args)}"
        async with acquire_with_retry(backend) as conn:
            total = int(await conn.fetchval(f"SELECT COUNT(*) FROM {fq_table('sessions')} WHERE {where}", *args))
            rows = await conn.fetch(
                f"""
                SELECT {self._summary_columns()} FROM {fq_table("sessions")} s WHERE {where}
                ORDER BY updated_at DESC LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}
                """,
                *args,
                limit,
                offset,
            )
            return SessionListResponse(
                items=[self._summary(conn, r) for r in rows], total=total, limit=limit, offset=offset
            )

    async def delete(self, bank_id: str, session_id: str, *, request_context: RequestContext) -> None:
        """Delete the session and its log. The document it saved stays: that is memory now, and is
        deleted as a document. A check or save still queued for it finds no session and does nothing."""
        engine = self._engine
        await self._authorize(bank_id, request_context, write=BankWriteOperation.DELETE_SESSION)
        backend = await engine._get_backend()
        async with acquire_with_retry(backend) as conn:
            status = await conn.execute(
                f"DELETE FROM {fq_table('sessions')} WHERE bank_id = $1 AND session_id = $2", bank_id, session_id
            )
        if status.endswith(" 0"):
            raise SessionNotFoundError(session_id)

    @staticmethod
    async def forget_saves(conn: Any, bank_id: str, document_id: str | None = None) -> None:
        """The document a session saved into is gone (deleted, or the bank cleared): start its saves over.

        A save appends the turns after ``saved_seq`` to the document, so against a missing document it
        would write only the tail. Back to 0, with no plan, the next save rewrites the whole
        conversation from the log. Runs on the caller's connection, inside the delete's transaction.
        ``document_id`` None means every session of the bank.
        """
        sql = f"UPDATE {fq_table('sessions')} SET saved_seq = 0, save_plan = NULL WHERE bank_id = $1"
        if document_id is None:
            await conn.execute(sql, bank_id)
        else:
            await conn.execute(f"{sql} AND document_id = $2", bank_id, document_id)
