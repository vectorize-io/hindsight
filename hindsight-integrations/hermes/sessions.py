"""Hermes' conversation as a live Hindsight session (``POST .../sessions/{id}/events``).

Hermes hands its memory provider the turn as OpenAI-style messages, which is the format the
session API takes, so the messages go through as they are, each with an id the provider derives
from the turn. Hindsight then saves each turn into memory (no per-turn retain from this side) and,
with the live session on, writes notes back when memory records a decision the turn goes against.
The provider hands those notes to the agent with the context it injects before each turn.

Plain request/response over stdlib HTTP: it runs on Hermes' sync worker threads, and works against
any server version (an older one answers 404 and the provider falls back to its per-turn retain).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class SessionNote:
    seq: int
    text: str
    quote: str = ""
    document_id: str = ""


@dataclass
class AppendResult:
    supported: bool
    notes: list[SessionNote] = field(default_factory=list)
    checking: bool = False


class SessionOptions(BaseModel):
    """How the session is set up (the API's ``session`` field), sent until the first append succeeds."""

    document_id: str
    live: bool = True
    retain: Literal["turns", "idle", "off"] = "turns"
    metadata: dict[str, str] = {}
    context: str | None = None
    strategy: str | None = None
    tags: list[str] | None = None
    observation_scopes: str | list[list[str]] | None = None


class _AppendBody(BaseModel):
    # OpenAI-style messages, passed through as Hermes built them.
    messages: list[dict[str, Any]]
    ack: int
    turn_end: bool | None = None
    session: SessionOptions | None = None


class _NoteSource(BaseModel):
    quote: str = ""
    document_id: str = ""


class _Note(BaseModel):
    seq: int
    text: str = ""
    source: _NoteSource | None = None


class _AppendReply(BaseModel):
    notes: list[_Note] = []
    checking: bool = False


class SessionLink:
    """One Hermes session's link to its Hindsight session. Notes are handed over until acknowledged:
    ``take_notes`` returns the ones not delivered yet, and the next request acknowledges them."""

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        bank_id: str,
        session_id: str,
        options: SessionOptions,
        timeout: float = 10.0,
    ) -> None:
        self._url = f"{api_url.rstrip('/')}/v1/default/banks/{quote(bank_id, safe='')}/sessions/{quote(session_id, safe='')}/events"
        self._api_key = api_key
        self._options: SessionOptions | None = options  # sent until the first append succeeds
        self._timeout = timeout
        self._acked = 0
        self._pending: list[SessionNote] = []
        self._lock = threading.Lock()
        self.supported = True

    def append(self, messages: list[dict[str, Any]] | None = None, *, turn_end: bool = False) -> AppendResult:
        body = _AppendBody(messages=messages or [], ack=self._acked, turn_end=turn_end or None, session=self._options)
        req = urllib.request.Request(self._url, data=body.model_dump_json(exclude_none=True).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                reply = _AppendReply.model_validate_json(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            if e.code in (404, 405):
                # A server without sessions: the provider goes back to its own per-turn retain.
                self.supported = False
                return AppendResult(supported=False)
            raise
        self._options = None
        notes = [
            SessionNote(
                seq=n.seq,
                text=n.text,
                quote=n.source.quote if n.source else "",
                document_id=n.source.document_id if n.source else "",
            )
            for n in reply.notes
        ]
        with self._lock:
            known = {n.seq for n in self._pending}
            self._pending += [n for n in notes if n.seq not in known]
        return AppendResult(supported=True, notes=notes, checking=reply.checking)

    def take_notes(self, wait_s: float = 0.0) -> list[SessionNote]:
        """The notes not handed over yet. With ``wait_s``, asks the session for up to that long while a
        check is still running (the check of the prompt that was just sent): a note found in time
        reaches the agent in this turn, not the next."""
        deadline = time.monotonic() + wait_s
        while True:
            result = self.append()
            if not result.supported or not result.checking or time.monotonic() >= deadline:
                break
            time.sleep(0.5)
        return self.pop_notes()

    def pop_notes(self) -> list[SessionNote]:
        """The notes earlier requests brought back and not handed over yet, without asking again."""
        with self._lock:
            notes, self._pending = self._pending, []
        if notes:
            self._acked = max(self._acked, *(n.seq for n in notes))
        return notes


NOTE_CREDIT = "> 🧠 **From Hindsight memory (live session #<note number>)** —"
# Every note is mentioned, so the user sees Hindsight at work even when it had nothing to add; the
# server counts both lines (the session's credit_count and dismissed_count).
NOTE_DISMISS = "> 🧠 **Hindsight note (live session #<note number>)** —"


def notes_block(notes: list[SessionNote]) -> str:
    """The notes as the context block the agent reads before its turn."""
    lines = [
        "# Hindsight live session",
        "Notes from this project's memory about what you are doing now. Check the quoted source before acting on one.",
        # The same credit the plugin's recall asks for, so the user sees memory at work.
        "Tell the user about every note, one line each, naming it by its number: one that shaped what you do, starting with "
        f'"{NOTE_CREDIT} <what it told you>"; one you did not need, starting with '
        f'"{NOTE_DISMISS} <what it said>; not needed here: <why>". '
        "Never use the first line for a note you did not use.",
    ]
    for n in notes:
        lines.append(f"\n- Note #{n.seq}: {n.text}")
        if n.quote:
            lines.append(f'  Source ({n.document_id}): "{n.quote}"')
    return "\n".join(lines)


def tool_call_messages(turn: int, call_id: str, name: str, args: Any, result: Any) -> list[dict[str, Any]]:
    """One tool call and its result, sent as soon as the tool returns so the live session sees it mid-turn."""
    return [
        {
            "id": f"t{turn}:call:{call_id}",
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": args or {}}}],
        },
        {"id": f"t{turn}:result:{call_id}", "role": "tool", "tool_call_id": call_id, "content": str(result)},
    ]


def turn_messages(
    messages: list[dict[str, Any]], turn: int, sent_calls: set[str] = frozenset()
) -> list[dict[str, Any]]:
    """This turn's messages (from its user message on), each with an id derived from the turn and its
    place in it: sending a turn twice stores it once, and the user message gets the same id the turn
    start already sent it with. Tool calls in ``sent_calls`` went in live as they ran
    (``tool_call_messages``) and are left out, along with their results."""
    start = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=0)
    out = []
    for k, m in enumerate(messages[start:]):
        if m.get("role") not in ("user", "assistant", "tool", "system"):
            continue
        if m.get("role") == "tool" and m.get("tool_call_id") in sent_calls:
            continue
        if m.get("tool_calls"):
            calls = [c for c in m["tool_calls"] if c.get("id") not in sent_calls]
            if not calls and not m.get("content"):
                continue
            m = {**m, "tool_calls": calls} if calls else {k2: v for k2, v in m.items() if k2 != "tool_calls"}
        out.append({**m, "id": f"t{turn}:{k}"})
    return out
