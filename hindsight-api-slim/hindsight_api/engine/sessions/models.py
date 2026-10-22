"""Request and response shapes of the sessions API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "tool", "system"]

MAX_MESSAGES_PER_APPEND = 500
MAX_EVENT_TEXT = 100_000


class ToolCallFunction(BaseModel):
    name: str = Field(max_length=256)
    arguments: str | dict[str, Any] = Field(
        description="The tool's arguments: a JSON object, or the JSON string OpenAI sends."
    )


class ToolCall(BaseModel):
    id: str = Field(min_length=1, max_length=256, description="Matched by the `tool` message that answers it.")
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ContentPart(BaseModel):
    """A part of a message's content, as in the OpenAI format. Only text parts are read."""

    type: str
    text: str | None = None


class SessionMessage(BaseModel):
    """One message of the conversation, in the OpenAI chat format most agents already keep, plus the
    client's own `id`. Unknown keys (a name, reasoning, ...) are ignored, so a client can send the
    messages it already has as they are."""

    id: str = Field(
        min_length=1,
        max_length=256,
        description="The client's id for this message, unique in the session. Sending it again is a no-op: "
        "an append that is retried after a lost response stores nothing twice.",
    )
    role: MessageRole
    content: str | list[ContentPart] | None = Field(
        default=None, description="What was said: a string, or a list of parts whose text parts are read."
    )
    tool_calls: list[ToolCall] | None = Field(default=None, description="The tools an assistant message calls.")
    tool_call_id: str | None = Field(
        default=None, max_length=256, description="On a `tool` message: the call it answers."
    )
    is_error: bool = Field(default=False, description="On a `tool` message: the tool failed.")


class SessionOptions(BaseModel):
    """How the session is set up. Applied when an append creates the session; later appends that
    carry options update the fields they set."""

    document_id: str | None = Field(
        default=None,
        max_length=512,
        description="The document the conversation is saved into. Defaults to `session:<session_id>`. "
        "Fixed once the session exists.",
    )
    live: bool | None = Field(
        default=None,
        description="Hindsight follows the session live and may send notes back when memory records a decision "
        "the work goes against. Off by default: the session is only saved.",
    )
    retain: Literal["turns", "idle", "off"] | None = Field(
        default=None,
        description="When the conversation is saved into its document: at the end of each turn and when the "
        "session goes idle (`turns`, the default), only when the session goes idle (`idle`: one extraction for "
        "the whole session), or never (`off`).",
    )
    tags: list[str] | None = Field(default=None, description="Tags the saved document gets.")
    context: str | None = Field(
        default=None,
        max_length=2000,
        description="How extraction should read the saved conversation (who is speaking). A generic "
        "user/assistant description by default.",
    )
    strategy: str | None = Field(default=None, max_length=256, description="The retain strategy saves use.")
    observation_scopes: Literal["per_tag", "combined", "all_combinations", "shared"] | list[list[str]] | None = Field(
        default=None,
        description="How observations built from the saved conversation are scoped, as on retain. The bank's "
        "default when unset.",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Free-form labels, merged across appends and copied onto the saved document. "
        "`harness` names the client (e.g. `claude-code`): the UI shows its logo.",
    )


class SessionState(BaseModel):
    diff: str | None = Field(
        default=None,
        description="For a coding agent: the `git diff` of everything it changed so far. An edit is checked "
        "against it.",
    )


class SessionAppendRequest(BaseModel):
    messages: list[SessionMessage] = Field(default_factory=list, max_length=MAX_MESSAGES_PER_APPEND)
    state: SessionState | None = Field(
        default=None,
        description="A snapshot of the work after these messages, said by no one. Never saved into memory.",
    )
    turn_end: bool = Field(
        default=False,
        description="The assistant finished its turn after these messages: the turn is saved into memory "
        "(with `retain: turns`). Sent again with nothing new in between, it is the same turn end.",
    )
    ack: int | None = Field(
        default=None,
        ge=0,
        description="The `seq` of the last note this client delivered. The response carries the notes after it. "
        "Omit it to get every note the session has not had acknowledged.",
    )
    session: SessionOptions | None = None


class SessionNoteSource(BaseModel):
    """Where a note comes from, so the agent can check it before acting on it."""

    document_id: str
    quote: str = Field(default="", description="The source's own sentence closest to the note, verbatim.")
    mentioned_at: datetime | None = None


class SessionNote(BaseModel):
    """A note for the agent. What Hindsight checked and the memories it used are in the session's
    event log (the note's `note` entry and its `check`)."""

    seq: int = Field(description="Its position in the log: acknowledge it by sending it back as `ack`.")
    text: str
    source: SessionNoteSource | None = None


class SessionAppendResponse(BaseModel):
    notes: list[SessionNote] = Field(description="Notes after `ack`, oldest first.")
    checking: bool = Field(description="A check is queued or running: another note may still come.")


class SessionEvent(BaseModel):
    seq: int
    event_id: str
    source: Literal["client", "hindsight"] = Field(
        description="`client`: an event the client sent. `hindsight`: Hindsight's own entry: a "
        "`check`, a `note`, or a `save` of the conversation into its document."
    )
    kind: str
    created_at: datetime
    payload: dict[str, Any]


class SessionEventsResponse(BaseModel):
    items: list[SessionEvent] = Field(description="Oldest first: poll again with `after` set to the last `seq`.")
    checking: bool


class SessionSummary(BaseModel):
    session_id: str
    document_id: str
    live: bool
    retain: str
    event_count: int
    check_count: int
    note_count: int
    credit_count: int = Field(
        description="Notes the agent told the user it used, each counted once: assistant lines carrying "
        "`From Hindsight memory (live session #<note seq>)`."
    )
    dismissed_count: int = Field(
        description="Notes the agent told the user it did not need, each counted once: assistant lines carrying "
        "`Hindsight note (live session #<note seq>)`."
    )
    last_seq: int
    saved_seq: int
    acked_seq: int
    checking: bool
    created_at: datetime
    last_event_at: datetime | None = Field(default=None, description="When the client last sent an event.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionListResponse(BaseModel):
    items: list[SessionSummary]
    total: int
    limit: int
    offset: int


class SessionSettings(BaseModel):
    """What a session keeps of its options (the `settings` column): the defaults applied, later
    appends' options merged over it."""

    live: bool = False
    retain: Literal["turns", "idle", "off"] = "turns"
    tags: list[str] | None = None
    context: str | None = None
    strategy: str | None = None
    observation_scopes: Literal["per_tag", "combined", "all_combinations", "shared"] | list[list[str]] | None = None


class SavedTurn(BaseModel):
    """One line of the transcript a save writes into the session's document."""

    role: str
    content: str
    timestamp: str


class CheckLog(BaseModel):
    """A check's entry in the session log: the moment judged and what came of it."""

    trigger: str
    moment_seq: int
    moment: str
    outcome: str
    reason: str | None = None
    error: str | None = None
    ms: int | None = None
    question: str | None = None
    context: str | None = None
    search_queries: list[str] | None = None
    answer: str | None = None
    abstained: bool | None = None
    memory_ids: list[str] | None = None
    # The reflect's token usage and its LLM call traces, as the engine reports them.
    usage: dict[str, Any] | None = None
    llm_calls: list[dict[str, Any]] | None = None


class NoteLog(BaseModel):
    """A note's entry in the session log."""

    text: str
    trigger: str
    check_seq: int
    memory_ids: list[str]
    source: SessionNoteSource | None = None
