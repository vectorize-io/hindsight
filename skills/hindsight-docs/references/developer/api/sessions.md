
# Live sessions

A **live session** is one conversation that your client streams into a bank while it happens: a
coding agent's run, a support chat, an assistant's day. Hindsight keeps it as a log, saves it into
memory at the end of each turn, and, with `live` on, follows it and sends notes back when memory
holds something the agent must hear right now.

It turns retain and reflect into a three-way conversation: the user, your agent, and Hindsight
following along.

**Figure: Session Lifecycle.** An animated diagram on the docs site; its narration, step by step:

- **Append & notes**
  1. The agent sends its new messages (user, assistant, tool calls and results), each with its own id. The first append creates the session.
  2. Hindsight writes them to the session log. A message id it already has is skipped, so a retried append stores nothing twice.
  3. The reply comes back right away: the notes the agent has not acknowledged yet, and whether a check is still running. The agent never waits.
  4. Anyone can read the log from a position. The control plane's Sessions view polls it to follow the session live.
- **Watch check**
  1. With watch on, some moments are worth a look: a new request, a plan, a command that changes things, an edit, a failure.
  2. The append queues a check and returns at once. The check waits until the session is quiet (1 s, or 3 s at most), so a plan and the command after it are one check.
  3. A check is one reflect over observations, with search queries and permission to abstain. A decision model settles most moments with no LLM call.
  4. Memory holds a decision this edit goes against.
  5. The answer becomes a note in the log. Most checks end silent, and that is logged too. At most three notes per session, never the same memory twice.
  6. The agent gets the note in the reply to its next append, and hands it over.
  7. Then it sends ack = the note's seq, so the note is not sent again.
- **Save**
  1. The agent says its turn is over.
  2. With retain: turns (the default), a turn end queues a save.
  3. The save appends the new turns to the session's document. Tool output and the git diff are left out. Memories are extracted as from any retained document.
  4. An append that leaves turns unsaved also starts an idle timer, one per session. When the session has been quiet for 10 minutes, the timer queues a save.
  5. There is no close: a session that finished and one that crashed end the same way. With retain: idle the session is saved only here, in one go; with off, never.

{/* Import raw source files */}

> **💡 Prerequisites**
>
Make sure you've completed the [Quick Start](./quickstart.md) to install the client and start the server.
## How it works

Your client does one thing: after something happens, it sends the new messages.

- **Append.** `POST /v1/default/banks/{bank_id}/sessions/{session_id}/events` with the messages since
  your last append. The first append creates the session. The response comes back in milliseconds
  and carries any notes waiting for you.
- **Follow.** With `live` on, each moment worth a look (a new request, a plan, a command that changes
  things, a file edit, a failure) queues one memory check in the background. Your append never waits
  for it. A check runs one [reflect](./reflect.md) with `abstain`: most moments end silently, and a note is
  written only when memory records a decision the moment does not state or goes against.
- **Save.** At the end of a turn (and every 300 events of a turn that keeps going), and when the session goes idle, the new turns are appended to the
  session's document, and memories are extracted from it like from any [retained](./retain.md) document.
  There is no "close": a session is idle once it has been quiet for a while (10 minutes by default),
  so a client that finishes and a client that crashes mid-turn end the same way, with nothing lost.
- **Read.** `GET .../events?after=N` returns the session's log from position `N`: your messages (one entry per text, tool call and tool result) and
  Hindsight's own entries (each check, each note, each save) in one order. Poll it to follow a
  session live; the control plane's **Sessions** view does exactly that.

### Safe to retry, any server

The protocol is plain request and response, so any server instance can take any request, and every
call is safe to repeat:

- Each message carries **your own id**, unique in the session. An append that is retried after a lost
  response stores nothing twice.
- **Notes are delivered until you acknowledge them.** Send `ack` = the `seq` of the last note you
  handed to your agent; the response carries only the notes after it.
- Checks and saves are queued jobs. A server that dies in the middle of one leaves it to the next
  worker, and a save that is retried writes the same turns once.

## Sending messages

### Python

```python
# Send the messages since the last append, in the OpenAI chat format your agent likely
# keeps already, each with your own id: sending one again (a retry after a lost response)
# stores nothing twice. The first append creates the session.
r = await client.sessions.append_session_events(
    "my-bank",
    session_id,
    SessionAppendRequest.from_dict({
        "session": {"live": True, "retain": "turns"},
        "messages": [
            {"id": "1", "role": "user", "content": "Half cents round the wrong way."},
            {"id": "2", "role": "assistant", "tool_calls": [{
                "id": "t1", "type": "function",
                "function": {"name": "edit_file", "arguments": '{"path": "pay/rounding.py"}'},
            }]},
            {"id": "3", "role": "tool", "tool_call_id": "t1", "content": "updated"},
        ],
    }),
)
print(r.notes, r.checking)
```

### Node.js

```javascript
// Send the messages since the last append, in the OpenAI chat format your agent likely
// keeps already, each with your own id: sending one again (a retry after a lost response)
// stores nothing twice. The first append creates the session.
const { data: appended } = await sdk.appendSessionEvents({
    client: apiClient,
    path: { bank_id: 'my-bank', session_id: sessionId },
    body: {
        session: { live: true, retain: 'turns' },
        messages: [
            { id: '1', role: 'user', content: 'Half cents round the wrong way.' },
            {
                id: '2',
                role: 'assistant',
                tool_calls: [{ id: 't1', type: 'function', function: { name: 'edit_file', arguments: '{"path": "pay/rounding.py"}' } }],
            },
            { id: '3', role: 'tool', tool_call_id: 't1', content: 'updated' },
        ],
    },
});
console.log(appended.notes, appended.checking);
```

### CLI

```bash
# One message from the flags; --id makes a retry store nothing twice. The first append creates the session.
hindsight session append my-bank "$SESSION_ID" --id 1 --role user --text "Half cents round the wrong way." \
  --live --retain turns

# Several messages at once (tool calls too), from a JSON array of OpenAI-style messages.
hindsight session append my-bank "$SESSION_ID" --messages-file "$MESSAGES_FILE"
```

### Go

```go
// Send the messages since the last append, in the OpenAI chat format your agent likely
// keeps already, each with your own id: sending one again (a retry after a lost response)
// stores nothing twice. The first append creates the session.
opts := hindsight.NewSessionOptions()
opts.SetLive(true)
opts.SetRetain("turns")

text := func(s string) hindsight.Content1 { return hindsight.Content1{String: &s} }
ask := hindsight.NewSessionMessage("1", "user")
ask.SetContent(text("Half cents round the wrong way."))
args := `{"path": "pay/rounding.py"}`
call := hindsight.NewSessionMessage("2", "assistant")
call.SetToolCalls([]hindsight.ToolCall{*hindsight.NewToolCall("t1",
	*hindsight.NewToolCallFunction("edit_file", hindsight.Arguments{String: &args}))})
result := hindsight.NewSessionMessage("3", "tool")
result.SetToolCallId("t1")
result.SetContent(text("updated"))

req := hindsight.NewSessionAppendRequest()
req.SetSession(*opts)
req.SetMessages([]hindsight.SessionMessage{*ask, *call, *result})
appended, _, err := client.SessionsAPI.AppendSessionEvents(ctx, "my-bank", sessionID).
	SessionAppendRequest(*req).Execute()
if err != nil {
	log.Fatalf("append: %v", err)
}
fmt.Println(appended.Notes, appended.Checking)
```

Messages are in the OpenAI chat format most agents already keep, plus your own `id` per message
(unknown keys are ignored, so you can send the messages you have as they are):

| Message | Fields |
|---|---|
| Something said | `role` (`user`, `assistant`, `system`), `content` (a string, or a list of parts whose text parts are read) |
| The assistant calling tools | `role: assistant`, `tool_calls`: `[{id, type: "function", function: {name, arguments}}]` (`arguments` a JSON string or object) |
| What a tool returned | `role: tool`, `tool_call_id`, `content`, `is_error` |

Up to 500 messages per append. Two more fields of the append:

- `state`: a snapshot of the work, said by no one: `{"diff": "..."}`, the `git diff` of everything a
  coding agent changed so far. An edit is checked against the whole change. Never saved into memory.
- `turn_end: true`: the assistant finished its turn. Sent again with nothing new in between, it is
  the same turn end.

Hindsight recognizes tools by what their names say, whichever agent named them: an edit (`Edit`,
`apply_patch`, `write_file`, `patch`), a command (`Bash`, `shell`, `terminal`), or anything else.

### Session options

Sent with an append (`session`), applied when the append creates the session; later appends that
carry options update what they set.

| Option | Meaning | Default |
|---|---|---|
| `live` | Hindsight follows the session and may send notes back; off, the session is only saved | `false` |
| `retain` | When the conversation is saved: `turns` (each turn end, every 300 events of a turn that keeps going, and when idle), `idle` (only when idle: one extraction for the whole session), or `off` | `turns` |
| `document_id` | The document it is saved into; fixed once the session exists | `session:<session_id>` |
| `metadata` | Labels, copied onto the saved document. `harness` names the client (e.g. `claude-code`) and the Control Plane shows its logo | — |
| `tags`, `context`, `strategy`, `observation_scopes` | Passed to the save's retain, as on [retain](./retain.md) | the bank's |

## Getting notes back

### Python

```python
# Notes come back with every append. `ack` is the seq of the last note you delivered:
# the response carries the notes after it, so a note is handed over until you confirm it.
delivered = 0
r = await client.sessions.append_session_events(
    "my-bank", session_id, SessionAppendRequest(ack=delivered)
)
for note in r.notes:
    source = note.source.document_id if note.source else "memory"
    print(f"#{note.seq} {note.text} (source: {source})")
    delivered = note.seq
```

### Node.js

```javascript
// Notes come back with every append. `ack` is the seq of the last note you delivered:
// the response carries the notes after it, so a note is handed over until you confirm it.
let delivered = 0;
const { data: polled } = await sdk.appendSessionEvents({
    client: apiClient,
    path: { bank_id: 'my-bank', session_id: sessionId },
    body: { ack: delivered },
});
for (const note of polled.notes) {
    console.log(`#${note.seq} ${note.text} (source: ${note.source?.document_id ?? 'memory'})`);
    delivered = note.seq;
}
```

### CLI

```bash
# With no message, an append just fetches the notes after --ack (the seq of the last note you delivered).
hindsight session append my-bank "$SESSION_ID" --ack 0
```

### Go

```go
// Notes come back with every append. `ack` is the seq of the last note you delivered:
// the response carries the notes after it, so a note is handed over until you confirm it.
delivered := int32(0)
poll := hindsight.NewSessionAppendRequest()
poll.SetAck(delivered)
polled, _, err := client.SessionsAPI.AppendSessionEvents(ctx, "my-bank", sessionID).
	SessionAppendRequest(*poll).Execute()
if err != nil {
	log.Fatalf("poll: %v", err)
}
for _, note := range polled.Notes {
	fmt.Printf("#%d %s\n", note.Seq, note.Text)
	delivered = note.Seq
}
```

Every append answers with just two fields: `notes` (the notes after your `ack`) and `checking`. A
note has its `seq` (send it back as `ack` once delivered), its `text`, and its `source`: the document
it comes from and that document's own sentence closest to the note (`quote`), so your agent can check
it. What caused a note and the memories it used are in the session's log (its `check` entry).
`checking` is `true` while a check is queued or running: another note may still come, so a client at
the end of a turn can poll a little longer.

The rules for notes: Hindsight waits for the session to settle before checking (a plan and the command
right after it are one check), never says the same memory twice, judges the same situation once,
and sends at most three notes per session.

Ask your agent to tell the user about every note, in its reply, with one of two lines naming the note
by its `seq`: a note it used starts with `🧠 **From Hindsight memory (live session #12)**`, and a note
it did not need with `🧠 **Hindsight note (live session #12)**`. Hindsight reads both in the agent's
messages as they stream in and counts each note of the session once: the session's `credit_count`
(notes used) and `dismissed_count` (notes shown but not needed).

## Saving into memory

### Python

```python
# The end of a turn saves the conversation into the session's document
# (`session:<id>` unless you named one), where memories are extracted from it.
await client.sessions.append_session_events(
    "my-bank",
    session_id,
    SessionAppendRequest.from_dict({
        "messages": [{"id": "4", "role": "assistant", "content": "Fixed: half cents round down."}],
        "turn_end": True,
        "ack": delivered,
    }),
)
```

### Node.js

```javascript
// The end of a turn saves the conversation into the session's document
// (`session:<id>` unless you named one), where memories are extracted from it.
await sdk.appendSessionEvents({
    client: apiClient,
    path: { bank_id: 'my-bank', session_id: sessionId },
    body: {
        messages: [{ id: '4', role: 'assistant', content: 'Fixed: half cents round down.' }],
        turn_end: true,
        ack: delivered,
    },
});
```

### CLI

```bash
# The end of a turn saves the conversation into the session's document.
hindsight session append my-bank "$SESSION_ID" --id 4 --role assistant --text "Fixed: half cents round down." --turn-end
```

### Go

```go
// The end of a turn saves the conversation into the session's document
// (`session:<id>` unless you named one), where memories are extracted from it.
reply := hindsight.NewSessionMessage("4", "assistant")
reply.SetContent(text("Fixed: half cents round down."))
turn := hindsight.NewSessionAppendRequest()
turn.SetMessages([]hindsight.SessionMessage{*reply})
turn.SetTurnEnd(true)
turn.SetAck(delivered)
if _, _, err := client.SessionsAPI.AppendSessionEvents(ctx, "my-bank", sessionID).
	SessionAppendRequest(*turn).Execute(); err != nil {
	log.Fatalf("turn end: %v", err)
}
```

What is saved: what the user and the assistant said, and one short line per tool call (the tool and
its main target, such as `Edit pay/rounding.py`). Tool output and `state` snapshots are not saved.
Each save appends to the document, so its memories are extracted from the new turns only.

## Following a session

### Python

```python
# Follow the session's log: your events and Hindsight's checks, notes and saves, in one order.
after = 0
log = await client.sessions.list_session_events("my-bank", session_id, after=after)
for entry in log.items:
    print(entry.seq, entry.source, entry.kind)
after = log.items[-1].seq if log.items else after  # poll again from the last entry you saw
```

### Node.js

```javascript
// Follow the session's log: your events and Hindsight's checks, notes and saves, in one order.
let after = 0;
const { data: log } = await sdk.listSessionEvents({
    client: apiClient,
    path: { bank_id: 'my-bank', session_id: sessionId },
    query: { after },
});
for (const entry of log.items) {
    console.log(entry.seq, entry.source, entry.kind);
}
after = log.items.at(-1)?.seq ?? after; // poll again from the last entry you saw
```

### CLI

```bash
# The session's log, one line per entry. Add --follow to keep printing it live (Ctrl-C to stop).
hindsight session events my-bank "$SESSION_ID"
```

### Go

```go
// Follow the session's log: your events and Hindsight's checks, notes and saves, in one order.
entries, _, err := client.SessionsAPI.ListSessionEvents(ctx, "my-bank", sessionID).After(0).Execute()
if err != nil {
	log.Fatalf("events: %v", err)
}
for _, entry := range entries.Items {
	fmt.Println(entry.Seq, entry.Source, entry.Kind)
}
// poll again with After(<the last entry's Seq>) to get only what is new
```

Each log entry has a `seq`, a `source` (`client` or `hindsight`), a `kind` and a `payload`.
Hindsight's entries:

- `check`: one check of a moment: the `trigger`, the `outcome` (`note`, `silent`, `repeat` when the
  memory was already told, `skipped`, `error`), how long it took, and everything sent to reflect
  (`question`, `context`, `search_queries`) with its decision path (`llm_calls`) and tokens.
- `note`: a note sent back, with `check_seq` pointing at its check.
- `handoff`: the moment a note first went back to the client, in the reply to an append:
  `note_seq` is the note and `with_seqs` the events that append logged (empty for a poll).
- `save`: the log range saved into the document.

## Listing and deleting

### Python

```python
sessions = await client.sessions.list_sessions("my-bank", limit=20)
for s in sessions.items:
    print(s.session_id, s.last_event_at, s.note_count)
```

### Node.js

```javascript
const { data: sessions } = await sdk.listSessions({
    client: apiClient,
    path: { bank_id: 'my-bank' },
    query: { limit: 20 },
});
for (const s of sessions.items) {
    console.log(s.session_id, s.last_event_at, s.note_count);
}
```

### CLI

```bash
hindsight session list my-bank
```

### Go

```go
sessions, _, err := client.SessionsAPI.ListSessions(ctx, "my-bank").Limit(20).Execute()
if err != nil {
	log.Fatalf("list: %v", err)
}
for _, s := range sessions.Items {
	fmt.Println(s.SessionId, s.GetLastEventAt(), s.NoteCount)
}
```

### Python

```python
# Deletes the session and its log. The document it saved stays: that is memory now.
await client.sessions.delete_session("my-bank", session_id)
```

### Node.js

```javascript
// Deletes the session and its log. The document it saved stays: that is memory now.
await sdk.deleteSession({ client: apiClient, path: { bank_id: 'my-bank', session_id: sessionId } });
```

### CLI

```bash
# Deletes the session and its log. The document it saved stays: that is memory now.
hindsight session delete my-bank "$SESSION_ID"
```

### Go

```go
// Deletes the session and its log. The document it saved stays: that is memory now.
if _, _, err := client.SessionsAPI.DeleteSession(ctx, "my-bank", sessionID).Execute(); err != nil {
	log.Fatalf("delete: %v", err)
}
```

## Access

Sessions go through the same extension hooks as other bank reads and writes
(`append_session_events`, `delete_session`; `list_sessions`, `get_session`,
`list_session_events`). A caller confined to a tag scope is refused (`403`): the checks search the
whole bank, so the notes could carry memories outside the caller's tags.

## Configuration

| Variable | What it does | Default |
|---|---|---|
| `HINDSIGHT_API_SESSION_CHECK_QUIET_MS` | A moment is checked once the session has been quiet this long | `1000` |
| `HINDSIGHT_API_SESSION_CHECK_MAX_WAIT_MS` | ...or at the latest this long after the moment appeared | `3000` |
| `HINDSIGHT_API_SESSION_IDLE_MS` | A session quiet this long is idle: its turns not saved yet are saved | `600000` |

The checks use reflect's [decision model](../configuration.md#decision-model) when one is set: most
moments are then settled without an LLM call.

The coding-agents plugin streams every Claude Code and Codex session into a live session (see
[Coding Agents](../../sdks/integrations/coding-agents.md)).
