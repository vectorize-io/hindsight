#!/usr/bin/env python3
"""
Sessions API examples for Hindsight (live sessions).
Run: python examples/api/sessions.py
"""
import asyncio
import os
import uuid

from hindsight_client import Hindsight
from hindsight_client_api.models import SessionAppendRequest

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
client = Hindsight(base_url=HINDSIGHT_URL)
session_id = f"chat-{uuid.uuid4().hex[:8]}"


async def main() -> None:
    # [docs:sessions-append]
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
    # [/docs:sessions-append]

    # [docs:sessions-notes]
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
    # [/docs:sessions-notes]

    # [docs:sessions-turn-end]
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
    # [/docs:sessions-turn-end]

    # [docs:sessions-follow]
    # Follow the session's log: your events and Hindsight's checks, notes and saves, in one order.
    after = 0
    log = await client.sessions.list_session_events("my-bank", session_id, after=after)
    for entry in log.items:
        print(entry.seq, entry.source, entry.kind)
    after = log.items[-1].seq if log.items else after  # poll again from the last entry you saw
    # [/docs:sessions-follow]

    # [docs:sessions-list]
    sessions = await client.sessions.list_sessions("my-bank", limit=20)
    for s in sessions.items:
        print(s.session_id, s.last_event_at, s.note_count)
    # [/docs:sessions-list]


    # [docs:sessions-delete]
    # Deletes the session and its log. The document it saved stays: that is memory now.
    await client.sessions.delete_session("my-bank", session_id)
    # [/docs:sessions-delete]

    # Cleanup (not shown in docs)
    await client.adelete_bank(bank_id="my-bank")
    await client.aclose()


asyncio.run(main())
