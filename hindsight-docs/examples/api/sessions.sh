#!/bin/bash
# Sessions API examples for Hindsight CLI (live sessions)
# Run: bash examples/api/sessions.sh

set -e

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
SESSION_ID="chat-$RANDOM$RANDOM"
MESSAGES_FILE=$(mktemp)
cat > "$MESSAGES_FILE" <<'EOF'
[
  {"id": "2", "role": "assistant", "tool_calls": [{"id": "t1", "type": "function",
    "function": {"name": "edit_file", "arguments": "{\"path\": \"pay/rounding.py\"}"}}]},
  {"id": "3", "role": "tool", "tool_call_id": "t1", "content": "updated"}
]
EOF


# =============================================================================
# Doc Examples
# =============================================================================

# [docs:sessions-append]
# One message from the flags; --id makes a retry store nothing twice. The first append creates the session.
hindsight session append my-bank "$SESSION_ID" --id 1 --role user --text "Half cents round the wrong way." \
  --live --retain turns

# Several messages at once (tool calls too), from a JSON array of OpenAI-style messages.
hindsight session append my-bank "$SESSION_ID" --messages-file "$MESSAGES_FILE"
# [/docs:sessions-append]


# [docs:sessions-notes]
# With no message, an append just fetches the notes after --ack (the seq of the last note you delivered).
hindsight session append my-bank "$SESSION_ID" --ack 0
# [/docs:sessions-notes]


# [docs:sessions-turn-end]
# The end of a turn saves the conversation into the session's document.
hindsight session append my-bank "$SESSION_ID" --id 4 --role assistant --text "Fixed: half cents round down." --turn-end
# [/docs:sessions-turn-end]


# [docs:sessions-follow]
# The session's log, one line per entry. Add --follow to keep printing it live (Ctrl-C to stop).
hindsight session events my-bank "$SESSION_ID"
# [/docs:sessions-follow]


# [docs:sessions-list]
hindsight session list my-bank
# [/docs:sessions-list]


# [docs:sessions-delete]
# Deletes the session and its log. The document it saved stays: that is memory now.
hindsight session delete my-bank "$SESSION_ID"
# [/docs:sessions-delete]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
rm -f "$EVENTS_FILE"
hindsight bank delete my-bank -y 2>/dev/null || true
