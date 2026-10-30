---
name: "Hindsight Memory"
description: "Use Hindsight (the connected MCP app) as the single long-term memory — save with its retain/sync_retain and retrieve with recall/reflect. Do not use OpenBot's built-in remember/search_memories tools."
---

# Hindsight Memory

When the **Hindsight** app is connected, Hindsight is your one and only long-term memory. It persists across threads, survives context compaction, and is shared with the other agents on the same bank. Do not rely on the thread alone to remember.

## Use Hindsight's tools — not the built-in memory

OpenBot also ships built-in memory tools (`remember`, `search_memories`, `set_memory_inclusion`, `forget_memory`, and a built-in `list_memories`). **Do not use those** — they write to a separate, per-agent store that is not shared and is not where the user keeps their memory. For anything memory-related, use the Hindsight MCP tools instead:

| To do this            | Use (Hindsight)          | Do NOT use (built-in) |
| --------------------- | ------------------------ | --------------------- |
| Save a fact           | `sync_retain` (or `retain`) | `remember`         |
| Look something up     | `recall`                 | `search_memories`     |
| Reason across memory  | `reflect`                | —                     |

If a tool name is ambiguous (e.g. two `list_memories`), prefer the Hindsight one that operates on the connected bank. When in doubt, use `recall`, `sync_retain`, and `reflect` — those are Hindsight's.

## Workflow

1. **Recall first.** Before answering anything about the user, the project, past decisions, or prior work, call `recall`. Use what it returns; don't guess when memory can tell you.
2. **Save with `sync_retain`.** When the user states a preference, decision, fact, goal, or a result worth keeping, call Hindsight's **`sync_retain`** (not `remember`), so it's stored and immediately recallable. Note the returned memory id to confirm it saved to Hindsight.
3. **Reflect** for synthesis questions ("what do you know about X", "how should you approach this").
4. Don't store secrets, transient chatter, or duplicates — recall first, and correct a wrong memory rather than adding a contradicting one.

## Notes

- Memory lives in the Hindsight **bank** from the connection's MCP URL; agents on the same bank share it, so give an agent its own bank URL for isolation.
- Retained facts are re-worded on extraction — recall by meaning, not exact wording.
