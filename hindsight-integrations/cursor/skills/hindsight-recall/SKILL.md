---
name: hindsight-recall
description: Search long-term memory for relevant context from past coding sessions using Hindsight MCP tools
---

# Hindsight Recall

## Trigger

Use when the user explicitly asks about past decisions, project context, preferences, or anything that may have been discussed in prior sessions.

## Workflow

1. Identify the key topic or question from the user's request.
2. Check if `<hindsight_memories>` in the current context already contains relevant results from the automatic session-start recall.
3. If session memory already covered it, use those memories directly.
4. If the user needs more specific or deeper recall, use the Hindsight MCP `recall` tool to search for additional memories.
5. If you want to reason over accumulated memories for architectural decisions, use the `reflect` tool.

## Guardrails

- Only use MCP tools for deeper recall when the session-start memory was insufficient.
- When memories conflict with current context, prefer current context and note the discrepancy.
- Do not expose raw memory metadata to the user unless asked.

## Output

- Relevant memories integrated into the response
- If MCP tools are not available, advise the user to check their `.cursor/mcp.json` configuration
