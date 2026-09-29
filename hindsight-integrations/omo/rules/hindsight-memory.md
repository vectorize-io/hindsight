---
description: Hindsight long-term memory - recall relevant context before complex tasks
alwaysApply: true
---

# Hindsight Memory

You have access to long-term memory via Hindsight MCP tools. Use them to provide continuity across sessions.

## When to recall

Before starting non-trivial tasks, use the `recall` tool to search for relevant memories:
- Extract 3-5 key terms from the user's request
- Search for prior solutions, debugging insights, or architectural decisions
- If relevant memories exist, use them to inform your approach
- Don't recall for trivial tasks (typo fixes, simple questions)

## When to retain

After completing significant work, use the `retain` tool to store:
- Solutions to problems that could recur
- Key architectural decisions and their reasoning
- User preferences for workflow, coding style, or tools
- Debugging insights that were hard to discover

Do NOT retain:
- Trivial changes or one-liners
- Application code (already tracked in git)
- Sensitive data (API keys, credentials, secrets)

## When to reflect

Use the `reflect` tool when the user asks for synthesis across multiple topics, or when you need to reason about patterns across many memories.
