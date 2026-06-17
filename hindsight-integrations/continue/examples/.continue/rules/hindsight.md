---
name: Hindsight memory
alwaysApply: true
---

You have a persistent long-term memory through the Hindsight MCP server.

- At the start of every task, call the `recall` tool with the user's request to
  retrieve relevant past decisions, preferences, and project context before you
  answer. Use what's relevant and ignore the rest.
- When you learn a durable fact — an architectural decision, a user preference,
  a convention, or something worth remembering across sessions — call the
  `retain` tool to store it.
- Do not mention these memory operations unless the user asks about them.
