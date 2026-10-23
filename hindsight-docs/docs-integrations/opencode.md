---
sidebar_position: 20
title: "OpenCode Persistent Memory with Hindsight | Integration"
description: "Add long-term memory to OpenCode with Hindsight. Automatically captures conversations and recalls relevant context across coding sessions."
---

# OpenCode

:::warning Retired — use the Coding Agents plugin
**The opencode plugin** is superseded by the [Coding Agents plugin](/sdks/integrations/coding-agents) — one
package covering Claude Code, Codex, opencode, Kilo, Cursor, Copilot, Grok, Antigravity, Devin and Cline and other CLI agents, with a per-repo memory bank they all share instead
of one bank per agent.

This integration is retired: it is no longer maintained and its published package gets no further updates. To switch:

```bash
cd /path/to/your/repo
npx @vectorize-io/hindsight-coding-agents install opencode
```

Memory does not move automatically — the banks are scoped differently and this agent's history cannot be imported (it is kept in an internal database). See
[Migrating from the per-agent plugins](/sdks/integrations/coding-agents#migrating-from-the-per-agent-plugins).
:::

[View Changelog →](/changelog/integrations/opencode)
