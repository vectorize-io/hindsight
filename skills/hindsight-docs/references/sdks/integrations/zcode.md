
# ZCode

> **⚠️ Retired — use the Coding Agents plugin**
>
**The ZCode integration** is superseded by the [Coding Agents plugin](coding-agents.md) — one
package covering Claude Code, Codex, opencode, Kilo, Cursor, Copilot, Grok, Antigravity, Devin and Cline and other CLI agents, with a per-repo memory bank they all share instead
of one bank per agent.

This integration is retired: it is no longer maintained and its published package gets no further updates. To switch:

```bash
cd /path/to/your/repo
npx @vectorize-io/hindsight-coding-agents install zcode
```

Memory does not move automatically — the banks are scoped differently and ZCode keeps no conversation history on disk to import. See
[Migrating from the per-agent plugins](coding-agents.md#migrating-from-the-per-agent-plugins).
[View Changelog →](../../changelog/integrations/zcode.md)
