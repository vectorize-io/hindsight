
# Codex

> **⚠️ Retired — use the Coding Agents plugin**
>
**The Codex CLI integration** is superseded by the [Coding Agents plugin](coding-agents.md) — one
package covering Claude Code, Codex, opencode, Kilo, Cursor, Copilot, Grok, Antigravity, Devin and Cline and other CLI agents, with a per-repo memory bank they all share instead
of one bank per agent.

This integration is retired: it is no longer maintained and its published package gets no further updates. To switch:

```bash
cd /path/to/your/repo
npx @vectorize-io/hindsight-coding-agents install codex --import-conversations
```

Two things move, and nothing else does.

**Your server moves automatically.** `install` reads `~/.hindsight/codex.json` and keeps the same
`apiUrl` and token — an empty URL still means the local daemon — so you are not silently switched
to Hindsight Cloud. Pass `--server` to change it.

**Your conversations are re-imported from local transcripts** by `--import-conversations`, as new
documents. They are not copied from the old bank: that bank's default was a single static bank
shared by every project, and its documents record only a session id, so working out which ones
belong to this repo means reading the local transcripts anyway.

**Nothing else carries over.** The old `recall*` and `retain*` settings, missions and bank-naming
options describe a pipeline this package replaced. Bank naming changes too — one bank per repo,
shared by every agent. See
[Migrating from the per-agent plugins](coding-agents.md#migrating-from-the-per-agent-plugins).
[View Changelog →](../../changelog/integrations/codex.md)
