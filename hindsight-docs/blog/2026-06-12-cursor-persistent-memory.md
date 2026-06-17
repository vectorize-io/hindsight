---
title: "Cursor Persistent Memory: One Bank for the Editor and the CLI"
authors: [benfrank241]
slug: "2026/06/12/cursor-persistent-memory"
date: 2026-06-12T12:00
tags: [cursor, cursor-cli, memory, persistent-memory, hindsight, coding-agents, tutorial]
description: "Two new Hindsight integrations give Cursor persistent long-term memory: one for the Cursor editor (with a workaround for the broken sessionStart additionalContext channel in Cursor 3.x) and one for Cursor CLI. Both can share the same memory bank."
image: /img/blog/cursor-persistent-memory.png
hide_table_of_contents: true
---

![Cursor Persistent Memory with Hindsight](/img/blog/cursor-persistent-memory.png)

[Cursor](https://cursor.com) has two surfaces a developer actually uses: the editor and the [Cursor CLI](https://cursor.com/docs/cli). Both run agents. Neither remembers anything between sessions. Today there are two new Hindsight integrations that fix that, and (this is the interesting part) they can both write to and read from the same memory bank. Switch from the editor to the CLI mid-task and the agent already knows what you decided yesterday.

This post is a walkthrough of both integrations: `hindsight-cursor` for the editor and `hindsight-cursor-cli` for the command-line agent. The story is mostly the same memory primitives wired into two different lifecycles, with one notable detail on the editor side: Cursor 3.x has a known bug in its session-start context-injection channel, and the editor integration ships with a clean workaround.

## TL;DR

<!-- truncate -->

- Two pip packages: `hindsight-cursor` (editor, first-party) and `hindsight-cursor-cli` (CLI, community-built by [@Korayem](https://github.com/Korayem)).
- Both use Cursor's lifecycle hooks: recall before each chat session/prompt, retain after each task or every N turns.
- **Editor integration also ships an MCP server** for explicit `recall` / `retain` / `reflect` tools during a session, on top of the automatic session-start recall.
- **Editor integration works around a Cursor 3.x bug** that broke the native `additionalContext` injection channel. Memories get written to a workspace rules file that Cursor's rules engine reliably picks up.
- **CLI integration has four hooks** (`sessionStart`, `beforeSubmitPrompt`, `stop`, `sessionEnd`) and requires Cursor CLI v0.45+.
- Point both at the same `bankId` and switching surfaces is transparent.

## Why Cursor Needs Persistent Memory

A new Cursor session is a fresh model with no context. It can read the files in your workspace, and it has whatever you wrote in your `.cursor/rules`. What it doesn't have is anything about previous sessions: the bug you debugged on Tuesday, the architecture decision you talked through last week, the convention you've established for how this codebase names test files.

You can manually pin context with rules files. That works for the things you remember to write down, but not for the things you didn't realize were important until later. Persistent memory closes that gap: every session retains itself, every new session recalls relevant past content automatically.

Two surfaces, one memory layer.

## `hindsight-cursor`: The Editor Integration

The editor integration is a `pip` plugin that installs into your project's `.cursor-plugin/hindsight-memory/` directory. It wires up two complementary mechanisms.

**Plugin hooks (automatic).** `session_start.py` runs when the agent processes the first prompt of each new chat. It runs a project-level recall against your bank and surfaces the result so the agent sees it before it answers. `retain.py` runs on the `stop` event after each task and stores the transcript.

**MCP server (on-demand).** The installer also writes `.cursor/mcp.json` to connect Cursor to Hindsight's MCP endpoint, giving the agent three explicit tools (`recall`, `retain`, `reflect`) that it can call mid-session when it needs memory beyond what was injected at start.

### The Cursor 3.x Workaround

Cursor's native channel for delivering session-start hook output is the `additionalContext` JSON field: the hook prints memory text on stdout, Cursor places it in the agent's system prompt. In Cursor 3.x that channel is broken (the bug is [acknowledged by Cursor staff](https://forum.cursor.com/t/sessionstart-hook-additional-context-is-never-injected-into-agents-initial-system-context/158452), still open against 3.6.31 as of writing). If `additionalContext` is the only delivery path, recalled memories never reach the model.

The plugin works around it by **also** writing the recalled memories to `<workspace>/.cursor/rules/hindsight-session.mdc` with `alwaysApply: true` in the frontmatter. Workspace rules files are reliably injected by Cursor's rules engine, so the agent sees memories on the very first prompt of every new chat.

A few details worth knowing:

- **Every new agent's first prompt has memories.** Cursor blocks prompt submission until the `sessionStart` hook returns; recall latency is typically under a second.
- **The rules file is regenerated at the top of every `sessionStart`.** Stale memories from a previous session don't linger.
- **The rules file is auto-`.gitignore`'d** in git workspaces (idempotent append; no-op for non-git workspaces).
- **`additionalContext` is still emitted to stdout** for forward-compat. If Cursor restores the native channel, the same plugin keeps working without code changes.

If you'd rather not have the plugin touch your workspace rules, set `useRulesFileFallback: false` and the plugin will fall back to the (currently broken) native channel. Useful only if you'd rather see the bug bite than have the plugin write a rules file.

### Install

```bash
pip install hindsight-cursor
cd /path/to/your-project
hindsight-cursor init --api-url https://api.hindsight.vectorize.io --api-token YOUR_KEY
```

Then **fully quit and reopen Cursor**: plugins load at startup.

For self-hosting, drop the `--api-url` / `--api-token` flags and run Hindsight locally (`docker run ghcr.io/vectorize-io/hindsight:latest`). The plugin can also auto-manage a local `hindsight-embed` daemon via `uvx` if you'd rather not run a server yourself.

## `hindsight-cursor-cli`: The CLI Integration

The CLI integration is a separate package, community-built by [@Korayem](https://github.com/Korayem), that wires four Cursor CLI hooks to the same recall/retain primitives.

| Hook | Action |
|---|---|
| `sessionStart` | Confirms Hindsight is reachable and pre-warms the local daemon if needed |
| `beforeSubmitPrompt` | Recalls relevant memories and injects them as `additional_context` |
| `stop` | Retains the conversation to long-term memory every configured N turns |
| `sessionEnd` | Forces a final retain so short sessions are still stored |

Two things this lifecycle does differently from the editor plugin:

- **Recall fires before every prompt, not just session start.** The CLI is a more turn-by-turn surface, and per-prompt recall means each new question gets context fitted to it rather than relying on a single broad project-level recall at the top.
- **`sessionEnd` guarantees a retain.** Short CLI sessions (one or two prompts, then exit) used to drop on the floor if `stop` only fired every N turns. The `sessionEnd` hook closes that gap.

### Install

```bash
pip install hindsight-cursor-cli
hindsight-cursor-cli install --api-url https://api.hindsight.vectorize.io --api-token YOUR_KEY
```

The installer copies the hook scripts to `~/.cursor/hooks/cursor-cli/`, writes `~/.cursor/hooks.json` (merged with any existing entries) with absolute paths to the scripts, and seeds `~/.hindsight/cursor-cli.json` for personal overrides. Restart Cursor CLI to load the hooks.

Cursor CLI v0.45+ is required for hooks support. The hook scripts use Python 3 stdlib only, with no `pip install` required at runtime.

## Sharing One Bank Across Both Surfaces

Both integrations write to a `bankId`. By default the editor uses `cursor` and the CLI uses `cursor-cli`. Point them at the same bank and they share memory:

```json
// ~/.hindsight/cursor.json AND ~/.hindsight/cursor-cli.json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "hsk_your_key",
  "bankId": "my-cursor-memory"
}
```

Now anything you retain from a CLI session shows up in the editor's session-start recall, and vice-versa. A decision you talked through with the CLI agent on the train Tuesday morning lands in your editor's first prompt Tuesday afternoon.

For per-project isolation across both surfaces, switch both configs to dynamic bank IDs:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "project"]
}
```

Both integrations derive `project` from the workspace path (basename). The CLI also supports `gitProject` if you want to share memory across worktrees of the same repo.

## What's the Same and What's Different

Most of the config surface is shared. The differences are about what each surface can do, not how Hindsight behaves.

| | `hindsight-cursor` (editor) | `hindsight-cursor-cli` (CLI) |
|---|---|---|
| Author | Hindsight team (first-party) | [@Korayem](https://github.com/Korayem) (community) |
| Hooks | `sessionStart`, `stop` | `sessionStart`, `beforeSubmitPrompt`, `stop`, `sessionEnd` |
| Recall trigger | Once per chat (session start) | Once per prompt |
| MCP server | Yes (`recall` / `retain` / `reflect` tools) | No |
| Cursor 3.x `additionalContext` workaround | Workspace rules-file fallback | n/a (uses different channel) |
| Local daemon auto-management | Yes (`uvx hindsight-embed`) | Yes |
| Default `bankId` | `cursor` | `cursor-cli` |
| Cursor version requirement | Cursor 3.x (works around the bug) | Cursor CLI v0.45+ |
| Runtime Python deps | stdlib only | stdlib only |

Both share: full `retainMode` semantics (`full-session` / `chunked`), `recallBudget` (`low` / `mid` / `high`), `recallMaxTokens`, `dynamicBankId` + `dynamicBankGranularity`, `HINDSIGHT_*` environment-variable overrides, and Hindsight Cloud or self-host connection.

## Tradeoffs

**Editor: the rules-file workaround puts a file in your workspace.** It's small, regenerated every session, and auto-`.gitignore`'d. Most people won't notice. If you'd rather not have the plugin write to your workspace at all, `useRulesFileFallback: false` reverts to the native channel, at the cost of no memory delivery until Cursor fixes the upstream bug.

**CLI: per-prompt recall costs more.** Every prompt triggers a Hindsight query before the agent runs. On Hindsight Cloud that's typically well under a second, but it's not free. Drop `recallBudget` to `"low"`, or set `autoRecall: false`, if you need to skip it.

**Both: retain extracts facts asynchronously.** The retain call returns when the transcript lands in the bank, not when the extractor finishes. Facts become recallable within seconds. That's fine for chat-style workflows where the next session is at least a minute away. For automated scripts that retain-then-recall in the same run, give the extractor a beat.

**Two banks vs. one is a real choice.** Sharing one bank is convenient but mixes editor and CLI usage patterns in the same fact set. Two banks gives each surface its own context but loses the cross-surface continuity. Most people will be happier with one shared bank; tag-based filtering (`retainTags`) can still differentiate which surface produced what if you need to.

## Setup

The fast path for both:

1. Sign up at [hindsight.vectorize.io](https://ui.hindsight.vectorize.io/signup) (free tier is enough).
2. Grab an API key from the dashboard.
3. Install whichever integration(s) you want:
   ```bash
   pip install hindsight-cursor                  # editor
   pip install hindsight-cursor-cli              # CLI
   ```
4. Run the installer once per integration:
   ```bash
   cd /path/to/your-project
   hindsight-cursor init --api-url https://api.hindsight.vectorize.io --api-token YOUR_KEY
   hindsight-cursor-cli install --api-url https://api.hindsight.vectorize.io --api-token YOUR_KEY
   ```
5. Restart Cursor (fully quit and reopen the editor; restart the CLI session). The agent has memory from the next prompt onward.

If you want both surfaces to share a bank, set the same `bankId` in `~/.hindsight/cursor.json` and `~/.hindsight/cursor-cli.json`.

## Recap

| | Cursor default | With Hindsight |
|---|---|---|
| Memory across sessions (editor) | None | Recalled at session start, retained on `stop` |
| Memory across prompts (CLI) | None | Recalled before each prompt, retained on `stop` + `sessionEnd` |
| On-demand memory tools in editor | None | MCP `recall` / `retain` / `reflect` |
| Per-project isolation | Manual rules | Optional via `dynamicBankId` |
| Shared memory across editor + CLI | n/a | One `bankId` across both configs |
| Cursor 3.x `additionalContext` bug | Hits you | Worked around via rules file |

## Next Steps

- **Hindsight Cloud:** [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io/signup)
- **Editor integration docs:** [Cursor + Hindsight](/sdks/integrations/cursor)
- **CLI integration docs:** [Cursor CLI + Hindsight](/sdks/integrations/cursor-cli)
- **Sources:**
  [`hindsight-integrations/cursor`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/cursor),
  [`hindsight-integrations/cursor-cli`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/cursor-cli)
- **Hindsight API reference:** [API quickstart](/developer/api/quickstart)
