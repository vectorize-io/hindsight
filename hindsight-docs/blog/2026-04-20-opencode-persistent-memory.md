---
title: "Your OpenCode Agent Forgets Everything Between Sessions. Here's the Fix."
authors: [DK09876]
date: 2026-04-20
tags: [opencode, memory, integrations, typescript, tutorial]
description: "OpenCode sessions are stateless by default. The Hindsight Coding Agents plugin adds persistent memory via automatic hooks and three explicit tools."
image: /img/blog/opencode-persistent-memory.png
hide_table_of_contents: true
---

![Your OpenCode Agent Forgets Everything Between Sessions. Here's the Fix.](/img/blog/opencode-persistent-memory.png)

OpenCode gives you a fast, terminal-native AI coding agent. But every session starts cold — no memory of what you built yesterday, what conventions you follow, or what decisions you've already made. Here's how to give it persistent long-term memory with one plugin.

<!-- truncate -->

## TL;DR

- OpenCode sessions are stateless — every conversation starts from scratch
- The Hindsight Coding Agents plugin adds persistent memory: one `npx` command installs it, and it registers itself in OpenCode's config
- Relevant memory is injected on the first prompt of a session, so the agent has context before it starts working
- Your conversations and your repo's git history flow into memory in the background — no setup step, no explicit save
- Each repository gets its own memory bank by default, shared by every coding agent you use on it
- Works with [Hindsight Cloud](https://ui.hindsight.vectorize.io) (the default), a self-hosted server, or a local daemon

---

## The Problem: No Persistent Memory Between Sessions

OpenCode is a powerful terminal-based coding agent. You can install plugins, wire up providers, and get real work done. But every session is isolated. There's no mechanism for the agent to remember what happened in previous sessions.

Ask your agent to use a specific test framework. It does. Next session, it doesn't remember. Explain your project's architecture, your deployment process, your naming conventions. All gone once the session ends.

For quick one-off tasks, this is fine. For ongoing project work where continuity matters, it's a real limitation.

---

## The Approach

[Hindsight](https://github.com/vectorize-io/hindsight) is a memory layer for AI agents. It extracts facts from conversations, retrieves them semantically at query time, and can synthesize reasoned answers from accumulated context.

The Hindsight Coding Agents plugin integrates Hindsight directly into OpenCode's plugin system as native tools — no MCP server needed. It works in two modes:

**Automatic** — the plugin handles these behind the scenes:
- Injects relevant memory on the first prompt of a session
- Writes the session back to memory every turn, plus a flush when the session goes idle
- Ingests the repo's git history in the background
- Keeps a curated set of **pages** (architecture, conventions, in-flight initiatives) that later sessions start from

**Explicit** — tools the agent can call directly:
- `hindsight_search_pages` — first stop for project questions (components, conventions, past decisions)
- `hindsight_read_page` / `hindsight_list_pages` — read pages in full
- `hindsight_reflect` — deep reasoning over the whole memory for "why" questions and exact decided values
- `hindsight_ingest_document` — store an external document, notes or durable findings
- `hindsight_capture_initiative` — record a new feature or plan as a page once it is agreed

```
Session start
    │
    ▼
┌──────────────────────────────────────────┐
│  First prompt                            │
│  + injected memory (automatic)           │
├──────────────────────────────────────────┤
│  OpenCode Agent                          │
│  ├── hindsight_search_pages   (explicit) │
│  ├── hindsight_reflect        (explicit) │
│  └── hindsight_ingest_document (explicit)│
└──────────────────────────────────────────┘
        │                   ▲
   write-back             recall
        │                   │
        ▼                   │
  ┌───────────────────────────────┐
  │   Hindsight bank (per repo)   │
  └───────────────────────────────┘
```

The automatic part handles most use cases. The explicit tools are there for when the agent (or you) wants to dig deeper or store something on purpose.

---

## Setup

### 1. Install the plugin

```bash
npx @vectorize-io/hindsight-coding-agents install opencode
```

The installer writes the plugin entry into `~/.config/opencode/opencode.json` for you — there is nothing to add by hand. On OpenCode 2.x, use `install opencode2` instead.

### 2. Choose where memory lives

On a terminal, `install` asks once where memory should live. Scripted installs pass `--server`:

**Option A: Hindsight Cloud** (the default)

Sign up at [Hindsight Cloud](https://ui.hindsight.vectorize.io), generate an API token, and pass it to the installer:

```bash
npx @vectorize-io/hindsight-coding-agents install opencode --server cloud --api-token <token>
```

**Option B: Self-hosted**

Run Hindsight on your own infrastructure (see the [installation guide](/developer/installation)) and point the plugin at it:

```bash
npx @vectorize-io/hindsight-coding-agents install opencode --server self-hosted --api-url http://localhost:8888
```

**Option C: Local daemon**

Run memory on your machine with a local `hindsight-embed` the plugin starts on demand. It needs `uv` on your PATH and an LLM key for fact extraction:

```bash
npx @vectorize-io/hindsight-coding-agents install opencode --server daemon
```

Re-running `install` never asks again: a config that already names a server is left alone.

### 3. Start OpenCode

```bash
opencode
```

The plugin loads with OpenCode. There is no setup command: the repo's git history and your conversations start flowing into its memory bank in the background as you work.

---

## What Happens Under the Hood

Here's the lifecycle of a session with the plugin active:

1. **Every turn** — the plugin resolves which bank to use from the directory the agent is working in, so a session that moves into another repo follows it
2. **First prompt** — the plugin injects memory once: by default a low-budget reflect synthesis; if that times out or fails, it falls back to page search, then recall
3. **During the session** — the agent can call `hindsight_search_pages`, `hindsight_read_page` or `hindsight_reflect` when it needs more, and store things deliberately with `hindsight_ingest_document` or `hindsight_capture_initiative`
4. **After each turn** — the session transcript is written back to the bank, and an idle flush captures the reply the per-turn pass can't see

---

## Configuration

Configuration is **one JSON file**: `~/.hindsight/coding-agent.json`. The installer writes it; you edit it to change settings.

Environment variables (`HINDSIGHT_API_URL`, `HINDSIGHT_API_TOKEN`, and one `HINDSIGHT_<FIELD_IN_CAPS>` per scalar setting) are a **fallback**: the file wins wherever it sets a value. They are meant for containers and CI that inject config instead of writing a file.

The file can also hold per-agent overrides (`harnesses.opencode`), per-directory overrides (`paths.<prefix>`) and per-repo overrides (`banks.<bankId>`).

OpenCode reads the file when it loads the plugin, so restart OpenCode after an edit.

### Config file example

```json
{
  "apiUrl": "http://localhost:8888",
  "bankId": "my-project",
  "autoInject": "pages",
  "reflectBudget": "mid"
}
```

### Key settings

| Setting | Default | What it does |
|---|---|---|
| `apiUrl` | `https://api.hindsight.vectorize.io` | Hindsight API base URL (`http://localhost:8888` for a local server) |
| `apiToken` | — | Bearer token for Hindsight Cloud |
| `bankId` | — | Explicit static bank; unset means one bank per repo |
| `bankIdTemplate` | `"coding-agent::{gitProject}"` | Format of the per-repo bank ID |
| `autoInject` | `"reflect"` | What to inject on the session's first prompt: `"reflect"`, `"pages"` or `"recall"` |
| `reflectBudget` | `"high"` | Budget for the agent's `hindsight_reflect` tool: `low`, `mid`, `high` |
| `retainSessions` | `true` | Write sessions back to memory; `false` stops it while recall and the tools keep working |
| `disabled` | `false` | Hard off-switch |

The full settings reference is in the [integration docs](/sdks/integrations/coding-agents).

---

## Scoping Memory with Bank IDs

By default, each repository gets its own bank: `coding-agent::{gitProject}`. A project at `/home/user/my-app` gets bank ID `coding-agent::my-app`, and every linked worktree of that repo shares it. The default doesn't name the agent, so OpenCode, Claude Code and Codex all share one memory per repo.

You can change that in `~/.hindsight/coding-agent.json`:

**Static bank ID** — one bank for everything, or one shared by a whole team:

```json
{ "bankId": "team-orders" }
```

**Split per agent** — keep OpenCode's memory apart from your other agents:

```json
{ "bankIdTemplate": "{harness}-{gitProject}" }
```

**Map paths to banks** — `mapPathToBank` maps an absolute path to a bank; the longest matching prefix wins and overrides everything else.

The template also accepts `{channel}` and `{user}` (from `$HINDSIGHT_CHANNEL_ID` / `$HINDSIGHT_USER_ID`) for multi-user setups. For more patterns, see the [per-user memory cookbook](/cookbook/recipes/per-user-memory).

---

## Pitfalls and Edge Cases

**Cold start on first session.** A new repo's bank starts empty, so the first session has little to recall. The git history ingest and each session's write-back fill it in the background. To pre-populate a bank, use the [Hindsight API](/developer/api/quickstart) or CLI directly.

**Local daemon start-up.** In daemon mode, the first start downloads the daemon and loads models, which takes a while. It runs in the background, so a session that begins before it is ready simply has no memory for a turn or two.

**Config edits need a restart.** OpenCode reads `~/.hindsight/coding-agent.json` when it loads the plugin. Restart OpenCode after you change it.

**Turn latency.** Injection on the first prompt costs time. `autoInject: "pages"` is faster than the default reflect synthesis, and `injectTimeoutMs` caps how long retrieval may take.

---

## Tradeoffs and Alternatives

**When not to use this:**
- If your sessions are always one-off questions with no continuity, the overhead of memory capture and recall isn't worth it
- If you're working on sensitive codebases where conversation content shouldn't leave the machine, use a self-hosted Hindsight instance

**Alternatives:**
- **Manual context files** — maintain a `CONTEXT.md` or `CLAUDE.md` in your project root. Works, but requires manual upkeep and doesn't scale across sessions
- **Git history** — your commit messages and PRs capture what changed, not why it was decided
- **Session export/replay** — some tools let you export sessions and replay them, but you can't inject 50 prior sessions into a context window

The plugin is most valuable when you have ongoing project work where decisions, preferences, and context accumulate over days or weeks.

---

## Recap

- OpenCode sessions are stateless by default
- The Hindsight Coding Agents plugin adds persistent memory: injection on the first prompt, write-back every turn, background git ingest, and explicit tools (search and read pages, reflect, ingest documents, capture initiatives)
- Configuration is one file, `~/.hindsight/coding-agent.json`; environment variables are only a fallback
- Memory is scoped per repo by default (`coding-agent::{gitProject}`); set `bankId` to share one bank

---

## Next Steps

- **[Sign up for Hindsight Cloud](https://ui.hindsight.vectorize.io)** — zero-config hosting, ready in seconds
- Install the plugin: `npx @vectorize-io/hindsight-coding-agents install opencode`
- Read the [Coding Agents integration reference](/sdks/integrations/coding-agents) for full configuration details
- Try the [quickstart](/developer/api/quickstart) if you prefer self-hosting
- Explore the [cookbook](/cookbook) for memory patterns across agent frameworks
