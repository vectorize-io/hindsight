---
title: "Your OpenCode Agent Forgets Everything Between Sessions. Here's the Fix."
authors: [DK09876]
date: 2026-04-20
tags: [opencode, memory, integrations, typescript]
description: "OpenCode sessions are stateless by default. The @vectorize-io/opencode-hindsight plugin adds persistent memory via automatic hooks and three explicit tools."
image: /img/blog/opencode-persistent-memory.png
hide_table_of_contents: true
---

![Your OpenCode Agent Forgets Everything Between Sessions. Here's the Fix.](/img/blog/opencode-persistent-memory.png)

OpenCode gives you a fast, terminal-native AI coding agent. But every session starts cold — no memory of what you built yesterday, what conventions you follow, or what decisions you've already made. Here's how to give it persistent long-term memory with one plugin.

<!-- truncate -->

## TL;DR

- OpenCode sessions are stateless — every conversation starts from scratch
- `@vectorize-io/opencode-hindsight` adds persistent memory via three tools (retain, recall, reflect) and automatic hooks
- Memories are injected into the system prompt on session start, so the agent has context before you say anything
- Conversations are auto-captured on idle, and memories survive context window compaction
- Works with [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) (zero setup) or self-hosted

---

## The Problem: No Persistent Memory Between Sessions

OpenCode is a powerful terminal-based coding agent. You can install plugins, wire up providers, and get real work done. But every session is isolated. There's no mechanism for the agent to remember what happened in previous sessions.

Ask your agent to use a specific test framework. It does. Next session, it doesn't remember. Explain your project's architecture, your deployment process, your naming conventions. All gone once the session ends.

For quick one-off tasks, this is fine. For ongoing project work where continuity matters, it's a real limitation.

---

## The Approach

[Hindsight](https://github.com/vectorize-io/hindsight) is a memory layer for AI agents. It extracts facts from conversations, retrieves them semantically at query time, and can synthesize reasoned answers from accumulated context.

The `@vectorize-io/opencode-hindsight` plugin integrates Hindsight directly into OpenCode's plugin system. It works in two modes:

**Automatic** — hooks handle everything behind the scenes:
- Recalls relevant memories when a session starts (injected into the system prompt)
- Retains the conversation when the session goes idle
- Preserves memories through context window compaction

**Explicit** — three tools the agent can call directly:
- `hindsight_retain` — store a specific fact or decision
- `hindsight_recall` — search memory for relevant context
- `hindsight_reflect` — synthesize a reasoned answer from accumulated memories

```
Session start
    │
    ▼
┌─────────────────────────────────────┐
│  System prompt                      │
│  + recalled memories (automatic)    │
├─────────────────────────────────────┤
│  OpenCode Agent                     │
│  ├── hindsight_retain  (explicit)   │
│  ├── hindsight_recall  (explicit)   │
│  └── hindsight_reflect (explicit)   │
└─────────────────────────────────────┘
        │                   ▲
      retain              recall
        │                   │
        ▼                   │
  ┌───────────────────────────────┐
  │        Hindsight bank         │
  └───────────────────────────────┘
```

The automatic hooks handle most use cases. The explicit tools are there for when the agent (or you) wants fine-grained control.

---

## Setup

### 1. Install the plugin

```bash
npm install @vectorize-io/opencode-hindsight
```

### 2. Register in opencode.json

```json
{
  "plugin": ["@vectorize-io/opencode-hindsight"]
}
```

### 3. Point at a Hindsight server

**Option A: Hindsight Cloud** (zero setup)

Sign up at [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup), generate an API token, and set two environment variables:

```bash
export HINDSIGHT_API_URL="https://api.hindsight.vectorize.io"
export HINDSIGHT_API_TOKEN="hsk_your_token"
```

**Option B: Self-hosted**

Run Hindsight locally or on your own infrastructure. See the [installation guide](/developer/installation) to get started:

```bash
uvx hindsight-server
export HINDSIGHT_API_URL="http://localhost:8888"
```

### 4. Start OpenCode

```bash
opencode
```

The plugin activates automatically. On your first session, it connects to Hindsight and begins capturing context.

---

## What Happens Under the Hood

Here's the lifecycle of a session with the plugin active:

1. **Session starts** — the `session.created` event fires, marking this session for memory injection
2. **First LLM call** — the system transform hook runs a recall query ("project context and recent work") and injects matching memories into the system prompt
3. **During the session** — the agent can call `hindsight_retain`, `hindsight_recall`, or `hindsight_reflect` explicitly as needed
4. **Session idles** — the `session.idle` event triggers auto-retain of the conversation transcript
5. **Context compaction** — if the context window fills up, the plugin retains the current conversation and injects recalled memories into the compaction context, so nothing important is lost

The auto-retain uses a sliding window controlled by `retainEveryNTurns` (default: 10). This prevents redundant storage while keeping recent context fresh.

---

## Configuration

You can configure the plugin at three levels (later wins):

1. **Config file** — `~/.hindsight/opencode.json` for user-wide defaults
2. **Plugin options** — inline in `opencode.json` for project-specific settings
3. **Environment variables** — for CI/CD or per-machine overrides

### Plugin options example

```json
{
  "plugin": [
    ["@vectorize-io/opencode-hindsight", {
      "hindsightApiUrl": "http://localhost:8888",
      "bankId": "my-project",
      "recallBudget": "high",
      "retainEveryNTurns": 5,
      "bankMission": "You are a coding assistant for a TypeScript monorepo. Focus on architecture decisions, test patterns, and deployment procedures."
    }]
  ]
}
```

### Key settings

| Setting | Default | What it does |
|---|---|---|
| `autoRecall` | `true` | Inject memories on session start |
| `autoRetain` | `true` | Capture conversation on idle |
| `recallBudget` | `mid` | How many memories to retrieve: `low`, `mid`, `high` |
| `retainEveryNTurns` | `10` | Auto-retain frequency (user turns) |
| `retainMode` | `full-session` | `full-session` upserts the whole conversation; `last-turn` creates chunked windows |
| `bankMission` | *(none)* | Guides what Hindsight extracts and how it reflects |
| `dynamicBankId` | `false` | Derive bank ID from project/agent context |

Full environment variable reference is in the [integration docs](/sdks/integrations/opencode).

---

## Scoping Memory with Bank IDs

By default, all sessions share a single `opencode` bank. For project isolation, you have two options:

**Static bank ID** — one bank per project:

```bash
export HINDSIGHT_BANK_ID="my-project"
```

**Dynamic bank ID** — derived automatically from context:

```bash
export HINDSIGHT_DYNAMIC_BANK_ID=true
```

With dynamic mode, the bank ID is composed from granularity fields (default: `agent::project`). A project at `/home/user/my-app` gets bank ID `opencode::my-app`.

For multi-user setups:

```bash
export HINDSIGHT_DYNAMIC_BANK_ID=true
export HINDSIGHT_CHANNEL_ID="team-backend"
export HINDSIGHT_USER_ID="alice"
```

This gives you `opencode::my-app::team-backend::alice` — full isolation per user, per project, per channel. For more patterns, see the [per-user memory cookbook](/cookbook/recipes/per-user-memory).

---

## Pitfalls and Edge Cases

**Memory feedback loops.** The plugin strips `<hindsight_memories>` tags from transcripts before retaining, preventing circular amplification. If you're building custom tooling that feeds OpenCode output into Hindsight directly, be aware of this.

**Cold start on first session.** The first session in a new bank has no memories to recall. The agent starts with a blank slate. Memories accumulate from the second session onward. To pre-populate a bank, use the [Hindsight API](/developer/api/quickstart) or CLI directly.

**Compaction timing.** The compaction hook retains the current conversation and recalls relevant context. If your Hindsight server is slow or unreachable, compaction proceeds without memory injection — the plugin never blocks the agent.

**Context window budget.** Recalled memories consume tokens from the context window. With `recallBudget: "high"`, you get more context but leave less room for the conversation. For long sessions, `mid` (the default) is a good balance.

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
- The Hindsight plugin adds persistent memory through automatic hooks (recall on start, retain on idle, preserve through compaction) and explicit tools (retain, recall, reflect)
- Configuration is layered: config file, plugin options, environment variables
- Memory is scoped via bank IDs — static for simple setups, dynamic for multi-project/multi-user isolation
- The plugin is non-blocking and gracefully degrades if Hindsight is unreachable

---

## Next Steps

- **[Sign up for Hindsight Cloud](https://ui.hindsight.vectorize.io/signup)** — zero-config hosting, ready in seconds
- Install the plugin: `npm install @vectorize-io/opencode-hindsight`
- Read the [OpenCode integration reference](/sdks/integrations/opencode) for full configuration details
- Try the [quickstart](/developer/api/quickstart) if you prefer self-hosting
- Set a `bankMission` that matches your project's focus — it significantly improves what gets extracted and recalled
- Explore the [cookbook](/cookbook) for memory patterns across agent frameworks
