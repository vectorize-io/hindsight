---
title: "The Memory Upgrade Every OpenClaw User Needs"
authors: [hindsight]
date: 2026-03-06
tags: [openclaw]
image: /img/blog/adding-memory-to-openclaw-with-hindsight.png
hide_table_of_contents: true
---

## TL;DR

<!-- truncate -->

- OpenClaw's built-in memory is file-based -- markdown files on disk with SQLite vector search. It works, but the agent has to decide what to remember. Hindsight automates the entire pipeline.
- Hindsight is open source and runs locally by default. Your conversations, extracted knowledge, and memory store never leave your machine unless you choose otherwise.
- One plugin install, three commands to set up. The `hindsight-embed` daemon bundles the full memory engine (API + PostgreSQL) into a single process.
- Memories auto-inject into context before each response -- no tool calls, no retrieval logic to write.
- For teams, an external API mode connects to a shared Hindsight server so multiple OpenClaw instances can share memory.

## The Problem

OpenClaw is an always-on AI assistant that lives in your messaging apps -- WhatsApp, Telegram, Slack, Discord, iMessage, and more. It connects to an LLM, executes tasks on your behalf, and communicates through the channels you already use.

OpenClaw has memory built in, and it's a thoughtful design. The system uses plain Markdown files on disk: daily notes in `memory/YYYY-MM-DD.md` for session-level context, and a curated `MEMORY.md` for long-term knowledge. A SQLite-backed vector index (using `sqlite-vec`) enables semantic search over these files, and there's even an experimental QMD backend that combines BM25 keyword search with vector retrieval.

But there's a fundamental constraint: **the agent has to decide what to remember**. The docs say it directly -- "If you want something to stick, ask the bot to write it." Memory is append-only text that the model must explicitly choose to save. Today's and yesterday's daily notes load automatically at session start, but anything older requires the agent to actively search with the `memory_search` tool.

In practice, this means:

- Important facts slip through because the model didn't think to write them down.
- The quality of memory depends on how well the LLM follows its own instructions to persist information.
- As daily notes accumulate, finding the right context requires the agent to search at the right time with the right query -- and models don't do this consistently.

There's also a data question that matters for OpenClaw users specifically. OpenClaw runs on *your* machine and talks to *your* messaging apps. The expectation is local-first, private by default. Any memory solution that routes your conversations through a third-party cloud service breaks that model.

## The Approach

[Hindsight](https://github.com/vectorize-io/hindsight) is an open-source memory engine that replaces OpenClaw's memory layer with automated, structured knowledge extraction. The key differences from the built-in system:

**Automatic capture, not manual.** Every conversation is captured after each turn without the agent needing to decide what's worth remembering. Hindsight extracts facts, entities, and relationships in the background -- the model doesn't need to be prompted to "save this."

**Structured knowledge, not flat text.** Instead of appending lines to a Markdown file, Hindsight extracts discrete facts ("production database runs on port 5433"), tracks entities (people, services, projects), and maps relationships between them ("auth service depends on Redis for sessions").

**Auto-recall, not tool-based retrieval.** OpenClaw's built-in memory exposes a `memory_search` tool that the agent can call, but models don't use search tools consistently. Hindsight sidesteps this entirely by injecting relevant memory into context *before* every agent response. The agent doesn't need to know the memory system exists.

**Feedback loop prevention.** When memories are injected into context before a response, they become part of the conversation. Without care, those injected memories would get re-stored and re-extracted as new facts, causing exponential growth and duplicates. The plugin automatically strips its own `<hindsight_memories>` tags before retention, preventing this loop.

**Local-first, open source.** Hindsight runs through `hindsight-embed`, a daemon that bundles the memory API and a PostgreSQL instance into a single process on your machine. No data leaves your environment. The entire codebase is open source.

```
┌─────────────────┐       ┌────────────────────────────────┐
│    OpenClaw      │       │    hindsight-embed daemon      │
│    Gateway       │──────▶│    ┌──────────┐ ┌───────────┐ │
│                  │◀──────│    │ Memory   │ │PostgreSQL │ │
│ WhatsApp/Slack/  │       │    │ API      │─│(embedded) │ │
│ Telegram/...     │       │    └──────────┘ └───────────┘ │
└─────────────────┘       └────────────────────────────────┘
                            runs locally · port 9077
```

## Implementation

### Step 1: Configure an LLM Provider

Hindsight needs an LLM for memory extraction. This is separate from your agent's primary model -- it runs in the background and handles fact/entity/relationship extraction.

```bash
# Option A: OpenAI (uses gpt-4o-mini)
export OPENAI_API_KEY="YOUR_API_KEY"

# Option B: Anthropic (uses claude-3-5-haiku)
export ANTHROPIC_API_KEY="YOUR_API_KEY"

# Option C: Gemini (uses gemini-2.5-flash)
export GEMINI_API_KEY="YOUR_API_KEY"

# Option D: Groq (uses openai/gpt-oss-20b)
export GROQ_API_KEY="YOUR_API_KEY"

# Option E: Claude Code (no API key needed)
export HINDSIGHT_API_LLM_PROVIDER=claude-code

# Option F: OpenAI Codex (no API key needed)
export HINDSIGHT_API_LLM_PROVIDER=openai-codex
```

You can also point it at any OpenAI-compatible endpoint, including OpenRouter:

```bash
export HINDSIGHT_API_LLM_PROVIDER=openai
export HINDSIGHT_API_LLM_MODEL=xiaomi/mimo-v2-flash
export HINDSIGHT_API_LLM_API_KEY=YOUR_API_KEY
export HINDSIGHT_API_LLM_BASE_URL=https://openrouter.ai/api/v1
```

A smaller, cheaper model is the right call here. Memory extraction doesn't need your most capable model.

### Step 2: Install the Plugin

```bash
openclaw plugins install @vectorize-io/hindsight-openclaw
```

You should see output confirming the install and that Hindsight takes over the memory slot:

```
Exclusive slot "memory" switched from "memory-core" to "hindsight-openclaw".
Installed plugin: hindsight-openclaw
```

This confirms Hindsight is replacing OpenClaw's built-in memory, not running alongside it.

### Step 3: Launch

```bash
openclaw gateway
```

The Hindsight daemon starts automatically on port 9077. You should see confirmation in the gateway output:

```
[Hindsight] ✓ Using provider: openai, model: gpt-4o-mini
```

That's the entire setup. No Docker, no database provisioning, no config files. Everything runs on your machine.

### Verifying It's Working

After a few conversations, check the gateway logs to confirm memory operations are happening:

```bash
tail -f /tmp/openclaw/openclaw-*.log | grep Hindsight
```

You should see lines like:

```
[Hindsight] Retained X messages for session ...
[Hindsight] Auto-recall: Injecting X memories
```

If you want to browse what your agent has learned, the Hindsight daemon includes a web UI:

```bash
uvx hindsight-embed@latest -p openclaw ui
```

### External API Mode: Shared Memory Across Instances

The default local daemon is ideal for a single OpenClaw instance. But if you're running multiple instances -- say, one on your laptop and one on a server -- or if your team wants shared agent memory, the plugin supports connecting to a remote Hindsight API server.

Configure via environment variables:

```bash
export HINDSIGHT_EMBED_API_URL=https://your-hindsight-server.example.com
export HINDSIGHT_EMBED_API_TOKEN=YOUR_API_TOKEN
openclaw gateway
```

Or in `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "hindsightApiUrl": "https://your-hindsight-server.example.com",
          "hindsightApiToken": "YOUR_API_TOKEN"
        }
      }
    }
  }
}
```

In this mode, no local daemon starts. The plugin performs a health check against the remote API on startup and routes all memory operations -- retain, recall, reflect -- through it. You can verify it's working in the logs:

```bash
# [Hindsight] External API mode enabled: https://your-hindsight-server.example.com
# [Hindsight] External API health check passed
```

> **Note:** Environment variables take precedence over plugin config. If you have both set, the env var wins.

> **Want to skip self-hosting entirely?** [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) works as the external API endpoint — just use your Cloud URL and API token above. This does mean your memory data leaves your machine, which breaks the fully-local setup. For personal use where privacy is paramount, stick with the local daemon. But for teams or multi-instance setups where shared memory matters more, Cloud is the fastest path.

## Memory Isolation

By default, the plugin creates separate memory banks based on the agent, channel, and user context -- so each unique combination gets its own isolated memory store. The bank ID is derived from configurable fields via `dynamicBankGranularity`:

```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "dynamicBankGranularity": ["provider", "user"]
        }
      }
    }
  }
}
```

In this example, memories are isolated per provider + user, meaning the same user shares memories across all channels within a provider.

Available isolation fields:
- `agent` -- the bot identity
- `channel` -- the conversation or group ID
- `user` -- the person interacting with the bot
- `provider` -- the messaging platform (Slack, Telegram, etc.)

The default is `["agent", "channel", "user"]`, which gives full isolation per user per channel per agent. Set `dynamicBankId: false` to use a single shared bank for all conversations. Use `bankIdPrefix` to namespace banks across environments (e.g. `"prod"`, `"staging"`).

## Retention and Recall Controls

The plugin ships with sensible defaults, but most behaviors are configurable.

**Retention** controls what gets stored:

| Option | Default | Description |
|--------|---------|-------------|
| `autoRetain` | `true` | Auto-retain conversations after each turn. Set `false` to disable. |
| `retainRoles` | `["user", "assistant"]` | Which message roles to include in retained transcript. |
| `retainEveryNTurns` | `1` | Retain every Nth turn. Values > 1 enable chunked retention with a sliding window. |
| `retainOverlapTurns` | `0` | Extra prior turns included when chunked retention fires. |

**Recall** controls what gets injected:

| Option | Default | Description |
|--------|---------|-------------|
| `autoRecall` | `true` | Auto-inject memories before each turn. Set `false` when the agent has its own recall tool. |
| `recallBudget` | `"mid"` | Recall effort: `low`, `mid`, or `high`. Higher budgets use more retrieval strategies. |
| `recallMaxTokens` | `1024` | Max tokens for recall response -- controls how much memory context is injected per turn. |
| `recallTypes` | `["world", "experience"]` | Memory types to recall. Excludes verbose `observation` entries by default. |
| `recallTopK` | unlimited | Hard cap on number of memories injected per turn. |
| `recallContextTurns` | `1` | Number of prior user turns to include when composing the recall query. |
| `recallPromptPreamble` | built-in string | Custom text placed above recalled memories in the injected context. |

Example: high-fidelity recall with multi-turn context:

```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "recallBudget": "high",
          "recallMaxTokens": 2048,
          "recallContextTurns": 3,
          "recallTopK": 10
        }
      }
    }
  }
}
```

## Pitfalls & Edge Cases

**Memory extraction is asynchronous.** Facts are extracted after each turn in the background. If you end a session and immediately start a new one, the most recent facts may still be processing. In practice this might be a second or two, so don't expect instant availability across sessions.

**Extraction quality depends on your model choice.** A very small or low-quality extraction model will miss nuanced technical details. `gpt-4o-mini` and `claude-3-5-haiku` are solid defaults -- capable enough for reliable fact extraction, cheap enough to run on every turn.

**The recall window is bounded.** The default `recallMaxTokens` of 1024 means not every relevant memory will appear in every response. Retrieval is relevance-ranked, so the most pertinent facts surface first, but be aware of the ceiling. You can increase this to 2048 or higher in config.

**Memory isolation defaults to per-agent + per-channel + per-user.** This means your Slack group chat memories don't bleed into your Telegram DM memories and vice versa. If you want unified memory across channels, adjust `dynamicBankGranularity` to just `["user"]` or set `dynamicBankId: false` for a single shared bank.

**The embedded PostgreSQL needs system libraries.** `hindsight-embed` bundles PostgreSQL via [pg0](https://github.com/vectorize-io/pg0). On minimal Docker images (e.g. `ubuntu:latest`), you'll need to install `libxml2` and `libreadline` before the daemon can start:

```bash
apt-get install -y libxml2 libreadline8t64
```

Check the pg0 README for your platform if you hit other missing library errors.

**PostgreSQL cannot run as root.** If you're running inside Docker, make sure you're using a non-root user. PostgreSQL's `initdb` refuses to run as root, and the daemon will fail with `initdb: error: cannot be run as root`. Create a user and switch to it:

```dockerfile
RUN useradd -m -s /bin/bash myuser
USER myuser
```

**First run downloads ~3GB of dependencies.** On the very first `openclaw gateway` launch, `hindsight-embed` downloads Python packages including PyTorch, sentence-transformers, and (on x86) CUDA libraries. This can take several minutes and may cause the daemon start to time out. The plugin auto-retries, and subsequent launches use the cached packages -- so this is a one-time cost.

**Debug logging.** If something isn't working as expected, enable `debug: true` in the plugin config. This produces verbose logging of recall queries, retention transcripts, bank ID derivation, and more.

## Tradeoffs & Alternatives

**When to stick with OpenClaw's built-in memory:**

- You prefer the transparency of plain Markdown files you can edit in any text editor and version control with Git.
- Your use case is lightweight and session-scoped -- daily notes plus `MEMORY.md` cover your needs.
- You don't want any additional processes running alongside the Gateway.

**When Hindsight is the better choice:**

- You want automated memory extraction without relying on the model to decide what to save.
- You need consistent auto-recall rather than hoping the agent calls `memory_search` at the right time.
- You want structured knowledge (facts, entities, relationships) rather than raw text.
- You need shared memory across multiple OpenClaw instances or team members.
- You want fine-grained control over what gets retained, what gets recalled, and how memory is isolated across agents and channels.

**Local daemon vs. external API:**

The local daemon is simpler and keeps everything on one machine -- the right default for personal use. External API mode adds network latency but enables shared memory and survives machine restarts. Since Hindsight is open source, you can self-host the server on your own infrastructure and keep the same data ownership guarantees.

## Recap

OpenClaw's built-in memory is file-based and transparent, but it depends on the agent deciding what to remember and when to search. Hindsight replaces this with automated extraction and auto-recall -- conversations are captured, facts are extracted in the background, and relevant knowledge is injected into context before every response.

The core insight: memory that works automatically is qualitatively different from memory that depends on model behavior. When the agent doesn't have to choose what to save or when to search, it just has the right context.

And because Hindsight is open source and local-first, you keep the same data ownership model that makes OpenClaw compelling in the first place.

## Next Steps

- Install the plugin and have a few conversations across different channels. Then open the web UI (`uvx hindsight-embed@latest -p openclaw ui`) to see what was captured.
- Experiment with different LLM providers for extraction and compare the quality of captured facts.
- Tune recall with `recallBudget`, `recallMaxTokens`, and `recallContextTurns` to find the right balance for your use case.
- Adjust `dynamicBankGranularity` if you want memories shared across channels or isolated per provider.
- Browse the [Hindsight source on GitHub](https://github.com/vectorize-io/hindsight) to understand the extraction pipeline.
- Read the [full integration docs](https://hindsight.vectorize.io/sdks/integrations/openclaw) for the complete configuration reference.
- If you're running multiple OpenClaw instances, try external API mode with a self-hosted Hindsight server for shared memory.
