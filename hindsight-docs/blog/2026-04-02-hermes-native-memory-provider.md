---
title: "Hindsight Is Now a Native Memory Provider in Hermes Agent"
authors: [benfrank241]
date: 2026-04-02
tags: [hermes, memory, hindsight, integration]
description: "Hermes Agent now supports pluggable memory providers. Here's why Hindsight is the backend to use, and how to set it up in two minutes."
hide_table_of_contents: true
---

Hermes Agent now ships with a pluggable memory provider system. Hindsight is one of the supported backends, and it's the one that leads on [the benchmark that actually tests memory at scale](/blog/2026/04/02/beam-sota).

<!-- truncate -->

---

## Why Hindsight on Hermes

Hermes ships with a built-in memory tool that saves notes to local markdown files. It works, but it captures what the model explicitly decides to write down, not what it implicitly learns from your conversations. Context doesn't accumulate automatically. If you ask Hermes to help you plan a sprint on Monday and then open a new session on Friday, it doesn't remember the project, the team, or the deadline unless you re-establish that context yourself.

Hindsight solves this with persistent memory, acting at two points in the Hermes lifecycle:

**Auto-recall.** Before every LLM call, the `pre_llm_call` hook calls `hindsight_recall`, a multi-strategy search across your memory bank, and injects relevant facts as ephemeral system prompt context. Hermes sees them before generating its response. You get relevant context from past sessions without repeating yourself.

**Auto-retain.** After every response, the `post_llm_call` hook calls `hindsight_retain`, which takes the user/assistant exchange and runs entity extraction on it, pulling out discrete facts, preferences, and relationships. These land in your memory bank and are available for recall in every future session. For synthesizing patterns across many memories, `hindsight_reflect` runs cross-memory reasoning to build higher-level models of what it knows.

The result: Hermes accumulates persistent memory about you across conversations, not just within them. You mention a product launch deadline once. A week later, in a new session on a different topic, Hermes already knows it. You didn't repeat yourself. You didn't paste in context. It was recalled automatically.

Of all the supported memory providers, Hindsight is the only one with published results on [BEAM](/blog/2026/04/02/beam-sota), the benchmark that tests memory at 10 million tokens, where context stuffing is physically impossible. Hindsight scores 64.1% at that tier. The next-best published result is 40.6%.

---

## Setting It Up

Setup is a single wizard command:

```bash
hermes memory setup    # select "hindsight"
```

Or configure manually:

```bash
hermes config set memory.provider hindsight
echo "HINDSIGHT_API_KEY=your-key" >> ~/.hermes/.env
```

Then confirm memory is active:

```bash
hermes memory status
```

Config lives at `~/.hermes/hindsight/config.json`:

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `cloud` | `cloud` or `local` |
| `bank_id` | `hermes` | Memory bank identifier |
| `budget` | `mid` | Recall thoroughness: `low` / `mid` / `high` |
| `memory_mode` | `hybrid` | `hybrid` (inject + tools), `context` (inject only), `tools` (tools only) |

---

## Local or Cloud

In local mode, Hindsight runs an embedded server with built-in PostgreSQL. The daemon starts automatically in the background on first use; no manual setup required. You need an LLM API key for memory extraction:

```json
{
  "mode": "local",
  "llm_provider": "groq",
  "llm_api_key": "your-groq-key"
}
```

Startup logs land at `~/.hermes/logs/hindsight-embed.log` if you need to debug.

For persistent memory across machines or shared across multiple Hermes instances, use cloud mode instead:

```bash
echo "HINDSIGHT_API_KEY=your-key" >> ~/.hermes/.env
```

Both modes use the same API. Switching is a one-line config change, not a migration.

---

## Get Started

- **Hermes integration docs**: [/sdks/integrations/hermes](/sdks/integrations/hermes)
- **BEAM benchmark results**: [Hindsight Is #1 on BEAM](/blog/2026/04/02/beam-sota)
- **Quick start**: [/developer/api/quickstart](/developer/api/quickstart)
- **GitHub**: [github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight)
- **Cloud**: [ui.hindsight.vectorize.io/signup](https://ui.hindsight.vectorize.io/signup)
