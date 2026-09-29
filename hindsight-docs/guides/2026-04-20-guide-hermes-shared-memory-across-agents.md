---
title: "Hermes Shared Memory Across Agents Setup"
authors: [benfrank241]
date: 2026-04-20
tags: [how-to, hermes, memory, multi-agent, configuration]
description: "Set up Hermes shared memory across agents with Hindsight, reuse one bank on purpose, and test that multiple agents can build on the same context safely."
image: /img/blog/guide-hermes-shared-memory-across-agents.png
hide_table_of_contents: true
---

If you want **Hermes shared memory across agents**, the setup is much simpler than most people expect. Multiple agents do not need a special federation layer. They just need to point at the same Hindsight backend and the same `bank_id`. Once that happens, the agents can retrieve and retain against the same memory space.

That simplicity is powerful, but it also means you need to be intentional. Shared memory only improves outcomes when the participating agents truly belong to the same workflow. If two unrelated agents write into the same bank, recall gets noisy fast. If two cooperating agents share a bank with a focused mission, the effect is the opposite, each one becomes smarter because it can build on what the others already learned.

This guide shows how to choose the right shared bank, how to configure multiple Hermes agents to reuse it, how to keep role boundaries without splitting the bank, and how to verify that one agent can recall memories produced by another. Keep the [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes), the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart), and the [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) open while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. Pick one shared `bank_id` for the workflow, not for your whole organization.
> 2. Configure each Hermes agent to use the same backend and the same bank.
> 3. Keep agent roles distinct in prompts and missions, not in separate banks, if collaboration is the goal.
> 4. Test with two agents, one writes a durable fact and the other recalls it.
> 5. Split the bank later if unrelated context starts polluting recall.

## Prerequisites

Before you share memory between Hermes agents, make sure:

- Every agent already uses the native Hindsight provider.
- You control the config for each agent.
- The agents really belong to the same project, customer workflow, or team process.
- You know what durable context should be shared.

If you still need the base provider setup, start with the [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes), the [docs home](https://hindsight.vectorize.io/docs), the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart), and the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall).

## Step by step

### 1. Choose a bank name that matches one real workflow

Good shared-bank examples:

- `support-escalations`
- `team-product-alpha`
- `launch-ops-q2`

Bad examples:

- `shared`
- `general`
- `everyone`

A bank should reflect one collaborative memory space. If you cannot explain which agents belong in it, the bank is probably too broad.

### 2. Configure the first Hermes agent

In the Hermes Hindsight config, set the shared bank explicitly:

```json
{
  "provider": "hindsight",
  "hindsight_api_url": "https://api.hindsight.vectorize.io",
  "api_key": "YOUR_HINDSIGHT_TOKEN",
  "bank_id": "team-product-alpha",
  "memory_mode": "hybrid",
  "prefetch_method": "recall"
}
```

This first agent becomes one writer and reader in the shared bank.

### 3. Configure the second agent with the same bank

Use the same backend and the same `bank_id` for the second agent too:

```json
{
  "provider": "hindsight",
  "hindsight_api_url": "https://api.hindsight.vectorize.io",
  "api_key": "YOUR_HINDSIGHT_TOKEN",
  "bank_id": "team-product-alpha",
  "memory_mode": "hybrid",
  "prefetch_method": "recall"
}
```

Now both agents are attached to the same memory space. Their roles can still differ in prompts, tool access, and operating rules, but their memory bank is shared.

### 4. Keep roles separate in instructions, not bank IDs

A common mistake is trying to preserve role differences by splitting the bank. That defeats the point of shared memory. Instead:

- keep one shared bank
- use different prompts or skills per agent
- use a focused retain mission so only durable workflow context gets stored

This is the same mental model discussed in [Team Shared Memory for AI Coding Agents](https://hindsight.vectorize.io/blog/team-shared-memory-ai-coding-agents) and [Adding memory to Codex with Hindsight](https://hindsight.vectorize.io/blog/adding-memory-to-codex-with-hindsight).

### 5. Test a real cross-agent handoff

A simple test looks like this:

1. Agent A learns: “The customer wants concise weekly summaries and the billing export must include tax codes.”
2. The session ends and retention completes.
3. Agent B is launched on the same shared bank.
4. Ask Agent B a question where those facts matter.

If the shared bank is working, Agent B should recall the retained context without you repeating it.

## Verifying it works

### Confirm both agents use the same bank ID

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get('HERMES_HOME', pathlib.Path.home() / '.hermes'))
path = base / 'hindsight' / 'config.json'
cfg = json.loads(path.read_text())
print(cfg.get('bank_id'))
PY
```

Run that check in each agent environment. The value should match exactly.

### Confirm cross-agent recall, not just single-agent recall

If Agent A remembers its own facts and Agent B does not, the problem is not Hindsight itself. It is that the two agents are not on the same bank.

## Troubleshooting / common errors

### The second agent does not recall anything

Usually the bank IDs do not match, or one agent is still pointing at a different Hindsight backend.

### Shared recall is too noisy

The bank is too broad. Split by workflow or team rather than letting every Hermes agent write into one general bank.

### Agents overwrite each other conceptually

That usually means the retain mission is too vague. Tighten what should count as durable shared memory.

## FAQ

### Should every Hermes agent in the company share one bank?

Usually no. Shared memory works best for one bounded workflow, not a whole organization.

### Can I combine a shared bank with different memory modes per agent?

Yes. The bank can stay shared while one agent uses `hybrid` and another uses `context`, if that matches the UX you want.

### Is one shared bank better than passing notes manually?

For ongoing workflows, yes. Shared memory reduces repeated handoff context and lets agents compound knowledge over time.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want one managed backend for all Hermes agents.
- Keep the [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes) open for full provider configuration details.
- Use the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) if you still need a backend.
- Read the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) and [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) to tune retrieval and storage.
- Compare the collaboration model in [Team Shared Memory for AI Coding Agents](https://hindsight.vectorize.io/blog/team-shared-memory-ai-coding-agents).
