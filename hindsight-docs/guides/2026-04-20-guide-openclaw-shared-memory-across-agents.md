---
title: "OpenClaw Shared Memory Across Agents Guide"
authors: [benfrank241]
date: 2026-04-20
tags: [how-to, openclaw, memory, configuration]
description: "Set up OpenClaw shared memory across agents with Hindsight, choose the right bank granularity, and verify that multiple agents reuse the same context safely."
image: /img/blog/guide-openclaw-shared-memory-across-agents.png
hide_table_of_contents: true
---

If you want **OpenClaw shared memory across agents**, the real question is not whether Hindsight can do it. It can. The question is how much of the default bank-isolation scheme you want to keep. Out of the box, the OpenClaw plugin isolates memory by `agent`, `channel`, and `user`, which is a safe starting point but also means two agents can talk to the same human and still behave like strangers.

That default is useful when each agent should keep a separate brain. It is the wrong layout when you deliberately want a support bot, an operations bot, and a personal assistant to share the same user context. In that case, the fix is simple: remove `agent` from the bank key, keep the user dimension, and make sure every agent points at the same Hindsight backend.

This guide shows how to configure that shared setup, how to decide between `["provider", "user"]` and `["user"]`, how to keep shared memory clean with a focused retain mission, and how to test that one agent can actually reuse what another agent learned. Keep the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw), the [docs home](https://hindsight.vectorize.io/docs), and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) open while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. Install and configure `hindsight-openclaw` normally on every agent.
> 2. Remove `agent` from `dynamicBankGranularity` so all agents derive the same bank for the same user.
> 3. Start with `["provider", "user"]`, not `["user"]`, unless cross-platform sharing is intentional.
> 4. Add a shared `retainMission` so the bank stores durable context instead of per-agent noise.
> 5. Restart every agent and test with the same user on two agents.

## Prerequisites

Before you share memory across agents, make sure these basics are already true:

- OpenClaw is installed and the gateway is healthy.
- `@vectorize-io/hindsight-openclaw` is installed for every agent that will participate.
- Every participating agent points at the same Hindsight backend, local or cloud.
- You know whether you want sharing per provider or across every provider.

If you have not installed the plugin yet, start there:

```bash
openclaw plugins install @vectorize-io/hindsight-openclaw
npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup
openclaw gateway status
```

It is also worth skimming the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall), the [Retain API reference](https://hindsight.vectorize.io/docs/api/retain), and the [OpenClaw shared memory post](https://hindsight.vectorize.io/blog/openclaw-shared-memory). Those three references make the tradeoffs much clearer.

## Step by step

### 1. Understand why agents do not share memory by default

The default bank granularity is conservative:

```json
["agent", "channel", "user"]
```

That means bank identity changes when any of these change:

- the OpenClaw agent identity
- the channel or room
- the user

If Agent A learns that a user prefers concise replies, Agent B will not see that preference later because the `agent` dimension is different. Nothing is broken, the agents are simply isolated.

You can inspect the current setting directly:

```bash
python3 - <<'PY'
import json, pathlib
path = pathlib.Path.home() / '.openclaw' / 'openclaw.json'
config = json.loads(path.read_text())
plugin = config['plugins']['entries']['hindsight-openclaw']['config']
print(plugin.get('dynamicBankGranularity', ['agent', 'channel', 'user']))
PY
```

If `agent` appears in the list, memory is still isolated per agent.

### 2. Choose the right shared-memory scope

For most multi-agent OpenClaw setups, the best default is:

```json
["provider", "user"]
```

This gives you one shared memory bank per user, per platform. The same user can move between multiple OpenClaw agents on Slack and keep continuity, but their Slack and Telegram memories still stay separate.

Use `["user"]` only if you intentionally want a single user bank shared across every platform too. That can be useful, but it raises more identity and privacy questions.

### 3. Update every agent to use the same bank pattern

Patch `~/.openclaw/openclaw.json` on each participating agent so the bank no longer includes `agent`:

```bash
python3 - <<'PY'
import json, pathlib
path = pathlib.Path.home() / '.openclaw' / 'openclaw.json'
config = json.loads(path.read_text())
entries = config.setdefault('plugins', {}).setdefault('entries', {})
plugin = entries.setdefault('hindsight-openclaw', {'enabled': True, 'config': {}})
plugin['enabled'] = True
cfg = plugin.setdefault('config', {})
cfg['dynamicBankId'] = True
cfg['dynamicBankGranularity'] = ['provider', 'user']
path.write_text(json.dumps(config, indent=2) + '\n')
print(f'Updated {path}')
PY
```

The key is consistency. If one agent uses `["agent", "channel", "user"]` and another uses `["provider", "user"]`, they will not share a bank even though both are configured correctly in isolation.

### 4. Use a retention mission built for shared memory

Shared banks are valuable because they compound context. They are dangerous when they accumulate every temporary detail. Add the same `retainMission` to each agent:

```bash
python3 - <<'PY'
import json, pathlib
path = pathlib.Path.home() / '.openclaw' / 'openclaw.json'
config = json.loads(path.read_text())
cfg = config['plugins']['entries']['hindsight-openclaw']['config']
cfg['retainMission'] = (
    'Extract durable user preferences, recurring tasks, project context, '
    'important constraints, and decisions that should help any cooperating '
    'agent. Ignore one-off chatter, duplicate tool output, and transient noise.'
)
path.write_text(json.dumps(config, indent=2) + '\n')
print(f'Updated {path}')
PY
```

If you want the mental model behind this setup, the [team shared memory post](https://hindsight.vectorize.io/blog/team-shared-memory-ai-coding-agents) and [Adding memory to Codex with Hindsight](https://hindsight.vectorize.io/blog/adding-memory-to-codex-with-hindsight) are useful comparisons.

### 5. Restart every agent and run a handoff test

Once the config is changed, restart the gateway for each agent:

```bash
openclaw gateway restart
```

Then test a real cross-agent flow:

1. Ask Agent A to learn something durable, for example: “Remember that I prefer concise summaries and my current launch date is May 15.”
2. Wait for the turn to finish so retention completes.
3. Ask Agent B a question where that context matters.
4. Confirm the second agent can answer with the retained preference or project fact.

## Verifying it works

Use two signals, not one.

### Confirm the config

```bash
python3 - <<'PY'
import json, pathlib
path = pathlib.Path.home() / '.openclaw' / 'openclaw.json'
config = json.loads(path.read_text())
plugin = config['plugins']['entries']['hindsight-openclaw']['config']
print('dynamicBankId:', plugin.get('dynamicBankId', True))
print('dynamicBankGranularity:', plugin.get('dynamicBankGranularity'))
print('retainMission:', plugin.get('retainMission'))
PY
```

### Confirm the behavior

The real test is not the file, it is whether a second agent feels informed. If Agent B still answers like it has never met the user, sharing is not working yet.

## Troubleshooting / common errors

### The agents still do not share memory

Usually one agent still includes `agent` or `channel` in the granularity list. Print the effective config on both agents and compare them line by line.

### Memory is too broad

If unrelated facts keep surfacing, your bank is too shared or your `retainMission` is too loose. Start by tightening the mission before you widen isolation again.

### Cross-platform sharing is happening when you did not want it

That means you used `["user"]` when `["provider", "user"]` was the safer fit.

## FAQ

### Should I ever share one bank across all users too?

Usually no. For most OpenClaw assistants, shared memory should be per user, not global. A fully shared bank is appropriate only for specialized team agents.

### Is `["provider", "user"]` better than `["user"]`?

For most deployments, yes. It preserves continuity within one platform without assuming a user identity is trustworthy across every platform.

### Do I need the same Hindsight API credentials on every agent?

Yes. If the agents point at different backends or different accounts, they are not sharing the same memory system.

### Can I combine this with the per-user guide?

Yes. This guide builds on the same idea. The earlier per-user guide focuses on channel sharing for one agent. This guide extends that model across multiple agents.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want one shared backend for every agent.
- Keep the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw) open for the full plugin reference.
- Review the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) if you still need to stand up Hindsight.
- Use the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) and [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) when you tune retrieval and storage.
- Compare the broader pattern in [Team Shared Memory for AI Coding Agents](https://hindsight.vectorize.io/blog/team-shared-memory-ai-coding-agents).
