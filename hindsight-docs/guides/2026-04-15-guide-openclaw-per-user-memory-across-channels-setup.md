---
title: "OpenClaw Per-User Memory Across Channels Setup Guide"
authors: [benfrank241]
date: 2026-04-15
tags: [how-to, openclaw, memory, configuration]
description: "Set up OpenClaw per-user memory across channels with Hindsight, choose the right bank granularity, and keep one user's context consistent everywhere."
image: /img/blog/guide-openclaw-per-user-memory-across-channels-setup.png
hide_table_of_contents: true
---

![OpenClaw Per-User Memory Across Channels Setup Guide](/img/blog/guide-openclaw-per-user-memory-across-channels-setup.png)

If you want **OpenClaw per-user memory across channels**, the key is changing how the Hindsight plugin derives bank IDs. By default, `hindsight-openclaw` isolates memory by `agent`, `channel`, and `user`, which is a safe default but also means the same person can feel like a stranger in every new DM, thread, or room. That is great for strict isolation. It is not great if you want a user's preferences and ongoing context to follow them from one conversation to another.

The good news is that you do not need a custom plugin or a second memory service. OpenClaw already supports this pattern. You just need to set `dynamicBankGranularity` to the right values, then verify that the bank layout matches the way your users move across channels. In most cases, `[
  "provider",
  "user"
]` is the sweet spot, because it lets one user share memory across all channels inside the same platform while still keeping Slack separate from Telegram.

This guide shows how to set that up safely, when to use `[
  "user"
]` instead, how to test the result, and how to avoid the most common memory-isolation mistakes. If you want the full plugin reference while you work, keep the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw), the [docs home](https://hindsight.vectorize.io/docs), and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby.

<!-- truncate -->

> **Quick answer**
>
> 1. Install and configure `hindsight-openclaw` normally.
> 2. Change `dynamicBankGranularity` from `[
>   "agent",
>   "channel",
>   "user"
> ]` to `[
>   "provider",
>   "user"
> ]` if you want one user's memory to carry across channels on the same platform.
> 3. Use `[
>   "user"
> ]` only if you intentionally want memory shared across platforms too.
> 4. Restart the gateway and test with the same user in two different channels.
> 5. Add a focused `retainMission` so the shared bank stores durable cross-channel context, not every one-off detail.

## Prerequisites

Before you change bank granularity, make sure a few basics are already true.

You should have:

- OpenClaw installed and running.
- `@vectorize-io/hindsight-openclaw` installed.
- A working Hindsight backend, either local, cloud, or external API.
- A recent plugin version, ideally 0.6 or later, because that is where the bank-granularity behavior is documented clearly and configured in `openclaw.json`.

If you have not installed the plugin yet, start there:

```bash
openclaw plugins install @vectorize-io/hindsight-openclaw
npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup
openclaw gateway
```

It also helps to know the difference between these three patterns before you begin:

- **Per-channel isolation**: same user gets separate memory in every room or DM.
- **Per-user across channels**: same user shares memory across channels, usually within the same provider.
- **Single shared bank**: everyone writes to the same bank, which is powerful for some team agents, but risky for normal one-to-one conversations.

If you want the mental model behind recall and retention, the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall), the [Retain API reference](https://hindsight.vectorize.io/docs/api/retain), and the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw) are the best references to skim first.

## Step by step

### 1. Understand the default bank layout

Out of the box, the OpenClaw Hindsight plugin derives bank IDs from:

```json
["agent", "channel", "user"]
```

That means a unique memory bank exists for each combination of:

- the bot identity
- the conversation or channel
- the user

This is conservative, and for good reason. It prevents context from leaking between conversations. But it also means the same person can talk to your agent in a Slack DM and then again in a Slack channel, and the second conversation starts from zero because the `channel` dimension changed.

You can inspect your current setting directly:

```bash
python3 - <<'PY'
import json, pathlib
path = pathlib.Path.home() / '.openclaw' / 'openclaw.json'
config = json.loads(path.read_text())
plugin = config['plugins']['entries']['hindsight-openclaw']['config']
print(plugin.get('dynamicBankGranularity', ['agent', 'channel', 'user']))
PY
```

If the result includes `channel`, you are still isolating memory per channel.

### 2. Pick the right granularity for your use case

There are two realistic answers for per-user continuity.

#### Option A, `[
  "provider",
  "user"
]`

This is the safest version of **per-user memory across channels**. The same user shares memory across all channels inside one provider, but not across different platforms.

Use it when:

- the same person talks to your OpenClaw agent in multiple Slack conversations
- you want continuity inside one provider
- you do **not** want Slack and Telegram memories mixed together automatically

This is the option I recommend first for most deployments.

#### Option B, `[
  "user"
]`

This is broader. It shares one user's memory across every provider and every channel, as long as the plugin identifies them as the same user.

Use it when:

- the same person truly has one identity across all the surfaces you care about
- you want memory to follow them everywhere
- you understand the privacy and identity-matching implications

This is powerful, but it deserves more caution. A user ID that is stable inside Slack is not automatically meaningful across Slack, Telegram, Discord, and SMS unless your deployment normalizes identity correctly.

### 3. Update `openclaw.json`

The most reliable way to change bank granularity is to patch `~/.openclaw/openclaw.json` directly with a small script.

For the recommended same-provider, cross-channel setup:

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

If you intentionally want memory to follow a user across providers too, use this instead:

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
cfg['dynamicBankGranularity'] = ['user']
path.write_text(json.dumps(config, indent=2) + '\n')
print(f'Updated {path}')
PY
```

The important thing is to leave `dynamicBankId` enabled. If you turn it off and set a single static `bankId`, you are no longer doing per-user memory across channels. You are moving toward a fully shared bank.

### 4. Add a retention rule that suits shared user memory

When a user's memory can span multiple channels, the bank becomes more valuable, but it can also get noisier. The safest way to keep it useful is a focused `retainMission`.

Here is a good starting point:

```bash
python3 - <<'PY'
import json, pathlib
path = pathlib.Path.home() / '.openclaw' / 'openclaw.json'
config = json.loads(path.read_text())
cfg = config['plugins']['entries']['hindsight-openclaw']['config']
cfg['retainMission'] = (
    'Extract user preferences, ongoing projects, recurring commitments, '
    'important context, and durable facts that should help across future '
    'conversations. Skip one-off chatter and temporary task noise.'
)
path.write_text(json.dumps(config, indent=2) + '\n')
print(f'Updated {path}')
PY
```

This matters because a per-user bank accumulates context from multiple conversations. Without a mission, the memory engine stores more raw conversational detail than most assistants need. With a mission, the bank becomes a reusable profile of the user and their ongoing work.

If you want examples of how shared memory behaves in other agents, [Adding memory to Codex with Hindsight](https://hindsight.vectorize.io/blog/adding-memory-to-codex-with-hindsight) and the [team shared memory post](https://hindsight.vectorize.io/blog/team-shared-memory-ai-coding-agents) are useful comparison reads.

### 5. Restart the gateway

After changing config, restart OpenClaw so the plugin reloads the new bank rules:

```bash
openclaw gateway restart
```

If you are unsure whether the config loaded, start with:

```bash
openclaw gateway status
```

Then restart.

### 6. Test the cross-channel behavior with a real user

Do not stop at file edits. Test the actual behavior.

A simple test looks like this:

1. In one channel, have the user tell the agent something durable, for example: "I prefer concise replies and I am planning a product launch next month."
2. Let the turn finish so retention can happen.
3. In a different channel on the **same provider**, have the same user ask a question where those facts would matter.
4. Check whether the agent recalls the preference and project context without being told again.

If you chose `[
  "provider",
  "user"
]`, this should work across channels on the same platform. If it does not, one of three things is usually wrong:

- `channel` is still present in the granularity list
- the provider does not identify the same human consistently across channels
- the retained context was too ephemeral to be useful

## Verifying it works

### Inspect the config

Print the effective granularity after your change:

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

### Verify the memory pattern, not just the config

The real signal is whether the same user feels known in a second channel. Ask something that depends on retained context, not a generic fact lookup.

### Watch gateway logs if needed

If you need a lower-level check, use the Hindsight log pattern from the main OpenClaw guide:

```bash
tail -f /tmp/openclaw/openclaw-*.log | grep Hindsight
```

You are looking for retain and recall activity after the config change.

## Troubleshooting common problems

### Problem: memory still feels isolated by channel

Double-check that `channel` is gone from `dynamicBankGranularity`. This is the most common miss.

### Problem: memory is unexpectedly shared too broadly

You probably switched to `[
  "user"
]` when you really wanted `[
  "provider",
  "user"
]`, or you disabled `dynamicBankId` and fell back to a shared static bank.

### Problem: the same user is not being recognized across channels

This is usually an identity issue, not a recall issue. Your provider must expose a stable user identity that stays consistent across the channels you expect to share memory.

### Problem: the bank fills with noisy conversation fragments

Tighten the `retainMission`. Shared user memory works best when it stores durable context, not every fleeting request.

### Problem: memory sharing works in one provider, but not another

That may be exactly what you configured. `[
  "provider",
  "user"
]` intentionally keeps providers separate.

## FAQ

### Should I use `[
  "provider",
  "user"
]` or `[
  "user"
]`?

Start with `[
  "provider",
  "user"
]`. It is safer and matches what most people mean by per-user memory across channels.

### Does this share memory across all channels automatically?

Only within the scope defined by your granularity. If `channel` is removed and the same user identity is stable, yes.

### Is this the same as using one global shared bank?

No. A global shared bank is usually `dynamicBankId: false` with one static `bankId`. Per-user shared memory still keeps users separate.

### Will this work for team agents?

Yes, but team setups may want a different pattern. If you need multiple OpenClaw instances to share memory, the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw) and the existing shared-memory post are the better reference points.

### What else should I tune after bank granularity?

Usually `retainMission`, `recallBudget`, and `recallContextTurns`. The [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) and [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) help you reason about those choices.

## Next Steps

- [Create a Hindsight Cloud account](https://hindsight.vectorize.io) if you want the fastest path to shared per-user memory across multiple channels or machines.
- Read the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw) for the full plugin configuration reference.
- Keep the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) handy if you want a clean setup baseline.
- Use the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) to understand what gets injected before a reply.
- Use the [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) to control what becomes durable cross-channel user memory.
- Compare related patterns like [Adding memory to Codex with Hindsight](https://hindsight.vectorize.io/blog/adding-memory-to-codex-with-hindsight) and the [team shared memory post](https://hindsight.vectorize.io/blog/team-shared-memory-ai-coding-agents) if you want broader design ideas.
