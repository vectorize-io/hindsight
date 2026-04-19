---
title: "Control Recall Injection in OpenClaw with Hindsight"
authors: [benfrank241]
date: 2026-04-15
tags: [how-to, openclaw, memory, recall]
description: "Control recall injection in OpenClaw with Hindsight, choose the right position, and tune memory placement for better context and prompt behavior."
image: /img/blog/guide-control-recall-injection-in-openclaw-with-hindsight.png
hide_table_of_contents: true
---

![Control Recall Injection in OpenClaw with Hindsight](/img/blog/guide-control-recall-injection-in-openclaw-with-hindsight.png)

If you need to **control recall injection in OpenClaw**, the main setting to learn is `recallInjectionPosition`. The Hindsight plugin can place recalled memories before your system prompt, after it, or as a user message. That sounds like a small implementation detail, but it changes how much weight the model gives to memory, how well large static prompts cache, and how easy the resulting behavior is to reason about.

Most people never change the default. That is fine until they hit a real prompt-design problem: a large static system prompt that should stay cache-friendly, a workflow where recalled facts need to sit closer to the user message, or a model behavior issue where memories feel too dominant or not visible enough. At that point, knowing where recall is injected becomes part of prompt engineering, not just plugin setup.

This guide shows what each injection mode does, how to switch it safely, which related recall settings matter, how to verify the result, and which mistakes to avoid. Keep the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw), the [docs home](https://hindsight.vectorize.io/docs), and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) open if you want the full reference while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. Use `prepend` if you want recalled memories injected before the system prompt, which is the default.
> 2. Use `append` if you want your large static system prompt to stay stable for caching and put memories after it.
> 3. Use `user` only when you intentionally want recalled memories to behave more like additional user context.
> 4. Restart the gateway after changing the setting.
> 5. Test with a real memory-heavy prompt and watch the logs, not just the config file.

## Prerequisites

Before changing recall injection, make sure:

- OpenClaw is already running with `@vectorize-io/hindsight-openclaw`.
- Auto-recall is enabled.
- You already have useful memories in the bank, so there is something to inject.
- You know whether your main problem is prompt layout, latency, or model behavior.

You can inspect the current recall settings with this small script:

```bash
python3 - <<'PY'
import json, pathlib
path = pathlib.Path.home() / '.openclaw' / 'openclaw.json'
config = json.loads(path.read_text())
plugin = config['plugins']['entries']['hindsight-openclaw']['config']
print('autoRecall:', plugin.get('autoRecall', True))
print('recallInjectionPosition:', plugin.get('recallInjectionPosition', 'prepend'))
print('recallBudget:', plugin.get('recallBudget', 'mid'))
print('recallMaxTokens:', plugin.get('recallMaxTokens', 1024))
print('recallContextTurns:', plugin.get('recallContextTurns', 1))
PY
```

If `autoRecall` is off, changing injection position will do nothing. That sounds obvious, but it is a common source of confusion.

It is also worth skimming the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) and the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw) before you change anything. They make it much easier to reason about whether a missing memory is a retrieval problem or just a placement choice.

## Step by step

### 1. Know what each injection position means

OpenClaw supports three values for `recallInjectionPosition`.

| Value | What it does | Best for |
|---|---|---|
| `prepend` | injects memories before the system prompt | default behavior, strongest memory framing |
| `append` | injects memories after the system prompt | large static prompts, prompt caching friendliness |
| `user` | injects memories as a user message | specialized prompt flows where memory should sit closer to the user turn |

A simple mental model:

- **`prepend`** says, "memory first, then the system rules."
- **`append`** says, "keep the system rules stable, then add memories."
- **`user`** says, "treat recalled context more like extra conversational input."

This is why the setting matters. You are not changing which memories are recalled. You are changing where those memories land in the model's input.

### 2. Keep `prepend` when you want the strongest default behavior

`prepend` is the default for a reason. It is the simplest mental model and the easiest place to start.

Set it explicitly like this:

```bash
python3 - <<'PY'
import json, pathlib
path = pathlib.Path.home() / '.openclaw' / 'openclaw.json'
config = json.loads(path.read_text())
plugin = config['plugins']['entries']['hindsight-openclaw']['config']
plugin['recallInjectionPosition'] = 'prepend'
path.write_text(json.dumps(config, indent=2) + '\n')
print(f'Updated {path}')
PY
```

Use `prepend` when:

- you want the most straightforward recall behavior
- you are debugging and want the least surprising setup
- you are not trying to optimize prompt caching on a large static system prompt

If you are unsure, stay here first.

### 3. Switch to `append` when prompt stability matters

The strongest case for `append` is a big static system prompt that benefits from staying unchanged between turns. When recalled memories are injected **after** it, that static section remains more cache-friendly.

Switch to `append` like this:

```bash
python3 - <<'PY'
import json, pathlib
path = pathlib.Path.home() / '.openclaw' / 'openclaw.json'
config = json.loads(path.read_text())
plugin = config['plugins']['entries']['hindsight-openclaw']['config']
plugin['recallInjectionPosition'] = 'append'
path.write_text(json.dumps(config, indent=2) + '\n')
print(f'Updated {path}')
PY
```

Use `append` when:

- your system prompt is large and mostly static
- you care about keeping that static prompt section stable across turns
- you want memory to influence the answer without being the very first thing in the model context

This is especially attractive for carefully designed production prompts where the instructions are long, expensive, and intentionally stable.

### 4. Use `user` only with intention

`user` mode is the most specialized option. It can be useful, but it is not the default for a reason.

Set it like this:

```bash
python3 - <<'PY'
import json, pathlib
path = pathlib.Path.home() / '.openclaw' / 'openclaw.json'
config = json.loads(path.read_text())
plugin = config['plugins']['entries']['hindsight-openclaw']['config']
plugin['recallInjectionPosition'] = 'user'
path.write_text(json.dumps(config, indent=2) + '\n')
print(f'Updated {path}')
PY
```

Use `user` when:

- you explicitly want memories to read as near-user context
- you are experimenting with a prompt shape where memory should sit closer to the incoming message
- you understand that this can change how the model interprets memory relative to instructions

In other words, this is an advanced tool. It is not the first thing I would reach for in a new setup.

### 5. Tune the settings that interact with injection position

Changing where memories are injected is only part of the story. These settings change how much memory shows up and how it is composed:

- `autoRecall`
- `recallBudget`
- `recallMaxTokens`
- `recallTopK`
- `recallContextTurns`
- `recallPromptPreamble`

A practical example for a heavier but still controlled recall setup:

```bash
python3 - <<'PY'
import json, pathlib
path = pathlib.Path.home() / '.openclaw' / 'openclaw.json'
config = json.loads(path.read_text())
plugin = config['plugins']['entries']['hindsight-openclaw']['config']
plugin['autoRecall'] = True
plugin['recallInjectionPosition'] = 'append'
plugin['recallBudget'] = 'mid'
plugin['recallMaxTokens'] = 1024
plugin['recallContextTurns'] = 2
path.write_text(json.dumps(config, indent=2) + '\n')
print(f'Updated {path}')
PY
```

If recall feels weak, the issue may be recall volume or relevance, not injection position. That is why the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) matters so much here.

### 6. Restart OpenClaw and test the behavior, not just the setting

After making your change:

```bash
openclaw gateway restart
```

Then run a real test, not a synthetic one-liner. Use a conversation where remembered context should obviously matter.

A good pattern:

1. Give the agent a durable fact, for example: "We deploy staging from `develop` and production from `main`."
2. Wait for the turn to finish so retention can happen.
3. Ask a related question on the next turn, for example: "Which branch should I use for staging?"
4. Compare the response quality before and after your injection-position change.

### 7. Turn on logs if you need to see more

If the behavior is unclear, use the same Hindsight log approach from the main OpenClaw guide:

```bash
tail -f /tmp/openclaw/openclaw-*.log | grep Hindsight
```

You are looking for evidence that:

- recall is actually firing
- memories are being injected
- the system is healthy after restart

If you want more signal, enable plugin debug logging:

```bash
python3 - <<'PY'
import json, pathlib
path = pathlib.Path.home() / '.openclaw' / 'openclaw.json'
config = json.loads(path.read_text())
plugin = config['plugins']['entries']['hindsight-openclaw']['config']
plugin['debug'] = True
path.write_text(json.dumps(config, indent=2) + '\n')
print(f'Updated {path}')
PY
```

## Verifying it works

### Check the config

Print the key settings after the change:

```bash
python3 - <<'PY'
import json, pathlib
path = pathlib.Path.home() / '.openclaw' / 'openclaw.json'
config = json.loads(path.read_text())
plugin = config['plugins']['entries']['hindsight-openclaw']['config']
for key in ['autoRecall', 'recallInjectionPosition', 'recallBudget', 'recallMaxTokens', 'recallContextTurns']:
    print(f'{key}:', plugin.get(key))
PY
```

### Check response behavior under a real prompt

The best verification is a real answer where memory placement should influence the result. Does the assistant pick up the recalled context naturally? Does it still respect your system rules? Does the response feel better or worse than before?

### Check whether the right problem was actually solved

If your goal was prompt caching, `append` may help. If your goal was stronger memory influence, `prepend` may still be better. If your goal was user-near context, `user` may be worth the tradeoff. The right answer depends on what you were trying to fix.

## Troubleshooting common problems

### Problem: changing the setting does not change anything

Most likely causes:

- `autoRecall` is off
- there are no useful memories to inject
- the gateway was not restarted

### Problem: `append` did not improve behavior

That can happen. `append` is mainly about placement and prompt stability, not retrieval quality. If recall content itself is poor, injection position will not save it.

### Problem: `user` mode feels weird or too strong

That is normal. You told the plugin to place memory closer to the user turn. Switch back to `prepend` or `append` if you want something less opinionated.

### Problem: memory seems too dominant

Try `append`, lower `recallMaxTokens`, or reduce `recallContextTurns`. Sometimes the issue is recall volume, not position.

### Problem: memory seems too weak

Try `prepend`, raise `recallBudget`, or increase `recallMaxTokens`. The [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) is also worth checking in case the right facts were never stored cleanly.

## FAQ

### Which recall injection position should most people use?

Start with `prepend`. It is the default and the easiest to reason about.

### When should I switch to `append`?

When you have a large, stable system prompt and want recalled memories added after it for better prompt-shape stability.

### When should I use `user`?

Only when you intentionally want recalled memories to behave more like user-side context. It is an advanced option.

### Does this change what Hindsight recalls?

No. It changes where the recalled memories are injected, not the retrieval algorithm itself.

### What else matters besides `recallInjectionPosition`?

Usually `autoRecall`, `recallBudget`, `recallMaxTokens`, and `recallContextTurns`. The [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw), [Recall API reference](https://hindsight.vectorize.io/docs/api/recall), and [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) are the right places to go deeper.

## Next Steps

- [Create a Hindsight Cloud account](https://hindsight.vectorize.io) if you want the fastest path to testing recall behavior across more than one machine or environment.
- Keep the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw) open for the full plugin reference.
- Keep the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby if you want a clean baseline configuration to compare against.
- Use the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) to reason about what gets retrieved before placement even matters.
- Use the [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) if you suspect the wrong facts are getting stored.
- Compare related memory workflows like [Adding memory to Codex with Hindsight](https://hindsight.vectorize.io/blog/adding-memory-to-codex-with-hindsight) and the [team shared memory post](https://hindsight.vectorize.io/blog/team-shared-memory-ai-coding-agents) if you want ideas for how prompt and memory layout interact in other agents.
