---
title: "Hermes Memory Modes with Hindsight, Hybrid, Context, Tools"
authors: [benfrank241]
date: 2026-04-14
tags: [how-to, hermes, memory, configuration]
description: "Choose the right Hermes memory mode for Hindsight. Learn when to use hybrid, context, or tools, how to switch modes, and how to verify recall."
image: /img/blog/guide-hermes-memory-modes-with-hindsight-hybrid-context-tools.png
hide_table_of_contents: true
---

![Hermes Memory Modes with Hindsight, Hybrid, Context, Tools](/img/blog/guide-hermes-memory-modes-with-hindsight-hybrid-context-tools.png)

If you are trying to choose between **Hermes memory modes with Hindsight**, the decision is really about one question: should memory be injected automatically, exposed as tools, or both? Hermes's native Hindsight provider supports three integration modes, `hybrid`, `context`, and `tools`, and each one changes how the model experiences memory during a conversation.

This matters more than it sounds. Teams often turn on Hindsight, see that memory works, and stop there. But the mode determines whether recall happens before every turn, whether the model can call `hindsight_recall` and `hindsight_reflect` directly, and whether your assistant behaves like a silent memory system or an explicit tool-using agent. Pick the wrong mode and the setup still looks healthy, but the behavior feels off.

This guide explains what each mode does, when to use it, how to switch safely, how `prefetch_method` changes the experience, and how to verify that your chosen mode is actually doing what you think it is. For the complete reference, keep the [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes), the [docs home](https://hindsight.vectorize.io/docs), and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby.

<!-- truncate -->

> **Quick answer**
>
> 1. Use **`hybrid`** if you want automatic recall and explicit Hindsight tools.
> 2. Use **`context`** if you want invisible auto-recall with no memory tools exposed.
> 3. Use **`tools`** if you want the model to call memory deliberately, not automatically.
> 4. Keep `prefetch_method="recall"` for speed, switch to `reflect` only when synthesized context is worth the latency.
> 5. Verify your choice with `hermes memory status`, `/tools`, and a real next-turn recall test.

## Prerequisites

Before you change modes, make sure:

- Hermes is already configured to use the native Hindsight provider.
- `hermes memory status` reports a healthy memory setup.
- You know where your Hermes Hindsight config file lives.
- You understand that memory mode affects recall behavior, not whether your bank exists.

Hermes stores the native Hindsight config at `~/.hermes/hindsight/config.json` by default. You can print it with:

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get("HERMES_HOME", pathlib.Path.home() / ".hermes"))
path = base / "hindsight" / "config.json"
print(path.read_text())
PY
```

If you have not validated the provider yet, do that first:

```bash
hermes memory status
```

It also helps to skim the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) and [Retain API reference](https://hindsight.vectorize.io/docs/api/retain), because they make the behavior of each mode easier to reason about. `memory_mode` controls how recall enters the conversation, while retain behavior continues to follow the provider's retention settings.

## Step by step

### 1. Understand what each mode actually changes

The three modes are not cosmetic. They shape how memory enters the agent loop.

| Mode | Auto-recall before each turn | Explicit Hindsight tools visible | Best for |
|---|---|---|---|
| `hybrid` | Yes | Yes | Most users, assistants that need both convenience and control |
| `context` | Yes | No | Clean UX, consumer-facing assistants, less tool noise |
| `tools` | No | Yes | Agents that should decide when to query memory explicitly |

Think of them like this:

- **`hybrid`** is the default general-purpose mode. Relevant memories are injected before the LLM sees the new user message, and the model can still call `hindsight_recall`, `hindsight_retain`, and `hindsight_reflect` explicitly if needed.
- **`context`** hides the tools and relies on automatic injection only. This is ideal when you want the assistant to feel seamless and not expose extra memory actions.
- **`tools`** removes automatic recall from the prompt-building path. The model has to choose to call memory tools. This is more deliberate, but it also means a poorly instructed model can forget to look.

If you remember one rule, make it this one: **`tools` mode is not broken when auto-recall disappears. That is the design.**

### 2. Inspect your current mode and prefetch method

Before you change anything, look at the current values:

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get("HERMES_HOME", pathlib.Path.home() / ".hermes"))
path = base / "hindsight" / "config.json"
cfg = json.loads(path.read_text())
print("memory_mode:", cfg.get("memory_mode", "hybrid"))
print("prefetch_method:", cfg.get("prefetch_method", "recall"))
PY
```

`prefetch_method` matters because it changes what automatic injection looks like:

- **`recall`** injects raw memory facts. It is faster and usually the right default.
- **`reflect`** injects a synthesized answer built from relevant memories. It is slower, but can be better when the assistant needs a coherent summary instead of a list of facts.

The simplest reliable starting point is:

- `memory_mode="hybrid"`
- `prefetch_method="recall"`

That gives you automatic context plus explicit tools, with the least latency overhead.

### 3. Switch to `hybrid` when you want the safest default

If you want both automatic recall and explicit tool access, set `hybrid`:

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get("HERMES_HOME", pathlib.Path.home() / ".hermes"))
path = base / "hindsight" / "config.json"
cfg = json.loads(path.read_text())
cfg["memory_mode"] = "hybrid"
cfg["prefetch_method"] = "recall"
path.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"Updated {path} to hybrid mode")
PY
```

Choose `hybrid` if any of these are true:

- you want the assistant to remember relevant context without being prompted
- you also want power users or the model itself to call `hindsight_reflect` explicitly for deeper synthesis
- you are still evaluating Hindsight and want the easiest mode to debug

This is also the best mode for internal assistants and technical users, where tool visibility is not a UX problem.

### 4. Switch to `context` when you want invisible memory

If you want recall to happen automatically but do **not** want the memory tools visible to the model, use `context`:

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get("HERMES_HOME", pathlib.Path.home() / ".hermes"))
path = base / "hindsight" / "config.json"
cfg = json.loads(path.read_text())
cfg["memory_mode"] = "context"
cfg.setdefault("prefetch_method", "recall")
path.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"Updated {path} to context mode")
PY
```

Use `context` when:

- you want a cleaner tool surface
- you are running a customer-facing assistant where fewer visible tools means less behavioral noise
- you trust automatic injection more than tool planning

For many production assistants, `context` is the best user experience. The model simply starts with relevant history. It does not need to remember that memory exists as a tool category.

### 5. Switch to `tools` when you want deliberate memory access

If you do not want automatic recall at all, use `tools`:

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get("HERMES_HOME", pathlib.Path.home() / ".hermes"))
path = base / "hindsight" / "config.json"
cfg = json.loads(path.read_text())
cfg["memory_mode"] = "tools"
path.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"Updated {path} to tools mode")
PY
```

This is the right choice when:

- you want the model to decide when memory is relevant
- you are optimizing for tighter prompt control
- you are experimenting with explicit agent strategies that use `hindsight_recall` and `hindsight_reflect` as first-class reasoning tools

The tradeoff is obvious but important: if the model is not prompted well, it may simply forget to use memory. `tools` mode gives you control, but less safety.

### 6. Decide whether `prefetch_method` should stay on `recall` or move to `reflect`

`prefetch_method` only matters when auto-recall is active, so it applies to `hybrid` and `context`.

Use this script to switch methods:

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get("HERMES_HOME", pathlib.Path.home() / ".hermes"))
path = base / "hindsight" / "config.json"
cfg = json.loads(path.read_text())
cfg["prefetch_method"] = "reflect"
path.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"Updated {path} to reflect prefetch")
PY
```

Here is the practical difference:

| Prefetch method | What Hermes gets | Speed | Best for |
|---|---|---|---|
| `recall` | raw relevant facts | faster | coding assistants, support, most chat workflows |
| `reflect` | synthesized summary across memories | slower | complex planning, open-ended reasoning, summarization |

If you are unsure, stay on `recall`. It is easier to reason about, easier to debug, and usually enough. Reach for `reflect` only when the model keeps needing a coherent memory summary rather than point facts.

### 7. Match the mode to the job

A quick practical map:

- **Personal assistant with recurring preferences and projects**: `hybrid`
- **Customer-facing assistant where you want seamless personalization**: `context`
- **Research or coding agent that should query memory intentionally**: `tools`
- **Planning-heavy assistant with deep history**: `hybrid` or `context` plus `prefetch_method="reflect"`

This is also where adjacent integrations are useful reference points. If you like the silent memory pattern, the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw) show the same idea in a different agent architecture. If you want a tool-centric workflow, [Adding memory to Codex with Hindsight](https://hindsight.vectorize.io/blog/adding-memory-to-codex-with-hindsight) is a good comparison.

## Verifying it works

Do not trust the config file alone. Verify behavior.

### Check provider status

```bash
hermes memory status
```

This confirms that the provider is active, not that the chosen mode behaves the way you expect. For that, you need live testing.

### Check tool visibility

Launch Hermes and inspect the tool list.

- In **`hybrid`**, you should see the Hindsight tools.
- In **`context`**, you should not.
- In **`tools`**, you should see the tools, but no automatic recall should happen before a turn.

### Run a next-turn recall test

Tell Hermes something it can remember:

```text
Remember that the design review is Thursday at 3 PM and we are prioritizing the mobile onboarding flow.
```

Then, on the next turn, ask:

```text
What do you remember about the design review?
```

Expected behavior:

- **`hybrid`** and **`context`**: relevant memory should already be in context before the model answers.
- **`tools`**: the model must choose to call a Hindsight tool, or it may answer without using memory.

### Verify your expectations against the chosen mode

Many false bug reports are really expectation mismatches:

- a user chooses `tools`, then expects auto-recall
- a user chooses `context`, then expects Hindsight tools in `/tools`
- a user chooses `reflect` prefetch, then is surprised by higher latency

The behavior has to match the mode you selected.

## Troubleshooting common mistakes

### Problem: `tools` mode seems to forget everything

That is often just a prompt issue. In `tools` mode, the model has to decide to use memory tools. If you want reliable automatic memory use, switch to `hybrid` or `context`.

### Problem: `context` mode works, but I cannot find `hindsight_reflect`

That is expected. `context` hides the explicit tools. Automatic recall is the feature, not tool exposure.

### Problem: `hybrid` shows tools, but no memory is injected

Check whether your Hermes build supports the required lifecycle hooks. The Hermes docs note that on older builds, only tools are registered and auto-injection is skipped.

### Problem: `reflect` prefetch feels slow

It is slower by design because it synthesizes context instead of injecting raw recalled facts. Switch back to `recall` if responsiveness matters more than coherence.

### Problem: nothing changes after editing the config file

Make sure you edited the correct file under `~/.hermes/hindsight/config.json`, then restart Hermes so the new config is loaded.

### Problem: memory appears stale or incomplete right after a user shares a new fact

Remember the async flow: retention happens after the response, then the new memory is available on the next turn. Test next-turn behavior, not same-turn behavior.

## FAQ

### Which mode should most people start with?

`hybrid`. It is the safest default and easiest to debug.

### When is `context` better than `hybrid`?

When you want invisible personalization and a cleaner tool surface. It is especially good for assistants that should feel simple and conversational.

### When is `tools` better than `hybrid`?

When you explicitly want the model to reason about whether memory should be consulted. It gives you more control, but less automation.

### Does `tools` mode disable retention too?

No. `tools` mode changes how recall enters the agent experience. Retention behavior is still governed by the provider's retention settings and lifecycle hooks.

### Should I use `prefetch_method="reflect"` everywhere?

Usually not. Use `recall` by default. Move to `reflect` when you have a real need for synthesized memory context.

### Where can I learn more about what Hindsight is injecting?

The [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) is the best place to understand fact retrieval, and the [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) explains how new information becomes memory in the first place. The [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes) tie those concepts back to Hermes configuration.

## Next Steps

- [Create a Hindsight Cloud account](https://hindsight.vectorize.io) if you want the fastest way to test mode changes without local infrastructure work.
- Read the [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes) for the complete config surface.
- Keep the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) open if you want a cleaner end-to-end setup reference.
- Read the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) to understand what `recall` prefetch actually returns.
- Read the [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) if you want to tighten what becomes memory.
- Compare the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw) and [Adding memory to Codex with Hindsight](https://hindsight.vectorize.io/blog/adding-memory-to-codex-with-hindsight) if you want examples of other memory interaction styles.
