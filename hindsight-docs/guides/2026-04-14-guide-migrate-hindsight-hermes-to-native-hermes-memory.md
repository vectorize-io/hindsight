---
title: "Guide: Migrate hindsight-hermes to Native Hermes Memory"
authors: [benfrank241]
date: 2026-04-14
tags: [how-to, hermes, memory, migration]
description: "Learn how to migrate hindsight-hermes to native Hermes memory without losing your bank, then verify recall, tools, and cross-session memory still work."
image: /img/blog/guide-migrate-hindsight-hermes-to-native-hermes-memory.png
hide_table_of_contents: true
---

![Guide: Migrate hindsight-hermes to Native Hermes Memory](/img/blog/guide-migrate-hindsight-hermes-to-native-hermes-memory.png)

If you need to **migrate `hindsight-hermes` to native Hermes memory**, the good news is that the move is smaller than it looks. Hermes now has a built-in Hindsight provider, so you no longer need the old pip plugin path from the original setup guide. In most cases, the migration is just: uninstall the old plugin package, point Hermes at the native provider, keep the same bank ID, and verify that automatic recall works on the next turn.

That last part matters. Many teams are nervous about migration because they assume it means rebuilding memory from scratch, re-teaching the agent, or manually exporting and importing data. Usually, none of that is required. If your old setup already stored memories in Hindsight, your bank already exists. The native provider can keep using it as long as you preserve the same backend and `bank_id`.

This guide walks through the safe path, including how to back up your current config, uninstall the deprecated plugin cleanly, switch to Hermes's native Hindsight provider, verify recall and tools, and fix the most common migration mistakes. If you want the broader setup reference while you work, keep the [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes) and the [Hindsight docs home](https://hindsight.vectorize.io/docs) open in another tab.

<!-- truncate -->

> **Quick answer**
>
> 1. Back up your current Hermes and Hindsight config.
> 2. Uninstall `hindsight-hermes` from the Hermes Python environment.
> 3. Run `hermes memory setup` and select `hindsight`, or update the native config manually.
> 4. Keep the same `bank_id` if you want to preserve your existing memories.
> 5. Run `hermes memory status`, then test recall on the next turn, not the same turn.

## Prerequisites

Before you start, make sure you know which of these setups you are migrating from:

- **Old plugin setup** from the earlier `hindsight-hermes` guide, where the package was installed into the Hermes virtual environment and registered through Python entry points.
- **Hindsight Cloud backend**, where your memories already live in Hindsight Cloud and Hermes reaches them through an API key.
- **Local Hindsight backend**, where your memories live in a local Hindsight service and Hermes points at it by URL.

You should also have:

- Hermes installed and working.
- A current Hindsight configuration that you can inspect.
- Access to the same API key or local backend that your old plugin used.
- A few known memories you can test with after the migration.

If you are not sure whether the old plugin is present, check the Python environment Hermes uses:

```bash
$HOME/.hermes/hermes-agent/venv/bin/python -m pip show hindsight-hermes
```

If that command prints package metadata, you are almost certainly on the older plugin path.

It is also worth skimming the newer [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes), the [quickstart](https://hindsight.vectorize.io/docs/quickstart), and the [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) before you begin. They are useful if you need to double-check how retention and recall behave after the cutover.

## Step by step

### 1. Back up your current config before you touch anything

Migration is usually straightforward, but the cheapest insurance is a copy of the files you are about to change. Back up the Hermes env file and the Hindsight config directory first:

```bash
mkdir -p ~/hermes-memory-migration-backup
cp -R ~/.hermes ~/hermes-memory-migration-backup/hermes
cp -R ~/.hindsight ~/hermes-memory-migration-backup/hindsight 2>/dev/null || true
```

If you are using cloud mode, this backup is mostly about convenience. Your memories live remotely, so the critical thing is preserving the same API credentials and bank ID. If you are using a local backend, the backup matters more because it gives you a fast rollback point for config and runtime logs.

You should also inspect your current Hindsight values before you migrate. In practice, the values that matter most are:

- the Hindsight API URL
- the bank ID
- whether you are using cloud or local mode
- whether the old plugin wrote to a shared or personal bank

If Hermes already has a native config file, inspect it directly:

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get("HERMES_HOME", pathlib.Path.home() / ".hermes"))
path = base / "hindsight" / "config.json"
print(path)
if path.exists():
    print(path.read_text())
else:
    print("No native Hermes Hindsight config yet")
PY
```

If your old setup relied on env vars, inspect `~/.hermes/.env` too:

```bash
grep '^HINDSIGHT_' ~/.hermes/.env || true
```

Write down the current `bank_id`. Reusing that bank is how you keep your existing memory history.

### 2. Remove the deprecated plugin package from the Hermes environment

The old `hindsight-hermes` path is deprecated. Hermes now ships a native provider, and you do not want both approaches competing for responsibility. Uninstall the plugin from the same Python environment Hermes uses:

```bash
uv pip uninstall hindsight-hermes --python $HOME/.hermes/hermes-agent/venv/bin/python
```

If you do not use `uv`, this fallback works too:

```bash
$HOME/.hermes/hermes-agent/venv/bin/python -m pip uninstall -y hindsight-hermes
```

Why uninstall at all if the new provider can coexist with the same backend? Because coexistence is not the goal. You want one memory path, one config surface, and one debugging story. Keeping the deprecated plugin around makes it much harder to tell whether a missing recall came from the old registration path, the new provider config, or a conflict between the two.

After uninstalling, confirm that Hermes no longer sees the plugin entry point:

```bash
$HOME/.hermes/hermes-agent/venv/bin/python - <<'PY'
import importlib.metadata
for ep in importlib.metadata.entry_points(group='hermes_agent.plugins'):
    print(f"{ep.name}: {ep.value}")
PY
```

If you still see an entry for `hindsight`, you are probably uninstalling from the wrong environment.

### 3. Configure the native Hindsight provider

The simplest path is the setup wizard:

```bash
hermes memory setup
```

When prompted, select **Hindsight** as the provider. If you are using Hindsight Cloud, enter the same API URL and API key you used before. If you are using local mode, point Hermes at the same local backend or choose local mode in the wizard so it can create the expected native config.

If you prefer to configure it manually, set the provider and write the same credentials Hermes should use going forward:

```bash
hermes config set memory.provider hindsight
printf '%s\n' 'HINDSIGHT_API_KEY=your-key' >> ~/.hermes/.env
printf '%s\n' 'HINDSIGHT_API_URL=https://api.hindsight.vectorize.io' >> ~/.hermes/.env
```

Then verify the native config exists:

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get("HERMES_HOME", pathlib.Path.home() / ".hermes"))
path = base / "hindsight" / "config.json"
cfg = json.loads(path.read_text())
print(json.dumps(cfg, indent=2))
PY
```

If you are preserving an existing bank, confirm the `bank_id` matches your previous setup. This is the key migration detail people miss. The memories are attached to the Hindsight bank, not to the deprecated plugin package itself. If the new provider points at the same backend and same bank ID, it can continue using the same memory store.

A minimal native cloud config looks like this:

```json
{
  "mode": "cloud",
  "api_url": "https://api.hindsight.vectorize.io",
  "api_key": "hsk_your_token",
  "bank_id": "hermes"
}
```

A minimal native local config looks like this:

```json
{
  "mode": "local",
  "llm_provider": "groq",
  "llm_api_key": "your-groq-key",
  "bank_id": "hermes"
}
```

If you want a deeper mental model of what the provider is doing during retention and recall, the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) and [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) are worth reading.

### 4. Decide whether to disable Hermes's built-in memory tool

The native Hindsight provider gives you automatic recall through lifecycle hooks and can also expose explicit Hindsight tools. Hermes still has its built-in `memory` tool, which writes local markdown notes. If both are active, the model may keep choosing the built-in tool out of habit.

If you want Hindsight to be the only memory path, disable the built-in tool:

```bash
hermes tools disable memory
```

This is especially helpful after migration because it removes ambiguity. When you test storage and recall, you know the result came from Hindsight, not from the old markdown path.

You can always turn it back on later:

```bash
hermes tools enable memory
```

### 5. Set the integration mode you actually want

The native provider supports `hybrid`, `context`, and `tools` modes. After migration, the safest default is usually `hybrid`, because it gives you automatic context injection and explicit tools at the same time.

Use this small script to set the mode explicitly:

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get("HERMES_HOME", pathlib.Path.home() / ".hermes"))
path = base / "hindsight" / "config.json"
cfg = json.loads(path.read_text())
cfg["memory_mode"] = "hybrid"
cfg.setdefault("prefetch_method", "recall")
path.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"Updated {path}")
PY
```

If you are migrating a setup that depended on explicit tools only, `tools` mode may be closer to the old mental model. If you want the easiest end-user experience, where relevant memories appear without the model deciding to search, `hybrid` or `context` is usually better.

### 6. Run a real migration test

Do not stop at `status`. Test the thing users actually care about: cross-session memory.

Start Hermes and store a simple fact:

```text
Remember that our launch date is May 21 and the API rollout depends on the billing migration.
```

Then ask a follow-up on the **next turn**, or in a new session:

```text
What do you remember about the launch plan?
```

The next-turn detail matters. Hindsight retention runs asynchronously after the response, and the provider's prefetch happens before the following turn. If you try to retain and recall in the exact same turn, you can conclude the migration failed when the system is actually behaving normally.

## Verifying it works

Use a layered verification flow, not a single check.

### `hermes memory status`

Start with the obvious one:

```bash
hermes memory status
```

You want confirmation that Hermes sees the Hindsight provider as active. If it does not, fix that before testing anything else.

### Config inspection

Confirm your important values did not change during setup:

- `mode`
- `bank_id`
- `memory_mode`
- `prefetch_method`

The fastest check is:

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get("HERMES_HOME", pathlib.Path.home() / ".hermes"))
path = base / "hindsight" / "config.json"
print(json.dumps(json.loads(path.read_text()), indent=2))
PY
```

### Tool visibility

If you are using `hybrid` or `tools`, launch Hermes and inspect the available tools. You should see `hindsight_retain`, `hindsight_recall`, and `hindsight_reflect` in the tool list.

### Cross-session behavior

This is the real success criterion. Tell Hermes something memorable, then ask about it on the next turn or in a new session. If the same bank ID and backend are in use, you should keep your memory history across the migration.

### Local backend health

If you use local mode, verify the daemon is reachable:

```bash
curl http://localhost:9077/health
```

A healthy response tells you the provider has something real to talk to. If you are using cloud mode, the better signal is a clean `hermes memory status` result plus successful live recall.

## Troubleshooting common migration problems

### Problem: the provider is configured, but nothing is recalled automatically

Check `memory_mode`. If it is set to `tools`, auto-recall is disabled by design. The model has to call `hindsight_recall` explicitly. Switch to `hybrid` or `context` if you want automatic memory injection before every turn.

### Problem: the old plugin still appears after uninstall

You are probably uninstalling from the wrong Python environment. Repeat the uninstall using the exact Hermes virtual environment path:

```bash
$HOME/.hermes/hermes-agent/venv/bin/python -m pip uninstall -y hindsight-hermes
```

Then rerun the entry point inspection command.

### Problem: the migration succeeded, but the agent seems to remember nothing

Confirm the new config uses the same `bank_id` and the same backend as the old setup. Changing either one effectively points Hermes at a different brain.

### Problem: tools show up, but auto-recall still does not happen

The Hermes docs note that lifecycle hooks require a build with `pre_llm_call` and `post_llm_call` support. On older Hermes versions, only the three tools are registered. In that case, the provider is partially present, but automatic injection is skipped.

### Problem: local mode feels broken right after first launch

Fresh local mode startup can take a while because the embedded Hindsight server and PostgreSQL need to initialize. Check the startup log:

```bash
cat ~/.hermes/logs/hindsight-embed.log
```

If you only waited a few seconds and assumed failure, give it a little more time and retry.

### Problem: recall fails right after retain

That can be normal. Retention is asynchronous. Test on the next turn. This is one of the easiest migration false alarms.

## FAQ

### Do I lose my existing memories when I migrate?

Usually, no. If the old plugin and the new provider both point at the same Hindsight backend and the same `bank_id`, the memory bank is the same. Migration changes the integration path, not the underlying stored memories.

### Do I need to export or import anything first?

Not for a normal plugin-to-native migration. The main migration work is configuration, not data conversion.

### Should I keep the same bank ID?

Yes, if your goal is continuity. Change the bank ID only if you intentionally want a fresh memory bank.

### Do I still need the built-in Hermes `memory` tool?

Not necessarily. Many teams disable it so Hindsight is the single source of truth for persistent memory. That makes testing and debugging much clearer.

### What is the best mode after migration?

For most users, `hybrid` is the safest default. You get automatic recall plus explicit tools. If you want invisible memory with no tools shown to the model, use `context`. If you want manual retrieval only, use `tools`.

### Where should I go if I want a more complete setup reference?

Keep the [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes), the [docs home](https://hindsight.vectorize.io/docs), and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby. If you want to compare how other agent integrations handle shared memory, the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw) and [Adding memory to Codex with Hindsight](https://hindsight.vectorize.io/blog/adding-memory-to-codex-with-hindsight) are useful reference points.

## Next Steps

- [Create a Hindsight Cloud account](https://hindsight.vectorize.io) if you want the fastest migration path with a managed backend.
- Read the [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes) for the full configuration surface.
- Read the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) if you want a smaller end-to-end Hindsight refresher.
- Use the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) when you need to reason about what auto-recall is actually returning.
- Use the [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) if you want to tune what gets stored in your bank.
- Compare adjacent workflows like the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw) to see how other agent frameworks approach automatic memory injection.
