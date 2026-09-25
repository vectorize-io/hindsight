---
title: "Does Hindsight + Hermes = AGI?"
authors: [benfrank241]
slug: "2026/09/25/hindsight-hermes-plugin-catalog"
date: 2026-09-25T15:00
tags: [hindsight, hermes, nous-research, plugins, memory-provider, integration, migration]
description: "Hermes moved its memory providers out of core and into the plugin catalog. Hindsight went first. Here is what changed, what you need to do about it, and where updates come from now."
image: /img/blog/hindsight-hermes-agi.png
hide_table_of_contents: true
---

![Yes.](/img/blog/hindsight-hermes-agi.png)

Yes.

No. Obviously not. But now that you are here: [Hermes Agent](https://github.com/NousResearch/hermes-agent) moved its memory providers out of the core codebase and into its plugin catalog this week, Hindsight went first, and if you are already running the two together you should know what that means for you.

It means nothing. That is the entire point of this post, and you can stop reading here if you want. For everyone who wants the detail, it is below.

<!-- truncate -->

## TL;DR

- **Nothing changes if you already use Hindsight with Hermes.** Your provider name, your settings, your memory bank and your data directories are all untouched.
- **Hermes migrates you automatically** on `hermes update` or on first agent start. You will see a one-line confirmation and nothing else.
- **The plugin now lives in the Hindsight repo** and is maintained by the Hindsight team rather than shipped inside Hermes.
- **New users install it from the catalog**: `hermes plugins install hindsight`, then `hermes memory setup`.
- **`hermes update` does not update the plugin.** `hermes plugins update hindsight` does. You can also freeze a version with `--ref`, or track our `main` branch.
- **One genuine gotcha**, and it predates this change: `hermes plugins enable` does not activate a memory provider.

## What actually happened

Nous Research is moving every memory provider out of the `hermes-agent` core tree and into its maintainer's own repository, published through the [plugin catalog](https://hermes-agent.nousresearch.com/docs/plugins). Their documentation puts it plainly:

> Memory providers are moving out of the Hermes tree into their maintainers' own repositories, published through the plugin catalog — Hindsight is the first.

The change landed upstream on 23 September, in a PR titled "hindsight moves to the plugin catalog (auto-migrated on first start / hermes update)." The bundled `plugins/memory/hindsight/` directory is gone from Hermes, and the plugin now lives in the Hindsight repository under `hindsight-integrations/hermes/`, maintained by us.

This is a good outcome for everyone. A memory provider bundled in someone else's tree moves at the speed of that tree's release cycle, and a bug in Hindsight's provider had to be fixed by people whose project is an agent, not a memory system. Now the people who wrote the memory system ship the fixes.

## If you already use Hindsight: nothing changes

Worth repeating, because "we moved your integration" is a sentence that normally comes with a weekend of work attached.

Your `memory.hindsight` settings stay valid. Your bank keeps its id and its contents. Your API key stays in `~/.hermes/.env`. Your `~/.hermes/hindsight/config.json` is untouched. The three tools keep their names: `hindsight_retain`, `hindsight_recall` and `hindsight_reflect`.

The migration is automatic. Run `hermes update`, or just start an agent with a Hindsight provider configured, and Hermes resolves the provider name against the catalog and installs the plugin for you. You will see a line like this:

```
✓ Memory provider 'hindsight' moved out of core — installed its plugin from the
  catalog (your memory.hindsight settings and data are unchanged).
```

Afterwards the plugin sits in `~/.hermes/plugins/hindsight/` and your `config.yaml` picks up `plugins.enabled: [hindsight]`. That is the whole migration.

We ran it on a machine still on the bundled provider to check. Two things are worth knowing before you do. The update is **interactive**: it asks whether to prepare the plugin's Python dependencies, and answers nothing on its own, so an unattended run will sit at the prompt rather than finish. And if you have local modifications in the Hermes tree, it stashes them before updating and asks whether to restore them, printing a `git stash apply <sha>` you will want to keep.

Afterwards, `~/.hermes/plugins/.install-metadata.json` records exactly what you got: the catalog pin, the source repo, and the resolved commit. Ours came through as version 1.0.1 pinned at `176f8c2d`.

One condition to be aware of: automatic installation depends on `security.allow_lazy_installs`, which is on by default. If you have turned it off, Hermes will not install anything behind your back. It logs a single line telling you to run `hermes plugins install hindsight` yourself.

## If you are starting fresh

Two commands:

```bash
hermes plugins install hindsight
hermes memory setup    # select "hindsight"
```

Hindsight is in the catalog, so the name is all you need on the first line. The setup wizard then installs dependencies into the Hermes venv via `uv`, walks you through configuration, and offers to seed the bank with a starter memory template. It warns before overwriting a bank that is already configured, and you can skip the template entirely.

Do not skip the second command. More on why below.

## The one thing that trips people up

This predates the move and catches people regardless, so it is worth stating clearly:

**`hermes plugins enable` is not what activates a memory provider.**

Hermes treats memory providers as `kind: exclusive`, and its plugin-enable gate deliberately skips them. What activates a provider is `memory.provider: hindsight` in your `config.yaml`, which `hermes memory setup` writes for you. Installing the plugin and enabling it without running setup leaves you with a provider that is present and not in use.

There is a related trap in embedded mode. `local_embedded` needs the `hindsight-all` package rather than just the client, and `pyproject.toml` deliberately does not declare it, because that would push the whole local ML stack onto cloud-mode users who have no use for it. The setup wizard installs it, and the plugin self-installs it as a backstop on the availability check. But `plugins install` on its own, with no `memory setup`, leaves embedded mode reporting "not available."

The rule that covers both: always run `hermes memory setup`.

## Three modes, unchanged

For completeness, since the move is a good moment to check you are on the right one.

| Mode | What it does | What it needs |
|---|---|---|
| `cloud` | Connects to Hindsight Cloud | An API key from [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io) |
| `local_embedded` | Hermes runs a local Hindsight daemon with built-in PostgreSQL | An LLM API key for extraction and synthesis; embeddings and reranking run locally |
| `local_external` | Points at a Hindsight instance you already run | A URL and an optional API key |

Embedded mode starts the daemon in the background on first use and stops it after five minutes of inactivity. It works with any OpenAI-compatible endpoint, so llama.cpp, vLLM and LM Studio are all fine: choose `openai_compatible` and give it a base URL. If you want the Hindsight web UI against your embedded instance, that is `hindsight-embed -p hermes ui start`.

## Updating: the habit to unlearn

This is the one thing that genuinely changes for you, and it is a habit rather than a command.

**Nothing updates the plugin on its own.** In particular, `hermes update` does *not* move it. It reinstalls the plugin's Python dependencies and leaves the checkout exactly where it is. If you came from the bundled provider, that is the assumption to drop: memory improvements no longer arrive as a side effect of updating Hermes.

There are three ways to decide which version you run.

### Follow the official pin

The default, and the right answer for almost everyone. This is what `hermes plugins install hindsight` gives you: the commit Nous reviewed and pinned in their catalog.

```bash
hermes plugins update hindsight    # move to the catalog's current pin
```

`hermes plugins list` flags the plugin `update_available` once your installed commit differs from the catalog's. Note that re-running `hermes plugins install hindsight` does **not** update it — it refuses with "already exists." `plugins update` is the command that re-pins.

Two consequences of the pin worth internalising. A change we merge does not reach you when we merge it; the pin has to move first, which takes a follow-up PR to `hermes-agent`. And Hermes re-fetches the published catalog at most once every six hours, so a freshly released version can take that long to even appear as available. If something we shipped today is not showing up, that is usually why.

### Pin a specific release

For a version you choose and freeze:

```bash
hermes plugins install vectorize-io/hindsight/hindsight-integrations/hermes \
  --force --ref <40-character-commit-sha>
```

Two things about `--ref`. It takes a full commit SHA and **rejects tag names**, so take the SHA from the [release notes](https://github.com/vectorize-io/hindsight/releases) rather than typing `v1.0.1`. And a `--ref` install is marked pinned, so `hermes plugins update hindsight` will deliberately refuse to move it. That is the point — you asked to be frozen. Install again with a new `--ref` when you want a different version.

### Track the latest development code

For fixes before they reach the catalog. Install from the source path rather than the catalog name:

```bash
hermes plugins install vectorize-io/hindsight/hindsight-integrations/hermes
hermes plugins update hindsight    # now a git pull of our main branch
```

Same update command, different meaning: installed this way, it follows our `main`. Unreviewed by definition — you get whatever is there the moment you run it. Useful if you are chasing a fix we just landed or testing something with us; not what we would run in production.

The catalog entry itself lives in Nous' repo at [`plugin-catalog/hindsight.yaml`](https://github.com/NousResearch/hermes-agent/blob/main/plugin-catalog/hindsight.yaml), which is what records the current pin.

**One update you should not defer.** If you installed between 14 and 21 September 2026, `hindsight-embed` 0.10.0 breaks `local_embedded` outright: its daemon probe cleared the calling thread's event loop, so the next client call failed with `Timeout context manager should be used inside a task`. That is why the client floor is 0.10.1 rather than the 0.6.1 the plugin needs at the API level. Here `hermes update` *is* the fix, because the problem is a dependency rather than the plugin code.

## One default worth knowing about

Recall now returns **observations only** by default, where it previously returned all three fact types.

Observations are Hindsight's consolidated layer: deduplicated beliefs grounded in evidence, carrying proof counts and freshness signals, refined as new facts arrive. Raw `world` and `experience` facts are the individual evidence underneath them. For per-turn context injection, observations are denser per token and stop the model being handed five raw facts that one observation already summarises.

If you want the old behaviour, set `"recall_types": "observation,world,experience"` in `~/.hermes/hindsight/config.json`. Note that this setting governs both auto-recall and the `hindsight_recall` tool, since the tool schema has no per-call types argument. Narrowing the default narrowed both.

## How it behaves per turn

Since the move is a reasonable moment to audit your configuration, the three settings that shape day-to-day behaviour:

`memory_mode` decides how memory reaches the model. `hybrid`, the default, does automatic context injection and exposes the three tools. `context` injects only and hides the tools. `tools` exposes them and injects nothing, which is the one to pick if you want the model deciding when to reach for memory.

`auto_recall` and `auto_retain` are both on by default, recalling before each turn and saving each turn. `recall_sync` is off, meaning recall runs in the background and lands on the following turn; turning it on recalls against the current message for better relevance at the cost of latency in the turn.

`recall_indicator` and `retain_indicator` print the `👁️ Hindsight` status lines. Leave them on while you are tuning, because seeing how many memories were injected is the fastest way to tell whether recall is doing anything useful. Turn them both off for anything customer-facing.

## So, AGI?

No.

What you get is an agent that remembers, which is a smaller claim and a more useful one. Hermes handles the reasoning and the interface. Hindsight handles what survives the end of the conversation: facts extracted from your turns, entities resolved across sessions, observations consolidated from repeated evidence, and a recall pass that puts the relevant subset back in front of the model before it answers.

That is not general intelligence. It is the difference between an assistant that asks your postgres version every morning and one that does not, and after a few weeks of use the second one is a meaningfully different product to live with.

The move to the catalog does not change any of that. It just means the people who build the memory system are now the people shipping it.

## FAQ

**Do I need to do anything?**
No. Run `hermes update` when you would have anyway, or just keep using Hermes. The migration happens on update or on first agent start, and it is a plugin install, not a data operation. The only people who need to act deliberately are those who have turned off `security.allow_lazy_installs`, who should run `hermes plugins install hindsight` once, and anyone who installed between 14 and 21 September 2026, who should update to clear the embedded-mode event loop bug.

**Will I lose any memories?**
No. Nothing about your bank is touched. The plugin is the code that talks to Hindsight, not the place your memories live. Whether your bank is on Hindsight Cloud, a local embedded daemon or an instance you run yourself, the data sits where it has always sat and the migration does not read or write it.

**Can I stay on the bundled provider?**
Not for long. The bundled `plugins/memory/hindsight/` directory has been removed from `hermes-agent`, so once you update past that point the catalog plugin is the only copy. While both existed, the bundled one won: provider lookup goes bundled, then `~/.hermes/plugins/`, then project, then entry point, first hit wins. That is why installing the plugin early was inert rather than harmful.

**Does this change anything about Hindsight Cloud or self-hosting?**
No. Modes, endpoints, API keys and bank behaviour are all unchanged. This is a change in how the Hermes-side connector is distributed, and it stops at the connector.

**How do I check what I am running?**
`hermes plugins list` shows the installed plugin and its version, which should be 1.0.1 or newer. Your active provider is `memory.provider` in `config.yaml`, and your Hindsight-specific settings are in `~/.hermes/hindsight/config.json`. If embedded mode is misbehaving, the startup log is `~/.hermes/logs/hindsight-embed.log` and the daemon's own log is `~/.hindsight/profiles/<profile>.log`.

**Where do I report a bug now?**
The Hindsight repository, not `hermes-agent`. That is the practical upside of the move: issues and fixes land with the team that maintains the memory system, and ship on the next catalog pin bump rather than waiting on an unrelated release cycle.

## Learn more

- [Hermes memory providers](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory-providers) for the upstream view of the migration
- [The Hindsight plugin in the Hermes catalog](https://hermes-agent.nousresearch.com/docs/plugins/hindsight)
- [Hermes Agent persistent memory with Hindsight](https://hindsight.vectorize.io/sdks/integrations/hermes) for the full configuration reference
- [The Fully Open Agent Memory Stack: Self-Hosting Hermes + Hindsight](https://hindsight.vectorize.io/blog/2026/07/17/hermes-hindsight-open-stack) for running the whole thing locally
- [Give Every Hermes Bot Its Own Memory](https://hindsight.vectorize.io/blog/2026/08/18/hermes-bot-mode-memory) on per-bot bank isolation
