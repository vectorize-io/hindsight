---
sidebar_position: 10
title: "Hermes Agent Persistent Memory with Hindsight | Integration"
description: "Add long-term memory to Hermes Agent with Hindsight. Install from the Hermes plugin catalog, then recall context automatically before every turn and retain conversations for future sessions."
---

{/* GENERATED from hindsight-integrations/hermes/README.md — edit that file, then run
    node hindsight-docs/scripts/sync-hermes-doc.mjs */}

:::tip
Using the **Hermes desktop app**? You can select and configure Hindsight entirely in Settings — no
terminal required. See [Hermes Desktop](/sdks/integrations/hermes-desktop).
:::

:::warning Deprecated: the standalone `hindsight-hermes` plugin
The old **`hindsight-hermes`** pip plugin (installed into the Hermes virtual environment and
registered through the `hermes_agent.plugins` entry point) is **deprecated** — on current Hermes
builds its tools fail with `{"error": "Timeout context manager should be used inside a task"}`.
Follow [Migrate hindsight-hermes to Native Hermes Memory](/guides/2026/04/14/guide-migrate-hindsight-hermes-to-native-hermes-memory)
to switch over while keeping the same memory bank.
:::

Long-term memory with knowledge graph, entity resolution, and multi-strategy retrieval. Supports cloud, local embedded, and local external modes.

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) memory-provider plugin, installed from
the Hermes plugin catalog. It used to ship inside Hermes as `plugins/memory/hindsight/`; Nous Research
moved every memory provider out of the core tree, so it is now maintained by the Hindsight team in
[vectorize-io/hindsight](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/hermes)
and the catalog pins it from there.

[View Changelog →](/changelog/integrations/hermes)

## Install

Hindsight is in the Hermes plugin catalog, so the name is all you need:

```bash
hermes plugins install hindsight
hermes memory setup           # select "hindsight"
```

`plugins install` asks for confirmation before enabling the plugin. Pass `--enable` (or
`--no-enable`) to skip the prompt, which is what you want in a script or an unattended run.

Dependencies in `pyproject.toml` are installed into the Hermes venv and survive `hermes update`.
Hermes asks before preparing them: an update that finds new plugin dependencies prints
`<plugin> declares Python dependencies` and waits for a yes. An unattended `hermes update` will
sit at that prompt rather than finish.

`hermes plugins enable` is *not* what activates a memory provider — Hermes treats providers as
`kind: exclusive` and its plugin-enable gate deliberately skips them. A provider is activated by
`memory.provider: <name>` in `config.yaml`, which `hermes memory setup` writes.

### How `local_embedded` runs

The Hindsight server is **not** a dependency of Hermes' venv. `local_embedded` starts it as a
separate process — an installed `hindsight-api` binary, else a `uvx hindsight-api` fallback that
gets its own environment — and talks HTTP to it. This venv keeps only `hindsight-client` and
`hindsight-embed`, both declared in `pyproject.toml`, so every mode installs with the plugin and
nothing has to be fetched at runtime.

That is not a size optimisation, it is the only shape that installs. `hindsight-all` pulls the whole
`hindsight-api-slim` tree, which cannot resolve against Hermes' pinned extras: api-slim needs
`protobuf>=7.35.1` while `mem0ai==2.0.10` and `modal==1.5.5` cap it below 7.0, and
`opentelemetry-semantic-conventions>=0.65b0` while `mistralai==2.4.8` caps it below 0.61. Relaxing
one pin only uncovers the next. It is also 482 MB on macOS arm64 and 3.2 GB on Linux x86_64 (the
CUDA wheels torch pulls there) that cloud-mode users would pay for nothing.

`_new_embedded_client` therefore composes what the old `hindsight.HindsightEmbedded` wrapper composed
internally — `hindsight_embed.get_embed_manager()` plus `hindsight_client.Hindsight` — instead of
importing it. **Existing embedded users keep everything**: the profile decides the daemon, so the
database (`~/.pg0/instances/hindsight-embed-<profile>/`) and the profile env
(`~/.hindsight/profiles/<profile>.env`) are the same files, read the same way. No migration step, no
re-setup.

The plugin no longer calls `tools.lazy_deps.install_specs` from anywhere. On package-manager Hermes
that is a retired shim which handed the process to the updater and exited instead of installing, so
every `hermes` invocation became a ~40s "update" that drained `hermes-gateway` and still left the
import missing (NousResearch/hermes-agent#126494, fixed upstream in af26acab73 to raise
`ImportError`). There is nothing left for it to install.

## Coming from the built-in provider

Hindsight used to ship inside Hermes as `plugins/memory/hindsight/`. Nous Research removed that copy
on 2026-09-23 and the provider now installs from the catalog instead. **You do not need to do
anything** — and your memories are not affected.

`hermes update`, and agent startup for Desktop users who never run it, calls Hermes'
`memory_provider_migration`: it sees `memory.provider: hindsight` configured, finds no provider on
disk, looks the name up in the plugin catalog and installs it at the reviewed commit pin. You'll see:

```
✓ Memory provider 'hindsight' moved out of core — installed its plugin from the catalog
  (your memory.hindsight settings and data are unchanged).
```

Your data never lived in the Hermes tree: memories are in your Hindsight bank — Hindsight Cloud, or
for `local_embedded` the profile directory `~/.hindsight/profiles/<profile>` and its embedded
PostgreSQL instance. Moving the provider code does not touch any of it, and your
`~/.hermes/hindsight/config.json` is read exactly as before.

If the migration can't run — offline, or the catalog fetch fails — Hermes prints the manual
one-liner rather than starting silently without memory:

```bash
hermes plugins install hindsight
```

## Updating

**Nothing updates the plugin on its own.** Whichever mode you choose below, the code only moves
when you run a command. In particular `hermes update` does *not* move it: it reinstalls the
plugin's Python dependencies and leaves the checkout where it is. If you used the built-in
provider, that is the one habit worth unlearning — memory improvements no longer arrive as a side
effect of updating Hermes.

Hermes re-fetches the published catalog at most once every 6 hours, so a freshly released version
can take that long to even appear as available.

### Follow the official pin (default)

What you get from `hermes plugins install hindsight`: the commit Nous reviewed and pinned in their
catalog.

```bash
hermes plugins update hindsight    # move to the catalog's current pin
```

`hermes plugins list` flags the plugin `update_available` once your installed commit differs from
the catalog's. Re-running `hermes plugins install hindsight` does **not** update it — it refuses
with "already exists"; `plugins update` is the command that re-pins.

### Pin a specific release

For a version you choose and freeze, install with an explicit commit:

```bash
hermes plugins install vectorize-io/hindsight/hindsight-integrations/hermes \
  --force --ref <40-character-commit-sha>
```

Replace the whole placeholder, angle brackets included: `<` and `>` are redirection operators in
most shells, so leaving them in makes the command fail to parse. And if you copy both lines, keep
the trailing `\` at the end of the first one or join them into a single line, since a `\` followed
by anything other than a newline is also a parse error.

`--ref` takes a full 40-character commit SHA and **rejects tag names**, so take the SHA from the
release notes of the [release](https://github.com/vectorize-io/hindsight/releases) you want rather
than typing `v1.1.0`. The copy button beside a commit on
[the plugin's history](https://github.com/vectorize-io/hindsight/commits/main/hindsight-integrations/hermes)
gives you the full SHA; the abbreviated one shown on screen is too short.

A `--ref` install is marked pinned, and `hermes plugins update hindsight` deliberately refuses to
move it — install again with a new `--ref` when you want a different version.

### Track the latest development code

For fixes before they reach the catalog. Install from the source path rather than the catalog name:

```bash
hermes plugins install vectorize-io/hindsight/hindsight-integrations/hermes
hermes plugins update hindsight    # now a git pull of our main branch
```

Unreviewed by definition — you get whatever is on `main` at the moment you run it.

The catalog entry lives in Nous' repo at
[`plugin-catalog/hindsight.yaml`](https://github.com/NousResearch/hermes-agent/blob/main/plugin-catalog/hindsight.yaml),
which is what records the current pin.

## Requirements

- **Cloud:** API key from [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io)
- **Local Embedded:** API key for a supported LLM provider (OpenAI, Anthropic, Gemini, Groq, OpenRouter, MiniMax, Ollama, or any OpenAI-compatible endpoint). Embeddings and reranking run locally — no additional API keys needed.
- **Local External:** A running Hindsight instance (Docker or self-hosted) reachable over HTTP.

## Setup

```bash
hermes memory setup    # select "hindsight"
```

The setup wizard installs nothing (dependencies come with the plugin), walks you through configuration, and offers to seed the bank with a **starter memory template** (a curated set of dispositions/instructions for common agent roles) — you can skip it, and it warns before overwriting an already-configured bank.

Or manually (cloud mode with defaults):
```bash
hermes config set memory.provider hindsight
echo "HINDSIGHT_API_KEY=your-key" >> ~/.hermes/.env
```

### Cloud

Connects to the Hindsight Cloud API. Requires an API key from [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io).

### Local Embedded

Hermes spins up a local Hindsight daemon with built-in PostgreSQL. Requires an LLM API key for memory extraction and synthesis. The daemon starts automatically in the background on first use and stops after 5 minutes of inactivity.

Supports any OpenAI-compatible LLM endpoint (llama.cpp, vLLM, LM Studio, etc.) — pick `openai_compatible` as the provider and enter the base URL.

Daemon startup logs: `~/.hermes/logs/hindsight-embed.log`
Daemon runtime logs: `~/.hindsight/profiles/<profile>.log`

To open the Hindsight web UI (local embedded mode only):
```bash
hindsight-embed -p hermes ui start
```

### Local External

Points the plugin at an existing Hindsight instance you're already running (Docker, self-hosted, etc.). No daemon management — just a URL and an optional API key.

## Config

Each Hermes profile has its own config file: `$HERMES_HOME/hindsight/config.json` (`~/.hermes/hindsight/config.json`
for the default profile). `hermes memory setup` writes it. The plugin reads the first of these that exists and is
not empty:

1. `$HERMES_HOME/hindsight/config.json` — this profile only
2. `~/.hindsight/config.json` — shared by every profile
3. Environment variables (see Environment Variables)

A profile has exactly one config. To give one profile different banks per repository, workspace, platform or user,
use `bank_id_template` or the multiple banks settings below; for fully separate setups, use
separate Hermes profiles.

Some keys also accept an older name: `apiKey` for `api_key`, `budget` for `recall_budget`, `prefetch_method` for
`recall_prefetch_method`, `llmApiKey` for `llm_api_key`.

### Connection

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `cloud` | `cloud`, `local_embedded`, or `local_external` |
| `api_url` | `https://api.hindsight.vectorize.io` (cloud), `http://localhost:8888` (local_external) | API URL. Falls back to `HINDSIGHT_API_URL` when unset. |
| `api_key` | — | Hindsight API key. Required for cloud, optional for local_external. Falls back to `HINDSIGHT_API_KEY` when unset; the setup wizard stores it there. |
| `timeout` | `120` | Seconds to wait for each Hindsight API request. Falls back to `HINDSIGHT_TIMEOUT` when unset. |

### Memory Bank

| Key | Default | Description |
|-----|---------|-------------|
| `bank_id` | `hermes` | Memory bank name (static fallback used when `bank_id_template` is unset or resolves empty) |
| `bank_id_template` | — | Optional template to derive the bank name dynamically. Placeholders: `{profile}`, `{workspace}`, `{project}`, `{platform}`, `{user}`, `{session}`. `{project}` is the name of the git repository Hermes runs in; every worktree of a repository resolves to the same name, and it is empty outside a repository or for one rooted at your home directory. Example: `hermes-{profile}` isolates memory per active Hermes profile, `{project}` gives each repository its own bank. Empty placeholders collapse cleanly (e.g. `hermes-{user}` with no user becomes `hermes`). |
| `mirror_to_own_bank` | `false` | Also write to `bank_id` when the template resolves to a different bank, so a profile keeps its own memory while working in project banks. |
| `additional_banks` | — | Extra banks to write to and recall from, in priority order. |
| `recall_additional_banks` | — | Extra banks to recall from but never write to, searched after the write banks. Alias `recallAdditionalBanks`, the name the Claude Code integration uses. A bank also listed in `additional_banks` stays writable. |
| `trusted_project_dirs` | — | Absolute folders whose git repositories may choose their own bank with a `.hindsight/config.toml`. Empty by default, so a cloned repository cannot redirect your memory. |
| `bank_mission` | — | Read but not applied to the bank. To set a bank's reflect mission, use the bank's config (control plane or `PATCH /v1/default/banks/{bank_id}/config`). |
| `bank_retain_mission` | — | Read but not applied to the bank. To set a bank's retain mission, use the bank's config, as above. |

#### Multiple banks

The primary bank is chosen in this order: a `.hindsight/config.toml` in a trusted repository, then `bank_id_template`, then `HINDSIGHT_BANK_ID` or `bank_id`. Writes go to the primary, then `bank_id` when `mirror_to_own_bank` is on, then each `additional_banks` entry. Recall searches the same banks and then each `recall_additional_banks` entry, all at once, each with the configured budget and `recall_max_tokens`; results are merged in that order and deduplicated by text. Reflect uses the primary bank only.

The three list settings take a JSON list or comma-separated text; use the JSON form for a value that contains a comma.

A failing extra bank is skipped for five minutes, with one warning when it goes down and one info line when it answers again; recall and writes cool down separately, and a bank misses the writes made while its writes are cooling down. An extra bank that has not answered after 80% of `timeout` counts as failing, so it never costs the primary its results. A failing primary bank fails the call exactly as it does with a single bank, and a write stops there, before any extra bank is touched.

A repository that is itself inside one of the `trusted_project_dirs` (a linked worktree is judged by where the worktree is) can name its bank in `.hindsight/config.toml` at its root or in any folder below it:

```toml
bank_id = "acme-billing"
```

The lookup stops at the repository root. Only `bank_id` is read.

### Recall

| Key | Default | Description |
|-----|---------|-------------|
| `recall_budget` | `mid` | Recall thoroughness: `low` / `mid` / `high` |
| `recall_prefetch_method` | `recall` | Auto-recall method: `recall` (raw facts) or `reflect` (LLM synthesis) |
| `recall_max_tokens` | `4096` | Maximum tokens for recall results |
| `recall_max_input_chars` | `800` | Maximum input query length for auto-recall |
| `recall_prompt_preamble` | — | Custom preamble for recalled memories in context |
| `recall_tags` | — | Tags to filter when searching memories: a JSON list (e.g. `["project-x", "team"]`) or comma-separated text (`"project-x, team"`) |
| `recall_tags_match` | `any` | Tag matching mode: `any` / `all` / `any_strict` / `all_strict` |
| `reflect_tags` | — | Tags that scope what reflect reasons over (the `hindsight_reflect` tool and `recall_prefetch_method: reflect`), as a JSON list or comma-separated text. Unset, reflect uses the whole bank; `recall_tags` does not apply to reflect. |
| `reflect_tags_match` | `any` | Tag matching mode for `reflect_tags` |
| `recall_types` | `observation` | Fact types surfaced by recall (both auto-recall and the `hindsight_recall` tool). Comma-separated string or JSON list. **Default narrowed to `observation` only** (see "Behavior change" below). Set to `observation,world,experience` to also include raw facts. |
| `recall_min_scores` | — | Minimum relevance per score field, as a JSON object (e.g. `{"reranker": 0.25}` or `{"semantic": 0.5}`). Applies to both auto-recall and the `hindsight_recall` tool. See the tip below. |
| `auto_recall` | `true` | Automatically recall memories before each turn |
| `recall_sync` | `false` | Recall synchronously against the *current* message each turn (higher relevance, adds recall latency). Default off: recall runs in the background and is injected on the next turn. |
| `recall_indicator` | `true` | Show a `👁️ Hindsight — recalled N memories` status line when auto-recall injects memory. Turn off for customer-facing agents. |

> **Behavior change — `recall_types` defaults to `observation` only.**
>
> Previously recall returned all three fact types. It now returns only observations.
>
> Per [Hindsight's docs](/developer/observations), observations are the **consolidated** knowledge layer Hindsight builds on top of raw facts: deduplicated beliefs grounded in evidence, refined as new facts arrive, with proof counts and freshness signals. Raw `world` / `experience` facts are the individual supporting evidence that feeds them. For per-turn context injection, observations are denser per token and avoid feeding the model multiple raw facts that one observation already summarizes.
>
> Restore the broad recall with `"recall_types": "observation,world,experience"` (string or JSON list) in `~/.hermes/hindsight/config.json`. This applies to **both** auto-recall and the `hindsight_recall` tool — both read the same `recall_types` setting (the tool schema has no per-call `types` argument), so narrowing the default narrows both paths.

> **Tip — `recall_min_scores` against off-topic recall.** Recall ranks, it does not judge relevance: a question
> unrelated to anything stored still comes back with up to `recall_max_tokens` of the least-bad memories. The
> `reranker` score separates the two cleanly — on a small bank relevant hits scored 0.7-0.97 and unrelated ones
> 0.0-0.1 — so `"recall_min_scores": {"reranker": 0.25}` in `~/.hermes/hindsight/config.json` turns that noise
> into an empty recall. The server only guarantees `reranker` and `final` floors; a `semantic` or `keyword` floor
> prunes just its own retrieval arm there, and a result found by another arm still comes back with a `null` score
> for that stage. The plugin therefore also checks every floor against the scores each result reports, and rejects
> a result that does not report the floored stage. Check the scores on your own bank before tuning a floor: with a
> multilingual embedding model, `{"semantic": 0.5}` separated relevant from unrelated queries better than the
> reranker did. Unset by default, so nothing changes unless you opt in.

### Retain

| Key | Default | Description |
|-----|---------|-------------|
| `auto_retain` | `true` | Automatically retain conversation turns. Sub-agent (`delegate_task`) runs and kanban workers are never auto-retained; their parent session keeps its own turns. |
| `retain_cron` | `false` | Also auto-retain scheduled (cron) runs. Off by default, so job output stays out of the user's memory; the `hindsight_retain` tool still works in cron runs. |
| `retain_async` | `true` | Process retain asynchronously on the Hindsight server |
| `retain_every_n_turns` | `1` | Retain every N turns (1 = every turn) |
| `retain_context` | `conversation between Hermes Agent and the User` | Context label for retained memories |
| `retain_tags` | — | Default tags applied to retained memories; merged with per-call tool tags. JSON list or comma-separated text. |
| `observation_scopes` | `shared` | How observations are grouped by tag during consolidation: `shared` (one scope for the whole bank, so observations merge across sessions; the `session:`/`parent:` tags stay on the facts for filtering), `combined` (one scope per exact tag set, which here means one per session), `per_tag` (one pass per tag), `all_combinations` (every tag subset, expensive), or a JSON list of tag lists for custom scopes (e.g. `[["user:alice"], ["project-x"]]`). |
| `retain_source` | — | Opt-in `metadata.source` attached to retained memories (identifies the storing client, e.g. `hermes`). Empty by default — no attribution tag ships unless you set it. |
| `retain_strategy` | — | Named retain strategy sent with every stored item (`HINDSIGHT_RETAIN_STRATEGY`). The bank must define it under `retain_strategies`; an unknown name is ignored by the server. Empty lets the bank decide. |
| `retain_indicator` | `true` | Show a `👁️ Hindsight — saving to memory…` status line when a turn is saved. Turn off for customer-facing agents. |
| `retain_user_prefix` | `User` | Label used before user turns in auto-retained transcripts |
| `retain_assistant_prefix` | `Assistant` | Label used before assistant turns in auto-retained transcripts |
| `prefetch_waits_for_retain` | `true` | Before the background recall for the next turn, wait until the turn just saved can be found, so recall includes it. Runs off the reply path and adds no response time. |
| `prefetch_retain_drain_timeout` | `10` | Most seconds that background recall waits for the save above before recalling anyway |

**When a save fails.** Automatic saves are not dropped on the first error:

- If the server can't be reached, times out or answers with a 5xx, the save is kept and sent again before the next one (and once more at exit), up to 5 tries. Each save carries its own operation id, so a resend the server already received is not stored twice.
- If the server accepts a save and its processing fails later, the plugin asks the server to retry it, up to 3 times. It checks before each new save, so a failure after the process exits is not retried; retry it from the control plane.
- If the bank keeps no document text (`store_document_text` off), it can't append to a session document. The plugin then stops appending and stores the turns in a separate document per process instead.
- An error that can never succeed (any other 4xx) is logged and the save is dropped.

### Integration

| Key | Default | Description |
|-----|---------|-------------|
| `memory_mode` | `hybrid` | How memories are integrated into the agent |

**memory_mode:**
- `hybrid` — automatic context injection + tools available to the LLM
- `context` — automatic injection only, no tools exposed
- `tools` — tools only, no automatic injection

### Local Embedded LLM

| Key | Default | Description |
|-----|---------|-------------|
| `llm_provider` | — | `openai`, `anthropic`, `gemini`, `groq`, `openrouter`, `minimax`, `ollama`, `lmstudio`, `openai_compatible`. The setup wizard suggests `openai`. |
| `llm_model` | — | Model name (e.g. `gpt-4o-mini`, `qwen/qwen3.5-9b`). The setup wizard suggests one per provider. |
| `llm_base_url` | — | Endpoint URL for `openai_compatible` (e.g. `http://192.168.1.10:8080/v1`). Falls back to `HINDSIGHT_API_LLM_BASE_URL`. |
| `llm_api_key` | — | LLM API key. Usually left out of the file: the setup wizard stores it in `$HERMES_HOME/.env` as `HINDSIGHT_LLM_API_KEY`. |
| `profile` | `hermes` | Name of the local daemon profile: its data, log and env file under `~/.hindsight/profiles/` are named after it. Give two Hermes profiles different values to keep their local memory apart. |
| `idle_timeout` | `300` | Seconds of inactivity before the daemon stops; `0` keeps it running. Falls back to `HINDSIGHT_IDLE_TIMEOUT`. |
| `port_health_grace_timeout` | `30` | Seconds to wait for a slow daemon health check before restarting it. Raise on busy or low-resource machines. `HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT` wins when set. |

The embedded daemon is a subprocess that cannot see the per-turn secret
scope, so it reads the key from `~/.hindsight/profiles/<profile>.env`
(materialized owner-only at setup and on config change). Key resolution
order is explicit config → secret scope → the on-disk profile env, and the
rewrite path is fail-closed: a build with no key never clobbers a profile
file that already holds one.

The plugin owns only the LLM, log-level and idle-timeout keys in that file. Anything else in it
(the port hindsight-embed records, a tenant extension and its `HINDSIGHT_API_TENANT_API_KEY`) is
left alone and never counts as a config change, and the plugin's client sends that tenant key to
the daemon.

## Tools

Available in `hybrid` and `tools` memory modes:

| Tool | Description |
|------|-------------|
| `hindsight_retain` | Store information with auto entity extraction; supports optional per-call `tags` |
| `hindsight_recall` | Multi-strategy search (semantic + entity graph) |
| `hindsight_reflect` | Cross-memory synthesis (LLM-powered) |

## Environment Variables

Set these in the profile's `$HERMES_HOME/.env`. A value in `config.json` wins over its variable, except where noted.

These are used whenever the matching config key is not set:

| Variable | Description |
|----------|-------------|
| `HINDSIGHT_API_KEY` | Hindsight API key (`api_key`) |
| `HINDSIGHT_API_URL` | API endpoint (`api_url`) |
| `HINDSIGHT_BANK_ID` | Bank name; when set, wins over `bank_id` in `config.json` (a `bank_id_template` still takes precedence) |
| `HINDSIGHT_TIMEOUT` | API request timeout in seconds (`timeout`) |
| `HINDSIGHT_LLM_API_KEY` | LLM API key for local_embedded (`llm_api_key`); `HINDSIGHT_API_LLM_API_KEY` also works |
| `HINDSIGHT_API_LLM_BASE_URL` | LLM endpoint for local_embedded (`llm_base_url`) |
| `HINDSIGHT_IDLE_TIMEOUT` | Daemon idle timeout in seconds (`idle_timeout`) |
| `HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT` | Daemon health-check grace in seconds; wins over `port_health_grace_timeout` |
| `HINDSIGHT_RETAIN_TAGS` | Default tags on retained memories (`retain_tags`) |
| `HINDSIGHT_RETAIN_OBSERVATION_SCOPES` | Observation scopes (`observation_scopes`) |
| `HINDSIGHT_RETAIN_SOURCE` | `metadata.source` on retained memories (`retain_source`) |
| `HINDSIGHT_RETAIN_STRATEGY` | Named retain strategy (`retain_strategy`) |
| `HINDSIGHT_RETAIN_USER_PREFIX` | Label before user turns (`retain_user_prefix`) |
| `HINDSIGHT_RETAIN_ASSISTANT_PREFIX` | Label before assistant turns (`retain_assistant_prefix`) |

These are used **only when there is no config file at all**. Once `hermes memory setup` has written
`config.json`, they are ignored; set the config key instead:

| Variable | Description |
|----------|-------------|
| `HINDSIGHT_MODE` | Mode: `cloud`, `local_embedded` or `local_external` (`mode`) |
| `HINDSIGHT_BUDGET` | Recall budget (`recall_budget`) |
| `HINDSIGHT_RETAIN_CONTEXT` | Label stored with each retained conversation (`retain_context`) |
| `HINDSIGHT_RETAIN_INDICATOR` | `false` hides the "saving to memory" status line (`retain_indicator`) |
| `HINDSIGHT_RECALL_INDICATOR` | `false` hides the "recalled N memories" status line (`recall_indicator`) |
| `HINDSIGHT_RECALL_SYNC` | `true` recalls against the current message before answering (`recall_sync`) |

## Client Version

Requires `hindsight-client >= 0.10.1` and, for `local_embedded`, `hindsight-embed >= 0.10.1`. A version below
the floor only logs a warning; `hermes plugins update hindsight` moves the environment.

The floor is 0.10.1 rather than the 0.6.1 this plugin needs at the API level because
`hindsight-embed` 0.10.0 breaks `local_embedded` outright: its daemon probe cleared the calling
thread's event loop, so the next client call failed with `Timeout context manager should be used
inside a task`. If you installed between 2026-09-14 and 2026-09-21, run `hermes update` (or
`hermes plugins update hindsight`) to move off it.

## Hermes Gateway (Telegram, Discord, Slack)

The provider works across every gateway platform. Hermes builds a fresh agent per message, and the
provider is re-initialized with it, so auto-recall runs for each turn regardless of platform.

Two settings are worth turning off for customer-facing bots: `recall_indicator` and
`retain_indicator`, which otherwise print a `👁️ Hindsight` status line into the user's channel.

## Disabling Hermes' built-in memory

Hermes has its own memory store backed by local markdown (`MEMORY.md`, plus a slimmer `USER.md`
profile). With both active the model may prefer the built-in one, so turn the flat-file stores off:

```bash
hermes config set memory.memory_enabled false
hermes config set memory.user_profile_enabled false   # optional: the USER.md profile
```

Setting both to `false` removes the built-in `memory` tool from the agent entirely. Re-enable later
by setting the same flags back to `true`.

If you keep the built-in stores on, every entry the agent adds or replaces in them is also saved to
every bank Hindsight writes to, tagged `builtin-memory`, `builtin-target:<memory|user>` and `builtin-action:<add|replace>`,
so a fact pruned from the size-capped file is not lost. Removals are not mirrored.

## Troubleshooting

**Tools don't appear in `/tools`** — the provider skips tool registration when it isn't configured.
Check `hermes memory status` reports `hindsight` as the active provider and `Status: available`. In
`memory_mode: context` the tools are hidden on purpose.

**`Status: not available` in `local_embedded`** — this plugin's own packages
(`hindsight-client`, `hindsight-embed`) are missing from the environment, which happens when a venv
rebuild dropped the plugin member. Run `hermes pm repair` and restart Hermes to rebuild them. The Hindsight server is *not* needed in that venv — it runs as a separate
process, and first use downloads it if no `hindsight-api` binary is present yet.

**`Timeout context manager should be used inside a task`** — `hindsight-embed` 0.10.0. Run
`hermes plugins update hindsight` to move to the 0.10.1 floor.

**Local daemon not starting** — check the logs:

```bash
cat ~/.hermes/logs/hindsight-embed.log     # startup
cat ~/.hindsight/profiles/<profile>.log    # daemon runtime
```

**Recall returns nothing** — memories need at least one retain cycle, and extraction is an LLM call.
Store a fact, then ask about it on a later turn.
