
{/* GENERATED from hindsight-integrations/hermes/README.md — edit that file, then run
    node hindsight-docs/scripts/sync-hermes-doc.mjs */}

> **💡 Tip**
>
Using the **Hermes desktop app**? You can select and configure Hindsight entirely in Settings — no
terminal required. See [Hermes Desktop](hermes-desktop.md).
> **⚠️ Deprecated: the standalone `hindsight-hermes` plugin**
>
The old **`hindsight-hermes`** pip plugin (installed into the Hermes virtual environment and
registered through the `hermes_agent.plugins` entry point) is **deprecated** — on current Hermes
builds its tools fail with `{"error": "Timeout context manager should be used inside a task"}`.
Follow Migrate hindsight-hermes to Native Hermes Memory
to switch over while keeping the same memory bank.
Long-term memory with knowledge graph, entity resolution, and multi-strategy retrieval. Supports cloud, local embedded, and local external modes.

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) memory-provider plugin, installed from
the Hermes plugin catalog. It used to ship inside Hermes as `plugins/memory/hindsight/`; Nous Research
moved every memory provider out of the core tree, so it is now maintained by the Hindsight team in
[vectorize-io/hindsight](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/hermes)
and the catalog pins it from there.

[View Changelog →](../../changelog/integrations/hermes.md)

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
`memory.provider: <name>` in `config.yaml`, which `hermes memory setup` writes. Setup also installs
the mode-dependent extras (`local_embedded` needs `hindsight-all`, not just the client), so
`plugins install` on its own leaves the provider reporting "not available" in embedded mode.

`local_embedded` mode needs `hindsight-all`, which `pyproject.toml` deliberately does not declare
(it would push the local-ML stack onto cloud-mode users). The setup wizard installs it, and
`embedded.py::_ensure_local_runtime` self-installs it on the availability check as a backstop.

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

The setup wizard installs dependencies automatically via `uv`, walks you through configuration, and offers to seed the bank with a **starter memory template** (a curated set of dispositions/instructions for common agent roles) — you can skip it, and it warns before overwriting an already-configured bank.

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

Config file: `~/.hermes/hindsight/config.json`

### Connection

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `cloud` | `cloud`, `local_embedded`, or `local_external` |
| `api_url` | `https://api.hindsight.vectorize.io` | API URL (cloud and local_external modes) |

### Memory Bank

| Key | Default | Description |
|-----|---------|-------------|
| `bank_id` | `hermes` | Memory bank name (static fallback used when `bank_id_template` is unset or resolves empty) |
| `bank_id_template` | — | Optional template to derive the bank name dynamically. Placeholders: `{profile}`, `{workspace}`, `{platform}`, `{user}`, `{session}`. Example: `hermes-{profile}` isolates memory per active Hermes profile. Empty placeholders collapse cleanly (e.g. `hermes-{user}` with no user becomes `hermes`). |
| `bank_mission` | — | Reflect mission (identity/framing for reflect reasoning). Applied via Banks API. |
| `bank_retain_mission` | — | Retain mission (steers what gets extracted). Applied via Banks API. |

### Recall

| Key | Default | Description |
|-----|---------|-------------|
| `recall_budget` | `mid` | Recall thoroughness: `low` / `mid` / `high` |
| `recall_prefetch_method` | `recall` | Auto-recall method: `recall` (raw facts) or `reflect` (LLM synthesis) |
| `recall_max_tokens` | `4096` | Maximum tokens for recall results |
| `recall_max_input_chars` | `800` | Maximum input query length for auto-recall |
| `recall_prompt_preamble` | — | Custom preamble for recalled memories in context |
| `recall_tags` | — | Tags to filter when searching memories |
| `recall_tags_match` | `any` | Tag matching mode: `any` / `all` / `any_strict` / `all_strict` |
| `recall_types` | `observation` | Fact types surfaced by recall (both auto-recall and the `hindsight_recall` tool). Comma-separated string or JSON list. **Default narrowed to `observation` only** (see "Behavior change" below). Set to `observation,world,experience` to also include raw facts. |
| `auto_recall` | `true` | Automatically recall memories before each turn |
| `recall_sync` | `false` | Recall synchronously against the *current* message each turn (higher relevance, adds recall latency). Default off: recall runs in the background and is injected on the next turn. |
| `recall_indicator` | `true` | Show a `👁️ Hindsight — recalled N memories` status line when auto-recall injects memory. Turn off for customer-facing agents. |

> **Behavior change — `recall_types` defaults to `observation` only.**
>
> Previously recall returned all three fact types. It now returns only observations.
>
> Per [Hindsight's docs](../../developer/observations.md), observations are the **consolidated** knowledge layer Hindsight builds on top of raw facts: deduplicated beliefs grounded in evidence, refined as new facts arrive, with proof counts and freshness signals. Raw `world` / `experience` facts are the individual supporting evidence that feeds them. For per-turn context injection, observations are denser per token and avoid feeding the model multiple raw facts that one observation already summarizes.
>
> Restore the broad recall with `"recall_types": "observation,world,experience"` (string or JSON list) in `~/.hermes/hindsight/config.json`. This applies to **both** auto-recall and the `hindsight_recall` tool — both read the same `recall_types` setting (the tool schema has no per-call `types` argument), so narrowing the default narrows both paths.

### Retain

| Key | Default | Description |
|-----|---------|-------------|
| `auto_retain` | `true` | Automatically retain conversation turns |
| `retain_async` | `true` | Process retain asynchronously on the Hindsight server |
| `retain_every_n_turns` | `1` | Retain every N turns (1 = every turn) |
| `retain_context` | `conversation between Hermes Agent and the User` | Context label for retained memories |
| `retain_tags` | — | Default tags applied to retained memories; merged with per-call tool tags |
| `retain_source` | — | Opt-in `metadata.source` attached to retained memories (identifies the storing client, e.g. `hermes`). Empty by default — no attribution tag ships unless you set it. |
| `retain_indicator` | `true` | Show a `👁️ Hindsight — saving to memory…` status line when a turn is saved. Turn off for customer-facing agents. |
| `retain_user_prefix` | `User` | Label used before user turns in auto-retained transcripts |
| `retain_assistant_prefix` | `Assistant` | Label used before assistant turns in auto-retained transcripts |

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
| `llm_provider` | `openai` | `openai`, `anthropic`, `gemini`, `groq`, `openrouter`, `minimax`, `ollama`, `lmstudio`, `openai_compatible` |
| `llm_model` | per-provider | Model name (e.g. `gpt-4o-mini`, `qwen/qwen3.5-9b`) |
| `llm_base_url` | — | Endpoint URL for `openai_compatible` (e.g. `http://192.168.1.10:8080/v1`) |

The LLM API key is stored in `~/.hermes/.env` as `HINDSIGHT_LLM_API_KEY`.

The embedded daemon is a subprocess that cannot see the per-turn secret
scope, so it reads the key from `~/.hindsight/profiles/<profile>.env`
(materialized owner-only at setup and on config change). Key resolution
order is explicit config → secret scope → the on-disk profile env, and the
rewrite path is fail-closed: a build with no key never clobbers a profile
file that already holds one.

## Tools

Available in `hybrid` and `tools` memory modes:

| Tool | Description |
|------|-------------|
| `hindsight_retain` | Store information with auto entity extraction; supports optional per-call `tags` |
| `hindsight_recall` | Multi-strategy search (semantic + entity graph) |
| `hindsight_reflect` | Cross-memory synthesis (LLM-powered) |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HINDSIGHT_API_KEY` | API key for Hindsight Cloud |
| `HINDSIGHT_LLM_API_KEY` | LLM API key for local mode |
| `HINDSIGHT_API_LLM_BASE_URL` | LLM Base URL for local mode (e.g. OpenRouter) |
| `HINDSIGHT_API_URL` | Override API endpoint |
| `HINDSIGHT_BANK_ID` | Override bank name |
| `HINDSIGHT_BUDGET` | Override recall budget |
| `HINDSIGHT_MODE` | Override mode (`cloud`, `local_embedded`, `local_external`) |

## Client Version

Requires `hindsight-client >= 0.10.1` and, for `local_embedded`, `hindsight-embed >= 0.10.1`. The plugin
auto-upgrades the client on session start if an older version is detected.

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

## Troubleshooting

**Tools don't appear in `/tools`** — the provider skips tool registration when it isn't configured.
Check `hermes memory status` reports `hindsight` as the active provider and `Status: available`. In
`memory_mode: context` the tools are hidden on purpose.

**`Status: not available` in `local_embedded`** — the embedded runtime (`hindsight-all`) isn't
installed. The plugin self-installs it on the availability check; if that is blocked
(`security.allow_lazy_installs: false`, or a sealed venv) install it yourself:
`uv pip install --python "$(hermes doctor --python-path)" hindsight-all`, or re-run
`hermes memory setup`.

**`Timeout context manager should be used inside a task`** — `hindsight-embed` 0.10.0. Run
`hermes plugins update hindsight` to move to the 0.10.1 floor.

**Local daemon not starting** — check the logs:

```bash
cat ~/.hermes/logs/hindsight-embed.log     # startup
cat ~/.hindsight/profiles/<profile>.log    # daemon runtime
```

**Recall returns nothing** — memories need at least one retain cycle, and extraction is an LLM call.
Store a fact, then ask about it on a later turn.
