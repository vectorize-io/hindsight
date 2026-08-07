
{/* GENERATED from hindsight-integrations/coding-agents/README.md — edit that file, then run
    node hindsight-docs/scripts/sync-coding-agents-doc.mjs */}

Long-term project memory for **coding agents**, backed by [Hindsight](https://vectorize.io/hindsight).
One package, several agents: a shared reflect-and-inject core with a thin entry point per agent
(**opencode**, **Kilo CLI**, **Cline CLI**, **Claude Code**, **Codex CLI**, **Antigravity CLI**, **Cursor CLI**, **GitHub Copilot CLI**, **Grok Build**). Ingestion is fully
automatic — there is no setup command: a repo's git history and conversations flow into its memory
bank in the background as you work.

The premise: most of a real fix is derivable from the code, but the _last mile_ often hinges on a
project-specific decision that isn't in the code at all — a rounding rule, a retry allowlist, a
tie-break policy. Those decisions live in git history and past conversations. This package puts them
in front of the agent at the moment it starts working, and keeps a curated set of **knowledge pages**
(architecture, conventions, in-flight initiatives) that future sessions start from.

## Install

```bash
npx @vectorize-io/hindsight-coding-agents install all          # every detected agent, wired natively
npx @vectorize-io/hindsight-coding-agents install claude-code  # or just one
npx @vectorize-io/hindsight-coding-agents uninstall all        # removes exactly what install added
```

`install` takes an explicit target — `all`, or one or more harness names. A bare
`npx @vectorize-io/hindsight-coding-agents install` changes nothing and prints the choice, so wiring every agent on
the machine is never something that happens by accident.

On a terminal it also asks **where memory should live** — Hindsight Cloud, a server you run, or a
local daemon on this machine (see Where memory lives). Scripted installs pass
`--server cloud|self-hosted|daemon` instead; it is asked only once, and never again on re-install.

### Per agent

Same command, only the harness name changes. Run after installing the package globally.

####  Claude Code

```bash
npx @vectorize-io/hindsight-coding-agents install claude-code
```

3 hooks in `~/.claude/settings.json`, MCP via `claude mcp add` (user scope), and the companion skill.

####  Codex CLI

```bash
npx @vectorize-io/hindsight-coding-agents install codex
```

3 hooks in `~/.codex/hooks.json` plus `[mcp_servers]` in `config.toml` (needs `codex_hooks = true`).

####  opencode

```bash
npx @vectorize-io/hindsight-coding-agents install opencode
```

A plugin entry in `~/.config/opencode/opencode.json` — native tools, no MCP needed.

####  Kilo CLI

```bash
npx @vectorize-io/hindsight-coding-agents install kilo
```

A plugin entry in `~/.config/kilo/kilo.json[c]`.

####  Cursor CLI

```bash
npx @vectorize-io/hindsight-coding-agents install cursor-cli
```

Hooks in `~/.cursor/hooks.json`, `~/.cursor/mcp.json`, and the companion skill.

####  GitHub Copilot CLI

```bash
npx @vectorize-io/hindsight-coding-agents install copilot-cli
```

`~/.copilot/hooks/`, `mcp-config.json`, and the companion skill.

####  Grok Build

```bash
npx @vectorize-io/hindsight-coding-agents install grok-build
```

Native hooks and MCP in `~/.grok/config.toml`, plus the companion skill.

####  Antigravity CLI

```bash
npx @vectorize-io/hindsight-coding-agents install agy
```

Lifecycle hooks, MCP, and the `Hindsight · <bank>` status line.

####  Devin CLI

```bash
npx @vectorize-io/hindsight-coding-agents install devin-cli
```

Hooks in `~/.config/devin/config.json` plus MCP. Needs Node 22.5+ — see below.

####  Cline CLI

```bash
npx @vectorize-io/hindsight-coding-agents install cline-cli
```

A native plugin via `cline plugin install`, plus MCP and the companion skill.

Uninstall the same way: `npx @vectorize-io/hindsight-coding-agents uninstall claude-code` (or `uninstall all`).

**Devin CLI needs Node 22.5 or newer.** Its hooks pass only a session id — the conversation itself
lives in `~/.local/share/devin/cli/sessions.db` — so reading it depends on Node's built-in
`node:sqlite`. Installing `devin-cli` checks for this first and refuses (with the reason) rather
than wiring hooks that could never retain anything. Every other agent works on any supported Node.

`install` copies what it needs into `~/.hindsight/coding-agents` and points each agent's wiring
there, so nothing depends on where you ran it from. **Updating** is the same command again — it
re-copies the runtime in place, leaving the wiring valid and every new session on the new version.

`install` merges the native wiring (hooks + MCP registration where the host wants them) into each
agent's own config, preserving everything already there; it is idempotent (re-run after moving the
package) and backs up any pre-existing file it touches as `<file>.hindsight-backup`. `uninstall`
removes only our entries. On Claude Code the install also ships a **companion skill**
(`~/.claude/skills/hindsight-coding-agent`) that teaches the agent how this memory works — what
"store this in hindsight" should do, the tool surface, per-repo configuration, debugging — so users
can ask the agent itself. Manual wiring per harness, if you prefer:

**opencode** installs directly — point `opencode.json` at the package dir:

```json
{ "plugin": ["/path/to/hindsight-coding-agents"] }
```

**Claude Code** and **Codex** get their full three-hook + MCP wiring from this package's own
installer — `npx @vectorize-io/hindsight-coding-agents install claude-code` / `install codex`. This package's `bin`
entries (`hindsight-claude-hook`, `hindsight-codex-hook`,
`hindsight-cursor-hook`) are the individual injection-only `UserPromptSubmit` entrypoints for a
minimal, hand-wired setup.

Adding an agent: hook-based → write a `HookSpec` entry point (see `src/cursor-hook.ts`) and register
a `hookAdapter` in `src/harness/registry.ts`; persistent-plugin → implement `HarnessAdapter`
(`src/core/types.ts`) fully (see `src/harness/opencode.ts`).

## Migrating from the per-agent plugins

The older per-agent integrations (`hindsight-claude-code`, `hindsight-cursor-cli`, `hindsight-codex`, …) are superseded by this package. Two things move; nothing else does.

**Your server moves automatically.** If `~/.hindsight/claude-code.json` or `~/.hindsight/codex.json`
exists, `install` adopts its endpoint — `hindsightApiUrl` → `apiUrl`, `hindsightApiToken` →
`apiToken`, and an empty URL means the local daemon, as it did there. The agent you are installing
is checked first, so wiring Codex takes Codex's server even if an old `claude-code.json` is still
lying around. You already chose where your memory lives; defaulting to Cloud instead would quietly
send your prompts somewhere else. Pass `--server` to override. (Those two are the only old plugins
that shipped a user config — Cursor CLI, Copilot CLI, opencode and Cline have no endpoint to carry.)

**Your conversations are re-imported from local disk**, as new documents:

```bash
cd /path/to/your/repo
npx @vectorize-io/hindsight-coding-agents install claude-code --import-conversations   # or: install codex --import-conversations
```

This re-extracts the transcripts the agent already wrote, so it costs tokens roughly in proportion
to the history imported, and it is safe to re-run (ingestion dedups by document id).

Local transcripts are the source rather than the old bank, because the old bank cannot be split by
repo. Its default was a **single static bank** — `dynamicBankId` defaults to false, so everything
landed in one bank named `claude_code` — and its documents record only `retained_at`,
`message_count` and `session_id`, nothing identifying the project. Working out which documents
belong to which repo means joining `session_id` back to the `cwd` in the local transcript, so the
transcripts are needed either way; going through them directly is simply the shorter path.

**How sessions are matched.** A conversation is imported only when the session itself records the
directory it ran in — never inferred from a file or folder name. Claude Code writes that directory
on its entries and Codex in its `session_meta` header, so both can be attributed exactly, including
sessions started in a subdirectory of the repo. Guessing was tempting (Claude names its history
folders after the project path) but unsafe: `/` and `.` both encode to `-`, so `repo-sub` is either
the subdirectory `repo/sub` or an unrelated sibling repo — and a wrong guess files someone else's
conversation into your bank. Sessions that record nothing are skipped and the count is reported.
The other harnesses (opencode, Kilo, Cursor, Cline, Copilot, Devin) keep history in internal SQLite
databases with unversioned schemas and are skipped with a reason.

**Nothing else is translated.** The old plugin's behavioural settings — 12 `recall*`, 7 `retain*`,
`bankMission`/`retainMission`, `dynamicBankGranularity` — describe a pipeline this package replaced,
and reinterpreting them would be guesswork. Bank naming changes too: this package uses one bank per
**repo** (`coding-agent::{gitProject}`) shared by every agent. To keep the old naming instead:

```jsonc
{ "bankIdTemplate": "{harness}::{gitProject}" } // reproduces the old per-agent naming
```

## Where memory lives

Three modes, chosen once when you install (`install` asks on a terminal; pass `--server` to script it):

| mode          | what runs                                 | needs                                    |
| ------------- | ----------------------------------------- | ---------------------------------------- |
| `cloud`       | Hindsight Cloud (default)                 | an API token                             |
| `self-hosted` | a Hindsight server you already run        | its URL                                  |
| `daemon`      | a local `hindsight-embed` on this machine | `uv` on PATH + an LLM key for extraction |

```bash
npx @vectorize-io/hindsight-coding-agents install claude-code --server daemon
npx @vectorize-io/hindsight-coding-agents install claude-code --server self-hosted --api-url http://localhost:8888
npx @vectorize-io/hindsight-coding-agents install claude-code --server cloud --api-token <token>
```

Re-running `install` never re-asks: a config that already names a server is left alone.

### Local daemon mode

Nothing to sign up for and nothing to host — memory runs on your machine. The plugin starts
`hindsight-embed` on demand at `127.0.0.1:9077` and points every agent at it.

- **A server already on the port is adopted, never restarted** — so one daemon serves every agent
  and every repo, and your own `hindsight-embed` is reused if you already run one.
- **Cold starts happen in the background.** The first start downloads the daemon and loads models,
  which takes longer than any hook is allowed to run, so it is launched detached at session start.
  A session that begins before it is ready simply has no memory for a turn or two — a daemon that
  isn't up is treated as an unreachable server, exactly like a Cloud or self-hosted outage, with the
  same error handling and the same diagnostics. Nothing downstream of the URL knows which mode it is.
- **It shuts down on idle**, after `daemonIdleTimeout` seconds. There is deliberately no
  stop-on-exit: one daemon is shared, so ending one session must not cut memory out from under
  another agent still working.
- **macOS additionally needs a current Rust toolchain.** `litellm` (a transitive dependency of the
  API) publishes wheels only for Linux and Windows, so a Mac compiles it from source through
  maturin and its crates pin a recent `rustc`. Install from [rustup.rs](https://rustup.rs) and keep
  it updated — an out-of-date toolchain fails as surely as a missing one. Linux and Windows install
  from wheels and need none of this.
- **Fact extraction runs locally**, so it needs an LLM. `HINDSIGHT_API_LLM_PROVIDER` wins if set;
  otherwise the first of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`;
  otherwise the Claude Code CLI, which needs no key. `install` tells you which it found.

Daemon settings keep the names the old per-agent Claude Code plugin used, so an existing
environment carries over unchanged:

| field               | env                             | default        | meaning                                    |
| ------------------- | ------------------------------- | -------------- | ------------------------------------------ |
| `serverMode`        | `HINDSIGHT_SERVER_MODE`         | `cloud`        | `cloud` \| `self-hosted` \| `daemon`       |
| `apiPort`           | `HINDSIGHT_API_PORT`            | `9077`         | port the local daemon listens on           |
| `daemonIdleTimeout` | `HINDSIGHT_DAEMON_IDLE_TIMEOUT` | `300`          | seconds of inactivity before it exits      |
| `daemonProfile`     | `HINDSIGHT_DAEMON_PROFILE`      | `coding-agent` | which local database it uses               |
| `embedVersion`      | `HINDSIGHT_EMBED_VERSION`       | `latest`       | which `hindsight-embed` release to run     |
| `embedPackagePath`  | `HINDSIGHT_EMBED_PACKAGE_PATH`  | —              | run a local checkout instead (development) |

Any `HINDSIGHT_API_*` variable you export is forwarded to the daemon, so server-side settings need
no equivalent here.

## Configuration

Configuration is **one JSON file**: `~/.hindsight/coding-agent.json`. Layering, later wins per field:

1. built-in defaults
2. environment variables — `HINDSIGHT_API_URL`, `HINDSIGHT_API_TOKEN`, and one per scalar setting
   (`HINDSIGHT_<FIELD_IN_CAPS>`), for containers and CI that inject config rather than write a file
3. the file's top level
4. its `harnesses.<name>` section — per-agent override

Environment variables are a **fallback**: the file wins wherever it sets a value, so adding env to
an existing setup changes nothing. The map-valued settings (`mapPathToBank`, `harnesses`, `banks`)
are file-only — nested branching doesn't survive flattening into one variable.

There is deliberately no repo-carried config file — per-repo bank routing is `mapPathToBank`,
per-agent differences are `harnesses.<name>`.

Each entry point knows which harness it _is_ (the opencode plugin is loaded by opencode, the codex
hook by Codex...), so one shared config serves several agents side by side:

```jsonc
{
  "apiUrl": "https://api.hindsight.vectorize.io",
  "harnesses": {
    "opencode": { "reflectTimeoutMs": 60000 },
    "claude-code": { "disabled": true }, // e.g. memory off for Claude only
  },
}
```

### Reference

| field                   | default                              | meaning                                                                                                                                                                                                                             |
| ----------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `apiUrl`                | `https://api.hindsight.vectorize.io` | Hindsight API base URL (set to `http://localhost:8888` for a local server)                                                                                                                                                          |
| `apiToken`              | —                                    | bearer token (Hindsight Cloud)                                                                                                                                                                                                      |
| `bankId`                | —                                    | **explicit static bank**; unset ⇒ per-repo dynamic resolution (below)                                                                                                                                                               |
| `dynamicBankId`         | dynamic iff no `bankId`              | force dynamic (`true`) or static (`false`) resolution                                                                                                                                                                               |
| `bankIdTemplate`        | `"coding-agent::{gitProject}"`       | dynamic bank id format; the default makes every agent share one bank per repo                                                                                                                                                       |
| `mapPathToBank`         | —                                    | absolute path → bank; **longest prefix wins**; overrides everything                                                                                                                                                                 |
| `resolveWorktrees`      | `true`                               | `{gitProject}`: linked worktrees share the main repo's bank                                                                                                                                                                         |
| `disabled`              | `false`                              | hard off-switch (inert plugin/hook — a no-memory baseline)                                                                                                                                                                          |
| `reflectTimeoutMs`      | `120000`                             | session-reflect timeout (hook harnesses additionally cap it at 25s to fit the host's hook window); on timeout the session runs without reflect (recorded)                                                                           |
| `pageRefreshEveryTurns` | `10`                                 | refetch the knowledge pages and re-inject the page roster + tool guide every N user turns                                                                                                                                           |
| `autoSeed`              | `true`                               | SessionStart: auto-seed a cold repo's bank from git history                                                                                                                                                                         |
| `seedLimit`             | `300`                                | auto-seed: most-recent-N-commits cap                                                                                                                                                                                                |
| `codebaseSurvey`        | `true`                               | SessionStart: headless survey of a cold repo's structure, run under the current harness's own CLI (claude/codex/antigravity/opencode), falling back to any available agent                                                          |
| `surveyModel`           | `haiku`                              | model for the survey — Claude recipe only (`claude -p --model`); other agents use their configured default                                                                                                                          |
| `surveyBudgetUsd`       | `2`                                  | survey spend cap — Claude recipe only (`claude -p --max-budget-usd`); other agents rely on their read-only sandbox                                                                                                                  |
| `retainSessions`        | `true`                               | plugin-harness write-back (opencode, Kilo): async upsert of the session transcript every turn, plus an idle flush that captures the reply the per-turn pass can't see (set `false` to opt out; hook harnesses always write on Stop) |
| `retainEveryTurns`      | `1`                                  | opencode write-back cadence (user turns)                                                                                                                                                                                            |
| `logLevel`              | `"info"`                             | plugin-log verbosity (`"debug"` \| `"info"` \| `"warn"` \| `"error"`); `HINDSIGHT_LOG_LEVEL` env overrides                                                                                                                          |
| `gitIngest`             | `"message"`                          | git depth for seeding AND staying current (same engine): `"message"` = commit messages only (one doc, re-upserted when HEAD moves); `"full"` = messages + per-commit full diffs (progressive, newest first); `"none"` = git off     |
| `harnesses.<name>`      | —                                    | per-harness override of any field above                                                                                                                                                                                             |
| `harness`               | `opencode`                           | **deepen engine only**: which session format `--conversations` is read as                                                                                                                                                           |

### Per-repo opt-in/out — `banks.<bankId>`

Per-repo control lives in the SAME file, keyed by the **resolved bank id** (shown in the session
banner) and applied AFTER bank resolution — so it works regardless of where the repo lives, and
survives directory moves:

```jsonc
{
  "banks": {
    "coding-agent::secret-client": { "disabled": true }, // blacklist: no memory at all
    "coding-agent::old-name": { "bank": "team::shared" }, // rename / converge banks
    "coding-agent::big-mono": { "gitIngest": "full", "retainSessions": false },
  },
}
```

Any behavioral field can be overridden per bank, and `bank` **renames the destination** (single
hop: the section is selected by the resolved id, the target is literal — several ids may converge
on one shared bank, and the target's own section is not consulted). Other bank-resolution fields
are ignored inside a bank section.

#### Recipe: two repos, one shared bank

Two ways, by what the natural key is:

**By resolved id** — you know the repo names; works wherever the repos live (and keeps working if
they move). Both ids converge on one literal target:

```jsonc
{
  "banks": {
    "coding-agent::backend": { "bank": "team::product" },
    "coding-agent::frontend": { "bank": "team::product" },
  },
}
```

**By path prefix** — the repos live under one directory; a single `mapPathToBank` entry covers
every repo (present and future) beneath it:

```jsonc
{
  "mapPathToBank": { "/Users/me/work/client-x": "client-x-memory" },
}
```

Rule of thumb: converge by **id** for a hand-picked set of repos; map by **path** when a folder is
the boundary ("everything I clone under `work/client-x` shares memory").

### Bank resolution

Coding memory is **per repository**. Resolution order for the working directory:

1. `mapPathToBank` — longest matching absolute-path prefix (mapping a repo root covers every
   subdirectory; deeper mappings win; overrides even an explicit `bankId`).
2. Static — `bankId` set (or `dynamicBankId: false`).
3. Dynamic — `bankIdTemplate` with placeholders:
   - `{gitProject}` — worktree-aware repo name: `git rev-parse --git-common-dir` resolves every
     linked worktree to the **main** worktree's basename, so all worktrees of a repo share one bank
     (bare repos use the bare dir name; non-git directories fall back to the dir basename)
   - `{project}` — plain working-directory basename
   - `{harness}` — the entry point asking (`opencode`, `claude-code`, `codex`, `antigravity-cli`, `cursor-cli`, `copilot-cli`)
   - `{channel}` / `{user}` — `$HINDSIGHT_CHANNEL_ID` / `$HINDSIGHT_USER_ID`

The default `"coding-agent::{gitProject}"` is **harness-neutral**, so opencode, Claude Code, and Codex
all share one memory per repo — use `"{harness}-{gitProject}"` to split per agent instead.

## Diagnostics & logging

Two files, two audiences:

**Leveled plugin log** (humans debugging): `$TMPDIR/hindsight-coding-agent/plugin.log` (override
`HINDSIGHT_LOG_FILE`) — timestamped `LEVEL [scope] message` lines from every component, including
the ingestion engine. Level defaults to `info`; set `"logLevel": "debug"` in config or
`HINDSIGHT_LOG_LEVEL=debug` for ad-hoc debugging (at `debug`, every diag event below is mirrored
here too, so one file tells the whole story).

**Structured diag events** (machines/harnesses): every reflect and page-fetch outcome is appended
as a JSON line to `/tmp/hindsight-plugin.log` (override with `HINDSIGHT_DIAG_FILE`):

```json
{
  "ts": "2026-07-27T07:05:52Z",
  "harness": "claude-code",
  "event": "reflect_ok",
  "ms": 14210,
  "chars": 792,
  "query": "..."
}
```

`reflect_failed` / `pages_failed` record the error; if you're comparing memory-on vs memory-off,
check this file — a run whose reflects failed is a no-memory run. Seed starts are logged as
`seed_started`.
