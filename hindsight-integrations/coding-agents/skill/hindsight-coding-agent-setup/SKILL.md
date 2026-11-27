---
name: hindsight-coding-agent-setup
description: Install, update, configure and troubleshoot this machine's Hindsight coding-agent memory plugin (the 🧠 banner). Use when the user wants to install or upgrade it, change its settings (per-repo opt-in/out, bank names, git depth, pages, server or daemon), or when memory looks broken, empty, or not updating (sync stuck) and needs diagnosing — logs, sync status, calling the Hindsight API directly.
---

<!-- GENERATED from README.md (its skill:begin regions) + skill-src/setup.md.
     Edit those, then run: npm run skill:build -->

# Hindsight Coding-Agent Memory — Setup & Troubleshooting

Reference for the `hindsight-coding-agents` plugin: installing and updating it, every configuration
setting, and how to diagnose it when memory looks wrong. For everyday use — storing, searching and
correcting memory — see the `hindsight-coding-agent` skill.

## Install / update

```bash
npx @vectorize-io/hindsight-coding-agents install all          # every detected agent, wired natively
npx @vectorize-io/hindsight-coding-agents install claude-code  # or just one
npx @vectorize-io/hindsight-coding-agents uninstall all        # removes exactly what install added
npx @vectorize-io/hindsight-coding-agents update               # refresh the runtime only, no rewiring
npx @vectorize-io/hindsight-coding-agents stats                # how often each agent uses Hindsight
```

`install` takes an explicit target — `all`, or one or more harness names. A bare
`npx @vectorize-io/hindsight-coding-agents install` changes nothing and prints the choice, so wiring every agent on
the machine is never something that happens by accident. **Updating is the same `install`
command again** — it re-copies the runtime in place.

Day to day you should not have to: once a day, a session start checks npm and re-stages a newer
runtime in the background (`autoUpdate`, on by default — set it to `false` to pin the version you
have). It only replaces a runtime that `npx` installed: a copy from `npm i -g`, a project dependency
or a local checkout is left alone. Running `update` or `install` yourself is different — it re-stages
the published release over whatever is installed, a local build included. `update` refreshes the copy
every wired agent already points at and deliberately touches no host config; re-run `install`
yourself after a release that adds a new hook, or to wire another agent.

## Local daemon settings (daemon mode)

Daemon settings keep the names the old per-agent Claude Code plugin used, so an existing
environment carries over unchanged:

| field               | env                             | default        | meaning                                                    |
| ------------------- | ------------------------------- | -------------- | ---------------------------------------------------------- |
| `serverMode`        | `HINDSIGHT_SERVER_MODE`         | `cloud`        | `cloud` \| `self-hosted` \| `daemon`                       |
| `apiPort`           | `HINDSIGHT_API_PORT`            | `9077`         | port the local daemon listens on                           |
| `daemonIdleTimeout` | `HINDSIGHT_DAEMON_IDLE_TIMEOUT` | —              | deprecated, ignored: the daemon no longer exits on its own |
| `daemonProfile`     | `HINDSIGHT_DAEMON_PROFILE`      | `coding-agent` | which local database it uses                               |
| `embedVersion`      | `HINDSIGHT_EMBED_VERSION`       | `latest`       | which `hindsight-embed` release to run                     |
| `embedPackagePath`  | `HINDSIGHT_EMBED_PACKAGE_PATH`  | —              | run a local checkout instead (development)                 |

Any `HINDSIGHT_API_*` variable you export is forwarded to the daemon, so server-side settings need
no equivalent here.

## Configuration

Configuration is **one JSON file**: `~/.hindsight/coding-agent.json`. Layering, later wins per field:

1. built-in defaults
2. environment variables — `HINDSIGHT_API_URL`, `HINDSIGHT_API_TOKEN`, and one per scalar setting
   (`HINDSIGHT_<FIELD_IN_CAPS>`), for containers and CI that inject config rather than write a file
3. the file's top level
4. its `harnesses.<name>` section — per-agent override
5. its `paths.<prefix>` section — per-directory override, applied after the bank is resolved
   (see [Per-directory overrides](#per-directory-overrides--pathsprefix))
6. its `banks.<resolvedBankId>` section — per-repo override, applied after the bank is resolved
   (see [Per-repo opt-in/out](#per-repo-opt-inout--banksbankid))

Environment variables are a **fallback**: the file wins wherever it sets a value, so adding env to
an existing setup changes nothing. The list-valued settings, `retainTags`, `recallTags` and `optInPaths`, take
a comma-separated value (`HINDSIGHT_RETAIN_TAGS="project:{gitProject},env:work"`); entries are
trimmed and blanks dropped.
The map-valued settings (`mapPathToBank`, `harnesses`, `paths`, `banks`, `retainMetadata`) are file-only —
per-key branching doesn't survive flattening into one variable. `maxParallelRetains` is available
as `HINDSIGHT_MAX_PARALLEL_RETAINS` for containers and CI.

`HINDSIGHT_CONFIG` moves the file itself — point it at another path for a container or a test
harness where `$HOME` is not the right anchor. It is still exactly one file; only its location
changes. (The other variables that are not settings are `HINDSIGHT_LOG_FILE`, `HINDSIGHT_DIAG_FILE`,
`HINDSIGHT_USAGE_FILE` and `HINDSIGHT_LOG_LEVEL` — see [Diagnostics & logging](#diagnostics--logging).)

### When a change takes effect

Config is read when a process starts — the file is not watched — so when an edit applies depends on
what reads it:

| host                                                                                                        | reads the file                                                      | an edit applies            |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------- |
| hook harnesses (Claude Code, Codex CLI, Cursor CLI, GitHub Copilot CLI, Grok Build, Antigravity CLI, Devin) | once per hook invocation — each hook is its own short-lived process | on your next prompt        |
| persistent plugins (opencode, opencode 2, Kilo CLI, Cline CLI, pi, DeepSeek Harness)                        | once per workspace, when the host loads the plugin                  | after restarting the agent |
| the MCP server behind the `hindsight_*` tools                                                               | once at startup                                                     | in your next session       |

`apiToken` is the exception. Every host re-reads it when the server rejects a request, so enabling
authentication or rotating the key is picked up on the next call with nothing to restart —
otherwise a rotation would leave a long-running agent failing every memory call until it was
restarted. Everything else follows the table: `apiUrl`, `disabled`, bank routing, `gitIngest`, and
the survey and page settings.

`hindsight_diagnose` reports both sides of that gap — what the file says now, and what the running
client is actually using.

### Opt-in only

By default every project gets memory — that is what makes the plugin zero-setup. If you would
rather nothing be remembered until you say so, turn memory off everywhere and name the projects
that may use it:

```jsonc
{
  "optInOnly": true,
  "optInPaths": ["~/work/client-x", "~/oss"],
}
```

Anything outside those paths is **inert**: no bank is created, nothing is retained, no seed runs,
and the agent behaves exactly as it would without the plugin. Approving costs nothing else —
`optInPaths` says _which projects_, not _which bank_, so an approved repo keeps its usual
`coding-agent::{gitProject}` name. Paths are prefixes, so approving `~/work` approves every repo
under it while each still gets its own bank.

A `mapPathToBank` entry counts as opted in too, since routing a path to a named bank already
declares that project. A bare `bankId` does not: it names a bank rather than a project, so it
cannot say which work may be remembered, and a privacy switch has to fail closed.

There is no per-repo opt-in file, for the same reason there is no repo-carried config at all: a
cloned repository must not be able to turn memory on.

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

| field                   | default                              | meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `apiUrl`                | `https://api.hindsight.vectorize.io` | Hindsight API base URL (set to `http://localhost:8888` for a local server)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `apiToken`              | —                                    | bearer token, required for Hindsight Cloud: with the Cloud URL and no token, memory stays off instead of sending requests Cloud would refuse. Picked up without restarting the agent: a long-lived host re-reads it after a rejected request, so enabling auth or rotating the key mid-session recovers on the next call                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `mcpServerName`         | `hindsight`                          | the name this package's MCP server is registered under in every agent, so it can sit beside a server of your own called `hindsight`. Set it with `install --mcp-name <name>`; letters, digits, `-` and `_`, up to 64. Top level only: never read from `harnesses`, `banks` or `paths`. Not used by Dcode                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `apiHeaders`            | —                                    | extra HTTP headers sent with every request, as a map of name to value. For a server behind a proxy that checks its own header, e.g. Cloudflare Access: `{"CF-Access-Client-Id": "<id>.access", "CF-Access-Client-Secret": "<secret>"}` (a Cloudflare service token). Fixed values only; `Authorization` stays the plugin's own when `apiToken` is set. Can be set per directory under `paths`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `bankId`                | —                                    | **explicit static bank**; unset ⇒ per-repo dynamic resolution (below)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `dynamicBankId`         | dynamic iff no `bankId`              | force dynamic (`true`) or static (`false`) resolution                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `bankIdTemplate`        | `"coding-agent::{gitProject}"`       | dynamic bank id format; the default makes every agent share one bank per repo                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `mapPathToBank`         | —                                    | absolute path → bank; **longest prefix wins**; linked worktrees inherit their main checkout's mapping; overrides everything                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `optInOnly`             | `false`                              | run memory ONLY in opted-in projects — everything else is inert, with no bank created; see [Opt-in only](#opt-in-only)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `optInPaths`            | —                                    | directories opted in, matched as prefixes with `~` expanded; each repo beneath and its linked worktrees are approved while keeping their own dynamic bank                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `resolveWorktrees`      | `true`                               | linked worktrees inherit the main checkout's bank identity, path approval, and mapping                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `retainTags`            | —                                    | extra tags on every document written by the integration, e.g. `["project:{gitProject}"]` — see **Recording where a memory came from** below                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `retainMetadata`        | —                                    | extra metadata on every document written by the integration, e.g. `{"repo": "{gitProject}"}`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `retainContext`         | see description                      | the `context` sent with every session write-back. Extraction reads it to decide whose claim a sentence is, so the default names both speakers, and asks for an agent's own "fixed" or "passes" to be stored as its claim rather than as fact. Accepts the same `{placeholder}` templates as `retainTags`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `keepToolInputs`        | —                                    | tools whose input is what the agent **said** to the user, kept in full as an assistant turn instead of a bare action line: a chat bridge's reply tool (Claude Code Channels: Telegram, Discord, …), or a command-line tool that delivers the reply. Keyed by the tool name as the transcript records it, the value names the input field holding the text: `{"mcp__plugin_telegram_telegram__reply": "text"}`. For a shell tool that also runs everything else, add a regex the field must match: `{"exec_command": {"field": "command", "match": "^send-reply "}}`. Add a named group `text` to the regex to keep only the message, e.g. `"^send-reply \"(?<text>[\\s\\S]*)\"$"`. Kept text is capped at 8000 characters. A call the host reports as failed stays a bare action line on Claude Code, Codex in code mode, opencode, opencode2 and kilo; on the other harnesses (and plain Codex function calls) a failed send is kept as said. Does nothing on antigravity-cli, cline-cli, copilot-cli, devin-cli, traecode and zcode, whose readers keep no tool calls. A bad rule is dropped with a warning. Global or per harness, not per bank; a `harnesses.<name>` entry replaces the global map rather than merging into it. File-only, like `recallOptions` |
| `recallTags`            | —                                    | tags every read is narrowed to: the automatic recall and reflect injected before a prompt, and `hindsight_reflect`. Same placeholders as `retainTags`, e.g. `["project:{gitProject}"]` on a shared bank. Unset reads the whole bank. See [Reading back only this repo's memories](#reading-back-only-this-repos-memories)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `recallTagsMatch`       | `"any"`                              | how `recallTags` match: `"any"` (any of the tags, plus untagged memories), `"all"`, `"any_strict"` or `"all_strict"` (untagged memories left out)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `manageBankConfig`      | `true`                               | let the plugin shape the bank's own configuration — the retain strategies it writes under, the `knowledge` entity-label group, and, on a bank that has none, the missions. Writing is **additive**: it adds what the bank does not define and never overwrites what is there, so your control-plane edits survive — the one exception is the extraction mode of its own strategies, which follows `retainExtractionMode`. Set `false` to keep it out of the bank config entirely — see **A bank you shape yourself** below                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `retainExtractionMode`  | `"concise"`                          | how the server extracts memories from sessions, commits and documents: `"concise"`, `"verbose"`, `"verbatim"` or `"chunks"` (store the text, no extraction). Every Stop writes the session back, so this is what each turn costs — `"verbose"` pulls more detail for several times the tokens. Kept in sync on the plugin's own retain strategies every session, so a change reaches existing banks too (not with `manageBankConfig: false`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `defaultBankConfig`     | —                                    | bank-config fields of your own for the banks the plugin shapes, e.g. `{"enable_observations": false, "enable_auto_consolidation": false, "mental_model_min_refresh_interval_seconds": 21600}`. Keys are the bank-config API's own field names. Written under the same additive rule as the rest: only where the bank does not define the field, so a bank the plugin creates is born with these instead of the server's defaults and a value you set in the control plane is never overwritten. Wins over the template on a key both name (`enable_observations`); `retain_strategies`, `entity_labels` and `retain_extraction_mode` are refused, since the plugin governs them itself. Ignored with `manageBankConfig: false` — see **A bank you shape yourself** below. File-only, like `recallOptions`; in a `banks.<id>` section it replaces the global map rather than merging into it                                                                                                                                                                                                                                                                                                                                                                         |
| `observationScopes`     | `"shared"`                           | how consolidation groups observations: `"shared"` (default) = ONE global scope per bank, so every agent on a repo builds one set of beliefs; also `"combined"` (the server default), `"per_tag"`, `"all_combinations"`, `[["t"]]`; `"per_source"` adds a scope per `source:` kind alongside the global one, so commit knowledge and conversation knowledge consolidate apart                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `disabled`              | `false`                              | hard off-switch (inert plugin/hook — a no-memory baseline)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `reflectTimeoutMs`      | `15000`                              | **automatic** session-reflect timeout; on hook harnesses the installer registers a 30s prompt-hook timeout, so going above ~20s also means raising that hook's `timeout` in the host's config, or the host kills the hook mid-reflect; on timeout or a 5xx the hook falls back to page search, then to a raw recall of observations (recorded)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `reflectToolTimeoutMs`  | `330000`                             | timeout for the agent-invoked `hindsight_reflect` tool — a call the agent waits on, whose high-budget synthesis on a populated bank runs for minutes. Defaults above the server's own reflect wall timeout (`HINDSIGHT_API_REFLECT_WALL_TIMEOUT`, 300s) so the server decides when to give up. Unset, it inherits an explicitly raised `reflectTimeoutMs`, but a short one never lowers it                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `injectTimeoutMs`       | `5000`                               | retrieval timeout in milliseconds for `autoInject: "pages"` and `autoInject: "recall"`; also the **shared** budget for page search then recall after a reflect timeout/5xx (recall gets only the time left after page search), and the limit on fetching the page list in the prompt and session-start hooks. Set e.g. `2000` to limit turn latency or `15000` for a slower self-hosted bank. Environment fallback: `HINDSIGHT_INJECT_TIMEOUT_MS`. On hook harnesses, injection must fit the host's prompt-hook timeout (30s by default); for reflect with fallback, allow for **`reflectTimeoutMs` + `injectTimeoutMs`**, and raise the host hook's `timeout` when needed                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `injectQueryMaxChars`   | unset                                | the most characters of the prompt that `autoInject: "pages"` and `autoInject: "recall"` search with, and the reflect fallback too. Reflect itself still gets the whole prompt. Unset, nothing changes: page search takes the first 500 characters and recall about 500 tokens (about 1,000 characters of English, 250 of Cyrillic). On a CPU-only server a long question can make the reranker use up all of `injectTimeoutMs`, so no memory arrives: set e.g. `300` there. Environment fallback: `HINDSIGHT_INJECT_QUERY_MAX_CHARS`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `reflectBudget`         | `"high"`                             | reflect budget for the `hindsight_reflect` tool: `"low"`, `"mid"` or `"high"`. Drop it on a large bank where high-budget synthesis exceeds the server's wall timeout. The automatic session-start reflect always uses `"low"` to fit its hook window and is unaffected                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `autoInject`            | `"reflect"`                          | what to inject on the session's **first prompt** (and again every `autoInjectEveryTurns`, for `"pages"`/`"recall"`): `"reflect"` = a low-budget reflect synthesis (on timeout/5xx it falls back to page search, then recall); `"pages"` = the pages matching the prompt by search; `"recall"` = the bank's memories recalled for the prompt (what it asks for is `recallOptions`, observations by default); `"none"` = nothing — the agent searches pages first and reflects only when they are too shallow. `"pages"` and `"recall"` are retrieval only (no LLM), their time budget is `injectTimeoutMs`, and when one finds nothing for the prompt it tries again on the next one, up to 5 prompts. A compaction counts as a new session start: the model lost what it was shown, so the first prompt after it injects again (Claude Code, Codex, Qwen Code, Factory Droid, Cursor, Kimi Code, opencode, Kilo, opencode2 and pi report their compactions)                                                                                                                                                                                                                                                                                                         |
| `autoInjectEveryTurns`  | `0`                                  | run a `"recall"` or `"pages"` `autoInject` again every N user turns, with the current prompt as the query, so a long session's memory follows its topic. `0` = first prompt only. Only the first run shows a notice. Ignored for `"reflect"`, which would put an LLM call in front of those prompts                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `autoReflect`           | `true`                               | **deprecated** — use `autoInject`. Still honoured (`false` = `autoInject: "none"`), ignored when `autoInject` is set, and logs a deprecation warning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `pageSearchLimit`       | `3`                                  | pages ONE search returns. Applies to every search of the bank — the `autoInject: "pages"` injection, the reflect fallback, and the agent's own `hindsight_search_pages` tool — because the limit lives on the client they all share                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `recallOptions`         | see description                      | overrides merged key-by-key into the body of every recall — the `autoInject: "recall"` source and the reflect fallback. Keys are the API's own recall parameters, passed straight through, so anything recall accepts is settable without a new option here; `query` is the one field it cannot replace. The default body is `{"types": ["observation"], "budget": "low", "max_tokens": 2000, "include": {"entities": null}}`, and what you set is merged over it one key at a time — `{"max_tokens": 4000}` changes the budget and leaves the rest alone. Observations are the consolidated layer, so they answer best per token, but a bank with **consolidation disabled never grows any** and the default recall comes back empty on it: set `{"types": ["world", "experience"]}` there, or `{"types": null}` for every type. Each injected observation ends with `(sources: N)`, how many facts back it, when the server sends that count. File-only, like `retainMetadata` — an object does not flatten into an env var                                                                                                                                                                                                                                       |
| `writeTools`            | `true`                               | `false` = the two tools that write to the bank (`hindsight_capture_initiative`, `hindsight_ingest_document`) are not registered, and the agent is never told to call them — not in the tool guide, not in the correction lines. For a host that denies those tools. The codebase survey still gets `hindsight_ingest_document`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `toolGuideExtra`        | —                                    | your own guidance on how the agent should treat memory, added **after** the built-in tool guide (wherever the full guide is sent) and after the crediting note in the `hindsight_search_pages` tool description. It adds to the built-in text, never replaces it. Set it per harness or per bank like any other key, e.g. `"Memory is a past record, not current state: verify any claim against the code before acting on it."`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `sessionContext`        | —                                    | mental models and pages whose **content** goes into the agent's context when a session starts, in the order listed: `mental_model:<id>` or `page:<id>` (a bare id is a mental model), e.g. `["mental_model:standing-rules"]`. For memory the agent must always have in front of it, like standing rules, rather than look up. Each is read once per session (no LLM call, 5 s limit each, all at once), so an update shows up in the next session. One that cannot be read, or has no content yet, is skipped with a warning in the plugin log. Same as the Hermes plugin's `session_context`. Env: comma-separated `HINDSIGHT_SESSION_CONTEXT`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `pageRefreshEveryTurns` | `1`                                  | refetch the pages and re-inject the page roster every N user turns (`0` also means 1). The full tool guide goes out once per session; later refreshes are a one-line reminder                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `pageTriggerType`       | `"cron"`                             | when NEW pages refresh, i.e. what keeping them current costs — `"cron"` (default) on `pageTriggerCron` only and only when actually stale, `"auto-refresh"` after every consolidation that produced new material, `"manual"` never on their own. Auto-refresh is the most current and by far the most expensive: one synthesis per page per consolidation. Maps to the page's `trigger.refresh_cron`, or `trigger.refresh_after_consolidation` in the Hindsight API (`true` for auto-refresh, `false` for manual)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `pageTriggerCron`       | `"H * * * *"`                        | schedule for `pageTriggerType: "cron"` — UTC, standard 5-field cron, e.g. `"0 3 * * *"`. The default is hourly, each page on its own hashed minute. Sets the page's `trigger.refresh_cron`, which the API treats as mutually exclusive with `refresh_after_consolidation`; a scheduled refresh is skipped when nothing changed. Write a field as `H` to give each page its own value there — see **Spreading refreshes with `H`** below                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `pages`                 | every page                           | per-page configuration for the seeded pages, keyed by page name (case-insensitive): `false` skips a page entirely, `{"source_query": "..."}` seeds it with your question instead of the built-in one. Omitted, all five pages are seeded with their built-in queries. This is the supported way to own a page's wording — the plugin re-syncs a page whose live query differs from the one it is configured to have, so a query edited through the API or the control plane is replaced on the next session. A skipped page is **not deleted**: one already seeded keeps its content and stops being re-synced. The scoping clause is appended to your query too, so a reworded page cannot start reporting a dependency's decisions as this project's. File-only, like `recallOptions`; in a `banks.<id>` section it replaces the global map rather than merging into it                                                                                                                                                                                                                                                                                                                                                                                           |
| `customPages`           | —                                    | pages of your own, seeded alongside the five above and keyed by the name they get: `{"Security posture": {"source_query": "...", "tags": ["knowledge:decision"]}}`. `source_query` is required; `tags` picks which facts feed the page and is optional — omitted, the page draws on everything the bank holds. A separate setting from `pages` on purpose, so that an unknown name there stays a typo warning rather than quietly creating a page. File-only, and replaced (not merged) by a `banks.<id>` section                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `autoSeed`              | `true`                               | SessionStart: auto-seed a cold repo's bank from git history                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `seedLimit`             | `300`                                | git-log document size: a full write keeps the newest commits, between half of this and this many (the window moves in steps of half), and logs how many older ones it drops                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `codebaseSurvey`        | `true`                               | SessionStart: headless survey of a cold repo's structure, run only under the current harness's own CLI (Claude Code, Codex, Antigravity, opencode v1); other agents get no survey                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `surveyModel`           | `haiku`                              | model for the survey — Claude recipe only (`claude -p --model`); other agents use their configured default                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `surveyBudgetUsd`       | `2`                                  | survey spend cap — Claude recipe only (`claude -p --max-budget-usd`); other agents rely on their read-only sandbox                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `surveyRefreshCommits`  | `20`                                 | re-run the survey at SessionStart once this many commits have accrued since the last one, so the structural pages track an architecture that keeps moving (`0` = survey a cold repo only, never again)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `retainSessions`        | `true`                               | session write-back, honored by every harness: hook harnesses write the transcript on Stop, Factory Droid also writes on its cancellation notification, and plugin harnesses (opencode, opencode 2, Kilo) upsert it every turn plus an idle flush that captures the reply the per-turn pass can't see. Set `false` - globally, per harness, or per bank - to stop writing transcripts (the background history import stops with it) while recall, git ingest and the memory tools keep working                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `skipOriginators`       | `[]`                                 | Codex only: don't write back a session started by another client, matched (exact match, case-sensitive) against the rollout's `session_meta.originator` — e.g. `["Claude Code"]` under `harnesses.codex`, so jobs Claude Code delegates to Codex are not saved twice. Sub-agent threads carry their client's originator too, so they are skipped with it, and `--import-conversations` skips the same sessions. Recall is unchanged; a skipped session logs `retain_skipped_originator`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `maxParallelRetains`    | `10`                                 | cap on concurrent retain-related requests: drain()'s per-op polls plus deepen's chat/git retain pools. The API rate-limits bursts, not single requests — if you see 429s, lower this rather than raising it                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `logLevel`              | `"info"`                             | plugin-log verbosity (`"debug"` \| `"info"` \| `"warn"` \| `"error"`); `HINDSIGHT_LOG_LEVEL` env overrides                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `autoUpdate`            | `true`                               | keep the installed runtime current by itself: once a day a session start asks npm for the published version and, when it is newer, re-stages `~/.hindsight/coding-agents` in the background. It rewires no host config, so a release adding a **new** hook entry point still needs a manual `install`. Set `false` to pin the installed version; `disabled` stops it too, since an inert plugin should stay inert. Only ever replaces a runtime installed the documented way, via `npx` — a copy installed with `npm i -g`, vendored as a project dependency, or built from a checkout is left to whoever manages it (update those the way you installed them), and it needs `npx` and `npm` on `PATH`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `gitIngest`             | `"message"`                          | git depth for seeding AND staying current (same engine): `"message"` = commit messages only (one doc, re-upserted when HEAD has a commit it lacks); `"full"` = messages + per-commit full diffs (progressive, newest first); `"none"` = git off                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `harnesses.<name>`      | —                                    | per-harness override of any field above                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `harness`               | `opencode`                           | **deepen engine only**: which session format `--conversations` is read as                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

By default a page refreshes **hourly, staggered**: `pageTriggerCron` is `"H * * * *"`, so every
page gets its own minute of the hour (see below) and a tick with nothing new to fold in is skipped
server-side. That keeps pages within an hour of the repo without paying auto-refresh's price — one
LLM synthesis per page per consolidation, on a repo that consolidates all day. Set
`pageTriggerType: "auto-refresh"` to go back to refreshing on every consolidation.

`pageTriggerType`/`pageTriggerCron` decide only **when** a page refreshes. **How** it refreshes
belongs to the server: Hindsight creates a page with a delta refresh (each pass edits the
page instead of rebuilding it) that doesn't reflect over sibling pages, and these settings merge
over those defaults rather than replacing them.

### Spreading refreshes with `H`

One `pageTriggerCron` is shared by every page in every repo you point this plugin at. So a literal
`"0 3 * * *"` does not schedule _a_ refresh at 03:00 — it schedules **all** of them at 03:00, five
pages per bank, on the same worker pool that serves retain. A session ingesting at 03:0x queues
behind the pile, and moving the hour just moves the pile.

Write a field as `H` and it is replaced, per page, by a value hashed from the bank id and the page
name. Each page gets its own slot, the same slot on every run:

| `pageTriggerCron`  | what each page gets                                    |
| ------------------ | ------------------------------------------------------ |
| `"H H * * *"`      | once a day, at its own minute and hour                 |
| `"H * * * *"`      | once an hour, at its own minute                        |
| `"H 3 * * *"`      | daily at 03:MM — spread inside the hour you chose      |
| `"H H(0-5) * * *"` | daily, spread across 00:00–05:59 only                  |
| `"0 3 * * *"`      | no `H`, no hashing — exactly what it says, all at once |

`H` is [Jenkins' syntax](https://www.jenkins.io/doc/book/pipeline/syntax/#cron-syntax) for the same
problem. It never reaches the API: the plugin resolves it to an ordinary cron expression
(`"41 17 * * *"`) when it creates the page, so the schedule you see in the control plane is a plain
one you can edit. Hashing spreads pages out, it does not partition them — two pages can still land
on the same minute, just not all of them.

**These settings apply to the pages a repo already has, too.** Every session compares each page
this plugin created — the seeded taxonomy and every captured initiative — against the config and
re-syncs the ones that differ, so a bank seeded before this default changed moves onto the hourly
schedule by itself, and a page you retriggered by hand in the control plane is put back on the
configured policy the next time an agent runs. The config
file is the source of truth for these pages: to give one a different schedule, change
`pageTriggerType`/`pageTriggerCron` (per bank, if it is only that repo) rather than editing the
page. Only the fields this plugin states are touched — a page's `mode`, its excluded siblings and
its minimum refresh interval are left exactly as they are.

### Customize Pages — `pages`

Every repo gets the same five pages. They are a **taxonomy**, not a summary of your source: each one
is synthesized from what the bank ingested — commit history and past conversations — and each is
pinned to one knowledge tier, so a page draws only on the facts the extractor routed to it.

| Page                           | What it answers                                                                                         | Tier tag                 |
| ------------------------------ | ------------------------------------------------------------------------------------------------------- | ------------------------ |
| `Component map`                | the main components/modules/subsystems, what each is responsible for, and how they depend on each other | `knowledge:component`    |
| `Core concepts`                | the domain abstractions and key entities — the vocabulary a developer has to know                       | `knowledge:concept`      |
| `Conventions and patterns`     | how THIS project does things: testing, error handling, naming, structure, how changes are made          | `knowledge:convention`   |
| `Key decisions and rationale`  | the significant technical decisions and the durable "why we do it this way" behind them                 | `knowledge:decision`     |
| `Initiatives and enhancements` | the major initiatives and features over time, linking out to each captured initiative's own page        | `knowledge:feature-work` |

`pages` says which of them to seed and what each one asks. Keys are the page names above, matched
ignoring case and surrounding spaces:

```jsonc
{
  "pages": {
    // don't seed this page at all
    "Component map": false,
    // seed it, but ask your question instead of the built-in one
    "Key decisions and rationale": {
      "source_query": "What did we decide about data retention, encryption and PII handling, and why? Prefer decisions that constrain what new code may do.",
    },
  },
}
```

Omit `pages` entirely — the default — and all five are seeded with their built-in queries.

**Fewer pages.** Each page costs one LLM synthesis per refresh, so a repo that only wants the
architecture ones turns the rest off:

```jsonc
{
  "pages": {
    "Initiatives and enhancements": false,
    "Conventions and patterns": false,
    "Key decisions and rationale": false,
  },
}
```

**Per repo**, like every other field — usually where this belongs, since what a page should ask is a
property of the project, not of your machine:

```jsonc
{
  "banks": {
    "coding-agent::payments-api": {
      "pages": {
        "Core concepts": {
          "source_query": "What are the payment domain's entities — orders, ledgers, settlement states — and what does each mean in OUR model?",
        },
      },
    },
  },
}
```

Four things worth knowing before you reach for it:

- **This is the only durable way to reword a page.** Every session compares each seeded page against
  the query it is configured to have and re-syncs the ones that differ, so a `source_query` edited
  through the API or the control plane is replaced the next time an agent runs. Setting it here makes
  your wording the configured one. When a re-sync does replace a query, the plugin now says which
  page in the plugin log rather than doing it silently.
- **Your query still gets the scoping clause.** The sentence that keeps a dependency's decisions off
  your project's page is appended to a custom query too — a bank holds facts about the libraries and
  services a repo merely uses, and without that clause a page will present them as yours.
- **The tier tag stays the taxonomy's.** It selects which facts the synthesis reads; rewording the
  question changes what is asked of those facts, not which ones are in scope.
- **`false` does not delete anything.** A page already seeded keeps its content and simply stops
  being re-synced — remove it in the control plane if you want it gone.

A name that matches no page above is ignored with a warning in the plugin log, so a typo fails
loudly instead of looking like it disabled something. Adding pages of your own is not what this
setting is for: the agent's `hindsight_capture_initiative` tool already creates pages, one per
initiative, each with its own query.

#### Pages of your own — `customPages`

`pages` only reworks the five above. To add a page, name it under `customPages`:

```jsonc
{
  "customPages": {
    "Security posture": {
      "source_query": "What are this project's security decisions — authn, secrets handling, PII, dependency policy — and what do they constrain in new code?",
      "tags": ["knowledge:decision"],
    },
    "Operational runbook": {
      "source_query": "How does this project get deployed, monitored and rolled back? What has broken in production, and what fixed it?",
    },
  },
}
```

`source_query` is required. `tags` is optional and picks which facts feed the page — one of the tier
tags above, or any tag you stamp on your own writes with `retainTags`. Omit it and the page draws on
everything the bank holds: a refresh matches tags with `all`, so no tags means no tag constraint,
**not** an empty page.

Your pages are seeded at the same root as the taxonomy, on the same refresh schedule, and re-synced
from the config the same way — reword one here and the live page follows on the next session. They
compose with `pages`, so trading two built-ins for one of your own is just both settings at once.

**Why it is a separate setting.** `pages` refuses a name that matches no seeded page, and that is
what makes a typo loud: if an unknown key there meant "create this page", `"Componnet map"` would
quietly create an empty second page instead of rewording the one you meant. For the same reason,
naming a seeded page under `customPages` is refused — reword it under `pages`.

A page you create yourself in the control plane is a third thing again, and the plugin never touches
it: the seed pass only visits the pages it is configured to own.

### A bank you shape yourself — `manageBankConfig`

Pointed at a bank, this plugin gives it the shape its ingestion needs: retain strategies for the
kinds of document it writes (`git`, `gitlog`, `conversation`, `document`, `survey`), a `knowledge`
entity-label group that routes facts to the pages, and — on a bank that has no missions of
its own — the coding missions.

**It only ever adds what is missing.** A strategy you defined, an edit you made to one of the
plugin's, a reworded label group, a mission you rewrote in the control plane: each is left exactly
as it is, on every session, forever. What the bank already says wins. The cost of that promise is
that a plugin release which _rewords_ an existing strategy or label does not reach a bank that
already has it. To take the current default back, clear that override on the bank (delete the
strategy, or the whole `retain_strategies` entry, in the control plane): the next session finds the
bank silent there and seeds it again.

One field is the exception: the extraction mode of the plugin's own four strategies (`git`,
`gitlog`, `conversation`, `document`) follows `retainExtractionMode` and is put back on every session
if it drifts — the same way a seeded page's query is. Change it in `coding-agent.json`, not in the
control plane. The strategies' other fields, and your own strategies, are still left alone.

**Your own defaults for new banks — `defaultBankConfig`.** Everything the template does not name, a
bank the plugin creates gets from the server: auto-consolidation on, observations on, a 60s floor
between mental-model refreshes. With one bank per repo those are the settings that cost money, and
a new repo an agent touches spawns a new bank with all of them switched on — while every bank you had
hand-tuned in the control plane sits at a cheaper baseline. `defaultBankConfig` names the baseline
once, in the plugin's config, and it travels in the same import that creates the bank:

```json
{
  "defaultBankConfig": {
    "enable_observations": false,
    "enable_auto_consolidation": false,
    "mental_model_min_refresh_interval_seconds": 21600
  }
}
```

The keys are the bank-config API's own field names, passed straight through, so anything the
bank-config accepts can go here. They follow the same additive rule as the template: written only
where the bank is silent, so an existing bank picks them up on its next session **except** for the
fields it already sets — a value you chose in the control plane stays, even one equal to the server
default, and so does a value an earlier plugin release seeded (the template writes
`enable_observations: true` alongside the missions; clear that override on the bank to let your
default in). On a key the template also names, your default wins. The three fields the plugin
governs itself — `retain_strategies`, `entity_labels` and `retain_extraction_mode` — are refused
with a warning: the first two are merged entry by entry, the last is `retainExtractionMode`'s.

Set `manageBankConfig: false` to keep the plugin out of the bank's configuration altogether,
`defaultBankConfig` included — the right setting for a bank you share with non-coding work, or one
you configure yourself. That bank should then define the five strategies above itself. Note that the
miss is **silent**: the server does not reject a retain naming a strategy the bank lacks, it logs a
warning and extracts with the bank's own configuration — so a commit diff, a session transcript and
a survey marker would all get the same generic treatment instead of the extraction each needs.
Pages are seeded either way; `pageTriggerType` governs what they cost.

Like every field here it can be set per bank, which is usually where it belongs:

```json
{
  "bankId": "my-global-bank",
  "banks": { "my-global-bank": { "manageBankConfig": false } }
}
```

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

### Per-directory overrides — `paths.<prefix>`

When a folder is the boundary for something other than the bank — typically a different server
or tenant for client work — key the override by directory instead of by bank id:

```jsonc
{
  "apiToken": "personal-key",
  "paths": {
    "~/work/client-x": { "apiToken": "client-x-key" },
    "~/work/acme": { "apiUrl": "https://hindsight.acme.internal", "apiToken": "acme-key" },
    "~/oss": { "serverMode": "daemon" },
  },
}
```

Every repo under the prefix, present and future, keeps its own bank but uses the entry's
settings. The longest matching prefix wins, and a linked worktree outside the tree uses its main
checkout's entry, as with `mapPathToBank`. The section applies after bank resolution and before
`banks.<bankId>`, so a bank section still wins for its one repo. Bank-resolution and approval
fields (`bankId`, `mapPathToBank`, `bank`, `optInOnly`, `optInPaths`, ...) are ignored inside a
path section. Any connection setting can be overridden — `serverMode`, `apiUrl`, `apiPort`,
`apiToken` — so one directory can use a local daemon while another uses Cloud or a self-hosted
server. The token is re-read on a rejected request, like the top-level `apiToken`, but only while
the directory still points at the same server.

### Bank resolution

Coding memory is **per repository**. Resolution order for the working directory:

1. `mapPathToBank` — longest matching absolute-path prefix (mapping a repo root covers every
   subdirectory; deeper mappings win; overrides even an explicit `bankId`).
2. Static — `bankId` set (or `dynamicBankId: false`).
3. Dynamic — `bankIdTemplate` with placeholders:
   - `{gitProject}` — worktree-aware repo name: `git rev-parse --git-common-dir` resolves every
     linked worktree to the **main** worktree's basename, so all worktrees of a repo share one bank
     (bare repos use the bare dir name). **Outside a repo** there is nothing for git to resolve, so
     it falls back to the basename of the directory the **session started in** — an agent that
     `cd`s into a subdirectory keeps writing to one bank, and a subdirectory gets its own bank only
     when you deliberately start a session there
   - `{project}` — plain working-directory basename
   - `{harness}` — the entry point asking (`opencode`, `claude-code`, `codex`, `antigravity-cli`, `cursor-cli`, `copilot-cli`)
   - `{channel}` / `{user}` — `$HINDSIGHT_CHANNEL_ID` / `$HINDSIGHT_USER_ID`

The default `"coding-agent::{gitProject}"` is **harness-neutral**, so opencode, Claude Code, and Codex
all share one memory per repo — use `"{harness}-{gitProject}"` to split per agent instead.

The bank is resolved on every turn from where the agent is working. A session that moves into a
directory with a different bank (say, `cd` across a `mapPathToBank` prefix) **follows the folder from
that turn on**: the new bank gets the turns from the move onward, never the ones already saved
elsewhere, and going back appends to the conversation the first bank already holds. Each move is
logged as a warning in `plugin.log`.

### Recording where a memory came from

With a bank per repo, the bank _is_ the answer to "where did this come from". On a deliberately
**shared** bank — one bank holding cross-project knowledge so facts recall everywhere — it isn't:
every memory looks alike. `retainTags` and `retainMetadata` stamp that provenance onto conversations,
git history and diffs, survey lifecycle documents, initiative markers, and documents saved through
`hindsight_ingest_document`:

```jsonc
{
  "bankId": "shared", // one bank for everything
  "retainTags": ["project:{gitProject}", "env:work"],
  "retainMetadata": { "repo": "{gitProject}" },
}
```

Recalls can then filter by `project:<repo>`, and every document shows which repository it came out
of. Both accept the same placeholders as `bankIdTemplate` — `{gitProject}`, `{project}`,
`{harness}`, `{channel}`, `{user}` — plus `{bankId}`, `{sessionId}`, `{timestamp}` and `{branch}`.
`{gitProject}` is worktree-aware here too, so every linked worktree of a repo stamps one name.
`{sessionId}` resolves to `unknown` for documents that do not originate from an agent session.
`{branch}` is the branch checked out when the memory is saved (`detached` on a detached HEAD,
`unknown` outside git).

The branch is also recorded on its own, with nothing to configure: every memory saved from a git
repo carries `metadata.branch`, so a fact learnt on a branch that was never merged can be told apart
from the project's current state. It is not a tag unless you add one, e.g.
`"retainTags": ["branch:{branch}"]`. History imported from before the plugin was installed carries
no branch, since today's checkout says nothing about where those sessions ran.

The plugin's own `source:` and `harness:` tags are reserved: entries in those namespaces are ignored
with a warning, so a document's agent attribution always reflects the agent that actually wrote it.

#### Reading back only this repo's memories

Tagging memories on the way in doesn't keep them apart on the way out: by default a recall or
reflect reads the whole bank, so in a shared bank every repo's answers draw on every other repo's
memories. `recallTags` narrows reads to the tags you give, with the same placeholders as
`retainTags`:

```jsonc
{
  "bankId": "shared",
  "retainTags": ["project:{gitProject}"],
  "recallTags": ["project:{gitProject}"],
}
```

It applies to the recall and reflect the plugin injects before a prompt and to the `hindsight_reflect`
tool. Placeholders that change on every save, `{timestamp}` and `{sessionId}`, would match nothing
here.
`recallTagsMatch` says how the tags match. The default, `"any"`, keeps memories carrying any of the
tags **plus untagged ones**, so knowledge shared across repos still answers. `"any_strict"` and
`"all_strict"` leave untagged memories out, and `"all"` requires every tag. Page search is not
narrowed yet: the server cannot filter pages by tag.

### One set of beliefs per repo

Every document this integration writes carries provenance tags — `source:chat`, `harness:<id>`,
`knowledge:<kind>`, plus anything from `retainTags`. Those tags say **who wrote** a memory; they are
what filters recall and draws each document's agent logo, and they stay on the facts.

They are not, however, a good boundary for
[observations](https://hindsight.vectorize.io/developer/observations). Consolidation's own
default (`combined`) builds one observation set per distinct tag set, so the same repository
worked on by two agents would grow two parallel sets of beliefs — one per harness — that never
merge, each blind to the other, at double the consolidation cost. Which agent happened to be typing
does not change whether a convention or a decision is true.

So the integration retains with `observationScopes: "shared"`: one global, untagged observation
scope per bank, which is what a bank already is — one project's memory. Set the field to change it:

```jsonc
{
  "observationScopes": "combined", // one observation set per distinct tag set (server default)
  "banks": {
    "coding-agent::mono": { "observationScopes": "per_tag" }, // per-repo, like any behavioral field
  },
}
```

### Splitting code from conversation — `per_source`

`shared` puts every document a repo produces into one belief set. `"per_source"` keeps that set and
adds one per origin, so "what the commits say" and "what was decided in conversation" can be asked
apart:

```jsonc
{ "observationScopes": "per_source" }
```

Each document consolidates into the global scope **plus** one named for each `source:` tag it
carries — `[[], ["source:chat"]]` for a session transcript, `[[], ["source:git"]]` for a commit
diff. Read an axis back with `tags: ["source:git"], tags_match: "exact"`, and the merged view with
`tags: [], tags_match: "exact"`.

A document carrying two `source:` tags gets a scope for each, and that is deliberate rather than
duplication. The commit-message seed is tagged `source:git` and `source:git-log`, so
`source:git-log` is fed only by the seed — what the commit _messages_ say — while `source:git` also
collects every per-commit diff under `gitIngest: "full"`. Two questions, two answers, each
deduplicated within itself by consolidation. A fact belonging to more than one axis is the point.

This cannot be expressed as a scope list. The server treats an explicit `list[list[str]]` as
unconditional — it is not filtered against the memory's own tags — so a configured
`[[], ["source:git"], ["source:chat"]]` writes every document into all three, and the `source:git`
scope fills with beliefs built from chat transcripts. Only a per-document decision separates them.

It costs one extra consolidation pass per document, and it reads only `source:`, so a volatile
provenance tag never becomes a scope. The global scope is still written first and unchanged, so the
untagged observations pages read are unaffected.

`"per_tag"` and `"all_combinations"` split further still, and an explicit `[["project:demo"], …]`
declares the scopes literally. `HINDSIGHT_OBSERVATION_SCOPES` sets the scalar modes; a scope list is
file-only. Changing this does not rewrite observations already consolidated under the old scoping —
they stay where they were built, and new work accrues under the new setting.

## Diagnostics & logging

All logs live in `~/.hindsight/coding-agents-logs/` (owner-only). Each file rotates to `<file>.1`
at 10 MB.

**Leveled plugin log** (humans debugging): `~/.hindsight/coding-agents-logs/plugin.log` (override
`HINDSIGHT_LOG_FILE`) — timestamped `LEVEL [scope] message` lines from every component, including
the ingestion engine. Level defaults to `info`; set `"logLevel": "debug"` in config or
`HINDSIGHT_LOG_LEVEL=debug` for ad-hoc debugging (at `debug`, every diag event below is mirrored
here too, so one file tells the whole story).

**Structured diag events** (machines/harnesses): every reflect and page-fetch outcome is appended
as a JSON line to `~/.hindsight/coding-agents-logs/diag.jsonl` (override with
`HINDSIGHT_DIAG_FILE`):

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
check this file — a run whose reflects failed is a no-memory run. When the failure was a timeout or
a 5xx, the hook falls back to page search and, if no page matches, to a raw recall of the
bank's memories: `reflect_fallback_pages` / `reflect_fallback_observations` record what each
step returned (`*_failed` when it errored). Seed starts are logged as
`seed_started`.

When `autoInject` names a retrieval source instead of reflect, that source records its own
outcome with the number of items it returned: `inject_pages` for `"pages"`, `inject_recall` for
`"recall"` (`inject_pages_failed` / `inject_recall_failed` when the call errored). No reflect
event is written on those turns — nothing reflected.

**Tool usage** (is the agent using Hindsight?): one JSON line per finished user turn in
`~/.hindsight/coding-agents-logs/usage.jsonl` (override `HINDSIGHT_USAGE_FILE`) — the `hindsight_*`
tools the agent called during that turn, and whether its reply credited Hindsight memory ("From
Hindsight memory"). The credit rate counts only turns that called a retrieval tool (search, list,
read, reflect); saving a document is not expected to be credited. Recorded at the end of each turn,
whether or not `retainSessions` writes the session back. It never leaves your machine. Summarize it per agent with:

```bash
npx @vectorize-io/hindsight-coding-agents stats
```

### Is the memory ready yet?

`hindsight_sync_status` — the agent-facing tool, `dist/status.js` for scripts — answers exactly
that: `"synced": true` means the seeded memory is queryable. It also reports gitlog freshness, how
far per-commit deepening has got, the codebase survey's state (`surveyBaseline` is the HEAD the last
survey started from, `surveyDocs` counts the findings documents that have landed, 0–4 — a baseline
with no findings retries automatically, at most three times per commit), and the extraction
operations still in flight. A survey that was skipped (no survey for this agent, or its CLI not
found), failed to launch or exited with an error leaves one `[survey]` line in the plugin log.

### Resetting a repo's memory

Delete its bank on the server. The bank is the **only** state this integration keeps, so the next
session in that repo is a true first open — seed and survey run again from scratch. There are no
client-side files to clean up.

Deleting a bank cannot be undone, so an agent never does it on its own: it names the repo and what
would be lost, and the owner deletes the bank themselves from the control plane.

### Marker documents you may notice

Two document ids exist for the machinery's own bookkeeping. Both are safe to ignore and safe to
delete:

- `survey-baseline:<sha>` — reads "🛰️ researching…" while a codebase survey runs and flips to
  "✅ completed" once its findings land. It is retained under the `survey` strategy, whose marker
  rule extracts **nothing** from a status marker, and it drives the re-survey cadence
  (`surveyRefreshCommits`) and `surveyBaseline` in sync status.
- `gitlog:<repo>` — the aggregated commit-message seed document, oldest commit first. Once seeded it
  is kept current by **appending** only the commits since the one its `gitlog-head:<sha>` tag
  names, so a new commit costs the extraction of its own message, not of the whole log. It is
  written in full again (bounded by `seedLimit`) only when an append cannot describe HEAD: the
  recorded commit is no longer an ancestor of HEAD (a rebase, a force-push), more than `seedLimit`
  commits are new, or the server cannot append safely.

### When memory seems to be missing

Failures never break the agent: a reflect, page fetch or retain that fails degrades to an ordinary
memoryless turn and is recorded in the logs. "No memory" is therefore a log question — check the
diag file for whether `session_start` and `deepen_started` ever fired for that bank. A session that
was already running when the plugin was installed has no SessionStart behind it; its first prompt
after the install self-heals.

### Calling the Hindsight API directly

To see what the server actually holds, call it with the URL and token the plugin uses: `apiUrl`
and `apiToken` from the config (in daemon mode, `http://127.0.0.1:<apiPort>`), and the bank id
from the session banner. A bank id like `coding-agent::my-repo` goes into the path as is.

```bash
API=https://api.hindsight.vectorize.io; BANK='coding-agent::my-repo'
AUTH="Authorization: Bearer <apiToken>"   # drop the -H "$AUTH" for a server without auth
curl -s -H "$AUTH" "$API/v1/default/banks/$BANK/stats"       # memory, document and pending counts
curl -s -H "$AUTH" "$API/v1/default/banks/$BANK/documents"   # what was retained (gitlog, survey, sessions)
curl -s -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"query": "retry policy"}' "$API/v1/default/banks/$BANK/memories/recall"
```

An empty bank, a 401, and a bank id that differs from the banner each point at a different fix:
the seed has not run yet, the token is wrong, or bank resolution picked another bank.
