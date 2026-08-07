# hindsight-cline

Persistent long-term memory for [Cline](https://github.com/cline/cline) via [Hindsight](https://github.com/vectorize-io/hindsight) — **no MCP required**.

Cline's [lifecycle hooks](https://docs.cline.bot/customization/hooks) run small scripts at key moments. This integration installs hooks that automatically **recall** relevant memories before Cline works and **retain** what happened when a task ends. Memory persists across sessions, so Cline builds on past context instead of starting cold.

## What It Does

- **Before a task / each message** — the `TaskStart` and `UserPromptSubmit` hooks recall relevant memories from Hindsight and inject them as context.
- **When a task ends** — the `TaskComplete` (and `TaskCancel`) hook retains the task's prompts and summary for future sessions.
- **Deterministic** — because it runs on hooks, memory happens automatically; it doesn't depend on the model deciding to call a tool (no MCP).

## Prerequisites

> ✨ **Recommended:** [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) — sign up free, get an API key, and skip self-hosting.

**Self-hosting alternative:**

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api  # starts on http://localhost:8888
```

> **Platform:** Cline hooks run on **macOS and Linux only** (no Windows). Hooks need Python 3.

## Installation

```bash
pip install hindsight-cline
```

Then, from your project directory:

```bash
hindsight-cline install --api-url https://api.hindsight.vectorize.io --api-token YOUR_KEY
```

Install globally (applies to all projects):

```bash
hindsight-cline install --global --api-url https://api.hindsight.vectorize.io --api-token YOUR_KEY
```

### Cline CLI

The Cline **CLI** discovers hooks from a different directory than the VS Code extension, so add `--cli`:

```bash
hindsight-cline install --cli --api-url https://api.hindsight.vectorize.io --api-token YOUR_KEY
```

To remove it later: `hindsight-cline uninstall` (add `--global` and/or `--cli` to match how you installed).

This copies four hook scripts (`TaskStart`, `UserPromptSubmit`, `TaskComplete`, `TaskCancel`) plus their `lib/` and `settings.json` into the directory your Cline client reads:

| Client | Project install (default) | Global install (`--global`) |
| --- | --- | --- |
| VS Code extension | `.clinerules/hooks/` | `~/Documents/Cline/Rules/Hooks/` |
| CLI (`--cli`) | `.cline/hooks/` | `~/.cline/hooks/` |

Commit the project directory to share the hooks with your team.

**Final step:**
- **VS Code extension:** enable hooks in Settings → Features → Hooks.
- **CLI:** nothing — the CLI reads its hooks directory automatically (no toggle). To install the hooks somewhere else, point the CLI at them with `CLINE_HOOKS_DIR=<dir>` or `cline --hooks-dir <dir>`.

## How It Works

```
Task starts ─ TaskStart ─────────► recall(task description) → inject memories
You send a message ─ UserPromptSubmit ─► recall(prompt) → inject memories
                                          (and append the prompt to the task transcript)
Task completes ─ TaskComplete ──► retain(accumulated transcript + summary)
Task cancelled ─ TaskCancel ────► retain(partial transcript)
```

Cline doesn't hand hooks a conversation transcript, so the integration accumulates each task's prompts in `~/.hindsight/cline/state/` and retains them at task end. Memories land in a single bank (`cline` by default).

## Configuration

Defaults live in the installed `settings.json`; put personal overrides in `~/.hindsight/cline.json` (stable across reinstalls). Common keys:

| Setting             | Default                  | Description                                                       |
| ------------------- | ------------------------ | ----------------------------------------------------------------- |
| `hindsightApiUrl`   | (empty)                  | Hindsight server URL. Empty → use a local server on `apiPort`.    |
| `hindsightApiToken` | `null`                   | API key for Hindsight Cloud.                                      |
| `bankId`            | `cline`                  | Memory bank for this integration.                                 |
| `autoRecall`        | `true`                   | Inject memories before tasks/prompts.                             |
| `autoRetain`        | `true`                   | Retain the task transcript when it ends.                          |
| `recallBudget`      | `mid`                    | Recall depth: `low` / `mid` / `high`.                             |
| `recallTypes`       | `["world","experience"]` | Memory types to recall.                                           |
| `dynamicBankId`     | `false`                  | Separate bank per project/session (see `dynamicBankGranularity`). |
| `debug`             | `false`                  | Log to stderr.                                                    |

Every key can also be set via `HINDSIGHT_*` environment variables (e.g. `HINDSIGHT_BANK_ID`, `HINDSIGHT_AUTO_RECALL=false`).

## Verifying Setup

1. Start Hindsight (`hindsight-api` or Hindsight Cloud) and run `hindsight-cline install` with your URL/key (add `--cli` for the Cline CLI).
2. VS Code extension only: enable hooks in Cline (Settings → Features → Hooks). The CLI needs no toggle.
3. Start a task — recalled memories appear in context as a `<hindsight_memories>` block.
4. Complete a task, then check the `cline` bank (via the API or dashboard) — a memory should appear.

You can smoke-test a hook without Cline:

```bash
echo '{"hookName":"UserPromptSubmit","prompt":"how do we authenticate?","taskId":"t1","workspaceRoots":["/tmp/x"]}' \
  | .clinerules/hooks/UserPromptSubmit
# → {"cancel": false, "contextModification": "<hindsight_memories>…", "errorMessage": ""}
```

## Development

```bash
uv sync
uv run pytest tests/ -v
```
