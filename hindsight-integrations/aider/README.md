# hindsight-aider

Persistent long-term memory for [Aider](https://aider.chat), powered by [Hindsight](https://github.com/vectorize-io/hindsight).

`hindsight-aider` is a drop-in wrapper for the `aider` command. It **recalls
relevant project memory before each session** (injected into Aider's context via
a read-only file) and **retains the session transcript afterwards** — so every
Aider session starts with what you've learned before and saves what it learns
for next time. Memory is scoped **per git repo**.

## How it works

Aider has no MCP client or per-prompt hook, but it does load **read-only context
files** and writes a **chat-history file**. The wrapper uses both:

- **Recall (before):** queries Hindsight, writes the results to
  `.aider.hindsight-memory.md`, and launches `aider --read .aider.hindsight-memory.md …`
  so the memory is in context. If you run `aider -m "fix the auth bug"`, that
  message is used as the recall query; otherwise a general project-context query.
- **Retain (after):** when Aider exits, the wrapper reads the slice of the
  chat-history file written during the session and retains it to the repo's bank.

Recall is once per session (Aider can't be hooked mid-conversation), which fits
its session-oriented workflow.

## Install

```bash
pip install hindsight-aider          # also needs aider installed: pip install aider-chat
export HINDSIGHT_API_TOKEN=hsk_...    # omit for an open self-hosted server
```

## Usage

Use it exactly like `aider` — all arguments pass straight through:

```bash
hindsight-aider                         # interactive, project memory loaded
hindsight-aider -m "add retry logic"    # one-shot; recall uses the message
hindsight-aider src/app.py tests/       # any aider args
```

Use [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
with `HINDSIGHT_API_URL=http://localhost:8888`.

## Configuration

Settings come from `~/.hindsight/aider.json` or environment variables:

| Setting | Env var | Default |
| --- | --- | --- |
| API URL | `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` |
| API token | `HINDSIGHT_API_TOKEN` | _(none; required for Cloud)_ |
| Bank id | `HINDSIGHT_AIDER_BANK_ID` | _(git repo name)_ |
| Auto-recall | `HINDSIGHT_AIDER_AUTO_RECALL` | `true` |
| Auto-retain | `HINDSIGHT_AIDER_AUTO_RETAIN` | `true` |
| Aider command | `HINDSIGHT_AIDER_COMMAND` | `aider` |

The bank defaults to the git repo name so a project's memory is shared with the
other Hindsight editor integrations on the same repo.

## Development

```bash
uv sync
uv run pytest tests -v -m 'not requires_real_llm'   # deterministic suite
uv run pytest tests -v -m requires_real_llm          # gated live retain/recall
```

## License

MIT
