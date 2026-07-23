
# Aider

Persistent long-term memory for [Aider](https://aider.chat), powered by [Hindsight](https://vectorize.io/hindsight). `hindsight-aider` is a drop-in wrapper for the `aider` command: it **recalls relevant project memory before each session** (injected into Aider's context via a read-only file) and **retains the session transcript after** — so each Aider session starts with what you've learned and saves what it learns. Memory is scoped **per git repo**.

## How It Works

Aider has no MCP client or per-prompt hook, but it loads **read-only context files** and writes a **chat-history file**. The wrapper uses both:

- **Recall (before):** queries Hindsight, writes the results to `.aider.hindsight-memory.md`, and launches `aider --read .aider.hindsight-memory.md …` so the memory is in context. `aider -m "fix the auth bug"` uses that message as the recall query; otherwise a general project-context query.
- **Retain (after):** when Aider exits, the wrapper reads the slice of the chat-history file written during the session and retains it to the repo's bank.

Recall is once per session (Aider can't be hooked mid-conversation), which fits its session-oriented workflow.

## Setup

```bash
pip install hindsight-aider aider-chat
export HINDSIGHT_API_TOKEN=hsk_...
```

Use it exactly like `aider` — all arguments pass through:

```bash
hindsight-aider                       # interactive, project memory loaded
hindsight-aider -m "add retry logic"  # one-shot; recall uses the message
hindsight-aider src/app.py            # any aider args
```

Use a [Hindsight Cloud](https://hindsight.vectorize.io) key, or a self-hosted server with `HINDSIGHT_API_URL=http://localhost:8888`. The bank defaults to the git repo name, so a project's memory is shared with the other Hindsight editor integrations on the same repo.

See the [package README](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/aider) for full configuration options.
