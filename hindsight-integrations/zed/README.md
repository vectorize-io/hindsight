# hindsight-zed

Automatic, always-on long-term memory for the [Zed](https://zed.dev) editor's AI
assistant, powered by [Hindsight](https://github.com/vectorize-io/hindsight).

When you chat with Zed's Agent Panel, relevant memory from past sessions on that
project is injected automatically — no manual tool calls, no slash commands —
and your conversations are retained so the next session builds on them.

## How it works

Zed has no AI-conversation hook, but it **always includes a project's
instruction file** (`.rules` / `AGENTS.md` / …) in every agent conversation, and
it stores conversations in a local `threads.db`. `hindsight-zed` runs a small
background daemon that uses both:

- **Auto-recall (passive injection):** when a Zed conversation updates, the
  daemon recalls relevant memory for that project and writes it into a fenced
  `<!-- HINDSIGHT -->` block in the project's instruction file. Zed includes
  that file automatically, so memory "just shows up" on the next turn. The block
  is written into the file Zed actually reads — it never hijacks or overwrites
  your existing `AGENTS.md`/`CLAUDE.md`.
- **Auto-retain (passive capture):** the daemon reads finished/updated threads
  from `threads.db` and retains their transcripts into the project's bank.

Memory is **per-project** by default — each git repo gets its own Hindsight bank,
so context from one codebase doesn't leak into another.

## Install

```bash
pip install hindsight-zed
hindsight-zed init --api-token YOUR_HINDSIGHT_API_KEY
```

`init` writes config to `~/.hindsight/zed.json` and installs a background service
(launchd on macOS, systemd user service on Linux) that runs automatically. After
that it's hands-off — open any project in Zed and memory works.

Use a [Hindsight Cloud](https://hindsight.vectorize.io) key, or point at a
self-hosted server with `--api-url http://localhost:8888`.

To use one shared bank across all projects instead of per-project:

```bash
hindsight-zed init --api-token ... --fixed-bank-id my-memory
```

## Commands

| Command | Description |
| --- | --- |
| `hindsight-zed init` | One-time setup: config + background daemon |
| `hindsight-zed status` | Whether the daemon is running |
| `hindsight-zed uninstall` | Stop and remove the daemon |
| `hindsight-zed run` | Run the daemon in the foreground (used by the service) |

## Configuration

Settings layer (later wins): defaults → `~/.hindsight/zed.json` → environment
variables (`HINDSIGHT_API_URL`, `HINDSIGHT_API_TOKEN`,
`HINDSIGHT_ZED_FIXED_BANK_ID`, `HINDSIGHT_ZED_AUTO_RECALL`, …). See
`hindsight_zed/config.py` for the full list.

## On-demand tools (optional)

The passive daemon above is the headline feature. For explicit recall/retain in
a conversation you can also add Hindsight's MCP server to Zed's `context_servers`
(see Zed's [MCP docs](https://zed.dev/docs/ai/mcp)) pointing at
`<api-url>/mcp/<bank-id>/`.

## Limitation

Zed exposes no per-prompt hook, so injection is **periodic** (refreshed when a
conversation updates), not query-aware on the exact keystroke. In practice the
relevant project memory is present in context; it just isn't recomputed against
each individual message the instant you send it.
