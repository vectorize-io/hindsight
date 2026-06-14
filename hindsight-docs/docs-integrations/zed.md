---
sidebar_position: 7
title: "Zed Persistent Memory with Hindsight | Integration"
description: "Add automatic long-term memory to the Zed editor's AI assistant with Hindsight. Recalls relevant project memory into every conversation and retains sessions — no manual steps."
---

# Zed

Automatic, always-on long-term memory for the [Zed](https://zed.dev) editor's AI assistant, powered by [Hindsight](https://vectorize.io/hindsight). When you chat with Zed's Agent Panel, relevant memory from past sessions on that project is injected automatically — no manual tool calls — and your conversations are retained so the next session builds on them.


## How It Works

Zed has no AI-conversation hook, but it **always includes a project's instruction file** (`.rules` / `AGENTS.md` / …) in every agent conversation, and it stores conversations in a local `threads.db`. `hindsight-zed` runs a small background daemon that uses both:

- **Auto-recall (passive injection):** when a Zed conversation updates, the daemon recalls relevant memory for that project and writes it into a fenced `<!-- HINDSIGHT -->` block in the project's instruction file. Zed includes that file automatically, so memory "just shows up" on the next turn. The block is written into the file Zed actually reads — it never hijacks your existing `AGENTS.md`/`CLAUDE.md`.
- **Auto-retain (passive capture):** the daemon reads finished and updated threads from `threads.db` and retains their transcripts into the project's Hindsight bank.

Memory is **per-project** by default — each git repository gets its own bank, so context from one codebase doesn't leak into another.

## Setup

```bash
pip install hindsight-zed
hindsight-zed init --api-token YOUR_HINDSIGHT_API_KEY
```

`init` writes config to `~/.hindsight/zed.json` and installs a background service (launchd on macOS, systemd user service on Linux). After that it's hands-off — open any project in Zed and memory works.

Use a [Hindsight Cloud](https://hindsight.vectorize.io) key, or point at a self-hosted server with `--api-url http://localhost:8888`. To share one bank across all projects, pass `--fixed-bank-id my-memory`.

## Commands

| Command | Description |
| --- | --- |
| `hindsight-zed init` | One-time setup: config + background daemon |
| `hindsight-zed status` | Whether the daemon is running |
| `hindsight-zed uninstall` | Stop and remove the daemon |

## Limitation

Zed exposes no per-prompt hook, so injection is **periodic** (refreshed when a conversation updates) rather than recomputed against each individual keystroke. In practice the relevant project memory is present in context for every turn.

See the [package README](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/zed) for full configuration options.
