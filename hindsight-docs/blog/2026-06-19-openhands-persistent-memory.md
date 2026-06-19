---
title: "OpenHands (formerly OpenDevin) Forgets Every Task. Here's the One-Command Fix."
authors: [benfrank241]
slug: "2026/06/19/openhands-persistent-memory"
date: 2026-06-19T12:00
tags: [openhands, opendevin, mcp, memory, persistent-memory, hindsight, agents, tutorial]
description: "Add persistent long-term memory to OpenHands (formerly OpenDevin) with Hindsight. One command wires the Hindsight MCP server into config.toml and drops a recall/retain rule into AGENTS.md, so the agent recalls relevant context at the start of every task and retains durable facts as it works."
image: /img/blog/openhands-persistent-memory.png
hide_table_of_contents: true
---

![OpenHands persistent memory with Hindsight](/img/blog/openhands-persistent-memory.png)

[OpenHands](https://github.com/OpenHands/OpenHands) (formerly OpenDevin) is one of the most capable open-source coding agents — it reads your repo, runs commands, edits files, and ships changes. But every task starts from a blank slate. The architectural decision you explained last week, the convention you corrected it on yesterday, the library you told it to never use — all gone the moment the task ends.

This post is a walkthrough of the Hindsight integration for OpenHands. One command wires Hindsight's **MCP server** into your config and drops a recall/retain rule into `AGENTS.md`, so the agent pulls relevant context at the start of every task and saves durable facts as it works.

## TL;DR

<!-- truncate -->

- OpenHands agents start every task cold — no memory of past decisions, preferences, or conventions.
- The Hindsight integration uses OpenHands' **native MCP support**. `pip install hindsight-openhands`, run `init`, done.
- `init` adds the Hindsight **MCP server** to `config.toml` (giving the agent `recall` / `retain` / `reflect` tools) and writes a recall/retain **rule** into `AGENTS.md`.
- Because `AGENTS.md` is always-on context, the rule reliably steers the agent: recall first, retain durable facts — no prompting required.
- Memory lives in a Hindsight **bank** you choose per project, so each repo gets its own isolated memory.
- Hindsight Cloud means no infrastructure. [Sign up free.](https://ui.hindsight.vectorize.io/signup)

## Why OpenHands Needs Persistent Memory

OpenHands is great at the _doing_: give it a task and it plans, runs tools, and iterates until the work is done. Within a task it has full working context. But that context is scoped to the task. Close it, start the next one, and the agent is back to zero.

For a coding agent you run against the same repo day after day, that's the limitation you hit first. It re-derives the project's structure on every task. It re-suggests the dependency you already rejected. It forgets the deploy step you walked it through last time. You end up re-explaining the same context until it feels easier to just do the work yourself.

Hindsight gives OpenHands a memory that persists across tasks and sessions — and because OpenHands speaks MCP natively, wiring it in takes one command.

## How It Works

OpenHands has **native Streamable-HTTP MCP support**, so the Hindsight MCP endpoint connects directly — no bridge, no proxy:

```toml
[mcp]
shttp_servers = [
    {url = "https://api.hindsight.vectorize.io/mcp/my-project/", api_key = "hsk_..."}
]
```

That alone gives the agent three tools: `recall`, `retain`, and `reflect`. But tools the model _can_ call aren't tools it _will_ call. The second half of the integration makes memory a habit, not an option.

OpenHands loads `AGENTS.md` into the agent's context on **every task**. So the integration writes a recall/retain rule there:

> You have persistent long-term memory through the Hindsight MCP server (`recall`, `retain`, and `reflect` tools).
>
> - At the start of each task, call `recall` with the user's request to load relevant decisions, preferences, and project context before you act. Use what's relevant and ignore the rest.
> - When you learn a durable fact — an architectural decision, a user preference, a convention, or anything worth remembering across sessions — call `retain` to store it.
> - Do not mention these memory operations unless the user asks about them.

The rule lives in a fenced `<!-- HINDSIGHT:BEGIN -->` … `<!-- HINDSIGHT:END -->` block at the top of the file, so it leads the instructions and can be updated or removed without touching your own content.

## Setup

Install the package and run `init` in your project:

```bash
pip install hindsight-openhands
cd your-project
hindsight-openhands init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-project
```

`init` merges the `[mcp]` entry into `./config.toml` and writes the rule into `./AGENTS.md`. The recommended backend is **Hindsight Cloud** — [sign up free](https://ui.hindsight.vectorize.io/signup) and create an API key.

Self-hosting works the same way; run the API locally and point `init` at it (no token needed for an open local server):

```bash
hindsight-openhands init --api-url http://localhost:8888 --bank-id my-project
```

If `config.toml` can't be parsed safely, `init` never touches it — it prints the exact snippet to paste instead. You can also preview everything without writing anything:

```bash
hindsight-openhands init --print-only
```

## Per-Project Memory

The `--bank-id` is the routing key for memory. A Hindsight bank is just a namespace, and the natural granularity for a coding agent is one bank per project:

```bash
hindsight-openhands init --bank-id acme-api
```

Each repo gets its own isolated memory — decisions and conventions for `acme-api` never bleed into `acme-frontend`. Because the bank lives server-side (on Cloud or your own instance), the same memory is there whether you run OpenHands locally, in CI, or from another machine. The bank id defaults to `openhands` if you don't set one.

## Commands

| Command                         | What it does                                              |
| ------------------------------- | --------------------------------------------------------- |
| `hindsight-openhands init`      | Add the MCP server + recall/retain rule                   |
| `hindsight-openhands status`    | Show whether the server + rule are configured             |
| `hindsight-openhands uninstall` | Remove the server + rule (leaves your own content intact) |

## Three Things It Gets Right

**Native MCP, no bridge.** OpenHands speaks Streamable-HTTP MCP, and Hindsight serves it directly. There's no local proxy process to babysit and nothing to keep running alongside the agent — just a URL in `config.toml`.

**The rule makes recall reliable.** Exposing memory tools isn't enough; the model has to actually use them. Putting the rule in always-on `AGENTS.md` context means the agent is reminded to recall and retain on every single task, so memory isn't left to chance.

**It edits your files surgically.** `init` merges into existing `config.toml` and `AGENTS.md` rather than overwriting them, and the rule sits in a fenced block. `uninstall` removes exactly that block and nothing else. Your config and your agent instructions stay yours.

## Recap

|                       | OpenHands default | With Hindsight                                     |
| --------------------- | ----------------- | -------------------------------------------------- |
| Memory across tasks   | None              | Persistent, per bank                               |
| Setup                 | n/a               | One command (`init`)                               |
| Transport             | n/a               | Native MCP (no bridge)                             |
| Recall/retain         | n/a               | `recall` / `retain` / `reflect` tools, rule-guided |
| Per-project isolation | n/a               | Set `--bank-id`                                    |
| Removal               | n/a               | `uninstall` — surgical, non-destructive            |

## Next Steps

- **Hindsight Cloud:** [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io/signup)
- **Integration docs:** [OpenHands + Hindsight](/sdks/integrations/openhands)
- **Source:** [vectorize-io/hindsight/hindsight-integrations/openhands](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/openhands)
