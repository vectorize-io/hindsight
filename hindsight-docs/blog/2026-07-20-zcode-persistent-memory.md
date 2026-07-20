---
title: "Give ZCode a Long-Term Memory That Follows You Between Tools"
authors: [benfrank241]
slug: "2026/07/20/zcode-persistent-memory"
date: 2026-07-20T12:00
tags: [hindsight, zcode, glm, agent-memory, persistent-memory, coding-agent]
description: "ZCode is Z.ai's GLM desktop coding agent. Add Hindsight for persistent memory via plain Python hooks, no MCP, shared with Claude Code and Cursor."
image: /img/blog/zcode-persistent-memory.png
hide_table_of_contents: true
---

![ZCode with Hindsight: persistent memory that follows you across coding tools](/img/blog/zcode-persistent-memory.png)

[ZCode](https://zcode.z.ai) is Z.ai's GLM desktop coding agent, built on Z.ai's GLM models. It is fast, it drives a capable model, and like almost every coding agent it starts each session with a blank slate. Close the app, reopen it tomorrow, and it has forgotten that your API is FastAPI on asyncpg, that you test with pytest-asyncio, and that you decided last week not to reach for an ORM.

This post shows how to add persistent memory to ZCode with Hindsight, and it does it without an MCP server and without changing how you use ZCode. You install a small set of hooks, and from then on ZCode recalls the relevant context before each prompt and retains each turn after it finishes. Better still, the agent memory it builds is not trapped inside ZCode: it lives in a Hindsight bank that Claude Code and Cursor read from too, so your context follows you between tools.

<!-- truncate -->

## TL;DR

- **Persistent memory for ZCode with no MCP.** ZCode embeds the Claude Code agent runtime and ships a native process-hook system, so Hindsight plugs in as plain Python hooks. No server, no workflow changes.
- **Two hooks do the work:** recall before each prompt (injected as extra context), retain after each turn.
- **Install two ways:** as a ZCode plugin from the Hindsight marketplace (no pip), or with the `hindsight-zcode` CLI installer.
- **One bank, many tools:** the same Hindsight bank powers Claude Code and Cursor, so memory follows you between agents and machines.
- **Zero runtime dependencies:** the hook scripts are pure Python standard library.

## Why hooks, not MCP

Most memory integrations attach through the Model Context Protocol: you run a server alongside your agent and expose memory as tools the model can call. That works, but it is one more process to manage, and it puts memory in the model's hands rather than making it automatic.

ZCode gives us a cleaner path. Because it embeds the Claude Code agent runtime, it reads the standard Claude Code hook schema, just from its own config namespace at `~/.zcode/cli/config.json`. Hindsight uses that to wire memory in at the lifecycle level: recall and retain happen around every turn, automatically, with nothing running alongside ZCode. The hook scripts are pure Python standard library, so there is no runtime dependency to install and nothing to keep updated.

## Setup

First, get a key. [Sign up free for Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) so there is no daemon to run, or self-host if you prefer (more on that below).

There are two ways to install, and the first needs no `pip` at all.

**As a ZCode plugin.** The hooks ship as a hooks-only plugin in the Hindsight marketplace, so ZCode can install them directly:

```
zcode plugins add-marketplace vectorize-io/hindsight
zcode plugins install hindsight-zcode
```

Installed this way, ZCode registers the hooks automatically. Provide your credentials with environment variables (`HINDSIGHT_API_URL`, `HINDSIGHT_API_TOKEN`) or a small config file at `~/.hindsight/zcode.json`:

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "hsk_your_token"
}
```

**With the CLI installer.** If you prefer a command-line setup, the `hindsight-zcode` package ships a one-time installer:

```bash
pip install hindsight-zcode
hindsight-zcode install --api-url https://api.hindsight.vectorize.io --api-token your-api-key
```

The installer copies the hook scripts to `~/.zcode/hooks/hindsight/`, merges its entries into `~/.zcode/cli/config.json` (preserving any hooks you already have) and flips `hooks.enabled` to `true`, and seeds `~/.hindsight/zcode.json` for your personal config. It never touches your real Claude Code config at `~/.claude/settings.json`. Restart ZCode and memory is live. To back it all out, `hindsight-zcode uninstall` strips only Hindsight's entries and leaves everything else in place.

## How it works

ZCode fires three hook events, and Hindsight maps one job onto each:

| Hook | Event | What it does |
|------|-------|--------------|
| `session_start.py` | `SessionStart` | Confirms Hindsight is reachable and pre-warms the local daemon if you are self-hosting |
| `recall.py` | `UserPromptSubmit` | Queries your bank for relevant memories and injects them as context, then stashes the prompt |
| `retain.py` | `Stop` | Pairs that prompt with the assistant's reply and stores the turn |

On `UserPromptSubmit`, the recall hook reads your prompt, searches your Hindsight bank, and emits a context block that ZCode injects before the turn reaches the model. It shows up as additional context the model can see, not as clutter in your transcript:

```
<hindsight_memories>
Relevant memories from past conversations...
Current time - 2026-03-27 09:14

- Project uses FastAPI with asyncpg, not SQLAlchemy [world]
- Preferred testing framework: pytest with pytest-asyncio [experience]
</hindsight_memories>
```

On `Stop`, the retain hook takes the prompt it stashed, pairs it with the response, and posts the turn to Hindsight. There is a small design detail worth knowing here: ZCode has no `SessionEnd` event, so retention rides `Stop` instead. Every turn is stored as it completes, each as its own memory, rather than being batched up at the end of a session. In practice that means nothing is lost if you close the app abruptly.

## The part that makes this different: one memory, many tools

Here is where ZCode plus Hindsight goes beyond "an agent that remembers." The memory does not belong to ZCode. It belongs to a Hindsight bank, and that same bank is what Claude Code and Cursor read from and write to through their own Hindsight integrations.

Picture a normal week. You are in ZCode with GLM, and you tell it your team deploys on Fridays only in emergencies. Two days later you are in Cursor finishing a feature, and Cursor already knows the deploy rule, because it came from the same bank. Switch to Claude Code on your laptop at home, and the project's conventions are there too. The agent changes; the memory does not.

This works cleanly precisely because ZCode embeds the Claude Code runtime and speaks the same hook schema. There is nothing special to configure to get the shared behavior beyond pointing the tools at the same bank, which is the default.

## Per-project memory

By default every ZCode session shares one bank, named `zcode`. That is the right choice for personal preferences and cross-project habits. When you want a project's memory kept separate, turn on dynamic bank IDs:

```json
{
  "dynamicBankId": true
}
```

With that set, Hindsight derives a distinct bank per working directory, so one repo's decisions never bleed into another's. If you want to understand the tradeoffs between one shared bank and many scoped ones, we wrote a whole field guide on [structuring agent memory](/blog/2026/07/16/bank-strategy-agent-memory).

## Cloud or self-hosted

The recommended path is Hindsight Cloud: drop your API URL and token in `~/.hindsight/zcode.json` and you are done. If you would rather keep everything local, run the `hindsight-embed` daemon and leave `hindsightApiUrl` empty. The `session_start.py` hook detects it on port `9077`:

```bash
uvx hindsight-embed
```

The daemon is not started for you, so launch it separately before you open ZCode.

## It complements ZCode's built-in memory

ZCode already ships a local, per-project memory under `~/.zcode/cli/memories/`. Hindsight does not replace it; it fills a different gap. ZCode's built-in store is local to one project on one machine. A Hindsight bank is a shared, cloud or self-hosted store that spans tools and machines, so your context follows you from ZCode to Claude Code to Cursor and from your work laptop to your home desktop. Use both: the local store for project scratch, Hindsight for the memory you want everywhere.

## Frequently asked questions

**Does this replace ZCode's built-in memory?**
No. ZCode's built-in store stays local to one project on one machine. Hindsight adds a shared, cloud or self-hosted bank that spans tools and machines. Run both.

**Does it need an MCP server?**
No. The integration is plain Python hook scripts that call Hindsight's REST API, so there is nothing running alongside ZCode.

**Which tools share the memory?**
Any tool pointed at the same Hindsight bank. ZCode, Claude Code, and Cursor all read from and write to it, so a fact you teach in one is recalled in the others.

**Does it work offline?**
Yes. Run the `hindsight-embed` daemon locally, leave `hindsightApiUrl` empty, and the hooks connect to it on port `9077`. Nothing leaves your machine.

**Will it slow down ZCode?**
Recall runs once before each prompt and retain runs once after each turn. The hook scripts are pure standard library with no runtime dependencies, so the overhead is a single lightweight API call per turn.

## Try it

If you already use ZCode, the fastest way to feel the difference is to install the hooks, work for a session, then start a fresh one the next day and watch it pick up where you left off. [Grab a free Hindsight Cloud key](https://ui.hindsight.vectorize.io/signup), add the marketplace, and install the plugin.

## Further reading

- [ZCode integration reference](/sdks/integrations/zcode): every configuration option in one place.
- [One memory for every AI tool](/blog/2026/04/07/one-memory-for-every-ai-tool): the case for a shared bank across agents.
- [Inside retain()](/blog/2026/07/13/inside-retain-agent-memory): what Hindsight actually stores each time ZCode retains a turn.
