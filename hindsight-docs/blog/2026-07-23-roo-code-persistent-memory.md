---
title: "Give Roo Code a Memory So Every Task Builds on the Last"
authors: [benfrank241]
slug: "2026/07/23/roo-code-persistent-memory"
date: 2026-07-23T12:00
tags: [hindsight, roo-code, agent-memory, persistent-memory, coding-agent, mcp]
description: "Roo Code runs autonomous coding tasks but starts each one from zero. Add Hindsight agent memory: recall context before every task, retain a summary after."
image: /img/blog/roo-code-persistent-memory.png
hide_table_of_contents: true
---

![Roo Code with Hindsight: recall context before each task, retain a summary after](/img/blog/roo-code-persistent-memory.png)

[Roo Code](https://github.com/RooVetGit/Roo-Code) is one of the fastest-growing autonomous coding agents in VS Code. You hand it a task, it plans, edits, runs commands, and works through the steps on its own. It is very good at the task in front of it. It is also completely amnesiac between tasks: the conventions it learned yesterday, the architecture decision you explained last week, the bug pattern it already debugged once, all gone the next time you start a task.

Hindsight fixes that. Install it once, and Roo recalls the relevant context before each task and retains what it learned after, so every task builds on the last instead of starting from zero.

<!-- truncate -->

## TL;DR

- One command installs persistent memory into Roo Code: `pip install hindsight-roo-code` then `hindsight-roo-code install`.
- It uses **both** of Roo's extension points: Hindsight's **MCP tools** (`recall`, `retain`) plus a **custom rules file** that tells Roo when to call them.
- **Before each task** Roo recalls relevant memories; **after each task** it retains a summary. It can also retain decisions mid-task.
- Scope memory per project (`.roo/`) or globally (`~/.roo/`).
- Works with [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) (recommended, free to start) or a self-hosted server.

## Why a task-based agent needs memory most

Chat assistants at least keep a conversation going. An autonomous agent like Roo works in discrete tasks: you start one, it runs to completion, and the slate wipes. That structure is great for focus and terrible for continuity. Every task re-learns your stack from scratch, re-discovers the same gotchas, and re-asks things you already answered.

Agent memory is what turns a sequence of isolated tasks into an agent that actually knows your project. The learnings from task one become the starting context for task fifty, so the agent gets more useful the longer you work with it, not less.

## Setup

First, get a Hindsight instance. The easy path is [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup): sign up free, grab an API key, nothing to run. (Self-hosting instructions are below if you prefer.)

Then install the integration:

```bash
# Install the CLI
pip install hindsight-roo-code

# Install into your project (defaults to Hindsight Cloud)
hindsight-roo-code install

# Restart Roo Code, memory is active
```

That is the whole setup. Restart Roo Code and start a task.

## How it works: MCP tools plus rules

Roo Code has two extension points, and this integration uses both together, which is what makes it feel automatic rather than manual.

The **MCP server** gives Roo the capability. The installer writes `.roo/mcp.json`, registering Hindsight's built-in `/mcp` endpoint and exposing two tools:

| Tool | What it does |
|------|--------------|
| `recall` | Searches your memory for context relevant to the current task |
| `retain` | Stores a decision, discovery, or summary immediately |

Both are auto-approved (`alwaysAllow`), so Roo can use them without stopping to ask for permission on every call.

The **rules file** supplies the behavior. The installer also writes `.roo/rules/hindsight-memory.md`, which is injected into every Roo system prompt and tells the agent when to reach for those tools:

```
New task starts
  └─ Rules tell Roo to call recall
       └─ Relevant memories injected into context

Agent working…
  └─ retain stores decisions and discoveries as they happen

Task ends
  └─ Rules tell Roo to call retain with a summary
       └─ Summary stored for the next session
```

MCP alone would give Roo memory tools it might never use. Rules alone could not store or retrieve anything. Together, the capability and the instructions produce an agent that remembers by default: it pulls the right context in at the start of a task and writes the important parts back out at the end.

## Project memory or global memory

By default the installer writes to `.roo/` in your current project, so memory is scoped to that codebase, which is usually what you want for a coding agent. One project's conventions do not bleed into another's.

```bash
hindsight-roo-code install                       # this project only
hindsight-roo-code install --project-dir /path   # a specific project
hindsight-roo-code install --global              # ~/.roo, applies everywhere
```

Use the global install for cross-project habits and preferences, and project installs for the details that only make sense inside one repo. If you want the full picture on scoping memory, we wrote a [field guide on bank strategy](/blog/2026/07/16/bank-strategy-agent-memory).

## Cloud or self-hosted

For **Hindsight Cloud**, the installer defaults to `https://api.hindsight.vectorize.io` and your API key is all you need.

For a **self-hosted** server, start Hindsight first and point the installer at it:

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api                                   # serves http://localhost:8888

hindsight-roo-code install --api-url http://localhost:8888
```

The MCP entry it writes looks like this, so you can always inspect or edit it:

```json
{
  "mcpServers": {
    "hindsight": {
      "type": "streamable-http",
      "url": "http://localhost:8888/mcp",
      "timeout": 30,
      "alwaysAllow": ["recall", "retain"]
    }
  }
}
```

## Verifying it works

1. Start Hindsight (or use Cloud) and run the installer.
2. Open Roo Code in your project.
3. In **Settings → MCP Servers**, `hindsight` should show as connected.
4. Start a task and watch the tool-call log. You should see `recall` fire at the beginning.

Teach it something in one task ("this repo uses pnpm, not npm; tests live next to the source"), then start a fresh task the next day and watch it recall that on its own.

## The memory follows you

Because the memory lives in a Hindsight bank rather than inside Roo Code, it is not locked to one tool. Point another Hindsight integration at the same bank and the context carries over. A decision Roo retained is available to Claude Code, Cursor, or any other tool reading from that bank. Roo is where the task runs; the memory is what survives it.

## Frequently asked questions

**Does this replace Roo Code's context?**
No. Roo still assembles its own working context for a task. Hindsight adds a durable layer that persists across tasks and sessions, and injects the relevant slice at the start.

**Do I have to approve every memory call?**
No. `recall` and `retain` are auto-approved in the MCP config, so they run without interrupting the task.

**Is my memory scoped to one project?**
By default yes. The project install writes to `.roo/`, so each repo has its own memory. Use `--global` if you want shared memory across all projects.

**Does it work offline?**
Yes. Self-host Hindsight, point the installer at `http://localhost:8888`, and nothing leaves your machine.

## Try it

If you already use Roo Code, the fastest way to feel the difference is to install the integration, work through a couple of tasks, then start a fresh one tomorrow and watch it pick up where you left off. [Grab a free Hindsight Cloud key](https://ui.hindsight.vectorize.io/signup) and run `hindsight-roo-code install`.

## Further reading

- [Roo Code integration reference](/sdks/integrations/roo-code): every option in one place.
- [Inside retain()](/blog/2026/07/13/inside-retain-agent-memory): what Hindsight stores each time Roo retains a summary.
- [One memory for every AI tool](/blog/2026/04/07/one-memory-for-every-ai-tool): why the store lives outside any single agent.
