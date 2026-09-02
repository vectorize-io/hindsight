---
title: "Four More Coding Agents That Remember Your Project"
authors: [benfrank241]
slug: "2026/09/02/coding-agents-050-four-new-harnesses"
date: 2026-09-02T12:00
tags: [hindsight, coding-agents, integrations, opencode, qwen-code, deepagents, pi, release]
description: "Coding Agents 0.5.0 adds pi, Qwen Code, DeepAgents Dcode, and opencode 2, taking the package to 16 agents that all share one memory per repository. Here's how coding agents work, and what changes when they remember."
image: /img/blog/coding-agents-050-four-harnesses.png
hide_table_of_contents: true
---

![Coding Agents 0.5.0 adds four new harnesses: pi, Qwen Code, DeepAgents Dcode, and opencode 2, all sharing one memory](/img/blog/coding-agents-050-four-harnesses.png)

There are a lot of coding agents now, and they're getting genuinely good. Every few weeks another one lands with a real point of view about how an agent should work: how it plans, what it's allowed to touch, whether it runs in your terminal or headless in CI.

`hindsight-coding-agents` 0.5.0 adds four of them — **pi**, **Qwen Code**, **DeepAgents Dcode**, and **opencode 2** — bringing the package to **16 supported agents**.

The reason to wire all sixteen isn't completeness. It's that they can share one memory.

<!-- truncate -->

## How a coding agent actually works

Strip away the interface and every one of these tools runs the same loop. You type a request. The agent reads files, greps for symbols, runs commands, forms a plan, and edits. When you close the session, that context is gone. Next time, it starts over.

That reset is the thing worth paying attention to, because of *what* gets lost.

The agent can always rebuild the mechanical part. It can re-read your code, re-derive the call graph, and work out what a function does, and modern agents are very good at this. What it cannot rebuild is the part that was never in the code to begin with.

Most of a real fix is derivable from the codebase. The **last mile** usually isn't. It hinges on a decision someone made once:

- Round half-up, because finance asked for it in a thread eight months ago.
- Don't retry that endpoint, because it isn't idempotent and we found out the hard way.
- When two records tie, prefer the older one, because of how the importer used to behave.

None of that is in the source. It's in git history, in pull request discussions, and in the conversations you already had with an agent last week. So every session, you re-explain it. And the agent, having no way to know better, confidently does the reasonable thing instead of the correct one.

## What changes when the agent remembers

`hindsight-coding-agents` attaches to the agent's own lifecycle. Nothing about how you work changes; the agent just stops starting cold.

**At session start**, the project's memory is seeded into context: what this repo is, its conventions, the decisions that already got made.

**On each prompt**, a recall runs against what you just asked and pulls back what's relevant to *that* task, rather than dumping everything it knows.

**During the session**, the agent has native `hindsight_*` tools, so it can go look something up on purpose. "Have we hit this before?" becomes a query rather than a guess.

**At the end**, the session is written back. What you decided this afternoon is what tomorrow's session starts from.

There's no setup step. Point it at a repo and its git history and conversations flow into a memory bank in the background as you work. On top of that sits a curated set of **knowledge pages** — architecture, conventions, in-flight initiatives — that future sessions read first.

## Sixteen agents, one memory

Here's the part that makes the count matter.

Agents working in the same repository **share one bank by default**, named `coding-agent::{gitProject}`. Not one bank per agent. One bank per project.

So what you tell pi is there when you open Prime Agent. A decision you made in Claude Code on Monday is available to Qwen Code on Thursday. If you try opencode 2 this week and go back to Codex next week, nothing is lost, because the memory was never the agent's in the first place — it belongs to the repo.

That turns switching agents from a cost into a free choice. Use the one whose planning you like for architecture work, the headless one in CI, the fast one for small edits. They're all reading and writing the same project memory.

Each agent still stamps its own name on everything it retains, so you can always see which one learned what.

## The four new agents

| Agent | Made by | Runs | Wires in as |
|---|---|---|---|
| **pi** | Earendil Works | Terminal | Extension, native tools |
| **Qwen Code** | Qwen | Terminal, headless | Hooks + MCP |
| **DeepAgents Dcode** | LangChain | Terminal, headless | Native Agent Plugin |
| **opencode 2** | opencode (beta) | Terminal | Plugin, native tools |

### pi

An extension entry in `~/.pi/agent/settings.json` plus a companion skill, with native tools and no MCP layer in between. **Prime Agent**, a fork of pi, is wired the same way in its own settings file — and per the rule above, the two share a repo's memory while staying individually attributable.

### Qwen Code

Qwen Code already has its own project-context convention: it reads `QWEN.md` files from your repo, so it starts with whatever you've written down by hand.

![Qwen Code at startup, reading three QWEN.md files](/img/blog/hindsight-coding-agents-qwen-code.png)

Hindsight adds the half you didn't write down. `QWEN.md` holds what someone remembered to document; the memory bank holds what actually happened — the decisions in git history and the conversations that produced them. Hooks land in `~/.qwen/settings.json`, alongside MCP and the companion skill.

### DeepAgents Dcode

LangChain's coding agent, and the one with the most opinionated plugin story: it has its own marketplace and plugin manager, and Hindsight installs as a **native Agent Plugin** through it rather than being bolted on. One `plugin.json` contributes the skill, the session lifecycle, and the `hindsight_*` tools together.

![DeepAgents Dcode v0.1.65](/img/blog/hindsight-coding-agents-dcode.png)

Dcode also runs headless (`dcode -n`), which is where project memory earns its keep: an agent running unattended in CI has nobody to ask, so the context has to already be there.

### opencode 2

The new opencode, currently in beta, shipping as `opencode2` alongside v1 rather than replacing it. It rewrote its plugin API from the ground up, so it's wired as a harness in its own right.

![opencode running locally](/img/blog/hindsight-coding-agents-opencode.png)

Both versions read the same config file and share a repo's bank, so you can try v2 without cutting yourself off from anything v1 already learned.

## Installing

One command, and it wires every agent it finds on the machine:

```bash
npx @vectorize-io/hindsight-coding-agents install all
```

![hindsight-coding-agents install all, wiring seven detected coding agents in 3.2 seconds](/img/blog/hindsight-coding-agents-install-all.png)

That output is worth a closer look, because it shows how little these hosts have in common. Seven agents, and **no two are wired the same way**: hooks merged into a JSON settings file, an extension registered, an MCP server added under user scope, a plugin patched into a YAML profile that applies to every profile on the machine.

A bare `install` with no target changes nothing and just prints the choices, so you can look before wiring anything. Naming one agent works too:

```bash
npx @vectorize-io/hindsight-coding-agents install qwen-code
```

Updating is the same command again.

## A note on what that takes

Behind the uniform install sits a genuinely varied set of host contracts, and a few of them are unforgiving in interesting ways.

Qwen Code is the best example. Its hook protocol matches Claude Code's field for field — same envelope, same output shape, same exit semantics — with one difference: **its timeouts are milliseconds, not seconds.** Write the usual `30` and you've registered a 30-millisecond hook. Worse, it looks fine in testing, because Qwen kills only the direct child process, so the orphaned work still finishes and memory still appears. The harness spec now carries an explicit `timeoutUnit`, and the tests fail if it's dropped.

That's the shape of most of this work. The install command is one line because the differences are absorbed here rather than by you.

## Also in 0.5.0

- **Per-source observation scoping**, so observations carry where they came from.
- **The installed runtime keeps itself current** instead of pinning whatever you first installed.
- **MCP tool safety annotations**, which some hosts require before they'll allow a call at all.
- **A failed git probe no longer forks a worktree into its own bank.**

0.5.1 followed with one fix worth having: the runtime auto-update wasn't firing. If you installed 0.5.0 on day one, update.

```bash
npm install -g @vectorize-io/hindsight-coding-agents@latest
```

## Learn more

- [Coding Agents changelog](https://hindsight.vectorize.io/changelog/integrations/coding-agents) — every release in full
- [Knowledge Pages for Coding Agents](/blog/2026/08/13/knowledge-pages-coding-agents) — the curated layer these agents read first
- [One Bank or Many? A Field Guide to Structuring Agent Memory](/blog/2026/07/16/bank-strategy-agent-memory) — how memory is scoped across repos and agents
