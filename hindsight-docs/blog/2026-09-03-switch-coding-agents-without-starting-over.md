---
title: "When Your Coding Agent Goes Down, Don't Start Over"
authors: [benfrank241]
slug: "2026/09/03/switch-coding-agents-without-starting-over"
date: 2026-09-03T12:00
tags: [hindsight, coding-agents, agent-memory, claude-code, codex, resilience, outage]
description: "Provider outages force you onto a different coding agent, and the new one knows nothing about your project. It doesn't have to be that way: when memory belongs to the repository instead of the agent, switching costs you nothing."
image: /img/blog/switch-coding-agents-without-starting-over.png
hide_table_of_contents: true
---

![When your coding agent goes down, switch to another one without losing your project's memory](/img/blog/switch-coding-agents-without-starting-over.png)

Your agent goes down mid-afternoon. Maybe it's a provider outage, maybe you hit a rate limit, maybe the model is having a bad day on this particular problem. You have another agent installed, so you switch.

And then you spend the next twenty minutes typing what you already explained this morning. That we round half-up because finance asked. That this endpoint isn't idempotent, so don't retry it. That the importer's tie-break rule is deliberate and no, please don't "fix" it.

The outage cost you a few minutes of downtime. The context reset cost you the rest of the afternoon.

<!-- truncate -->

## The real cost of switching isn't the switch

Having a second agent installed is easy. Most people already do. What makes switching expensive is that everything the first agent learned about your project lives *inside* that agent.

Each one keeps its own session history in its own format and its own directory. Each has its own convention for project context: `CLAUDE.md`, `QWEN.md`, `AGENTS.md`. None of them reads the others'.

So the second agent starts exactly where the first one started on day one. It can read your code, which is a lot. What it can't read is everything that isn't in the code: the decisions, the constraints, the three things you tried last week that didn't work. That's the part you typed by hand, into a tool that is currently down.

This is the thing worth noticing. **You didn't lose the agent. You lost the context you'd built up inside it.** The agent is replaceable in seconds. The context isn't, because it only ever existed in one place.

## Memory that belongs to the repo, not the agent

`hindsight-coding-agents` attaches to **16 coding agents** — Claude Code, Codex CLI, Cursor CLI, GitHub Copilot CLI, opencode, Qwen Code, DeepAgents Dcode, pi, and the rest — and by default they all share one memory bank per repository, named `coding-agent::{gitProject}`.

Not one bank per agent. One bank per project.

That single default is what makes an outage boring. What Claude Code learned this morning is what Codex reads this afternoon, because the memory was never Claude Code's. It belongs to the repository, and every agent you point at that repository reads and writes the same bank.

You switch agents. You keep working.

## What actually carries over

| | Agent-owned memory | Repo-owned memory |
|---|---|---|
| **Switching agents** | Start from zero | Start where you left off |
| **Project conventions** | Re-explained by hand | Already in the bank |
| **Past sessions** | Locked in the agent that had them | Readable by all 16 |
| **Git history** | Whatever the agent re-reads | Ingested continuously |
| **A new agent's first session** | Useless until it learns | Useful immediately |
| **Two engineers, two agents** | Two private contexts | One shared project memory |

Concretely, the agent you switch *to* starts with:

- **The project's accumulated decisions** — the conventions and constraints that came out of past sessions, whichever agent those sessions happened in.
- **Knowledge pages** — the curated layer covering architecture, conventions, and in-flight initiatives, which future sessions read first.
- **Git history** — ingested continuously, so commit messages and their reasoning are already in the bank.
- **Past conversations from every agent**, not just the one you're using now.

Attribution survives the switch too. Each agent stamps its own name on what it retains, so you can still tell which agent learned what. Shared memory doesn't mean anonymous memory.

## What about the session that was in flight?

This is the honest question, and the answer depends a little on which agent dropped.

Session write-back isn't a single event at the end. **Plugin-style harnesses** — opencode, opencode 2, Kilo — upsert the transcript **every turn**, plus an idle flush that catches the reply a per-turn pass can't see. **Hook-style harnesses** — Claude Code, Codex, Qwen Code and most others — write the transcript when the session stops.

So a normal exit loses nothing. A hard crash mid-turn can lose the tail of that session, and I'd rather say so than pretend otherwise.

There's a second net underneath, though: the conversations on disk are imported in the background as you work. A transcript that never made it through a Stop hook is still a file on your machine, and it gets picked up. The window where something is genuinely gone is narrow, and it's the current turn rather than the afternoon.

## You can't install this during an outage

The catch is timing. Memory that only starts recording after the incident doesn't help you during it. The bank has to already have your project in it when you need to switch.

The good news is that setup is one command and no ongoing effort:

```bash
npx @vectorize-io/hindsight-coding-agents install all
```

That wires every agent it finds on the machine. There's no configuration step and no "start recording" button. Point it at a repo and git history and conversations flow into the bank in the background as you work.

Wire up the agents you *might* switch to, not just the one you use daily. The whole point is that the fallback already knows things when you reach for it.

## Outages are just the obvious case

An outage is the version of this that hurts loudest, but it isn't the common one. The same property pays off constantly:

**Rate limits.** You hit a cap at 4pm. Finish the work somewhere else instead of stopping.

**Picking the right tool for the task.** Agents genuinely differ. One plans large refactors well, another is faster on small edits, another runs headless in CI. Without shared memory, using three agents means maintaining three separate mental models of your project, by hand. With it, using the best one for each task costs nothing.

**Trying something new.** A new agent ships every few weeks. Evaluating one normally means it's useless for the first day while it learns your project. Sharing the bank means a new agent is useful in its first session.

**Teams.** Two engineers on the same repo using different agents are contributing to the same project memory instead of two private ones.

The pattern underneath all four is the same: **your agent should be a choice you can revisit, not a lock-in you accumulate context inside.** That only works if the context lives somewhere the agent doesn't own.

## The short version

Provider outages are going to keep happening. The useful question isn't how to avoid them; it's how much they cost when they do.

If your project's memory lives inside one agent, an outage costs you that agent *and* everything it knew. If memory belongs to the repository, an outage costs you a few keystrokes to start a different one.

## Learn more

- [Four More Coding Agents That Remember Your Project](/blog/2026/09/02/coding-agents-050-four-new-harnesses) — the 16 supported agents, and how each one wires in
- [One Bank or Many? A Field Guide to Structuring Agent Memory](/blog/2026/07/16/bank-strategy-agent-memory) — when to share a bank and when to split it
- [Knowledge Pages for Coding Agents](/blog/2026/08/13/knowledge-pages-coding-agents) — the curated layer every agent reads first
