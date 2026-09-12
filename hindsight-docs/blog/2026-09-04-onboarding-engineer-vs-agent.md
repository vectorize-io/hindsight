---
title: "Onboarding an Engineer vs. Onboarding an Agent"
authors: [benfrank241]
slug: "2026/09/04/onboarding-engineer-vs-agent"
date: 2026-09-04T14:00
tags: [agent-memory, onboarding, coding-agents, engineering-culture, knowledge]
description: "You already know how to get a new engineer productive: docs, pairing, and time. Agents need the same thing and get none of it, because every session is their first day."
image: /img/blog/onboarding-engineer-vs-agent.png
hide_table_of_contents: true
---

![Onboarding an engineer vs onboarding an agent: one accumulates context, the other starts over every session](/img/blog/onboarding-engineer-vs-agent.png)

A new engineer joins on Monday. By Friday they can ship something small. Within a month they're arguing about architecture, and somewhere around month three they start saying "we tried that, it didn't work."

Nobody thinks this is remarkable. It's just onboarding, and every team has a version of it.

Now think about how a coding agent joins your team. It reads the code, brilliantly, in seconds. And then the session ends and it forgets everything, and the next session it does the same thing again. It is on its first day, permanently.

<!-- truncate -->

## What onboarding actually transfers

It's worth being precise about what a new engineer learns, because it isn't one thing and the parts have very different sources.

**The code.** They read it. This is the part we spend the least time worrying about, because it's self-serve and the artifact is right there.

**The conventions.** Some are documented. Most are absorbed from review comments, from noticing what other files look like, from someone saying "we don't do it that way here" once.

**The decisions.** Why the retry policy skips that endpoint. Why rounding is half-up. Why the importer's tie-break looks backwards. This lives in git history, in pull request threads, and in the heads of people who were there.

**The dead ends.** What was tried and abandoned. This is the most valuable category and the least written down, because failures don't produce artifacts. Nobody merges a PR titled "this approach doesn't work."

An agent gets the first category for free and essentially none of the other three.

## Agents are better at reading and worse at remembering

The comparison is lopsided in both directions, which is what makes it interesting.

An agent reads a codebase faster than any human, doesn't get bored tracing a call graph, and holds more of it at once than a person can. On raw comprehension of what the code *does*, it wins outright, on day one, every time.

But a new engineer has one thing the agent doesn't: **the days accumulate**. Monday's conversation is available on Tuesday. The correction you gave in review persists. Nobody re-explains the deployment story every morning, because the person you explained it to on Wednesday is the same person on Thursday.

For an agent without memory, every day is Monday. And that's a strange asymmetry: the fastest reader on the team is also the only one who can never learn anything that isn't in the code.

## What we do to compensate, and why it doesn't hold

Faced with this, everyone reaches for the same two tools.

**We write it down.** `CLAUDE.md`, `AGENTS.md`, a conventions doc. This is real work and it helps. But it's a document someone has to remember to update, it captures only what someone thought to write at the moment they wrote it, and it goes stale silently, which means the agent trusts a description of a codebase that has moved on.

**We re-explain it.** Every session, we type the context again. It works, it's exhausting, and it doesn't scale past a couple of people. It's also the exact thing we'd consider a management failure if we did it to a human: if a new hire needed the same explanation every morning, you wouldn't blame them, you'd conclude something was broken about how they were being brought up to speed.

Neither is a discipline problem. Both are compensating for a missing capability.

## The thing onboarding actually is

Strip it down and onboarding is a knowledge transfer problem with three properties.

It's **incremental** — you don't learn it in one sitting, you accumulate it.

It's **contextual** — you learn the thing you need when the situation calls for it, not all of it up front.

And it's **derived from history** — most of what a senior engineer knows about a codebase came from watching what happened to it, not from reading a document about it.

That's a useful description because it tells you what a fix has to look like. Not a bigger document. Something that accumulates over time, surfaces the relevant part when the situation calls for it, and is built from what actually happened rather than what someone remembered to record.

Which is a description of memory.

## What changes when the agent accumulates

Give an agent a memory bank on a repository and the shape of the relationship changes in a way that maps closely onto human onboarding.

**Corrections stick.** Tell it once that this endpoint isn't idempotent, and that's available next week. You've stopped repeating yourself, which is the single biggest tax of the current arrangement.

**Git history becomes context.** Commit messages and their reasoning are ingested continuously, so the "why" that lives in your history is available without anyone transcribing it into a document.

**It gets more useful over time.** This is the part that feels most like a colleague. A new engineer is a net cost in week one and an asset by month three. An agent without memory is exactly as useful on day 90 as on day 1, and that flatness is the real ceiling.

**It survives a change of tool.** Because the memory belongs to the repository rather than the agent, switching between agents doesn't reset anything. The closest human analogy would be a colleague whose knowledge of your codebase transfers intact to whoever replaces them, which is not something teams get to have.

## Where the analogy breaks

Two ways, and both matter.

An agent's memory is **more literal**. A person forgets the irrelevant and abstracts the rest into judgment. A memory system keeps things, which is why consolidation, scoping, and knowing what to ignore are engineering problems rather than emergent behaviour.

And an agent has **no social sense of the codebase**. It doesn't know that a module is politically sensitive, or that a particular refactor keeps getting proposed and rejected for reasons nobody writes down. It only knows what got captured.

So the goal isn't a synthetic colleague. It's narrower and more achievable: stop making the fastest reader on your team start from nothing every morning.

## Learn more

- [One Bank or Many? A Field Guide to Structuring Agent Memory](/blog/2026/07/16/bank-strategy-agent-memory) — how a project's memory is scoped
- [Four More Coding Agents That Remember Your Project](/blog/2026/09/02/coding-agents-050-four-new-harnesses) — the agents that share one project memory
- [Knowledge Pages for Coding Agents](/blog/2026/08/13/knowledge-pages-coding-agents) — the curated layer an agent reads first
