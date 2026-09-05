---
title: "Stop Growing Your System Prompt"
authors: [benfrank241]
slug: "2026/09/04/stop-growing-your-system-prompt"
date: 2026-09-04T13:00
tags: [agent-memory, context, system-prompt, retrieval, rag, agents]
description: "The default fix for an agent that forgets is to put more in the prompt. It works until it doesn't, and the way it fails is quiet: unconditional context that costs every turn and gets less relevant as it grows."
image: /img/blog/stop-growing-your-system-prompt.png
hide_table_of_contents: true
---

![Stop growing your system prompt: unconditional context costs every turn and gets less relevant as it grows](/img/blog/stop-growing-your-system-prompt.png)

Every agent starts with a small system prompt. Then it gets something wrong, and you add a line. Then it gets something else wrong, and you add a paragraph. Six weeks later the prompt is two thousand tokens of accumulated correction, nobody remembers why half of it is there, and deleting any of it feels risky.

This is the most natural thing in the world to do, and for a while it's genuinely the right move. The question is what happens after that.

<!-- truncate -->

## Why it works at first

Putting something in the prompt has real advantages, and it's worth being honest about them before arguing against it.

It's **immediate**. You add a line, the behaviour changes on the next turn. No indexing, no pipeline, no waiting.

It's **certain**. The model definitely saw it. There's no retrieval step that might have missed, no ranking that might have buried it. For a rule you actually need enforced every time, that certainty is worth a lot.

It's **inspectable**. The prompt is right there in your code. You can read it, diff it, and reason about it.

None of that stops being true at scale. What changes is the cost.

## The cost is unconditional

A system prompt is paid on every turn, regardless of what the turn is about.

Two thousand tokens of accumulated instruction gets sent when the user asks you to rename a variable, and again when they ask for a migration plan, and again for a typo fix. The token cost is the obvious part and honestly the least interesting one. Two other things matter more.

**It crowds out the actual task.** Context is finite even when it's large, and everything spent on standing instruction is unavailable to the thing you're doing right now. In a long session, prompt bloat is competing directly with the code the agent needs to read.

**Relevance falls as size grows.** A model attending to twenty highly relevant lines behaves differently than one attending to two hundred lines of which twenty apply. You haven't just added noise around the signal; you've made the signal proportionally smaller. Adding a rule can genuinely make an existing rule less likely to be followed, which is a maddening thing to debug because the line you added is fine and the line that broke is untouched.

## It only ever grows

Here's the structural problem. Adding to a system prompt is trivially easy and deleting from it is frightening.

Adding takes a moment and has an obvious motivation: something just went wrong. Deleting requires knowing that a line is obsolete, which usually means knowing why it was added, when, and whether the situation that caused it still exists. Nobody knows that six months later.

So prompts ratchet. Every incident adds a line, no incident removes one, and the natural equilibrium is "as large as we can tolerate." That's not a discipline failure. It's what happens to any structure where one direction is cheap and the other is scary.

## What "put it in the prompt" is really doing

It helps to separate two things that get bundled together.

**Instructions** are things you want the agent to do, always, regardless of the task. Use this style. Never touch that directory. Ask before running migrations. These are genuinely unconditional, and the prompt is exactly the right place for them.

**Knowledge** is things that are true about your project. Why the retry policy excludes that endpoint. What was decided about rounding, and when. What someone tried last quarter that didn't work.

Knowledge is the part that grows without bound, and it's also the part that's almost never relevant to the current task. There is a lot of it, you need a small slice at a time, and which slice depends entirely on what you're doing right now.

Putting knowledge in the prompt means paying for all of it to get any of it.

## Retrieval is the other option

The alternative isn't "remember less." It's to keep the knowledge somewhere queryable and pull the relevant part per turn.

That's what a memory system does, and the design consequence worth understanding is **budget**. In Hindsight, a recall runs under an explicit budget rather than returning whatever it finds. At the default `fixed` setting, a `low` budget pulls 100 items per retrieval method per fact type, `mid` pulls 300, and `high` pulls 1000. There's an `adaptive` mode that sizes the budget as a ratio of the request's own `max_tokens` instead.

The point isn't the specific numbers. It's that the cost is **bounded and per-request** rather than fixed and permanent. A trivial turn can retrieve almost nothing. A hard architectural question can pull a lot. Neither one pays for the other.

The second consequence is that knowledge stops needing a curator. A system prompt only contains what someone thought to write down, at the moment they thought to write it. A memory bank built from git history and past conversations contains what actually happened, including the decisions nobody remembered to document because they were made in a pull request thread at 6pm.

## So what should the prompt hold?

Keep it. Make it the instruction layer.

- **Standing behaviour** you want on every single turn.
- **Hard constraints**, especially anything dangerous.
- **Style and format** rules the agent should never deviate from.

Move out anything that's really a record: decisions and their reasoning, architectural history, what was tried before, why a piece of code is shaped the way it is. Those belong somewhere retrievable, where they can be dated, attributed, and returned only when they bear on the question.

A good test: for each line in your prompt, ask whether it would still be worth sending if the current task were completely unrelated to it. If the answer is no, it's knowledge wearing an instruction's clothes.

## The honest limits

Retrieval isn't free and it isn't magic.

It can **miss**. A prompt line is guaranteed to be seen; a retrieved memory has to be found first. For anything where being ignored is unacceptable, the prompt is still the right answer, and that's precisely why the instruction layer shouldn't be emptied out.

It adds a **dependency**. Something has to be running, and something has to have been captured.

And it needs **time to become useful**. A memory bank on day one knows less than a well-written prompt. The crossover comes later, and the trade only pays if you're working on something long-lived.

For a script you'll run twice, put it in the prompt. For a codebase you'll be in for a year, the prompt will lose that race, and it will lose it slowly enough that you won't notice which line broke.

## Learn more

- [Knowledge Pages for Coding Agents](/blog/2026/08/13/knowledge-pages-coding-agents) — a curated knowledge layer that maintains itself
- [Cross-Encoder Reranking: The Last Stage of Agent Memory Recall](/blog/2026/08/28/cross-encoder-reranking-agent-memory) — how a bounded retrieval budget gets spent
- [Knowledge Graphs vs. Vector Search for Agent Memory](/blog/2026/08/24/knowledge-graphs-vs-vector-search-agent-memory) — why retrieval runs several arms at once
