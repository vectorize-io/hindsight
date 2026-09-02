---
title: "Give Every Hermes Bot Its Own Memory"
authors: [benfrank241]
slug: "2026/08/18/hermes-bot-mode-memory"
date: 2026-08-18T12:00
tags: [hindsight, hermes, bot-mode, multi-agent, agent-memory, shared-memory]
description: "Nous just shipped Bot Mode for Hermes: named bots, each with its own memory. With Hindsight, that memory is persistent and precisely scoped per bot."
image: /img/blog/hermes-bot-mode-memory.png
hide_table_of_contents: true
---

![Hermes Bot Mode with Hindsight: give each named bot its own memory bank, or share one across a collaborating room](/img/blog/hermes-bot-mode-memory.png)

Nous Research just shipped [Bot Mode](https://x.com/NousResearch/status/2089429432612147572) for Hermes. Your agent profiles become a roster of named **Bots**, and each one carries its own role, model, skills, and profile picture. Bots can use any model and even talk to each other in a shared room. Build a specialist once, and use it forever.

The line in that announcement that matters most for us is a small one: **each Bot has its own memory.** That is exactly the seam where [Hindsight](https://vectorize.io/hindsight) fits, and where a roster of bots either becomes a genuine team or a set of goldfish with names.

<!-- truncate -->

## TL;DR

- **Bot Mode** turns Hermes agent profiles into named bots, each with its own role, model, skills, and **memory**. Rooms of 2 to 6 bots can collaborate.
- A bot is only as useful as what it remembers between sessions. Hermes already uses **Hindsight** as a native memory provider, so making that memory persistent is one setup command.
- The important choice is *scope*. In Hindsight, a **bank** is a recall boundary, so you decide it per bot:
  - **One bank per bot** → each specialist has private, isolated memory.
  - **One shared bank per room** → a team of bots that remembers together.
  - **Hybrid** → a private bank each, plus a shared team bank.
- Set it with a single field, `bank_id`, in the Hindsight provider config.

## What Bot Mode is

Bot Mode, shipped this week and now on by default in Hermes Desktop, lets you define a set of named bots instead of a single agent. Each bot is "a named teammate with its own memory, skills, and chat," and any bot can run on any model.

![Hermes Bot Mode: the New Agent dialog, creating a named bot with its own memory, skills, and chat](/img/blog/hermes-bot-mode-new-agent.png)

The interesting part is collaboration: you can put two to six bots in a room, where they take turns over a few serial rounds per message, respond when mentioned, or pass. A research bot, a coding bot, and an editor bot can now sit in one room and work a problem together.

It is all MIT licensed, same as Hermes and Hindsight.

## Why memory is the make-or-break

A roster of specialist bots sounds great until the specialists forget everything the moment a session ends. A research bot that cannot recall what it already found repeats itself. A coding bot that forgets your conventions relearns them every morning. And a *room* of bots has a second, subtler problem: you have to decide what they should and should not share. If the support bot can read the internal strategy bot's memory, you have a leak, not a feature.

So "each Bot has its own memory" is the right primitive, but persistence and scope are what make it real. That is precisely what Hindsight adds.

## The bank is the boundary

In Hindsight, memory lives in a **bank**, and a bank is a hard recall boundary: a bot can only recall what is in the bank it is pointed at. Hermes already ships a native Hindsight provider, so each bot's memory is auto-recalled before it answers and auto-retained after. The only decision left is which bank each bot uses, and that decision is the whole design:

- **One bank per bot.** Give each named bot its own `bank_id`. The research bot accumulates research memory, the support bot accumulates support memory, and neither can see the other. Isolation is enforced by storage, not by prompting.
- **One shared bank per room.** Point a collaborating room of bots at the same `bank_id`, and they remember *together*. What the research bot learns, the writing bot can recall on its next turn. The room becomes more than the sum of its bots because their context compounds.
- **Hybrid.** Give each bot a private bank for its own working memory, plus a shared team bank for the conclusions the room agrees on. This is the pattern that scales, and it is the same [bank-strategy](https://hindsight.vectorize.io/blog/2026/07/16/bank-strategy-agent-memory) our multi-agent users already rely on.

## Setup

If your Hermes already uses Hindsight, you are most of the way there. From a fresh install:

```bash
hermes memory setup      # choose "hindsight"
hermes memory status     # confirm the provider and bank
```

In Hermes Desktop, this lives in **Settings → Memory & Context**, and it is scoped per bot. The **Applies to** selector at the top picks which bot the settings belong to; set that bot's **Memory Provider** to Hindsight and give it its own **Bank ID**. That single field is the boundary: same bank id means shared memory, different bank ids mean isolated memory.

![Hermes Desktop Memory & Context settings, scoped per agent, with Hindsight as the memory provider and a per-bot Bank ID](/img/blog/hermes-bot-mode-memory-settings.png)

Prefer the terminal? The same settings live at `~/.hermes/hindsight/config.json` (the `bank_id` field, default `hermes`, or set `HINDSIGHT_BANK_ID`). Either way, auto-recall runs on a `pre_llm_call` hook and auto-retain on `post_llm_call`, and the provider recalls fresh memory per message even in gateway mode, so a bot stays coherent across platforms and across turns.

Memory can live in [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) or a server you run. Because both Hermes and Hindsight are MIT licensed and Hindsight self-hosts in one Docker command, a whole roster of bots with persistent memory can run entirely on your own hardware.

## Two setups worth copying

**A solo specialist that compounds.** Build a "release notes" bot once, give it its own bank, and every release it drafts adds to what it knows about your product's voice and history. Six months in, it is not a fresh model with a good prompt; it is a bot that has read everything it ever wrote.

**A room that thinks together.** Put a researcher, a coder, and a reviewer in one room on a shared bank. The researcher's findings land in memory, the coder recalls them while implementing, and the reviewer recalls both when it checks the work. No one re-explains anything, because the memory is the shared surface they all read and write.

## Why it fits

Bot Mode gave every bot a slot for memory. Hindsight fills it with something durable and scoped: persistent across sessions, isolated or shared by a single bank id, self-hostable, and already wired into Hermes as a native provider. The bank model maps almost exactly onto how you would draw a team of specialists on a whiteboard, which is the sign of a good primitive.

Spin up your roster, decide who shares a brain and who keeps their own, and let them get better every time they run. Start free on [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup), or read the [Hermes integration guide](https://hindsight.vectorize.io/sdks/integrations/hermes) first.

---

**Learn more:**
- [Hermes integration](https://hindsight.vectorize.io/sdks/integrations/hermes) — the native memory provider, setup, and config
- [One Bank or Many? Structuring Agent Memory](https://hindsight.vectorize.io/blog/2026/07/16/bank-strategy-agent-memory) — how to scope memory per bot or per room
- [Building multi-agent systems with shared memory](https://hindsight.vectorize.io/guides/2026/04/21/guide-building-multi-agent-systems-with-shared-memory) — isolation and sharing patterns that hold up
