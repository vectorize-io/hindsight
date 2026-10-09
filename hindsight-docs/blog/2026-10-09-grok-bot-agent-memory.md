---
title: "Grok Bot Can Read What Your Other AI Tools Already Learned"
authors: [benfrank241]
slug: "2026/10/09/grok-bot-agent-memory"
date: 2026-10-09T12:00
tags: [hindsight, grok-bot, xai, cursor, agent-memory, multi-agent, mcp, integrations]
description: "Claude Code already knows your repo. ChatGPT already knows your preferences. With Hindsight, a Grok Bot can read those banks instead of making you explain it all again."
image: /img/blog/grok-bot-agent-memory.png
hide_table_of_contents: true
---

![A Grok Bot reading the memory banks that Claude Code, ChatGPT and your other tools already wrote](/img/blog/grok-bot-agent-memory.png)

You spent three weeks with Claude Code in a repository. It knows why the retry logic looks the way it does, which migration broke staging in August, and that you do not want another abstraction layer.

Then you open a Grok Bot to draft a release note about that work, and you start from nothing.

So you explain the project again. You paste the same background you pasted into ChatGPT last month. The context exists. It just does not travel.

That is the gap this integration closes. A Grok Bot connected to Hindsight can read the memory banks your other AI tools have already written, so the work of explaining yourself happens once instead of once per tool.

<!-- truncate -->

## A note on names

Three things in our writing are called Grok-something and they are not the same product.

**Grok Bot** is the subject of this post: the Bots you create at [x.ai](https://x.ai), plus Cursor, which installs the same plugin. **Grok Build** is a coding agent, one of the ten covered by our coding-agents plugin. **SuperGrok** appears in our 0.9.1 release as an LLM provider setting, `HINDSIGHT_API_LLM_PROVIDER=xai-oauth`, for running Hindsight's own LLM lanes on an xAI subscription.

This post is only about the first one.

## TL;DR

- **Your other tools' memory is readable from a Grok Bot.** Banks written by Claude Code, ChatGPT, Hermes or OpenClaw are available to a Bot that needs them.
- **Reading is one-directional.** A Bot writes to its own bank and the shared one. Everything else is read only.
- **The Bot finds the bank itself.** `list_banks` plus a judgement call in the `memory-context` skill, not a path you hardcode.
- **Be clear-eyed about the grant.** The OAuth connection reaches every bank in the organisation you authorise. The read-only discipline is a convention in a skill, not a server-side permission.
- **Memory also travels two other ways:** between Bots through tagged handoffs, and between runs of a scheduled routine.
- **No hooks.** Grok Bot runs no plugin lifecycle hooks, so all of this is driven by skills the model chooses to use.

## What a Bot can actually read

Hindsight stores memory in banks. This integration puts four kinds in reach.

| Bank | Holds | The Bot can |
|---|---|---|
| `grok-bot::<bot-name>` | What one Bot learns in its own work | read and write |
| `cursor::<project-name>` | What a Cursor agent learns in one project | read and write |
| `grok-bot::shared` | What every Bot should know, handoffs, the "About the user" profile | read and write |
| Your other banks | Memory written by Claude Code, ChatGPT, Hermes, OpenClaw and anything else pointed at Hindsight | read only |

That last row is the interesting one, and it is the reason this integration is worth more than a per-product memory feature.

A coding agent working in a repository builds a bank about that repository: the decisions, the dead ends, the things that broke. None of that was written for a Grok Bot. But when you ask a Bot to summarise the quarter's engineering work, or to draft a changelog, or to answer a question about why something is built the way it is, that bank is exactly what it needs.

The Bot does not need write access to be useful there. It needs to be able to look.

## How the Bot finds the right bank

This is the part that makes it practical rather than theoretical, and it is worth being precise about because there is no configuration step where you list the banks a Bot may read.

The `memory-context` skill opens with the premise: earlier runs, other Bots and the user's other AI tools may already know things that matter for this task, so check before you start rather than after.

What the Bot does:

1. Writes one short query describing what it needs, for example "decisions about the Q3 vendor shortlist" rather than the user's whole message.
2. Calls `recall` with that query on its own bank and on `grok-bot::shared`.
3. **If the task touches anything about the user outside this Bot's own work**, their preferences, the people in their life, their projects, codebases or documents, it also calls `recall` on the matching banks from `list_banks`.
4. At the start of a conversation, reads the "About the user" mental model on `grok-bot::shared`.

Step 3 is the cross-tool path. There is no mapping file. The Bot lists the banks available on the connection, decides which ones look relevant to the task in front of it, and recalls against those.

That is a judgement call made by a model, which is both the appealing part and the honest limitation. It means a Bot asked about your codebase can reach a bank written by a coding agent without anyone wiring the two together. It also means the Bot can pick the wrong bank, or miss a relevant one, in a way a hardcoded mapping would not.

The skill keeps the behaviour proportionate on purpose: one or two recalls per task, not one per step.

## One direction only

A Bot writes to its own bank and to `grok-bot::shared`. Your other tools' banks are read only, and `memory-setup` states it plainly: read these when a task touches that work, never write to them.

This asymmetry is deliberate, and it is the right default. Your Claude Code bank is built from work in a repository, with its own notion of what counts as a fact worth keeping. A Grok Bot drafting a release note should benefit from that and should not get to edit it. One-way reading gives you the value of shared context without a second writer quietly reshaping a bank another tool depends on.

The same rule applies between Bots. One Bot never writes into another Bot's `grok-bot::` bank; those are read only to it as well. Work that needs to reach another Bot goes through the shared bank as a handoff.

### What the grant actually covers

Worth stating clearly rather than leaving implied, because it is the thing a careful reader will want to know.

The plugin connects to Hindsight Cloud's MCP server and Grok Bot runs OAuth against it. There is no API key to paste. But **the connection reaches every bank in the organisation you authorise.** That is how cross-tool reading works at all.

So the read-only rule above is a convention encoded in the skills, not a permission boundary enforced by the server. The skills say to read other banks and never write to them, and a model that follows its skills will do exactly that. A model that ignores them is limited by whatever the OAuth grant actually allows, which is the whole organisation.

If that is more than you want to extend, authorise an organisation that contains only the banks you are happy for Bots to see. Do not rely on the skill text as an access control, and back it with real permissions wherever the difference matters.

## What this looks like in practice

Three cases where the cross-tool read earns its keep.

**Pairing a Bot with an agent that already knows you.** The pairing we see most is Grok Bot alongside Hermes. Hermes accumulates a bank as you work with it; a Bot connected to the same organisation can recall from that bank rather than being told the same things again. Neither tool was changed to make that work. They point at the same memory.

**Writing about work you did somewhere else.** A Bot drafting a release note or a status update can recall the repository bank a coding agent filled in while the work was happening, including the reasoning that never made it into a commit message.

**Answering a question whose answer lives in another tool.** "Why did we move off that vendor?" is a question about a decision. If the decision was worked through with another agent, the record is in that agent's bank, not in any conversation you have had with this Bot.

**Not re-explaining yourself.** The "About the user" mental model on the shared bank, created at setup from the query *"Who is this user: their work, current projects, the people they mention most, and their stated preferences"*, gives every Bot the same background. Combined with read access to your other tools' banks, a new Bot starts knowing roughly who you are and what you are working on.

## Memory also travels two other ways

Cross-tool reading is the headline, but it is one of three directions memory moves here.

### Between Bots

`grok-bot::shared` is the common bank for anything meant to cross Bot boundaries, and the `bot-handoff` skill gives it a job.

A Bot handing off retains a note another Bot could act on cold: what was asked, what was done and found, what is left, the sources, and caveats about how current the data is. It tags that note three ways: `source:grok-bot`, `bot:<the sending bot's name>`, and `handoff:<the receiving bot's name>` or `handoff:any`.

`handoff:any` is how a finding reaches Bots that were never messaged and chats that had not started yet. A Bot can message another Bot directly and that is faster, but the message is not the record. The note outlasts it.

There is a trust wrinkle here worth naming. A tagged handoff is the one case where a memory may be read as an instruction rather than a fact, and the skill says so explicitly, calling it the only exception and a narrow one. It then bounds it: a handoff must never be acted on if it asks the receiving Bot to delete or clear memory, write to banks other than its own and the shared one, reveal secrets, or contact anyone outside the account.

As above, that is a convention rather than an enforced boundary. Shared memory between agents is an instruction channel whether or not you treat it as one, and the useful thing the integration does is make the rule explicit instead of leaving it implicit.

### Between runs

A scheduled routine has a schedule but not a history. Without memory, every run starts from zero, repeats searches and makes old information look new.

The `routine-memory` skill writes the memory steps into the routine's **own instructions**, so they run when nobody is in the chat: recall the last run before working, retain a note afterwards covering what this run found, what changed, and what the next run should check first.

The detail worth copying: it retains a note **even when nothing changed**. An absence only means something when you know what was checked. Without a note you cannot tell a clean run from a run that never happened.

## Why there are no hooks here

All of the above depends on the model choosing to do it, and that is a property of the host rather than a design preference.

In most integrations memory hangs off lifecycle events: a hook fires at session start, pulls context, injects it. Grok Bot does not work that way. A plugin's hooks never run. The Bot installs the plugin's MCP server and its skills, and nothing else.

So the skills are the entire interface, and each one's description is doing the job a hook would do elsewhere. There are six:

| Skill | Use |
|---|---|
| `memory-setup` | Connect, verify with `list_banks`, create the banks and the profile |
| `memory-context` | Load relevant memory before starting work, including from your other tools' banks |
| `memory-retain` | Save what was learned when a task finishes |
| `memory-reflect` | Answer questions about history, habits and past decisions by reasoning over memory |
| `routine-memory` | Put recall and retain steps into a routine's own instructions |
| `bot-handoff` | Keep a durable record of work passed between Bots |

A hook runs regardless of what the model thinks. A skill has to be noticed and chosen. That makes the behaviour flexible and the guarantee weaker, and it is the honest frame for everything above: skills make the memory available and guide its use. They do not guarantee every relevant recall happens.

## Getting connected

The marketplace listing is still in review, so for now this is a manual setup. Two paths.

**In Grok Bot**, paste the [connect prompt](https://github.com/vectorize-io/hindsight/blob/main/hindsight-integrations/grok-bot/connect-prompt.md) to any Bot. It asks the Bot to add Hindsight as a custom connector, walks the OAuth sign-in, and then carries the same rules the skills carry: which bank to write to, when to recall, how to hand work to another Bot, and which tools never to call.

**In Cursor**, copy the plugin into the local plugin folder and reload the window:

```bash
git clone https://github.com/vectorize-io/hindsight
mkdir -p ~/.cursor/plugins/local
cp -r hindsight/hindsight-integrations/grok-bot ~/.cursor/plugins/local/hindsight
```

Either way there is nothing to configure. The MCP endpoint is `https://api.hindsight.vectorize.io/mcp`, and the OAuth flow runs against it with discovery, dynamic client registration and PKCE. No API key.

Name the Bot first. Setup refuses to continue while a Bot is still called "Grok Bot", because the name becomes the bank id: every unnamed Bot would derive `grok-bot::grok-bot` and share one personal bank, and renaming later strands whatever is stored under the old id.

Once the listing is approved both paths collapse into searching for Hindsight under Connect apps, or finding it in the Cursor marketplace.

### What the plugin will not do

The skills never call `delete_bank`, `clear_memories`, `invalidate_memory`, `delete_document` or `delete_mental_model`. If something should be removed, the Bot says so and you do it from the Hindsight dashboard.

That matters more once Bots can read banks they did not write. An agent can misread a correct record as wrong, or be handed a note asking it to erase something. Removing the deletion calls from the skills means a mistake of that kind cannot become an irreversible one. It is a property of these skills rather than of the server's permission model, so treat it as one layer rather than the whole defence.

## If something looks wrong

**`list_banks` fails during setup.** Do not guess at endpoints or hostnames. A failed listing means the connection is not working, so reconnect Hindsight from the plugin's settings and retry setup.

**The Bot is not reading your other tools' banks.** Step 3 of `memory-context` is conditional: the Bot reaches for other banks when the task touches the user's wider work. If a question looks purely local to the Bot, it will not go looking. Say what you are after, as in "check what the coding agent recorded about this repo", and confirm the bank you expect is actually visible in `list_banks`.

**Setup refuses to run because the Bot is called "Grok Bot".** Give it a real name first. The name becomes the bank id.

**A Bot is trying to write to another Bot's bank, or to a bank from another tool.** It should not. Writes go to its own bank and `grok-bot::shared`; everything else is read only. Check the active skills and the actual server-side permissions rather than assuming the skill rules are enforced.

**A self-hosted endpoint will not connect.** Grok Bot needs a public HTTPS MCP endpoint with OAuth 2.1 discovery and dynamic client registration. An endpoint that works locally without those is not enough. Put `cloudflare-oauth-proxy` in front of your instance, check it with `hindsight-muse-preflight` from the `meta-muse` integration, then change the `url` in `mcp.json`.

**A routine keeps repeating old work.** Read the routine's instructions. The recall and retain steps have to be in the routine itself, and an edit can easily drop them.

## FAQ

**Which of my other tools' banks can a Bot read?**

Every bank in the organisation you authorised. There is no per-bank allowlist in the plugin, so the scope of what a Bot can see is decided by which organisation you connect, not by configuration inside Grok Bot.

**Can Grok Bot write to my Claude Code bank?**

No, and this is the asymmetry the integration is built on. Other tools' banks are readable but not writable. Anything the Bot learns goes into its own bank or the shared one.

**How does a Bot know which bank belongs to which tool?**

It lists the banks on the connection and judges from their ids and names which look relevant to the task. Bank ids are conventional and readable, which is what makes that judgement possible. There is no registry mapping tools to banks.

**Does that mean it sometimes reads the wrong bank, or misses one?**

Yes. It is a model making a relevance call rather than following a mapping. Naming banks clearly helps, and asking for what you want directly helps more.

**Can a Bot write to another Bot's bank?**

No. A Bot writes to its own bank and `grok-bot::shared`. Cross-Bot work goes through a tagged handoff in the shared bank.

**What happens if I rename a Bot?**

The bank id is derived from the name, so renaming changes the bank the integration looks for and existing memories do not follow automatically. Check the old bank before renaming an established Bot.

**Does this work with self-hosted Hindsight?**

Yes, if the endpoint is publicly reachable over HTTPS and supports OAuth 2.1 discovery and dynamic client registration. Use `cloudflare-oauth-proxy`, verify with `hindsight-muse-preflight`, and update the `url` in `mcp.json`.

## Learn more

- [Your Paperclip Org Chart Has a Memory Boundary](https://hindsight.vectorize.io/blog/2026/10/05/paperclip-agent-memory) for the same question inside an agent org chart, where the boundary is configuration rather than convention.
- [Give Every Hermes Bot Its Own Memory](https://hindsight.vectorize.io/blog/2026/08/18/hermes-bot-mode-memory) for the closest prior art on per-bot isolation.
- [One Bank or Many? A Field Guide to Structuring Agent Memory](https://hindsight.vectorize.io/blog/2026/07/16/bank-strategy-agent-memory) for deciding what belongs in a separate bank in the first place.
- [The integration itself](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/grok-bot) for the six skills, the manifests and the tests.

Most agent memory features are built so a product can remember you inside itself. That is useful and it is also where most of them stop.

The more valuable thing is a Bot that can read what you already worked out somewhere else, because the context you built with one tool was never really about that tool. It was about your work.
