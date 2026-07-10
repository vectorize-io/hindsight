---
title: "One Memory, Three Surfaces: A Day With Cursor, OpenClaw, and Vapi"
authors: [benfrank241]
slug: "2026/07/10/shared-memory-across-ai-tools"
date: 2026-07-10T12:00
tags: [hindsight, agent-memory, cursor, openclaw, vapi, shared-memory, workflow]
description: "We pointed our code editor, our Slack agent, and our voice agent at one shared Hindsight memory. Here is a day in that workflow, and what actually changed."
image: /img/blog/shared-memory-across-ai-tools.png
hide_table_of_contents: true
---

![One Hindsight memory shared across Cursor, OpenClaw, and Vapi](/img/blog/shared-memory-across-ai-tools.png)

At 9:14 this morning, one of us told Cursor we were dropping the ORM and standardizing on a thin repository layer for data access. At 2:30, a teammate who was not in that conversation asked our Slack bot why a query was hand-written, and it explained the repository decision without anyone re-typing it. On the drive home, another engineer asked our voice agent "what did we land on for data access?" and got the same answer out loud.

Three different tools. Three different surfaces. One memory. Nobody repeated themselves, because none of these tools actually forgot.

<!-- truncate -->

## TL;DR

- We run three AI tools on the same project: [Cursor](/blog/2026/06/12/cursor-persistent-memory) for building, [OpenClaw](/blog/2026/05/27/openclaw-codebase-memory) as our Slack/Discord agent, and [Vapi](/blog/2026/06/03/vapi-persistent-memory) for voice.
- Each one has a Hindsight integration that recalls relevant memory before it answers and retains durable facts as it works.
- The trick is one line of config: **point all three at the same bank**. A fact retained by any tool is recalled by all of them.
- The result is not "three AI tools." It is one assistant that happens to have three front doors.

## The setup is one shared bank

Every Hindsight integration scopes its memory to a **bank**, a single isolated store. Inside a code repo, the editor integrations default the bank to the repo name, so they line up on their own. A chat bot and a voice agent are not "in a repo," so we do the obvious thing and give all three the same bank id.

```text
# Cursor: in the repo it uses the project bank by default
#   (or set an explicit bank id in the integration config)

# OpenClaw: it defaults to a per-conversation bank, so switch it
#   to one static shared store —  dynamicBankId: false, bankId: "acme-web"

# Vapi: pass the same bank when you build the webhook —
#   HindsightVapiWebhook(bank_id="acme-web")
```

The syntax differs per tool, but the move is identical: aim every surface at one bank. That is the whole idea. There is no bus to wire up, no sync job, no per-tool copy of the truth. Each tool reads and writes the one bank. When memories overlap, Hindsight consolidates them into durable [observations](/blog/2026/05/21/agent-memory-consolidation) rather than stacking three near-identical notes, so the shared store stays clean instead of turning into a pile. For the longer version of this argument, see [one memory for every AI tool](/blog/2026/04/07/one-memory-for-every-ai-tool). This post is about what it feels like to actually live in it.

## A day in the shared memory

**Morning, in Cursor.** The decision gets made where decisions get made: in the editor, mid-task. "We're dropping the ORM, everything goes through a repository." Cursor's agent applies the change, and because it retains durable facts as it works, the decision and the reasoning land in the bank. We did not run a memory command. We just made the call and kept coding.

**Afternoon, in Slack.** A teammate three time zones away opens a PR, sees a hand-written query, and asks our OpenClaw agent about it in a channel. OpenClaw recalls against their question, finds this morning's decision, and answers with the actual context: we moved to a repository layer, here is why, here is where it lives. The teammate was not in the editor and not in the standup. They inherited the decision anyway, from the same memory that Cursor wrote to.

**Evening, on a call.** Someone is away from their laptop and phones our Vapi voice agent to sanity-check an approach before tomorrow. "What did we decide about data access?" The voice agent recalls from the same bank and says it back. No screen, no search, no Slack scrollback. The context followed them off the keyboard entirely.

Three surfaces touched the same fact within one day, and the fact was written exactly once.

## What actually changed

We have run agents without shared memory for a long time, so the contrast is sharp.

- **We stopped re-explaining ourselves per tool.** Before, every surface was its own island: teach Cursor, separately teach the Slack bot, separately brief whoever was on the voice agent. Now the second and third tool already know.
- **Teammates inherit decisions they were not present for.** The person who asked in Slack never saw the Cursor session. The memory carried the decision to them, which is the part that quietly changes how a team works.
- **Context survives the surface switch, not just the session.** Leaving the editor used to mean leaving the context behind. Now moving from code to chat to voice keeps the thread, because the thread lives in the bank, not in any one app.
- **The tools disagree less.** When three assistants read from one consolidated memory, they tend to give one answer instead of three slightly different ones.

None of this required a new workflow. It required deleting one: the workflow of repeating yourself.

## Why it holds up as it grows

A shared memory is only useful if it stays fast and accurate as it fills with months of decisions across three noisy sources. Two things carry it. Consolidation keeps the store distilled, merging related facts about the same entity and resolving that "the data layer," "the repository change," and "the ORM removal" are the same decision rather than three. And retrieval has to stay sharp at volume, which is the regime [BEAM](/blog/2026/04/02/beam-sota) tests, at 10 million tokens where you cannot just stuff everything into context. Hindsight scores 64.1% there; the next-best published result is 40.6%. That headroom is what lets one bank back three tools without degrading into noise.

## Wire up your own

If you already run more than one AI tool on a project, this is a same-afternoon change:

1. Install the Hindsight integration for each tool you use. Each recalls before it answers and retains as it works.
2. Give them all the **same bank id**. In a repo, the editors align automatically; point the chat and voice agents at that same bank (for a per-conversation tool like OpenClaw, switch it to one static shared bank).
3. Point them at [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server.

Then make a decision in one tool and ask about it in another. The second tool already knows.

## Frequently asked questions

**Do the tools step on each other's memory?**
No. They read and write one bank, and consolidation merges overlapping facts into single observations instead of duplicating them. More writers make the memory richer, not messier.

**Does everything have to be in the same repo?**
No. The repo name is just the default bank for editor integrations. Any tool can be pointed at any bank id, which is exactly how a chat bot and a voice agent join a project's memory.

**Is this only for coding tools?**
No, and that is the point. The three surfaces here are an editor, a chat agent, and a voice agent. Anything with a Hindsight integration can share the bank.

**What about private or per-user context?**
Use separate banks for what should stay separate. Sharing is a choice you make by handing out the same bank id, not a default that leaks everything everywhere.

## Further reading

- [One memory for every AI tool I use](/blog/2026/04/07/one-memory-for-every-ai-tool): the concept behind the shared bank.
- [Cursor persistent memory](/blog/2026/06/12/cursor-persistent-memory), [OpenClaw codebase memory](/blog/2026/05/27/openclaw-codebase-memory), and [Vapi persistent memory](/blog/2026/06/03/vapi-persistent-memory): the three integrations from this post.
- [The consolidation problem in agent memory](/blog/2026/05/21/agent-memory-consolidation): how overlapping facts become one clean observation.
