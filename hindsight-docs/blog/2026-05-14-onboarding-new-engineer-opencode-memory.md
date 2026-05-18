---
title: "Onboarding a New Engineer Onto Five Months of OpenCode Memory"
description: "A new engineer joins a codebase the team has been working in for five months. With OpenCode + Hindsight, day one looks very different from the README."
authors: [benfrank241]
date: 2026-05-14T12:00
tags: [memory, agents, hindsight, opencode, onboarding, teams, use-case, tutorial]
image: /img/blog/opencode-onboarding-memory.png
hide_table_of_contents: true
---

![Onboarding a New Engineer Onto Five Months of OpenCode Memory](/img/blog/opencode-onboarding-memory.png)

Devon joined the team on Monday. The orders service has been Maya's project since January. The README is two paragraphs and three releases stale. The actual decisions, the things Maya knows that aren't written down anywhere, live in five months of Slack threads, a few abandoned Notion pages, and Maya's head.

If Maya is unavailable for a week, Devon is going to spend that week mostly stuck.

<!-- truncate -->

Except, since late January, Maya has been using [OpenCode](https://opencode.ai) with [the Hindsight plugin](/blog/2026/04/20/opencode-persistent-memory). Every session has been writing to a shared Hindsight memory bank. Devon points his own OpenCode at that same bank on day one.

The rest of the week looks different.

---

## The Setup

Maya joined in January. The orders-service rewrite has been her main work since week two. Late that month, the team installed `@vectorize-io/opencode-hindsight` and pointed every member's OpenCode at a shared memory bank with the bank ID `team-orders`. Five engineers have used it on and off. Maya has been the most consistent contributor.

The plugin captures conversations after idle, runs reflect periodically, and injects relevant memories into each session's system prompt. (The [integration tutorial](/blog/2026/04/20/opencode-persistent-memory) covers how this actually works.) For the past five months, every decision discussed in an OpenCode session, every dead end, every "we tried X and it broke because Y," has been written to that bank.

Devon's onboarding contract is the same as anyone's: ship a small change to the orders service in his first week. The difference is that his OpenCode session already knows things he doesn't.

---

## Day One: "Why Postgres for the Orders Service?"

The README says, in one line, "We use Postgres." That's it. Devon, used to MongoDB from a previous job, asks his OpenCode why.

The agent surfaces a memory from January 31:

> *Maya, week 2: Tried MongoDB for the order-aggregate query that powers the merchant dashboard. The historical-order rollup required `$lookup` chains four levels deep, which hit MongoDB's pipeline depth at around 50,000 orders per merchant. Migration to Postgres + a materialized view took three days. Decision documented in session `j2n4p`.*

Devon spends ninety seconds reading. He now knows not just what database the team uses, but the specific class of query that ruled out the obvious alternative. He doesn't bother Maya. If he ever proposes "let's revisit Mongo for the new feature," he'll already know the failure mode he'd have to address.

This is the modal moment in onboarding. A question whose surface answer is a one-liner ("we use Postgres") and whose real answer is a two-paragraph story. The README has the one-liner. The story used to be in someone's head.

---

## Day Two: "Why Is `getCustomerOrders` Deprecated?"

There's a `@deprecated` tag on the function and no comment explaining why. Devon asks.

The agent recalls two memories: the original deprecation conversation from March 14 (an N+1 surfaced in load testing, replaced by `getCustomerOrdersBatch`), and a memory from March 22 where another engineer, Sam, had tried to revive the deprecated function for a quick fix and been corrected mid-conversation.

The first memory tells Devon why. The second memory tells Devon that he isn't the first person to wonder, and what happened the last time someone tried to bring it back. That second memory matters more than it sounds. Conventions that have been re-litigated and reaffirmed are stronger than conventions that exist by default. The agent treats Sam's correction as evidence that the deprecation is durable, not provisional.

---

## Day Three: A 2 a.m. Memory About the Deploy

Devon's first PR is small. A new field on the order schema, a thin endpoint, a one-line migration. He's ready to deploy.

He asks his OpenCode if there's anything special about deploying the orders service. The agent surfaces a memory from March 9:

> *2:14 a.m. incident: orders-service deploy failed because auth-service was on v0.6.1. Orders-service requires auth-service >= v0.6.2 for the new token format. The `ORDERS_REQUIRE_NEW_AUTH` config flag must be set after both services are deployed, not before. Captured during the postmortem session.*

Devon checks auth-service's version (0.6.4, fine), confirms the config flag will be set in the right order, and deploys. The deploy works. He never learns the lesson the hard way, because the agent learned it for the team in March.

The detail that matters here is the *story*, not the rule. "Set the flag after both deploys" by itself sounds like an arbitrary checklist item. With the incident attached, it becomes something Devon understands and can reason about if the situation drifts.

---

## Day Five: "We Don't Use Lodash Anymore"

Devon writes a small helper that reaches for `lodash.groupBy`. OpenCode interjects in the conversation. The team standardized on the native `Object.groupBy` (ES2024) in February, the Lodash dependency was removed in March, and three PRs since have re-introduced and removed it. The convention isn't in any style guide. It's not in the README. It's in the agent.

Devon swaps in the native helper. The PR goes up clean. The reviewer doesn't have to leave the same comment they've left three times in three months.

Of all the moments this week, this is the smallest, and the one that compounds the most. Conventions that the team has agreed on but never written down are exactly the kind of knowledge that drains away every time someone leaves. With memory, they don't.

---

## Can Multiple Engineers Share an AI Coding Agent's Memory?

Yes, when the agent is configured with a memory layer that supports shared memory banks. Hindsight organizes memories into memory banks, each addressed by a bank ID. Pointing every engineer's OpenCode session at the same bank ID means every session reads from and writes to the same store. Reflect runs against the combined memory, so synthesized observations cover the whole team's work, not one engineer's slice. Contradiction handling reconciles disagreements between authors as they happen.

In practice this is one configuration value in the OpenCode plugin. Same bank ID, same memory.

The easiest way to stand up a shared memory bank is [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup). Sign up, generate an API token, and have every team member set `HINDSIGHT_API_URL`, `HINDSIGHT_API_TOKEN`, and `HINDSIGHT_BANK_ID` to the same values. No infrastructure to run; the bank is shared from the first session.

---

## What Doesn't Belong in Shared Agent Memory

Not everything. Anything that needs to stay out of an LLM prompt path also stays out of agent memory: secrets, customer PII, regulatory-sensitive context, anything under NDA. Hindsight memory is for technical decisions, conventions, dead ends, and architecture context. It complements docs, ADRs, and runbooks. It doesn't replace them, and it doesn't try to.

A team using shared agent memory still needs:

- A README that orients
- ADRs for decisions that need formal review
- Runbooks for production operations
- An offboarding process that scrubs an engineer's individual context if they leave

The agent memory is the layer that captures everything else. The unglamorous middle of the documentation pyramid that almost no team writes down.

---

## What Devon's Week Looks Like Without It

The same five days, with everything serialized through Maya. The README orients him. Maya answers the Postgres question over coffee on Monday. The deprecated function gets explained over Slack on Tuesday and forgotten by Thursday. The deploy gotcha bites him at 11 p.m. on Wednesday, he reverses out, files a ticket, and asks Maya in the morning. Lodash makes it into the PR and gets caught by a teammate in code review. He ships his change by Friday.

Maya spends about six hours that week answering Devon's questions. Devon asks them well, but they're the same six hours she'd spend onboarding the next engineer in three months, and the next one after that.

The shared memory bank doesn't remove Maya from Devon's onboarding. It removes her from the questions that have already been answered.

---

## Conclusion

The artifact a new engineer needs is the structured record of decisions, conventions, and dead ends that the team has accumulated. Most teams don't have that artifact, because writing it down is expensive and it goes stale fast.

A team that has been using OpenCode + Hindsight for a few months has been writing it down as a side effect of doing their normal work. The README still exists. The ADRs still exist. The agent memory is the layer that captures what those don't, and on day one of a new engineer, it's the most useful documentation in the repo.

If you haven't set up the integration yet, [the OpenCode + Hindsight plugin tutorial](/blog/2026/04/20/opencode-persistent-memory) walks through it. Five minutes of configuration. Five months of compounding memory.

**Further reading:**
- [Your OpenCode Agent Forgets Everything Between Sessions](/blog/2026/04/20/opencode-persistent-memory) for the integration mechanics
- [Your Claude Code Subagents Don't Share What They Learn](/blog/2026/05/06/claude-code-subagents-shared-memory) on the shared-memory pattern in a different harness
- [The Missing Layer in Every Agent Harness](/blog/2026/05/04/agent-harness-needs-memory) on why this gap exists in the first place
- [What Is Agent Memory?](https://vectorize.io/what-is-agent-memory) for the foundational concepts
