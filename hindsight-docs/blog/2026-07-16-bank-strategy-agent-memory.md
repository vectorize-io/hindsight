---
title: "One Bank or Many? A Field Guide to Structuring Agent Memory"
authors: [benfrank241]
slug: "2026/07/16/bank-strategy-agent-memory"
date: 2026-07-16T12:00
tags: [hindsight, agent-memory, banks, multi-tenancy, architecture, how-it-works]
description: "Every Hindsight integration tells you to set a bank_id, but not how to decide what a bank should be. A bank is a recall boundary: how to scope agent memory, and when to reach for tags instead."
image: /img/blog/bank-strategy-agent-memory.png
hide_table_of_contents: true
---

![One bank or many: how to scope agent memory in Hindsight](/img/blog/bank-strategy-agent-memory.png)

Every Hindsight integration tells you to set a `bank_id`. Almost none tell you how to decide what a bank should *be*. That decision looks trivial, but it shapes what your agent can actually recall. Scope it too wide and one user's memory bleeds into another's; scope it too narrow and the agent cannot reach the thing it needs, because that thing lives in a different bank it cannot see.

This is the schema-design decision for agent memory. Here is how to make it.

<!-- truncate -->

## TL;DR

- A **bank is a recall boundary**. `recall`, `retain`, and `reflect` all operate inside one bank, and there is no cross-bank query. So "should these two things share a bank?" really means "should a memory stored by A be recallable by B?"
- Use a **separate bank** for anything that is a hard isolation boundary (a tenant, a customer, an untrusted context). Use **tags inside one bank** for soft partitions you sometimes want to filter and sometimes want to cross-reference.
- Banks are created lazily on first use, so a new `bank_id` string is a new, empty memory. That is the failure mode behind most fragmentation: a bank per conversation means every conversation starts blank.
- Several integrations expose this as config: a static `bankId`, or a `dynamicBankId` composed from context fields like `user`, `project`, and `agent`.
- You can change strategy later, but recall history is scoped to the bank, so picking well up front saves a backfill.

## The one idea: a bank is a recall boundary

A bank in Hindsight is a complete, isolated store: the memories retained from conversations, the documents indexed for retrieval, the entities extracted from them, the knowledge graph connecting those entities, and the bank's own disposition and directives. Banks are isolated from each other, and there is no built-in query that spans banks. Every `recall`, `retain`, and `reflect` call names exactly one `bank_id` and stays inside it.

That single fact is the whole design tool. Instead of asking "how should I organize memory," ask one question per boundary:

> If A retains a memory, should B be able to recall it?

If **yes**, A and B belong in the same bank. If **no**, they belong in different banks. Every pattern below is just that question applied to a different A and B: two users, two projects, two agents, two channels.

One more mechanical detail matters. You do not pre-create banks. The first time you use a `bank_id`, Hindsight creates it with default settings. There is a real upside here (no provisioning step) and a real trap: a `bank_id` you have never written to is simply an empty bank, and a typo or an unstable id silently gives you a fresh, blank memory instead of an error. Bank identity is load-bearing.

## The main axis: what does one bank represent?

Here are the common strategies, each framed by the recall-boundary question, with the case where it bites.

| Strategy | One bank per... | Good for | Where it bites |
|---|---|---|---|
| **Global** | everything | a single-user tool, a solo dev assistant | the moment a second user shows up, their memories mix |
| **Per-user** | end user | SaaS products, personal assistants | one user with many projects gets all of them blurred together |
| **Per-project / per-repo** | codebase or workspace | coding agents, project work | a user's cross-project context does not follow them |
| **Per-agent** | agent role | multi-agent systems with distinct roles | agents that *should* pool context cannot |
| **Shared / team** | a group, deliberately | one memory across many surfaces or teammates | needs an explicit opt-in; it is not the default |

None of these is correct in the abstract. The right pick falls out of who your A and B are:

- **A consumer assistant** where each person's memory must never touch anyone else's: **per-user**. The user is the isolation boundary, so it is the bank boundary.
- **A coding agent** where the useful memory is about *this codebase* (its conventions, its decisions, its layout): **per-repo**. This is exactly what the Aider integration does by default, deriving the bank from the git repo name so every editor working on that repo shares one project memory.
- **A crew of specialized agents** (a planner, a researcher, a reviewer) that should each keep their own experience: **per-agent**. But the moment you want them to collaborate on shared context, point them at one **shared** bank instead.

## The second axis most people miss: tags inside a bank

Here is the mistake I see most: reaching for a new bank every time you want to separate two *kinds* of memory inside the same trust domain. You do not need a new bank for that. You need tags.

Hindsight lets you attach `tags` when you retain a memory, and filter by them when you recall or reflect. Tags are a within-bank partition, applied at query time, so the same bank can hold many labeled slices and you decide per query which slices to look at.

```python
# retain with a scope label
client.retain(bank_id="acme", content="We deploy on Fridays only in emergencies",
              tags=["project:web"])

# recall only within that slice
client.recall(bank_id="acme", query="deploy policy?",
              tags=["project:web"], tags_match="all")
```

The `tags_match` mode is the part worth knowing, because the default is friendlier than people expect:

- **`any`** (default): OR matching, and it *includes untagged memories*. Good when tags are hints, not walls.
- **`all`**: AND matching, still includes untagged memories.
- **`any_strict` / `all_strict`**: same matching, but exclude untagged memories.
- **`exact`**: the memory's tag set must equal the query's.

That default is the tell for when tags are the wrong tool. Because `any` includes untagged memories, tags are a *soft* partition: convenient for organizing, not a security control. If a memory absolutely must never surface in the wrong context, do not rely on a tag filter someone can forget to pass. Put it in its own bank.

So the real decision is two-level:

- **Hard isolation** (a tenant, a customer, an untrusted source, anything where a leak is a bug): **separate bank**. Isolation is enforced by the storage boundary, not by remembering to filter.
- **Soft partition** (organizing one trust domain into projects, topics, or users you sometimes want to filter and sometimes want to see together): **one bank, tags**.

A worked example. A single-tenant internal assistant serving one company can live in one bank, with `tags` like `project:web`, `project:billing`, and `team:sre`, so a question about billing can be scoped tight or left open to draw on everything. But a multi-tenant SaaS where each customer is a different company must give each customer its own bank. Tags organize; banks isolate.

Beyond tags, recall can also filter by fact `types` and by time (`created_after` / `created_before`, and a `query_timestamp` for asking "what did we know as of this date"), so a single bank stays queryable in more than one dimension. But tags are the primary knob for structuring what lives together.

## You do not have to hand-roll this

The strategies above are not just concepts you implement by hand. Several integrations expose bank scoping directly as configuration, and reading how they do it is the fastest way to internalize the model.

Some integrations take a single static bank id, which is the whole "shared bank" pattern in one line. In the "one memory, three surfaces" setup, OpenClaw is pinned to a static `bankId` and the Vapi webhook is constructed with the same `bank_id`, so a voice call and a coding session write to and read from one store.

Others derive the bank from context. The Claude Code integration has a `dynamicBankId` switch and a `dynamicBankGranularity` list that names which context fields to combine into the id, choosing from `agent`, `project`, `session`, `channel`, and `user`. Set it to `["user"]` and you get per-user banks; set it to `["agent", "project"]` and you get a bank per agent per project. Paperclip exposes the same idea as a `bankGranularity` defaulting to `["company", "agent"]`, so memory is scoped to an agent's role in a company rather than to a single run. And the oh-my-pi example spells the choice out as three named modes: `global`, `per-project`, and `per-project-tagged`, the last of which is exactly "one bank plus project tags."

`per-project-tagged` is worth pausing on, because it is the two axes combined into one recommendation: one bank for a trust domain, tags for the projects inside it. When you are unsure, that is usually the shape you want.

## Does one bank or many affect performance?

Less than people tend to assume, so it is rarely the right thing to optimize for. All memories live in a single table, with `bank_id` as a column on each row, rather than a separate table or database per bank. Splitting into many banks does not add provisioning, and putting everything in one does not create a monolithic structure of its own. Recall is scoped by a `bank_id` filter, and on the default Postgres backend each bank gets its own vector index, so one bank's search is largely insulated from how much another bank holds.

So the number of banks is usually not the lever that decides whether recall is fast. "One big bank so it scales" and "many small banks so each stays fast" are both weak reasons to pick a structure. Decide based on correctness, who should recall what, and treat performance as a separate question.

## Anti-patterns

**A bank per conversation, as your actual strategy.** Some integrations fall back to a conversation-scoped bank when nothing else is configured. That is a fine *default* and a bad *strategy*: because banks are created fresh on first use, every new conversation resolves to a new, empty bank, so the agent never remembers anything across sessions. The past memory is not deleted, it is just unreachable, because nothing points back to that conversation's id. If your agent "forgets everything between sessions," this is almost always why: check what the bank id actually resolves to.

**One global bank in a multi-tenant app.** The classic leak. It works perfectly in the demo with one user and becomes an incident with the second. Tenancy boundaries are bank boundaries, full stop.

**Over-fragmentation.** The opposite failure. A bank per (user × project × agent × session) feels tidy and starves recall: each bank holds so little that the agent almost never has enough context to be useful. Recall is only as good as what shares the bank. When in doubt, prefer fewer banks and lean on tags.

**Unstable bank ids.** Because a new id is a new empty bank, deriving the id from something that changes when it should not (an absolute path that differs per machine, a session token, a display name that gets edited) silently splits one memory into many. Derive bank ids from stable identities: a user id, a repo name, a tenant id.

## A decision checklist

Run each boundary through this in order:

1. **Is it a tenancy or security boundary?** (different customers, different untrusted sources) → **separate banks**, always. Do not negotiate this with tags.
2. **Does memory from A ever need to reach B?** No → separate banks. Yes → keep reading.
3. **Same trust domain, just organizational?** (projects, topics, teams inside one tenant) → **one bank, tags**.
4. **Should these agents collaborate on shared context?** → **one shared bank** they all point at.
5. **Whatever you pick, is the bank id stable?** Derive it from a durable identity, not from something incidental.

## You can change your mind, at a cost

Bank strategy is not a one-way door, but it is not free to reverse either. Because recall is scoped to a bank, splitting one bank into many (or merging many into one) means moving memories between banks and re-establishing history where you want it, not flipping a config flag. It is very doable, and much easier to do before you have a year of accumulated memory than after. Spend the ten minutes on the checklist now.

## Further reading

- [Inside retain()](/blog/2026/07/13/inside-retain-agent-memory): what actually gets stored each time you write to a bank.
- [One memory for every AI tool](/blog/2026/04/07/one-memory-for-every-ai-tool): pointing multiple agents at one shared bank.
- [Give every agent you run in Omnigent a persistent memory](/blog/2026/07/14/omnigent-persistent-memory): a bank per agent, with a conversation-scoped fallback.
