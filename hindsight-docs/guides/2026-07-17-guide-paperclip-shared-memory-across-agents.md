---
title: "Guide: Shared Memory Across All Your Paperclip Agents"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, paperclip, multi-agent, memory]
description: "Give every Paperclip agent org-wide shared memory with Hindsight — install the plugin once and one agent's lessons become available to all the others."
image: /img/guides/guide-paperclip-shared-memory-across-agents.svg
hide_table_of_contents: true
---

![Guide: Shared Memory Across All Your Paperclip Agents](/img/guides/guide-paperclip-shared-memory-across-agents.svg)

The reason to reach for **shared memory** in Paperclip is not that a single agent forgets — it is that every agent forgets *independently*. You install the `@vectorize-io/hindsight-paperclip` plugin once, and from then on every agent in your Paperclip instance gets automatic recall before each run and retain after each run. The interesting decision is not whether to add memory, but whether all your agents should draw from **one institutional bank** or keep separate ones.

This guide is about that decision. Because the plugin operates at the event layer — `agent.run.started` triggers recall, run output and comments trigger retain — you get to choose the *shape* of memory with a single config value. Point every agent at one shared bank and a lesson your triage agent learns is instantly available to your review agent and your deploy agent. Or scope memory per company and per agent so each stays in its own lane. Same plugin, same install, different bank strategy.

Below covers the "install once, all agents remember" model, when one shared bank beats per-agent banks, when isolation actually matters, and how automatic recall and retain behave once many agents share the same memory. For the plumbing — installing the plugin and wiring credentials — the [setup guide](https://hindsight.vectorize.io/guides/2026/04/16/guide-paperclip-memory-with-hindsight) already covers it, and this guide links there rather than repeating it.

<!-- truncate -->

> **Quick answer**
>
> 1. Install the plugin once — see the [setup guide](https://hindsight.vectorize.io/guides/2026/04/16/guide-paperclip-memory-with-hindsight).
> 2. For shared memory, set `dynamicBankId` to `false` and give every agent the same `bankId`.
> 3. For isolation, leave `dynamicBankId` at `true` and pick a `bankGranularity`.
> 4. Recall-before and retain-after then run automatically for every agent, into whichever bank you chose.
> 5. Verify by having one agent store a fact and a *different* agent recall it.

## Install once, every agent remembers

The plugin is installed at the Paperclip-instance level, not per agent. Once `@vectorize-io/hindsight-paperclip` is in place, it subscribes to Paperclip's event system and applies to every agent uniformly:

- On `agent.run.started`, it fetches the issue and calls recall, caching the result in plugin state for that run.
- While the agent runs, it can call `hindsight_recall(query)` for targeted lookups and `hindsight_retain(content)` to store a decision immediately.
- On `issue.comment.created`, it retains the full comment body, attributed to the comment author when present and otherwise the issue assignee.
- With `autoRetain` at its default of `true`, run output is retained after every run.

You do not touch each agent's code. Add an agent to Paperclip tomorrow and it inherits the same recall-before / retain-after behavior automatically. The only question left is which bank each agent's memory flows into — and that is the config choice this guide is about.

## One shared bank vs per-agent banks

The plugin's bank behavior is driven by two fields (see the [integration reference](https://hindsight.vectorize.io/docs/integrations/paperclip) for the full table):

- **`dynamicBankId`** (default `true`) — derives the bank ID from `bankGranularity`, so different agents/companies land in different banks.
- **`bankId`** — a static bank used when `dynamicBankId` is `false`. Every agent sharing this value reads and writes the same memory bank.

That gives you two postures.

**One shared institutional bank.** Set `dynamicBankId` to `false` and give every agent the same `bankId`:

```
{bankId}   ← static shared bank (dynamicBankId = false)
```

Now what any one agent learns is written to a single bank, and every other agent recalls from it. Your triage agent notices a customer always wants Slack alerts before email; your notification agent recalls that preference on its next run without anyone re-encoding it. This is the org-wide institutional memory model: the fleet gets smarter as a whole, not one agent at a time.

**Per-agent (or per-company) banks.** Leave `dynamicBankId` at its default of `true` and let `bankGranularity` shape the key. The default is `["company", "agent"]`:

```
paperclip::{companyId}::{agentId}   ← default: per company + agent
paperclip::{companyId}              ← company granularity (shared across that company's agents)
paperclip::{agentId}                ← agent granularity (one agent's memory across companies)
```

`company` granularity is the interesting middle ground: agents within one company share memory, but companies stay isolated from each other — institutional memory *per tenant* rather than across your whole instance.

## When to isolate an agent

Shared memory is powerful, which is exactly why isolation should be a deliberate choice rather than an accident. Reach for per-agent or per-company banks when:

- **Tenants must not bleed together.** In a multi-tenant setup, one company's context recalled into another company's run is a data-isolation problem. Keep `companyId` in the granularity, or add `"user"` for per-user isolation:

  ```
  paperclip::{companyId}::{agentId}::user::{userId}   ← per-user (GDPR-friendly)
  ```

- **Agents have conflicting worldviews.** If a "conservative reviewer" agent and an "aggressive optimizer" agent share a bank, each pollutes the other's context with lessons that contradict its own role. Isolation keeps each agent's memory coherent with its job.
- **You are experimenting.** A new agent you are still tuning shouldn't write half-formed conclusions into the bank the whole fleet relies on. Give it its own bank until you trust it, then promote it into the shared one.

The rule of thumb: share by default when agents collaborate on the same domain; isolate when the boundary between them is a correctness or compliance boundary.

## Connect Paperclip to Hindsight

Installing the plugin, self-hosting or pointing at Hindsight Cloud, and wiring the API key are covered end to end in the existing setup guide. Follow it first, then come back here to choose your bank strategy:

- [Guide: Add Paperclip Memory with Hindsight](https://hindsight.vectorize.io/guides/2026/04/16/guide-paperclip-memory-with-hindsight)

Once the plugin is installed and configured, the only change needed for shared memory is the `dynamicBankId` / `bankId` setting in **Settings → Plugins → Hindsight Memory**.

## How automatic recall and retain behave across agents

With a shared bank, the recall-before / retain-after cycle becomes a fleet-wide feedback loop rather than a per-agent one:

- **Recall is fleet-wide read.** When any agent starts a run, its recall query pulls from everything every agent has written to the shared bank. A brand-new agent's very first run already benefits from months of accumulated fleet memory.
- **Retain is fleet-wide write.** Every run's output, every stored decision, and every issue comment flows into the same bank. Memory is keyed to the bank, not the run ID, so it accumulates rather than resetting each run.
- **Recall budget still applies per run.** `recallBudget` (`low`, `mid`, `high`) controls how thorough each recall is; it does not change *which* bank is read. A larger shared bank does not force slower recall — tune the budget to taste.
- **Attribution is preserved.** Even in a shared bank, retained facts carry their author (comment author or issue assignee), so recall can surface who established a given convention.

The practical consequence: agents stop rediscovering the same context independently. One agent's hard-won lesson becomes the whole fleet's starting knowledge on the next run.

## Verify that shared memory is working

The test that actually proves *shared* memory (not just per-agent memory) uses two different agents against one bank:

1. Configure a shared bank (`dynamicBankId` = `false`, same `bankId` on both agents).
2. Have **agent A** run and store a distinctive fact — for example, that deploys must wait for the on-call approval.
3. Trigger **agent B** (a different agent) on a related issue.
4. Confirm agent B's recall surfaces agent A's fact without you re-encoding it.

If agent B sees what agent A learned, cross-agent shared memory is live. If it does not, the two agents are almost certainly landing in different banks — check that both have `dynamicBankId` = `false` and the exact same `bankId`.

## Common mistakes

### Expecting sharing while `dynamicBankId` is `true`

The default granularity is `["company", "agent"]`, so each agent gets its own bank. Shared memory requires either a coarser granularity (like `company`) or a static `bankId` with `dynamicBankId` = `false`.

### Mismatched `bankId` values

A single typo means two agents write to two different banks and never see each other's memory. For a shared bank, every agent must have the identical `bankId` string.

### Sharing a bank across hostile tenants

Company-wide or instance-wide sharing is great for collaboration and wrong for isolation. If two companies must not see each other's data, keep `companyId` in the granularity.

### Verifying with one agent

A single agent recalling its own retained fact only proves per-agent memory. To prove *shared* memory, the writing agent and the reading agent must be different.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works identically — point `hindsightApiUrl` at your instance (for example `http://localhost:8888`). The setup guide covers both paths.

### Can I mix shared and isolated agents in one instance?

Yes. Bank behavior is driven by config, so you can point a core set of collaborating agents at a shared `bankId` while leaving other agents on the default per-agent granularity.

### Does a bigger shared bank slow recall down?

Not meaningfully. Recall thoroughness is governed by `recallBudget`, not bank size. Keep it at `mid` for a balance, or `low` for speed.

### How do I move from per-agent to shared later?

Switch `dynamicBankId` to `false` and set a common `bankId`. Existing per-agent banks stay where they are; new memory accumulates in the shared bank going forward.

## Next Steps

- Set up the plugin first with the [Paperclip setup guide](https://hindsight.vectorize.io/guides/2026/04/16/guide-paperclip-memory-with-hindsight)
- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Paperclip integration docs](https://hindsight.vectorize.io/docs/integrations/paperclip)
