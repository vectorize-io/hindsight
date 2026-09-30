---
title: "One Bank, Two Audiences, Two Briefs"
authors: [benfrank241]
slug: "2026/09/30/per-scope-consolidation-strategies"
date: 2026-09-30T14:00
tags: [hindsight, agent-memory, consolidation, observations, multi-tenant, privacy, deep-dive]
description: "Hindsight 0.10.2 lets each observation scope consolidate under its own mission. The company-wide view can record trends while a user's own scope keeps the detail, from the same facts."
image: /img/blog/per-scope-consolidation.png
hide_table_of_contents: true
---

![Per-scope consolidation: the same facts consolidated under different missions for different audiences](/img/blog/per-scope-consolidation.png)

In July we wrote a [field guide to structuring agent memory](https://hindsight.vectorize.io/blog/2026/07/16/bank-strategy-agent-memory), and framed the central decision as a trap with two jaws:

> Scope it too wide and one user's memory bleeds into another's; scope it too narrow and the agent cannot reach the thing it needs, because that thing lives in a different bank it cannot see.

Hindsight 0.10.2 loosens that. You can keep one bank, so everything stays reachable, and give each audience inside it a different brief.

<!-- truncate -->

## TL;DR

- **Each observation scope can now consolidate under its own mission**, set by `consolidation_strategies`.
- **Exactly four settings** are per-strategy. Everything else stays bank-wide.
- **First match wins, whole.** A later strategy never fills in an earlier one's gaps.
- **One real trap:** a strategy can never claim the untagged scope, and nothing in the docs says so.
- `observation_scope_limits` still works, but it matches differently, which makes a naive migration widen what you thought you were narrowing.

## What was already per-scope, and what wasn't

A quick recap, because the vocabulary matters.

An **observation scope** is the tag set that consolidation groups observations under. You choose the shape at retain time with `observation_scopes`: `combined` folds a memory's whole tag set into one pass, `per_tag` runs one pass per tag, `all_combinations` runs one per subset, and `shared` runs a single pass over the empty, untagged scope.

So a memory tagged `user:dana`, `team:exec`, `company:acme` can already build three separate sets of observations, one per audience. That part is old.

What you could not do was tell those audiences apart. The only per-scope setting was the observation cap, through `observation_scope_limits`, which mapped a scope pattern to an integer. The **mission was bank-wide.** Every scope got the same instructions about what to record.

The docs put the consequence plainly:

> Each scope builds its own observations. But with a single mission, the company-wide scope gets the same detail as Dana's own — names, deal sizes, anything said in confidence — just visible to more people.

That is a confidentiality problem wearing a configuration problem's clothes. The separation existed; the discretion did not.

## The same facts, two briefs

`consolidation_strategies` attaches a mission to a scope pattern:

```json
[
  {
    "scopes": [{"tags": ["company:*"]}],
    "observations_mission": "This scope is shared with the whole company. Record only general, industry-level trends. Never name a specific company, person, deal size or funding stage.",
    "max_observations_per_scope": 20
  },
  {
    "scopes": [{"tags": ["team:*"]}],
    "observations_mission": "Record decisions the team must act on."
  }
]
```

The result, from the same underlying facts:

> Dana's scope keeps "Acme Robotics signed a 3-year lease for a 4,000 GPU cluster", while the company scope gets "Companies are leasing GPU clusters to run open-weight models".

Nothing was redacted after the fact. The company scope was never told the details in the first place, because each scope gets its own consolidation call with its own mission. One scope's settings never reach another's call.

Set it per bank with `consolidation_strategies`, globally with `HINDSIGHT_API_CONSOLIDATION_STRATEGIES`, or in the control plane under Bank, Configuration, Observations.

## Exactly four things a strategy can change

This is worth stating precisely, because a feature that looks like "override anything per scope" is not what shipped. A strategy may set four fields:

| Field | What it does |
|---|---|
| `observations_mission` | The brief for this audience |
| `max_observations_per_scope` | How many observations the scope may hold |
| `consolidation_source_facts_max_tokens` | How much evidence each call sees |
| `consolidation_source_facts_max_tokens_per_observation` | Per-observation share of that budget |

The comment above the list explains the rule behind it: these are kept to "what actually varies per audience: the brief, how many observations the scope may hold, and how much source evidence each consolidation call is shown. Everything else stays bank-wide."

Notably **not** per-strategy: the semantic dedup threshold. It stays at the bank-wide 0.97. That turns out to be fine, because dedup only ever compares observations within the same scope, so a redacted company observation is never a merge candidate against Dana's detailed one. The two features compose without being aware of each other.

## First match wins, whole

Patterns are globs. Within one pattern, every glob must match some tag. Across a strategy's `scopes`, any one matching claims it. Across strategies, the first one in list order that claims a scope takes it entirely.

That last part is stricter than it sounds, and deliberately so:

> The first strategy in list order that claims the scope wins, whole. When two strategies both claim a scope (`["company:*"]` and `["company:acme"]`, say), only the earlier one applies — its settings, and for anything it leaves unset, the bank-wide value. A later strategy never fills in the earlier one's gaps.

So a list of `[{company:acme, cap 5}, {company:*, mission X}]` gives `company:acme` a cap of 5 and the **bank-wide** mission, not mission X. Order your specific rules before your general ones, and expect them to be complete.

The commit history records why it works this way. An earlier version resolved each setting independently, taking the mission from the first strategy that set one and the cap from the first that set one:

> That let two strategies silently blend on one scope, so nobody could tell which strategy a scope was actually using; with one winner, the control plane can show it.

Debuggability beat flexibility. That is usually the right call for configuration that decides what an agent is allowed to remember about people.

There are two matching modes. The default, `"all"`, is containment: the scope has all the pattern's tags and may have more. `"exact"` requires an exact cover. There is no `"any"`, because a strategy's patterns are already alternatives, so "any of these tags" is written as one pattern per tag.

## The trap: your global scope cannot be claimed

Here is the thing we did not document, and should have.

**A strategy can never claim the untagged scope.** Both matchers refuse an empty tag set outright. The containment matcher is literally:

```python
return bool(tags) and all(any(fnmatchcase(t, g) for t in tags) for g in globs)
```

That leading `bool(tags)` means no pattern matches the global scope. Not `["company:*"]`, not a bare `["*"]`, nothing. The docstring acknowledges it in passing — "the untagged scope never matches" — and the test suite asserts it. But the strategy section of the documentation never mentions it at all.

This matters more than it first appears, because `shared` is the documented remedy for volatile tags. If your retain calls carry per-session ids, the advice is to consolidate under `shared` so those sessions fold together instead of each spawning its own scope. Take that advice and you now have a scope that is permanently on bank defaults, that no strategy can reach, and nothing tells you.

If you need a differently-briefed catch-all, give it a real tag and target that, rather than relying on `shared`.

## Migrating from `observation_scope_limits`

The old field is deprecated but genuinely still honoured, as a fallback for the cap only. The precedence runs: the winning strategy's cap if it set one, then `observation_scope_limits`, then the bank default.

Two consequences that are easy to get wrong.

A mission-only strategy does not take the cap with it. If a strategy claims a scope but sets no `max_observations_per_scope`, the cap still comes from `observation_scope_limits` — and it falls back **past** any later strategy to get there. There is a test named exactly that: `test_the_winners_missing_cap_falls_back_past_later_strategies`.

And the two fields do not match the same way. `observation_scope_limits` is exact-cover only. Strategies default to containment. Copy a rule across verbatim and it will match **more** scopes than it used to, silently, unless you add `"tags_match": "exact"`. Nothing currently flags this as a migration step, which is why it is here.

Worth knowing too: the deprecation is prose only. There is no runtime warning, so nothing will appear in your logs to prompt the move.

## Other limits worth knowing

**Negation is not expressible.** You cannot write "has a company tag but no user tag". Patterns are positive only.

**On Oracle, strategies work but reconciliation does not.** Semantic dedup uses Postgres-only SQL, so it is skipped on Oracle regardless of the configured threshold. Since the cap is what stops a scope filling with near-duplicates, an Oracle deployment leans harder on `max_observations_per_scope`.

**Validation depends on which door you use.** Saved through the API or the control plane, a malformed strategy is rejected: the schema forbids unknown keys, so a `"scope"` typo for `"scopes"` is a 400 rather than a rule that silently never fires. Set through the environment variable, there is no such check. Entries parse leniently and bad ones are dropped without complaint, and an unrecognised `tags_match` quietly becomes `"all"` rather than erroring. If you configure by env, check the control plane afterwards to confirm the strategy is actually claiming what you think.

**Nothing marks an observation with the strategy that produced it.** Recall across scopes and you get observations written under different missions side by side, distinguished only by the scope tags they carry. The scopes endpoint enumerates scopes with counts, and the preview shows which strategy owns which scope, but that is a question you ask of the configuration rather than of a result.

## Where this leaves the one-bank question

The July post's advice was to choose your bank boundary carefully because it is a recall boundary, and it still is. What changes is that the boundary no longer has to carry the whole burden of confidentiality.

A bank is where recall can reach. A scope is now where a brief applies. You can let an agent reach across a whole organisation's memory while still deciding, per audience, what the memory is permitted to say.

## FAQ

**Can a strategy change the deduplication threshold for its scope?**
No. Only the four fields above. The semantic dedup threshold stays bank-wide at 0.97. In practice this does not get in the way, because dedup only compares observations inside the same scope, so the company scope's summaries and Dana's detailed ones are never candidates to merge with each other.

**What happens to a scope no strategy claims?**
It uses the bank-wide values, and the control plane labels it "Default". There is no implicit catch-all, which is the other reason the untagged-scope limitation bites: you cannot write a fallback strategy that sweeps up everything unmatched.

**If I write a strategy with no settings, does it block later ones?**
No. A strategy that sets nothing is dropped when the config is parsed, so it never claims a scope and never hides a later one. That is deliberate, because the control plane saves as you type and half-written rules should not take effect.

**Can I give the `shared` scope its own mission?**
No, and this is the one genuinely surprising limitation. No pattern matches an empty tag set. If you need a differently-briefed catch-all, tag it and target the tag.

**Is `observation_scope_limits` going away?**
It is deprecated, not removed, and it still works as a fallback for the observation cap. It is consulted after strategies. Note that it only ever carried the cap, which is why strategies exist: the mission is the part that needed to vary per audience.

**Does one scope's mission leak into another's observations?**
No. Each resolved scope gets its own consolidation call, so one scope's settings never reach another's. The observation is written with that scope's tags.

**How do I check which strategy is actually claiming a scope?**
The control plane shows the owning strategy per scope, and the scopes endpoint enumerates the scopes in a bank with their observation counts. Worth doing after any change, especially if you configured strategies through the environment variable rather than the API.

## Learn more

- [One Bank or Many? A Field Guide to Structuring Agent Memory](https://hindsight.vectorize.io/blog/2026/07/16/bank-strategy-agent-memory) on choosing the boundary in the first place
- [Bring the Facts, Not the Beliefs](https://hindsight.vectorize.io/blog/2026/09/23/bring-facts-not-beliefs) on what consolidation does and does not collapse
- [The Consolidation Problem in Agent Memory](https://hindsight.vectorize.io/blog/2026/05/21/agent-memory-consolidation) on importance, merge, decay and eviction as a framework
- [What's new in Hindsight 0.10.2](https://hindsight.vectorize.io/blog/2026/09/29/version-0-10-2) for the rest of the release
