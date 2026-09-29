---
title: "Guide: Team Memory with OpenCode — Shared Banks for Engineering Conventions"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, opencode, multi-agent, memory]
description: "Give a whole engineering team one shared OpenCode memory bank with Hindsight, so every developer's sessions draw from the same conventions, known bugs, and past decisions."
image: /img/guides/guide-opencode-team-shared-memory.svg
hide_table_of_contents: true
---

![Guide: Team Memory with OpenCode — Shared Banks for Engineering Conventions](/img/guides/guide-opencode-team-shared-memory.svg)

If you run OpenCode across an engineering team, the highest-leverage move is **team memory**: a single shared Hindsight bank that every developer's OpenCode sessions read from and write to. Instead of each person's setup relearning the same conventions, known bugs, and past decisions, the whole team draws from one institutional memory.

Most OpenCode-with-Hindsight setups default to per-project or per-developer memory, which is the right choice for isolated work. But a coding convention learned by one developer, a footgun someone hit last month, or the reason a past architecture decision went the way it did are all things the *team* should remember, not just the person who was there. A shared bank turns those one-off discoveries into recall that shows up for everyone.

This guide is about strategy, not installation. It covers when to share a bank versus isolate one, how static and dynamic bank IDs serve a team, and how to keep recall clean as the bank grows. The plugin setup itself is already covered — see [Add OpenCode Memory with Hindsight](https://hindsight.vectorize.io/guides/2026/04/16/guide-opencode-memory-with-hindsight) — so here we focus on the team-memory decisions on top of it.

<!-- truncate -->

> **Quick answer**
>
> 1. Decide what belongs in shared team memory: conventions, known bugs, past decisions.
> 2. Point everyone's OpenCode plugin at the same Hindsight backend.
> 3. Set a static `HINDSIGHT_BANK_ID` (e.g. `acme-platform-team`) so all sessions share one bank.
> 4. Keep truly private or per-repo work on separate banks instead.
> 5. Verify that one developer's stored convention shows up in another developer's session.

## Why a team needs shared coding memory

On a team, the same knowledge gets rediscovered over and over. One developer learns that the payments service must be deployed before the API, another spends an afternoon rediscovering it. Someone documents why the retry logic uses exponential backoff, and six weeks later a new hire proposes ripping it out.

Per-developer memory does not fix this, because each person's OpenCode only remembers their own sessions. Shared team memory does: when the payments-deploy ordering is retained once, every teammate's OpenCode recalls it on the next relevant session. The value compounds with team size — the bigger the team, the more redundant relearning a shared bank eliminates.

Good candidates for a team bank:

- **Conventions** — the repo prefers pnpm, strict TypeScript, conventional commits.
- **Known bugs and footguns** — "the staging DB resets nightly," "don't call the billing API in a loop."
- **Past decisions** — why a library was chosen or rejected, why a service boundary sits where it does.

## A shared team bank vs per-developer banks

The plugin lets you choose where memory lands. The two common shapes:

- **One shared team bank.** Everyone's OpenCode points at the same bank ID. Every session contributes to and reads from the shared institutional memory. Best for conventions and decisions that should be common knowledge.
- **Per-developer or per-project banks.** Each developer (or each repo) gets an isolated bank. Best for private scratch work, experiments, or when two projects should never see each other's context.

These are not mutually exclusive across a company — you might run a shared bank for a team's core platform repos and isolated banks for unrelated side projects. The question to ask for any given stream of work is simply: *should another teammate's OpenCode recall this?* If yes, it belongs in the shared bank.

**When to isolate instead of share:**

- The work is exploratory and you do not want half-baked findings surfacing for teammates.
- Two projects are unrelated and shared recall would just add noise.
- Content is sensitive and should not spread across the team's sessions.

## Static vs dynamic bank IDs for teams

The plugin resolves which bank to use from its bank-ID configuration, and this is the lever that makes a bank team-wide or per-context. The full mechanics are in the [setup guide](https://hindsight.vectorize.io/guides/2026/04/16/guide-opencode-memory-with-hindsight) and the [integration docs](https://hindsight.vectorize.io/docs/integrations/opencode); here is how each maps to team memory.

### Static bank ID — the team default

A static bank ID is the simplest way to give a team one shared memory. Every developer sets the same value, so all their OpenCode sessions read and write the same bank:

```bash
export HINDSIGHT_BANK_ID="acme-platform-team"
```

Because the priority order is `defaults < ~/.hindsight/opencode.json < plugin options < env vars`, you can standardize the team bank in a checked-in `opencode.json` or a shared `~/.hindsight/opencode.json`, and still let an individual override it with an env var when they need to.

```json
{
  "plugin": [
    ["@vectorize-io/opencode-hindsight", {
      "bankId": "acme-platform-team",
      "autoRecall": true,
      "autoRetain": true
    }]
  ]
}
```

### Dynamic bank IDs — team plus per-project isolation

When a team works across several repos and you want isolation *within* the team, enable dynamic bank IDs. The plugin composes the bank from granularity fields (default `agent::project`, with `agent`, `project`, `channel`, and `user` available):

```bash
export HINDSIGHT_DYNAMIC_BANK_ID=true
```

With `agent::project`, each project gets its own bank automatically while the shared agent name keeps naming consistent across the team — every developer on the same repo lands in the same bank without hand-editing config. If you need per-user separation inside a shared setup (for example a shared agent serving multiple people), the `channel` and `user` fields cover that:

```bash
export HINDSIGHT_CHANNEL_ID="acme-platform"
export HINDSIGHT_USER_ID="user123"
```

Rule of thumb: **static** when the whole team should share exactly one bank; **dynamic** when the team spans projects and you want automatic per-project banks with consistent naming.

## Connect OpenCode to Hindsight

This guide assumes the Hindsight plugin is already installed and pointed at your backend. If it is not, follow the standard setup first — adding `@vectorize-io/opencode-hindsight` to your OpenCode config, pointing it at a local server or Hindsight Cloud, and enabling auto-recall and auto-retain:

- [Add OpenCode Memory with Hindsight](https://hindsight.vectorize.io/guides/2026/04/16/guide-opencode-memory-with-hindsight)

Once that is working, the only team-specific change is the bank-ID strategy above.

## Keeping team recall clean at scale

A shared bank is more valuable than a personal one, but it also collects more content, so recall hygiene matters more. A few practices keep a team bank sharp:

- **Tag retained memory and filter recall.** The plugin supports `retainTags` and `recallTags` (with `recallTagsMatch` modes `any`, `all`, `any_strict`, `all_strict`). Tag by area — `payments`, `infra`, `frontend` — so a frontend session can bias recall toward frontend memory instead of the whole team's history.
- **Tune the recall budget.** `recallBudget` (`low`, `mid`, `high`) controls how much context is pulled in. Start at `mid`; a larger team bank may warrant `low` for focused sessions and `high` only when broad context genuinely helps.
- **Control retain frequency.** `retainEveryNTurns` throttles auto-retain so the bank captures meaningful state without saving every micro-turn as separate noise.
- **Keep exploratory work out.** Route half-baked or experimental sessions to an isolated bank so the shared bank stays authoritative.

## Verify that shared memory is working

Team memory is working when one developer's session surfaces in another's. Test it across two machines or two accounts:

1. Developer A runs OpenCode with the shared `HINDSIGHT_BANK_ID` and stores a convention (e.g. "deploy payments before the API").
2. Let the session idle so auto-retain writes it to the shared bank.
3. Developer B runs OpenCode pointed at the same bank.
4. Developer B asks about deploy ordering.
5. Recall should surface the convention Developer A stored — even though B never learned it firsthand.

If B's session recalls A's convention, the shared bank is live. You can also confirm explicitly by asking OpenCode to use `hindsight_recall` against the shared bank.

## Common mistakes

### Sharing one bank across unrelated projects

A shared bank is for a team's common knowledge, not for merging unrelated repos. If two projects should never see each other's context, give them separate banks (static per project, or dynamic with `project` in the granularity).

### Not aligning bank IDs across the team

Team memory only works if everyone resolves to the *same* bank. A typo in one person's `HINDSIGHT_BANK_ID` silently splits them onto a private bank. Standardize the value in shared config.

### Letting exploratory work pollute the shared bank

Half-finished experiments retained to the team bank surface as noisy recall for everyone. Keep scratch work on an isolated bank.

### Testing retain before the session idles

Auto-retain fires on `session.idle`. If you check the shared bank before the first developer's session goes idle, the memory may not be stored yet.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — just point every teammate's plugin at the same `HINDSIGHT_API_URL`. Cloud simply makes a single shared backend easy to reach for a distributed team.

### How do I make sure the whole team writes to the same bank?

Use a static `HINDSIGHT_BANK_ID` set in shared config, or dynamic bank IDs with a consistent agent name so per-project banks are named identically for everyone.

### Can we share some memory but isolate other work?

Yes. Run a shared bank for common conventions and decisions, and separate banks for private or unrelated work. The bank-ID strategy decides where each session's memory lands.

### Won't a shared bank get noisy as the team grows?

It can, which is why recall hygiene matters. Use tags, an appropriate recall budget, and retain throttling, and keep exploratory work out of the shared bank.

## Next Steps

- Follow the plugin setup in [Add OpenCode Memory with Hindsight](https://hindsight.vectorize.io/guides/2026/04/16/guide-opencode-memory-with-hindsight)
- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend for the team
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [OpenCode integration docs](https://hindsight.vectorize.io/docs/integrations/opencode)
