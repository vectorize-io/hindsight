---
title: "Guide: A Codex CLI Memory Bank Strategy Across Repos, Bugs, and Decisions"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, codex, bank-strategy, memory]
description: "A per-repo memory bank strategy for Codex CLI with Hindsight, so repo conventions, bug history, and engineering decisions stay retrievable without unrelated projects bleeding together."
image: /img/guides/guide-codex-memory-bank-strategy-across-repos.svg
hide_table_of_contents: true
---

![Guide: A Codex CLI Memory Bank Strategy Across Repos, Bugs, and Decisions](/img/guides/guide-codex-memory-bank-strategy-across-repos.svg)

A good **Codex memory bank strategy** is not one giant coding bank that every project writes into. Once you use Codex CLI across several repos, a single shared bank starts recalling the frontend app's styling rules while you debug a database migration, and the signal you actually want gets buried. The highest-leverage setup is per-repo banks: each repository keeps its own conventions, recurring bug history, and engineering decisions retrievable, without unrelated projects bleeding together.

This guide is about *how you organize memory*, not how you install the hooks. Codex CLI already exposes the lifecycle points Hindsight needs — recall before each prompt, retain after each turn — so the interesting decision is where those memories land. Get the bank layout right and Codex remembers why the auth retry logic changed in one repo without dragging in an unrelated repo's release quirks.

If your Codex hooks are not installed yet, do that first with the [Codex setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-codex-memory-with-hindsight), then come back here to decide how banks should be laid out. Keep the [docs home](https://hindsight.vectorize.io/docs) and [quickstart](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. Install the Codex hooks with the [setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-codex-memory-with-hindsight).
> 2. Enable per-repo banks: set `dynamicBankId: true` with `dynamicBankGranularity: ["agent", "project"]`.
> 3. Retain the things worth remembering — conventions, fragile areas, past debugging, and decisions.
> 4. Use a shared bank (a fixed `bankId`) only for cross-repo team standards.
> 5. Verify that a later session in the same repo recalls what an earlier one stored.

## Why coding memory gets messy fast

Coding conversations are dense and repetitive. Across a week you touch several repos, each with its own build system, test framework, and history of the same three bugs coming back. If every one of those turns retains into a single `codex` bank, recall has to sift a mixed pile of unrelated context on every prompt.

The failure mode is noisy recall. You ask Codex to fix a query in the API repo and it surfaces the frontend's lint conventions because they were retained recently and score close enough. Nothing is broken — the memories are real — but the wrong repo's context is competing for the recall budget. Isolation at the bank level is what keeps recall sharp: if the API repo's memories live in their own bank, they are the only thing recall considers for that repo.

## One bank per repo

The clean default for Codex is one memory bank per repository. Hindsight supports this directly through dynamic bank IDs, so you do not have to name banks by hand:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "project"]
}
```

Save that in `~/.hindsight/codex.json`. With `dynamicBankId` enabled, the bank ID is derived from the fields in `dynamicBankGranularity` — `"project"` is the working directory and `"agent"` is the agent name. Running Codex in `~/projects/api` and `~/projects/frontend` then stores and recalls memories separately, without cross-repo leakage.

Bank isolation in Hindsight is strict — no data crosses between banks — so per-repo banks give each codebase a private memory that persists across sessions. Reopen the same repo tomorrow and Codex starts with what that project has already learned.

## What to retain

Per-repo banks are only as good as what you put in them. The four categories that pay off most for coding work:

- **Conventions** — the repo's real rules: "this project uses FastAPI with asyncpg, not SQLAlchemy," the lint command, the test runner, the branch/release process. These are the things you re-explain to a fresh session most often.
- **Known-fragile areas** — the modules that break in surprising ways, the migration that must run before a deploy, the endpoint with the flaky timeout. Retaining "here be dragons" notes stops Codex from relearning them the hard way.
- **Past debugging** — when a bug is finally understood, that root-cause story is worth keeping. Recurring bugs come back; a bank that remembers "the auth 401s were caused by clock skew, fixed by X" turns a repeat incident into a fast recall.
- **Engineering decisions** — why the retry logic changed, why a library was chosen over an alternative, what tradeoff a refactor accepted. Decisions are expensive to reconstruct from a diff and cheap to recall.

With the default `retainMode` of `"full-session"`, the finished session transcript is retained and Hindsight extracts facts from it, so most of this is captured just by working normally and letting a session end. The point of the categories is to know what a *useful* session looks like — one where the convention or decision actually got stated.

## Connect Codex to Hindsight

This is a strategy guide, not an install walkthrough. The full install — the hook bundle, connecting to Hindsight Cloud or a local daemon, and enabling `codex_hooks` — is covered end to end in the [Codex setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-codex-memory-with-hindsight). Set that up first, then apply the bank layout below.

## A shared bank for team repos

Per-repo isolation is the default, but some knowledge genuinely spans repositories. Team-wide standards — the org's commit conventions, the shared CI pipeline, security review requirements, the internal library everyone depends on — belong in one place so every repo's Codex sessions can draw on them.

For that, use a fixed bank instead of a dynamic one. Turn `dynamicBankId` off and set a stable `bankId` for the shared bank:

```json
{
  "dynamicBankId": false,
  "bankId": "team-standards"
}
```

A practical layout is per-repo banks for day-to-day work and a single shared bank for the conventions that must follow you everywhere. Because banks that share the same `bankId` are shared across the Hindsight integrations pointing at them, a standard retained once by one teammate's Codex is recallable by the rest of the team working the same shared bank.

## Avoid one all-purpose bank

The tempting shortcut is to leave everything on the default single `codex` bank and never think about it again. That works until it doesn't: the more repos write into one bank, the more every recall competes with unrelated context, and the harder it is to trust what comes back.

If you find recall surfacing the wrong project's details, that is the signal to split. Turn on `dynamicBankId` for per-repo isolation and reserve a named shared bank for the genuinely cross-cutting standards. A shared-everything bank only makes sense for truly global, personal habits — not for a portfolio of distinct codebases.

## Verify that memory is working

The bank layout is doing its job when memory stays scoped to the repo it came from. A good test sequence:

1. In one repo, have Codex record a convention or decision — for example the preferred lint command or why a retry was added.
2. End the session so the transcript is retained.
3. Start a fresh Codex session in the **same** repo and ask about that decision. It should come back.
4. Now run Codex in a **different** repo and ask the same question. With per-repo banks, it should *not* surface the first repo's answer.

If step 3 recalls and step 4 stays clean, isolation is working. If recall misses in step 3, confirm both runs derived the same bank ID; if step 4 leaks, check that `dynamicBankId` is actually enabled. While testing, `retainEveryNTurns` defaults to `10`, so set it to `1` temporarily so retain fires every turn.

## Common mistakes

### Leaving everything on one bank

The default `bankId` is `codex`, and every repo shares it unless `dynamicBankId` is enabled. If you never split, expect noisy cross-repo recall once you have more than a couple of projects.

### Putting team standards in a per-repo bank

A convention that should apply everywhere loses its value if it only lives in one repo's bank. Cross-cutting standards belong in a fixed shared `bankId`.

### Expecting one repo to recall another's context

Bank isolation is strict. Per-repo banks are private by design — do not expect the API repo to recall the frontend repo's decisions. That separation is the feature.

### Testing retain before a session ends

With `retainMode` set to `"full-session"`, the transcript is retained per session. If you check for recall mid-session or before enough turns have passed, the memory may not be stored yet — lower `retainEveryNTurns` while testing.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server or the local `hindsight-embed` daemon works the same way. The bank strategy is identical regardless of backend — see the [setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-codex-memory-with-hindsight) for connection options.

### Per-repo banks or one shared bank?

Per-repo banks for day-to-day coding, so projects stay isolated. Add one fixed shared bank for standards that must follow you across every repo.

### How are per-repo banks named?

With `dynamicBankId` enabled, Hindsight derives the bank ID from `dynamicBankGranularity` — by default the agent name and the working directory — so each project directory gets its own bank automatically.

### What should I not bother retaining?

Transient scratch work and one-off commands rarely earn recall budget. Focus on conventions, fragile areas, root-cause debugging, and decisions — the context you would otherwise re-explain to every fresh session.

## Next Steps

- Install the hooks first with the [Codex setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-codex-memory-with-hindsight)
- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Codex integration docs](https://hindsight.vectorize.io/docs/integrations/codex)
