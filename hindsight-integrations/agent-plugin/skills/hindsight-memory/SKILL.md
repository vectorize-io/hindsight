---
name: hindsight-memory
description: Long-term memory for the agent via Hindsight. Use to recall relevant past context before answering, retain durable facts as you learn them, and reflect over accumulated memory for the "why" behind a decision. Load whenever continuity across sessions matters — the user refers to earlier work, states a lasting preference, or asks a question that prior context could answer.
---

# Hindsight long-term memory

This plugin connects the agent to **Hindsight**, a long-term memory engine. Memory
persists across sessions in a **bank** (scoped by `HINDSIGHT_BANK_ID`), so what you
retain now is available to recall in future conversations.

The `hindsight` MCP server exposes the tools below. Prefer these over guessing from
scratch when the answer might live in past context.

## When to recall (read memory)

Call **`recall`** at the start of a task, or whenever the user:

- refers to earlier work, a past decision, or "the thing we set up",
- states a preference or constraint that may already be recorded,
- asks a question that accumulated project/user context could answer.

```
recall(query: "how do we deploy the API and which region")
```

`recall` runs semantic + keyword + graph + temporal retrieval and returns the most
relevant memories. Ground your answer in what comes back, and say when nothing
relevant was found rather than inventing continuity.

## When to retain (write memory)

Call **`retain`** when you learn something **durable and reusable** — worth having in
a future session, not just this one:

- stable user preferences ("prefers pnpm; deploys from `main` only"),
- project facts and decisions ("staging DB is Postgres 16 on Neon"),
- outcomes and gotchas ("the flaky test was a timezone bug, fixed in #482").

```
retain(content: "The user deploys the API to us-east-1 via GitHub Actions on push to main.")
```

Do **not** retain transient chatter, secrets, or anything the user asked you to keep
out of memory. Retain the fact, not the whole transcript.

## When to reflect (reason over memory)

Call **`reflect`** when a single recall is too shallow and you need synthesized
reasoning over everything remembered — the *why* behind a behavior, or a judgment that
weighs many facts together:

```
reflect(query: "What has repeatedly caused our CI to flake, and what should we standardize?")
```

`reflect` is slower and disposition-aware; use it deliberately, not for lookups.

## Bank scope

All tools operate on the bank selected by the connection (`HINDSIGHT_BANK_ID`, default
`default`). One bank = one memory store — keep a project's or a user's memory in its
own bank so context stays isolated and relevant. You never pass a bank id to a tool;
it is implicit from the endpoint.
