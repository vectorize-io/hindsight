---
title: "Guide: Add Aider Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, aider, coding-agents, memory]
description: "Add Aider memory with Hindsight using the hindsight-aider wrapper, so each Aider session recalls relevant project context before it starts and retains the transcript after."
image: /img/guides/guide-aider-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Aider Memory with Hindsight](/img/guides/guide-aider-memory-with-hindsight.svg)

If you want **Aider memory with Hindsight**, the cleanest setup is the `hindsight-aider` wrapper. It runs in place of the `aider` command, recalls relevant project memory before each session, and retains the session transcript after Aider exits. That gives Aider long-term memory across coding sessions instead of forcing every new session to rediscover the same repo context.

This is a good fit for Aider because Aider has no MCP client or per-prompt hook, but it does load read-only context files and write a chat-history file. The wrapper uses both: it injects recalled memory as a read-only file at launch, and reads the session slice of the chat history to retain when you exit. Memory is scoped per git repo, so a project's memory stays isolated from unrelated work.

This guide walks through installing the wrapper, pointing it at your Hindsight backend, understanding the per-repo bank strategy, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-aider aider-chat`.
> 2. Set `HINDSIGHT_API_TOKEN` (Cloud) or `HINDSIGHT_API_URL` (self-hosted).
> 3. Run `hindsight-aider` exactly like `aider` — all arguments pass through.
> 4. The bank defaults to the git repo name, so memory is per project.
> 5. Verify that a later session remembers what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- Aider installed and working (`aider-chat`)
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- A git repository, since Aider memory is scoped per repo

## Step 1: Install the wrapper

Install the wrapper alongside Aider itself.

```bash
pip install hindsight-aider aider-chat
```

`hindsight-aider` is a drop-in replacement for the `aider` command. It does not change how Aider works — it wraps the launch so memory is recalled before the session and retained after it.

## Step 2: Point the wrapper at Hindsight

For Hindsight Cloud:

```bash
export HINDSIGHT_API_TOKEN="hsk_..."
```

For a self-hosted Hindsight server:

```bash
export HINDSIGHT_API_URL="http://localhost:8888"
```

Then use it exactly like `aider` — all arguments pass through:

```bash
hindsight-aider                       # interactive, project memory loaded
hindsight-aider -m "add retry logic"  # one-shot; recall uses the message
hindsight-aider src/app.py            # any aider args
```

## How the wrapper uses memory

Aider has no MCP client and no per-prompt hook, so the wrapper works at the two points Aider does expose:

- **Recall (before):** it queries Hindsight, writes the results to `.aider.hindsight-memory.md`, and launches `aider --read .aider.hindsight-memory.md …` so the memory is in context. With `hindsight-aider -m "fix the auth bug"`, the message is used as the recall query; otherwise it runs a general project-context query.
- **Retain (after):** when Aider exits, the wrapper reads the slice of the chat-history file written during the session and retains it to the repo's bank.

Recall happens once per session because Aider cannot be hooked mid-conversation. That fits its session-oriented workflow: each session starts with what the project has learned and saves what it learned on exit.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Per-repo memory banks

The bank defaults to the **git repo name**, so a project's memory is isolated from other projects and shared with the other Hindsight editor integrations working on the same repo.

That means if you also use Hindsight with another editor or coding agent on the same repository, they draw from the same project memory. A convention learned in one tool is available in the others.

If you want a different bank strategy — for example a shared bank across several related repos — see the [package README](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/aider) for the full configuration options.

## Verify that memory is working

A good test sequence is:

1. run `hindsight-aider` in a repo
2. make a change and let Aider record a decision or convention
3. exit the session so the transcript is retained
4. run `hindsight-aider` again in the same repo
5. ask about the earlier decision

For example:

- session one refactors the auth module and notes why the retry logic changed
- session two asks what the current auth conventions are

If the recalled memory file surfaces the earlier decision, the setup is working.

## Common mistakes

### Running outside a git repo

Aider memory is scoped per repo, so run the wrapper from inside the repository you want memory for.

### Expecting mid-session recall

Recall runs once at launch. If you want fresh recall for a new task, start a new session or pass the task with `-m`.

### Testing retain too early

The transcript is retained when Aider exits. If you check before exiting, the session may not have been stored yet.

### Assuming memory is global

By default each repo gets its own bank. That is usually what you want, but do not expect one repo to recall another repo's context.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — set `HINDSIGHT_API_URL`.

### Does this change how I use Aider?

No. All arguments pass through, so you use `hindsight-aider` exactly like `aider`.

### How is memory scoped?

Per git repo, using the repo name as the bank by default.

### Is this similar to other coding-agent integrations?

Yes in spirit. Aider uses read-only context files and chat history instead of a plugin or MCP, but the recall-before / retain-after pattern is the same. Compare the [OpenCode integration](https://hindsight.vectorize.io/docs/integrations/opencode) and [the Claude Code integration](https://hindsight.vectorize.io/docs/integrations/claude-code).

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Aider integration docs](https://hindsight.vectorize.io/docs/integrations/aider)
