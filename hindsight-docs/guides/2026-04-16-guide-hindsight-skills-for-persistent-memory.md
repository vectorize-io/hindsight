---
title: "Guide: Use Hindsight Skills for Persistent Memory"
authors: [benfrank241]
date: 2026-04-16
tags: [how-to, skills, coding-agents, memory]
description: "Use Hindsight Skills for persistent memory in Claude Code, OpenCode, or Codex so user preferences and project lessons survive across sessions."
image: /img/blog/guide-hindsight-skills-for-persistent-memory.png
hide_table_of_contents: true
---

![Guide: Use Hindsight Skills for Persistent Memory](/img/blog/guide-hindsight-skills-for-persistent-memory.png)

If you want **Hindsight Skills for persistent memory**, the goal is to install a reusable skill into your coding assistant so it can retain lessons, recall relevant context before work begins, and reflect across what it has learned over time. Instead of relying on one giant static prompt, the assistant gets a repeatable memory workflow that compounds across sessions.

This is especially useful for coding assistants because they keep running into the same categories of context: project conventions, user preferences, bug workarounds, architecture decisions, and commands that worked last time. Hindsight Skills turn those into durable memory rather than ephemeral chat history.

This guide covers the installer flow, local vs cloud mode, what the skill actually provides, and how to verify that it is storing the right kinds of information. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) open while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. Install the Hindsight Skill for your assistant.
> 2. Choose local mode or cloud mode.
> 3. Configure your backend and bank settings.
> 4. Let the assistant retain and recall project knowledge automatically.
> 5. Verify that a later session remembers a real preference or lesson.

## Supported platforms

Hindsight Skills support common coding assistants such as:

- Claude Code
- OpenCode
- Codex CLI

The important point is that the same memory pattern can be reused across several tool surfaces rather than reimplemented separately for each one.

## Local mode vs cloud mode

### Local mode

Local mode is best when:

- one developer wants private memory on one machine
- all data should stay local
- you want minimal external dependencies

### Cloud mode

Cloud mode is best when:

- a team wants shared project knowledge
- several developers should benefit from the same memory bank
- you want memory available across more than one machine

If you want the easiest managed path, use [Hindsight Cloud](https://hindsight.vectorize.io).

## Step 1: Install the skill

The recommended installer is:

```bash
curl -fsSL https://hindsight.vectorize.io/get-skill | bash
```

You can also target a specific platform.

```bash
# Claude Code
curl -fsSL https://hindsight.vectorize.io/get-skill | bash -s -- --app claude

# OpenCode
curl -fsSL https://hindsight.vectorize.io/get-skill | bash -s -- --app opencode

# Codex CLI
curl -fsSL https://hindsight.vectorize.io/get-skill | bash -s -- --app codex
```

If you already know you want cloud mode:

```bash
curl -fsSL https://hindsight.vectorize.io/get-skill | bash -s -- --app claude --mode cloud
```

## Step 2: Choose what kind of memory you want

The skill can help store three especially valuable categories:

- **user preferences** such as coding style or response format
- **procedure outcomes** such as commands that fixed an issue
- **learnings** such as architecture decisions or recurring pitfalls

That makes the skill more useful than a static `AGENTS.md`-style context file. A static file captures what someone remembered to write down. A memory skill can capture what actually happened.

For the retrieval behavior beneath the skill, review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall). For storage behavior, review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Step 3: Configure local or cloud mode

### Local mode setup

The skill uses local Hindsight tooling and stores data on your machine.

Reconfigure local mode with:

```bash
uvx hindsight-embed configure
```

### Cloud mode setup

Cloud mode connects the skill to a shared Hindsight backend. Typical inputs are:

- API URL
- API key
- bank ID

This is the better choice when several developers should share the same project memory.

## How the skill actually helps the assistant

Once installed, the assistant can:

- **retain** useful project facts and lessons
- **recall** relevant context before non-trivial work
- **reflect** across memories when a synthesized answer is more useful than a raw list

That means the assistant can remember things like:

- preferred test commands
- deployment gotchas
- architecture choices
- repo-specific conventions

The result is not that the model becomes magically omniscient. The result is that it starts from a better context baseline in later sessions.

## What to store in a shared team bank

If you use cloud mode with a team bank, be deliberate.

Good candidates:

- project conventions
- domain rules
- known environment quirks
- shared debugging knowledge

Things to store carefully:

- individual preferences, which should be named explicitly
- anything that should remain private to one developer

If the team wants one memory bank per project, keep the bank naming explicit and consistent.

## Verify that the skill works

A simple test is:

1. tell the assistant a real preference or project fact
2. complete the task or end the session
3. start a fresh session
4. ask about that preference or fact

If the answer comes back accurately, the skill is working.

A better second test is to see whether the assistant recalls a lesson before work starts, not only when explicitly prompted.

## Common mistakes

### Treating the skill like a generic prompt snippet

It is more useful as a memory workflow than as a static block of instructions.

### Using a shared bank for personal-only context

Project knowledge and individual preferences should not be mixed carelessly.

### Expecting the skill to replace all documentation

It complements documentation. It does not replace docs, READMEs, or runbooks.

### Never testing recall in a fresh session

You only know the memory works if it survives the session boundary.

## FAQ

### Does this only work with Claude Code?

No. The skill pattern also supports OpenCode and Codex CLI.

### Should I start with local or cloud mode?

Start local for personal use. Start cloud for team-shared memory.

### Is this the same as using a plugin?

Not exactly. A skill is a reusable capability pattern for the assistant, not just a backend hook.

### Can I still use direct integrations too?

Yes. Skills and integration packages can complement each other depending on the tool.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want team-shared memory
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Compare coding workflows in [Adding Memory to Codex with Hindsight](https://hindsight.vectorize.io/blog/adding-memory-to-codex-with-hindsight)
