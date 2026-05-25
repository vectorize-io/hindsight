---
title: "Building a Hermes Coding Assistant That Remembers Your Codebase"
authors: [benfrank241]
date: 2026-05-25
tags: [hermes, coding, memory, agents, tutorial, hindsight]
description: "Hermes Agent with Hindsight remembers your codebase across sessions, conventions, past bugs, architectural decisions. Setup in 2 minutes."
image: /img/blog/hermes-coding-assistant-codebase-memory.png
hide_table_of_contents: true
---

![Building a Hermes Coding Assistant That Remembers Your Codebase](/img/blog/hermes-coding-assistant-codebase-memory.png)

Every AI coding session starts from zero.

You open a new chat, paste in context, your stack, your conventions, the architectural decision you made last week, the bug you spent two days on in March. Then you do it again next session. And the one after that. The problem isn't that AI coding tools are bad at coding. It's that they have no memory of your codebase.

Hermes Agent with Hindsight is the exception. Each session adds to what it knows about your project. By session 20, Hermes already knows your module boundaries, naming conventions, known fragile areas, and the root cause of that recurring auth issue. You stopped explaining; it started knowing.

This post covers what Hermes actually extracts from coding sessions, how to set it up in two minutes, and the three workflows where persistent codebase memory has the highest leverage.

<!-- truncate -->

---

## What Hermes Remembers About Your Codebase

Hermes doesn't store transcripts. What Hindsight extracts and retains are facts, atomic, retrievable pieces of knowledge pulled from your conversations.

After a typical coding session, facts like these enter memory automatically:

- `"Project uses ESM modules, not CommonJS, always use .js extensions in imports"`
- `"The auth middleware fails silently on expired refresh tokens, known issue as of March 14"`
- `"SQLAlchemy was removed in favor of raw asyncpg after performance testing in February"`
- `"Team convention: all async handlers wrapped in handle_errors() decorator"`

None of these require you to explicitly tell Hermes to remember them. Hindsight's write pipeline extracts them from the natural flow of your session, from the questions you ask, the bugs you describe, the decisions you explain along the way.

What doesn't become memory: raw file contents, line-by-line code, verbose terminal output. The extraction step is itself a filter. Conversational filler, repeated context-setting, procedural noise, none of it survives. What remains is a growing index of codebase facts that Hermes carries into every future session.

The lifecycle runs at both ends of each turn:

**Before each turn:** Hindsight prefetches the most relevant memories from your history and injects them into the system prompt. Hermes sees that context before it sees your message.

**After each response:** Your conversation is retained asynchronously. Hindsight extracts facts in the background. What you discuss this turn becomes searchable starting next turn.

---

## Two-Minute Setup

Hermes v0.14.0 (v2026.5.16) ships with a native Hindsight integration. Setup is a single wizard command:

```bash
hermes memory setup # select "hindsight"
```

Confirm memory is active:

```bash
hermes memory status
```

Config lives at `$HERMES_HOME/hindsight/config.json`. The defaults work for most workflows:

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `cloud` | `cloud` or `local` |
| `bank_id` | `hermes` | Memory bank identifier, change per project |
| `budget` | `mid` | Recall thoroughness: `low` / `mid` / `high` |
| `memory_mode` | `hybrid` | Auto-recall + explicit tools |
| `prefetch_method` | `recall` | `recall` (fast) or `reflect` (LLM synthesis) |

The `memory_mode` controls how memory surfaces in Hermes:

- **`hybrid`** (default): Memories auto-injected before every turn, plus `hindsight_recall`, `hindsight_retain`, and `hindsight_reflect` tools available to the model
- **`context`**: Auto-recall only, nothing explicit, nothing to think about
- **`tools`**: Model must call `hindsight_recall` explicitly; nothing injected automatically

For coding work, `hybrid` is the right default. You get automatic recall on every turn plus the ability to ask Hermes to explicitly surface what it knows about a specific component.

For deployment: if you work across machines or share memory with a team, use cloud. If you want everything local with no external dependencies, local mode runs an embedded PostgreSQL daemon in the background. First startup takes about a minute while the database initializes; subsequent starts are fast. Startup logs land at `~/.hermes/logs/hindsight-embed.log`.

> **Migrating from the older `hindsight-hermes` plugin?** Uninstall it first (`uv pip uninstall hindsight-hermes --python $HOME/.hermes/hermes-agent/venv/bin/python`), then run the setup wizard. The native provider replaces everything the plugin did. Full details in [Hindsight is now a native memory provider in Hermes Agent](/blog/2026/04/06/hermes-native-memory-provider).

---

## Three Workflows Where Codebase Memory Matters

### Starting a New Session on Existing Work

Without memory, resuming work on an existing project means context-setting before any actual work happens: paste the README, explain the tech stack, re-establish what you were doing last time, remind Hermes about the convention it should already know. On a complex project, that overhead eats 10–15 minutes of every session.

With memory, Hermes starts each session with the accumulated facts from your previous sessions already injected into its context. It knows the stack. It knows the conventions. It knows what you were debugging last week.

The first message of a session becomes the actual work.

What gets injected is controlled by the `budget` setting. At `mid` (the default), Hindsight fetches the 10–15 most relevant memories, enough to cover your project's core facts without flooding the context window. At `high`, retrieval runs deeper: more context, more tokens. For most coding workflows, `mid` is the right balance. If you're jumping back into a complex investigation across many modules, `high` is worth the extra cost.

### Debugging Recurring Issues

The highest-leverage value of codebase memory is pattern recognition across sessions. Some bugs aren't one-off, they're symptoms of a deeper architectural issue that surfaces in different forms over months.

Without memory, you debug each instance independently. You might trace the same root cause three separate times without connecting the dots.

With memory, Hermes recalls the previous instances. Describe a new failure mode, and it surfaces related context: the root cause it identified two months ago, the workaround that held until the next refactor, the component that keeps appearing in these failures.

The kinds of facts that pay off here:

- `"The rate limiter bypasses auth checks for requests with X-Internal: true header, source of two privilege escalation near-misses"`
- `"Async task queue silently drops jobs when Redis connection resets, needs explicit ACK handling, not fire-and-forget"`
- `"GraphQL resolver N+1 pattern reappears after every new schema addition, needs DataLoader enforcement flagged in code review"`

These aren't facts you'd think to paste at the start of a debugging session. They're the institutional knowledge that separates debugging blindly from debugging with full context.

`prefetch_method: reflect` is worth enabling for this workflow. Instead of retrieving individual facts through semantic search, `reflect` asks the LLM to synthesize a coherent summary across all related memories before injecting it. It's slower, but for "help me understand this class of bug" queries, the synthesized context is more useful than a list of individual facts.

### Onboarding to Someone Else's Code

If you join a project where a colleague has been working with Hermes and Hindsight using a shared bank, the memory bank already has context from their sessions.

Query what Hermes knows about a specific module:

```
What do you know about the payments module?
```

```
What quirks or known issues have come up in the auth service?
```

```
What were the reasons we moved off SQLAlchemy?
```

Hermes surfaces the accumulated facts from previous sessions: architectural context, known edge cases, past decisions and the reasoning behind them, without anyone having to write it down in a README, a wiki page, or a Slack thread that nobody can find.

This isn't documentation. It's the institutional knowledge that never makes it into documentation.

---

## What Good Codebase Memory Looks Like

After 30+ sessions on a project, a well-built memory bank typically covers:

**Project conventions:** Module structure and import patterns, error handling requirements, naming conventions that aren't obvious from the code, linting rules that differ from the defaults.

**Known fragile areas:** Components that break under specific load or input conditions, integration points that have caused production incidents, edge cases the test suite doesn't cover.

**Architectural history:** Dependencies that were replaced and why, patterns considered and rejected, performance characteristics discovered through testing rather than docs.

**Team preferences:** Code review priorities, deployment gotchas, things that work differently than the official docs say.

Most of this accumulates automatically from normal sessions. The exception: major architectural decisions and team preferences benefit from explicit statement. When you make a significant call, tell Hermes the rationale:

```
We're switching to asyncpg from SQLAlchemy because connection pooling
under our load profile caused intermittent timeouts above ~200 concurrent
requests. The fix wasn't tuning, it was the ORM abstraction. Remember this.
```

In `hybrid` mode, the model can also call `hindsight_retain` to explicitly flag something for retention. But the background extraction catches most of what matters without that step.

**Before/after: what memory recall looks like in a session**

Without memory, a session opening might look like:

> "I'm working on a Python service that uses asyncpg for database access. We removed SQLAlchemy in February due to connection pool issues under load. All async handlers should be wrapped in `handle_errors()`. Help me debug this intermittent 500..."

With memory, Hermes already has these facts injected. You open with:

> "Help me debug this intermittent 500 in the payment handler."

The stack, the convention, the architectural context, already there.

---

## Team Codebases: Shared Memory Banks

By default, `bank_id` is `hermes`. Every developer on the same project can share a memory bank by setting the same `bank_id`:

```json
{
 "bank_id": "payments-service",
 "mode": "cloud"
}
```

When multiple developers use Hermes on the same codebase with a shared bank, the memory compounds from all their sessions. Knowledge that one developer builds up, a tricky module's undocumented behavior, a hard-won debugging insight, a deployment gotcha, becomes available to the rest of the team automatically.

A few considerations:

- **What compounds well:** Codebase facts, architectural decisions, known issues, conventions. These describe the codebase, not the person, safe to share.
- **What to keep separate:** Personal workflow preferences, unrelated personal context. Use a separate bank for those.
- **Bank naming:** One bank per project, not one bank per developer. A shared bank named `hermes` that three people use without coordinating turns into noise; a shared bank named `payments-service` that the payments team all uses turns into institutional memory.

---

## Advanced: Seeding a Structured Mental Model

Organic extraction from sessions is the primary way Hindsight builds codebase knowledge. But you can front-load context explicitly, useful when starting on an existing codebase, or when critical conventions should be in the bank before the first session runs.

Hindsight exposes two operations via the SDK and API outside of Hermes, feeding the same memory bank Hermes draws from:

**Ingesting existing docs.** Upload architecture notes, ADRs, or conventions files. Hindsight runs fact extraction on the content and stores the results as memories in the bank. Once ingested, those facts are available to Hermes on the next session, no waiting for organic extraction to catch up.

**Creating a mental model.** Define a curated summary built from a source query, "What are the coding conventions for this project?" Hindsight runs a reflect operation, synthesizes the answer from all ingested and session-extracted knowledge, and saves the result. Set `refresh_after_consolidation` to true and the model re-derives itself as new facts arrive. Mental models are checked first during reflect calls, before individual observations and raw facts, so the pre-computed answer is returned without re-deriving it on the fly.

The combination of ingested project docs and session-extracted facts gives Hermes a complete picture from two angles: what was deliberately documented, and what was discovered through use.

That's also what makes autonomous coding agents tractable. An agent with rich, current mental models, conventions, architecture, known fragile areas, has enough structured context to make decisions and drive changes without a human briefing it at the start of every task. Persistent session memory is how you get to that baseline.

See the [Hindsight mental models docs](https://hindsight.vectorize.io/developer/api/mental-models) for the full API reference.

---

## The Longer You Use It, the Less You Explain

Hermes with Hindsight is the only coding workflow where context accumulates across sessions. Every session adds to what it knows about your codebase. Other tools reset. This one doesn't.

The value compounds with use. Session one, Hermes knows nothing about your project. Session five, it knows the stack and conventions. Session 30, it knows the project's history, its fragile areas, the decisions that shaped its current shape. At that point you've stopped explaining those things, not because you skipped the context, but because you never needed to provide it again. And once that mental model is rich enough, you're not just talking to a coding assistant. You're working alongside an agent that knows the codebase as well as you do.

Set it up with `hermes memory setup`, or start with the [Hindsight integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes).

---

**Further reading:**
- [What Is Agent Memory?](https://vectorize.io/what-is-agent-memory/), foundational concepts behind how AI agents retain context
- [Hindsight Is Now a Native Memory Provider in Hermes Agent](/blog/2026/04/06/hermes-native-memory-provider), setup guide, memory modes, and config reference
- [Adding Persistent Memory to Codex with Hindsight](/blog/2026/04/08/adding-memory-to-codex-with-hindsight), the same pattern applied to OpenAI Codex
- [Best AI Agent Memory Systems in 2026](https://vectorize.io/articles/best-ai-agent-memory-systems/), comparison of all major agent memory frameworks
