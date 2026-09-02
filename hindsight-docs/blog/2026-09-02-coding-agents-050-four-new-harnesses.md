---
title: "Four New Coding Agents, and Four Ways to Fail Silently"
authors: [benfrank241]
slug: "2026/09/02/coding-agents-050-four-new-harnesses"
date: 2026-09-02T12:00
tags: [hindsight, coding-agents, integrations, opencode, qwen-code, deepagents, pi, release]
description: "Coding Agents 0.5.0 adds pi, Qwen Code, DeepAgents Dcode, and opencode 2, bringing the total to 16 harnesses. Each one broke a different assumption, and three of the four failed without producing a single error."
image: /img/blog/coding-agents-050-four-harnesses.png
hide_table_of_contents: true
---

![Coding Agents 0.5.0 adds four new harnesses: pi, Qwen Code, DeepAgents Dcode, and opencode 2](/img/blog/coding-agents-050-four-harnesses.png)

`hindsight-coding-agents` 0.5.0 adds four harnesses: **pi**, **Qwen Code**, **DeepAgents Dcode**, and **opencode 2**. That brings the package to **16 supported agents**, all sharing one reflect-and-inject core.

The interesting part isn't the count. It's that three of these four integrations, when wired the obvious way, **failed without producing a single error**. They installed cleanly, started cleanly, and did nothing. Each one broke a different assumption, and each is worth knowing about if you're building against these hosts yourself.

<!-- truncate -->

## The four

| Harness | Package | How it integrates |
|---|---|---|
| **pi** | `@earendil-works/pi-coding-agent` | Extension, shared with Prime Agent |
| **Qwen Code** | `qwen-code` | Hooks, Claude Code protocol |
| **DeepAgents Dcode** | LangChain's `deepagents-code` | Native Agent Plugin, Hooks V2 + MCP |
| **opencode 2** | `@opencode-ai/cli@beta` | Plugin, v2 API |

Install any of them the same way:

```bash
hindsight-coding-agents install qwen-code
```

`install all` wires every harness it finds. Ingestion is automatic from there: a repo's git history and conversations flow into its memory bank in the background, with no setup command to run.

![hindsight-coding-agents install all, wiring seven detected coding agents in 3.2 seconds](/img/blog/hindsight-coding-agents-install-all.png)

Worth reading that output closely, because it's the whole design in one screen. Seven hosts, and **no two are wired the same way**: hooks merged into a JSON settings file, an extension registered, an MCP server added under user scope, a plugin patched into a YAML profile that applies to every profile on the machine. Every one of those is a different host contract. Behind all of them sits the same reflect-and-inject core.

## opencode 2: same name, nothing else in common

opencode v2 installs alongside v1 as a separate binary, `opencode2`. It also rewrote the plugin contract end to end.

A v1 plugin is a function returning a bag of named hooks. A v2 plugin is `{id, setup(ctx)}`, where `ctx` hands out per-domain registration. The mapping is a near-total rewrite:

| opencode v1 | opencode 2 |
|---|---|
| `chat.message` | `ctx.session.hook("prompt")` |
| `experimental.chat.system.transform` | `ctx.session.hook("context")` |
| `tool: {...}` | `ctx.tool.transform(d => d.add(...))` |
| `event (session.idle)` | `ctx.event.subscribe()` |
| `client.session.messages()` | `ctx.session.context({sessionID})` |

Here's the part that matters. **Hand either host the other's export and the plugin loads, registers nothing, and reports no error.** No warning, no failed import, no missing-hook complaint. The agent simply runs without memory, exactly as if you had never installed anything.

There's a second trap underneath it. Both CLIs read the *same* `~/.config/opencode/opencode.json`, and v1 rejects the entire file if it sees v2's `plugins` key. So you cannot solve this with a second config entry; one entry has to serve both.

What makes that possible is a difference in how each host resolves a plugin *directory*: v1 follows `package.json` `main`, while v2 ignores `main` and loads `<dir>/index.js`. One directory, two entry points, each host finding its own. Installing either harness wires both, and uninstalling either removes the shared entry.

![opencode running locally](/img/blog/hindsight-coding-agents-opencode.png)

opencode. Version 2 ships as a separate `opencode2` binary that installs *alongside* v1 and reads the very same `opencode.json`, which is exactly what forces one shared plugin entry rather than two.

## Qwen Code: the same protocol, in different units

Qwen Code speaks Claude Code's hook protocol field for field. Same stdin envelope (`session_id`, `transcript_path`, `cwd`, `hook_event_name`), same `hookSpecificOutput.additionalContext` output, same exit semantics, and a `settings.json` shape byte-for-byte what the installer already emitted.

So the harness spec is Claude Code's with three deltas, and the first one is nasty: **Qwen's hook timeouts are milliseconds. Everywhere else they're seconds.**

Write the usual `30` and `60` and you've just registered **30-millisecond hooks**. Recall gets killed before it can return, and memory silently stops arriving.

Now the part that makes it genuinely dangerous: *that misconfiguration looks fine in testing.* Qwen spawns hooks without `detached: true` and kills only the direct child, so the orphaned work still completes. You watch a session, memory shows up, everything seems wired. It isn't; you're seeing a process that outlived its own timeout.

The fix is structural rather than a corrected constant. The harness spec now declares a `timeoutUnit`, the installed values are `30000/30000/60000`, and the lifecycle tests normalize through it. Changing `30_000` to `30`, or dropping the unit entirely, now fails a test instead of shipping dead hooks.

![Qwen Code at startup](/img/blog/hindsight-coding-agents-qwen-code.png)

Qwen Code. Its hooks land in `~/.qwen/settings.json` in a shape byte-for-byte identical to Claude Code's, which is what makes the millisecond timeout so easy to miss.

## DeepAgents Dcode: a Python repr in the transcript

LangChain's Dcode registers as a native Agent Plugin: a root `plugin.json` contributing the shared skill, the Hooks V2 `SessionStart` / `UserPromptSubmit` / `Stop` lifecycle, and the `hindsight_*` MCP server, installed through Dcode's own marketplace.

Running it against `deepagents-code 0.1.65` turned up three things.

**The transcript lags the Stop event**, which makes `last_assistant_message` load-bearing. But that field is computed as `str(content)` — a *Python repr* whenever the provider returns content blocks rather than a plain string. The effect was that roughly 1.8 KB of encrypted reasoning payload got retained as the assistant's turn, and because the repr never compared equal to the transcript's clean text, an already-flushed reply was appended again every single turn.

That one is now guarded family-wide: any harness surfacing that field has to declare a decoder.

**Dcode rejects unannotated MCP calls in headless mode.** The six read-only tools needed `readOnlyHint` annotations, without which recall and the knowledge-page tools were unusable under `dcode -n`.

**Dcode can't host the codebase survey.** `hindsight_ingest_document` writes, and Dcode's headless runtime gates writes by design. The survey falls back to another agent's CLI, which is the same path eight other harnesses already take.

![DeepAgents Dcode v0.1.65](/img/blog/hindsight-coding-agents-dcode.png)

DeepAgents Dcode **v0.1.65**, the exact version the three fixes above were found against.

## pi: one config key, two hosts

pi and Prime Agent are the same shape — Prime Agent is a fork — so they now share one extension adapter and one installer.

The bug was smaller and just as quiet. Both hosts read the same `pi` key in `package.json`, so that key could only ever name one bundle. **The host it didn't name loaded the other's bundle and reported the wrong harness.** Memory worked; it was just filed under the wrong agent.

The key is gone. `hindsight-coding-agents install pi` and `install prime-agent` are now the only route for either, each with its own config section and its own agent stamped on the documents it retains.

## The pattern

Four integrations, and the failure mode was the same shape in three of them: **the integration reports success and does nothing.**

- opencode 2 loads a plugin that registers no hooks, silently.
- Qwen Code registers hooks that time out in 30 ms, and hides it by letting orphaned processes finish.
- Dcode retains a Python repr that never matches the transcript, so it duplicates instead of erroring.
- pi loads a bundle under the wrong harness name.

None of these throws. Every one of them passes a casual smoke test. That's the argument for testing an agent integration on what it *wrote to memory* after a real session, rather than on whether the install command exited zero — which is what the lifecycle tests now assert per harness.

## Also in this release

- **Per-source observation scoping**, so observations can be scoped by where they came from.
- **The installed runtime keeps itself current**, rather than pinning whatever version you first installed.
- **MCP tool safety annotations** across the tool surface.
- **A failed git probe no longer forks a worktree into its own bank.**
- **Bank config is only ever added to, never overwritten.**

0.5.1 followed immediately with one fix worth noting: the runtime auto-update never actually fired, because the registry returned a 406 on the `/latest` media type. If you installed 0.5.0 on day one, upgrade.

```bash
npm install -g @vectorize-io/hindsight-coding-agents@latest
```

## Learn more

- [Coding Agents changelog](https://hindsight.vectorize.io/changelog/integrations/coding-agents) — every release in full
- [Knowledge Pages for Coding Agents](/blog/2026/08/13/knowledge-pages-coding-agents) — the curated layer these agents read from
- [One Bank or Many? A Field Guide to Structuring Agent Memory](/blog/2026/07/16/bank-strategy-agent-memory) — how memory is scoped across repos and agents
