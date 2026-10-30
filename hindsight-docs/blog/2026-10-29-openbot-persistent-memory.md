---
title: "Your OpenBot Agents Forget Between Threads. Give Them Shared Memory."
authors: [benfrank241]
date: 2026-10-29
tags: [openbot, memory, integrations, mcp, tutorial]
description: "Connect an OpenBot agent to a Hindsight bank over MCP and its memory outlives the thread, survives compaction, and is shared with every teammate."
image: /img/blog/openbot-persistent-memory.png
hide_table_of_contents: true
---

![Your OpenBot agents forget between threads, here is shared memory that does not](/img/blog/openbot-persistent-memory.png)

[OpenBot](https://openbot.run) runs a team of AI teammates on your own machine. Claude, Codex, Grok, Gemini, OpenCode, Cursor and Cline, each with its own workspace, its own task queue, and the ability to message each other. The agents are persistent. They sit there between sessions, waiting for work.

OpenBot memory is not persistent in the same way. It lives inside a thread, and it belongs to one agent. This integration gives your agents a memory that outlives both.

<!-- truncate -->

## TL;DR

- Connect an OpenBot agent to a Hindsight **bank** over MCP, by adding one URL MCP server
- Point an agent at a bank and it gets `recall`, `retain`, `sync_retain` and `reflect`, plus mental-model and document tools
- Memory outlives the thread, survives context compaction, and survives restarts
- Every agent pointed at the **same bank reads the same memory**, so what Codex learns, Claude knows
- Setup is a URL and a header
- Works with [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) or a self-hosted server

---

## The problem: OpenBot memory lives and dies with the thread

An OpenBot agent survives a restart. The thread it was working in does not survive itself.

Every thread compacts as it fills. The early turns get summarized away to make room, and the early turns are usually where you explained the architecture, the deploy rules and the three things that always break. Start a new thread and you start cold.

The second half is worse, because it does not look like a problem. Agents can message each other, but what one learns stays with that one. You tell Codex your test conventions. Claude has never heard of them. You end up briefing each teammate separately, which is the overhead a team of agents was supposed to remove.

OpenBot does ship its own memory, the `remember` and `search_memories` tools, and it works. It is also a few kilobytes per agent and private to that agent. A report in OpenBot's tracker ([#2463](https://github.com/nightly-labs/openbot/issues/2463)) counted 177 memories across 17 agents, 15 contradicted and 31 stale. That is per-agent memory at team scale.

---

## What we built: shared memory over MCP

[Hindsight](https://github.com/vectorize-io/hindsight) is an open-source long-term memory engine for agents. It extracts facts from conversations, retrieves them semantically, and can reason over an entire store rather than fetching the nearest few chunks. Memory is organized into **banks**.

The integration connects an OpenBot agent to Hindsight over MCP. Connect an agent to a bank and it gains:

- **`recall`** to search memory semantically, across every thread that bank has seen
- **`retain`** and **`sync_retain`** to write facts back, from any thread
- **`reflect`** to synthesize an answer over the whole bank instead of a handful of retrieved snippets

A bank is a shared destination rather than a per-agent store. Two agents on one bank read and write the same memory. That single property is what changes how a team of agents behaves.

Two things are worth separating here. The MCP connection is a **capability**: you add it once in the Local server's settings and its tools reach every agent. The optional **[Hindsight Memory skill](https://hindsight.vectorize.io/skills/openbot/hindsight-memory/SKILL.md)** is the **habit**: recall before acting, retain what is durable. The skill is enabled per agent, with no all-agents switch, though Claude agents load it automatically from `~/.agents/skills/<slug>/`.

---

## Set it up in two minutes

In OpenBot, open the **Local** server's settings → **MCP** → **Connect to a custom MCP**, pick **Streamable HTTP**, and enter the server URL and an `Authorization` header:

```
URL:    https://api.hindsight.vectorize.io/mcp/<bank>/
Header: Authorization: Bearer hsk_your_key
```

The connection is added at the server level, so its tools reach every agent; enable the memory skill on the agents you want using it.

That URL defaults to [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup). Self-hosting is the same shape with a different host:

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api
```

Then use `http://localhost:8888/mcp/<bank>/` as the URL. A local server with no auth needs no key, so leave that field blank.

---

## The walkthrough

Pick an agent and tell it something true about your project:

> We only deploy on Fridays after the staging smoke test passes. Remember that.

It calls `retain`. Now close that thread. Start a **fresh thread**, with a **different agent**, and ask:

> When are we allowed to deploy?

It calls `recall` and answers with your rule, including the smoke test condition, from a conversation it was never part of. No context file, no re-briefing, nothing pasted.

That is the whole demo, and it is worth running once yourself. Watching a second agent answer from the first one's conversation lands differently than reading about it.

---

## What happens under the hood

The lifecycle is short, and knowing it explains most of the behavior you will see:

1. **The agent connects.** OpenBot opens an MCP connection to the bank in your URL, and Hindsight's tools appear in that agent's tool list alongside its built-in ones.
2. **The agent recalls.** A `recall` query searches the whole bank, not the current thread, and the matches come back as tool output in the live context window.
3. **The agent retains.** `retain` is asynchronous: it returns immediately while extraction and consolidation run in the background. `sync_retain` blocks until the memory is stored and immediately recallable, which is what you want for read-after-write.
4. **The thread compacts.** The facts are in the bank rather than the thread, so compaction costs you the turns and not the knowledge.
5. **A second agent connects.** Same URL, same bank, and its first `recall` returns what the first agent wrote.

The distinction that matters is between what sits in context and what sits in the bank. Context is rented and compaction reclaims it. The bank is yours until you delete it.

---

## Scoping memory with banks

The `<bank>` segment of the URL is the scope, and it is the main design decision you make here. Point two agents at one bank and they share everything. Point them at different banks and they share nothing.

- **One bank per project** is the sensible default. Every teammate working that repo sees the same conventions, decisions and gotchas.
- **One bank per team** suits context that outlives any single project, like review standards or release process.
- **One bank per agent** is available if you want isolation, but it gives up the shared layer, which is the reason to run this at all.

Because the bank is part of the URL, each connection is scoped to exactly one bank, and agents never have to pass a `bank_id`. Changing the scope means changing the server entry, not a setting buried inside the agent.

---

## Pitfalls and edge cases

**Cold start.** A new bank is empty, and the first thread has nothing to recall. Memory accumulates from the second thread onward, so judge this on day three rather than minute one.

**Read-after-write.** Because `retain` is asynchronous, a fact you retain and then immediately query in the same turn may not be searchable yet. Use `sync_retain` when the agent needs to read back what it just wrote.

**Recall spends context.** Recalled memories arrive as tool output and consume tokens from the same window that compaction is already pressuring. More recall is not automatically better.

**A shared bank shares mistakes too.** If one agent retains something wrong, every agent on that bank will read it. That is the cost of the thing that makes this useful, and it means a bank deserves the same care as any other shared resource.

---

## The honest part: you now have two memories

When Hindsight is connected, an agent has **both** sets of memory tools, Hindsight's and OpenBot's built-in ones, and the two stores do not share. An agent can still write something to built-in memory where no other agent will ever see it.

The Hindsight Memory skill exists for exactly this reason. It steers the agent toward Hindsight's tools consistently rather than leaving the choice to chance, and in practice an agent with the skill enabled settles into one store.

The real fix is upstream. There is an open feature request ([#3682](https://github.com/nightly-labs/openbot/issues/3682)) for a pluggable memory provider, so an external memory could take over properly instead of sitting alongside. One source of truth, not two. It is a request rather than a shipped feature, so until it lands, the skill is the practical answer.

---

## Next steps

- **[Sign up for Hindsight Cloud](https://ui.hindsight.vectorize.io/signup)** and generate a key, which takes about a minute
- Add the Hindsight MCP URL in OpenBot's Local server settings (**MCP → Connect to a custom MCP**)
- Enable the Hindsight Memory skill on the agents you care about
- Read the [OpenBot integration reference](https://hindsight.vectorize.io/sdks/integrations/openbot) for the full tool list and self-hosting detail
- Point two different agents at one bank and run the walkthrough above
