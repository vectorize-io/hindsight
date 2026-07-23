---
title: "Guide: Per-Agent vs Shared Memory Banks in Strands Agents"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, strands, multi-agent, memory]
description: "Decide between per-agent memory banks (isolation) and a shared bank (institutional memory) when several Strands Agents collaborate, using Hindsight's native @tool retain, recall, and reflect."
image: /img/guides/guide-strands-per-agent-vs-shared-memory.svg
hide_table_of_contents: true
---

![Guide: Per-Agent vs Shared Memory Banks in Strands Agents](/img/guides/guide-strands-per-agent-vs-shared-memory.svg)

When you build a multi-agent system with the Strands Agents SDK, the biggest memory decision is **per-agent vs shared memory** banks: does each agent keep a private bank, or do several agents write into one shared bank? Hindsight scopes every memory operation to a `bank_id`, so this choice is just which value you pass to `create_hindsight_tools(bank_id=...)` for each agent — but it decides whether your agents stay isolated or build institutional memory together.

A single Strands agent is easy: one user, one bank. The interesting question shows up when a planner hands off to a researcher, a researcher hands off to a writer, and you have to decide what each of them is allowed to remember about the others. Per-agent banks give you clean isolation and no cross-contamination. A shared bank gives the whole crew one memory, so a fact one agent learns is available to all of them on the next run.

This guide is a **bank strategy** guide, not another install walkthrough — the [setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-strands-memory-with-hindsight) already covers wiring the tools. Here we focus on when to isolate, when to share, and how to run a hybrid where agents read a shared bank but write to their own.

<!-- truncate -->

> **Quick answer**
>
> 1. Give each Strands agent Hindsight tools with `create_hindsight_tools(bank_id=...)`.
> 2. Use a **per-agent bank** when agents must not contaminate each other's context.
> 3. Use a **shared bank** when the crew should build one institutional memory.
> 4. For a hybrid, point `memory_instructions()` at the shared bank (read) and give the retain tool a private bank (write).
> 5. Verify isolation by storing a fact with one agent and checking who can recall it.

## The bank strategy decision

Every Hindsight tool operates on exactly one bank per call, and bank isolation is strict — there is no cross-bank leakage. In Strands terms, the `bank_id` you pass to `create_hindsight_tools()` (and to `memory_instructions()`) is the whole strategy. Two agents that share a `bank_id` share a memory; two agents with different `bank_id` values are invisible to each other.

So the decision reduces to a single question you ask for each pair of collaborating agents: **should what one agent learns be visible to the other?**

- If yes, they share a bank.
- If no, they each own a bank.
- If "sometimes" — read shared, write private — you use the hybrid below.

Because the tools are plain functions passed to `Agent(tools=[...])`, you can mix strategies freely inside one system. A planner can own a private bank while three workers share a common one. Nothing about the SDK forces a single global choice.

## When each agent should own its bank

Give each agent its own `bank_id` when the agents play genuinely different roles and mixing their memories would hurt more than help.

Good cases for per-agent banks:

- **Distinct personas or dispositions.** A skeptical reviewer agent and an optimistic brainstormer agent should not blur their learned context together.
- **Different users behind different agents.** If each agent fronts a different end user, per-user banks give you the hard separation the platform guarantees.
- **Noisy or throwaway workers.** A scratch agent that generates lots of low-value observations should not pollute a bank that a careful agent relies on.
- **Security or tenancy boundaries.** When one agent must never see another's data, separate banks are the boundary.

The wiring is one bank per agent:

```python
from strands import Agent
from hindsight_strands import create_hindsight_tools

planner = Agent(tools=create_hindsight_tools(bank_id="planner"))
researcher = Agent(tools=create_hindsight_tools(bank_id="researcher"))
writer = Agent(tools=create_hindsight_tools(bank_id="writer"))
```

Each agent gets its own `hindsight_retain`, `hindsight_recall`, and `hindsight_reflect` tools, all scoped to its bank. What the researcher retains is never recalled by the writer.

## When agents should share a bank

Use one `bank_id` across several agents when the crew is really one team working toward one outcome and should accumulate shared institutional memory.

Good cases for a shared bank:

- **A pipeline about one subject.** Planner, researcher, and writer all working on the same report benefit when a fact the researcher finds is recalled by the writer.
- **Project or session memory.** Everything the team learns about one project belongs in one place, keyed by project rather than by agent.
- **Handoffs that must carry context.** When agent A hands off to agent B, a shared bank means B starts with what A already established, without you serializing it by hand.

The wiring is the same bank for every agent:

```python
from strands import Agent
from hindsight_strands import create_hindsight_tools

BANK = "project-atlas"

planner = Agent(tools=create_hindsight_tools(bank_id=BANK))
researcher = Agent(tools=create_hindsight_tools(bank_id=BANK))
writer = Agent(tools=create_hindsight_tools(bank_id=BANK))
```

Now `hindsight_retain` from any agent writes to `project-atlas`, and `hindsight_recall` from any agent reads the whole team's memory. `hindsight_reflect` synthesizes across everything the crew has stored, which is exactly what you want when several agents each contributed pieces of the picture.

## Connect Strands to Hindsight

This guide assumes the tools are already wired up. If you have not installed and connected `hindsight-strands` yet, follow the [Strands setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-strands-memory-with-hindsight) — it covers `pip install hindsight-strands`, pointing the tools at Hindsight Cloud or a local server with `configure()`, and the native `@tool` pattern that gives each agent `hindsight_retain`, `hindsight_recall`, and `hindsight_reflect`. Then come back here to choose banks.

## A hybrid: shared read, private write

Often you want the best of both: each agent should **read** the team's shared memory but **write** only to its own bank, so private working notes don't leak into the shared record until you promote them.

Hindsight makes this natural because reading and writing are separate tools with their own bank scope. Inject the shared bank through `memory_instructions()` (read), and give the retain tool a private bank (write):

```python
from strands import Agent
from hindsight_strands import create_hindsight_tools, memory_instructions

SHARED = "project-atlas"          # read: team institutional memory
PRIVATE = "researcher"            # write: this agent's own notes

# Recall the shared bank into the system prompt (read side)
shared_context = memory_instructions(bank_id=SHARED)

# Retain/recall tools scoped to the private bank (write side)
tools = create_hindsight_tools(bank_id=PRIVATE)

researcher = Agent(
    tools=tools,
    system_prompt=f"You are a research agent.\n\n{shared_context}",
)
```

The agent starts every run with the team's shared context injected via `memory_instructions()`, but anything it decides to store through `hindsight_retain` lands in its private bank. To promote a finding into shared memory, retain it explicitly to `SHARED` — for example with a separate tool set scoped to the shared bank, or by giving one "publisher" agent write access to the shared bank while the rest stay read-only against it.

You can also narrow what a shared bank returns per agent with recall tags, so several agents share a bank but each pulls its own slice; see `recall_tags` in the [Strands integration docs](https://hindsight.vectorize.io/docs/integrations/strands).

## Verify the bank strategy

Confirm isolation and sharing behave the way you intended:

1. With agent A, store a distinctive fact (e.g. "the API deadline is March 3").
2. With agent B, ask about that fact.
3. If A and B **share** a bank, B should recall it. If they are **isolated**, B should not.
4. For the hybrid, confirm that a private write from A is *not* visible to B until you promote it to the shared bank, but shared context *is* injected into both.

If the result matches your intended strategy, the banks are wired correctly. If not, check that each agent's `bank_id` (and the bank passed to `memory_instructions()`) is exactly what you expect.

## Common mistakes

### Sharing a bank when you needed isolation

If two agents accidentally share a `bank_id`, one agent's noisy or role-specific memories bleed into the other. When agents have different personas or serve different users, give them separate banks.

### Isolating agents that needed to collaborate

The opposite failure: a pipeline where each agent owns a private bank, so handoffs lose context and the writer can't recall what the researcher found. If the crew works on one outcome, share the bank.

### Mismatched read and write banks in the hybrid

In a shared-read/private-write setup, `memory_instructions()` and the retain tool point at different banks on purpose. If you accidentally point them at the same bank, you lose the private-write boundary. Keep the two `bank_id` values distinct and deliberate.

### Building `bank_id` from something unstable

If the bank value changes between runs (a random ID, a timestamp), agents can't recall their own earlier memory. Derive `bank_id` from something stable — the agent role, the project, or the user.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — point the tools at `http://localhost:8888` instead of the Cloud URL. The bank strategy is identical either way.

### Can agents share some memory but not all?

Yes — that is the hybrid pattern. Read a shared bank via `memory_instructions()` while writing to a private bank via the retain tool. You can also share one bank and use `recall_tags` so each agent pulls its own slice.

### Does reflect work across a shared bank?

Yes. `hindsight_reflect` synthesizes an answer across everything in whichever bank it's scoped to, so pointed at a shared bank it reasons over the whole crew's contributions.

### How many banks is too many?

There's no hard limit — banks are cheap and isolation is strict. Let the collaboration structure decide: one bank per isolation boundary, shared banks per team or project.

## Next Steps

- Start with the [Strands setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-strands-memory-with-hindsight) if you haven't wired the tools yet
- Try [Hindsight Cloud](https://hindsight.vectorize.io) for a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Strands integration docs](https://hindsight.vectorize.io/docs/integrations/strands)
