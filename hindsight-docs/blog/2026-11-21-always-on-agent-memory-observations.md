---
title: "How to Keep an Always-On Agent's Memory Up to Date"
description: "Store a scheduled agent's memory in Hindsight. New facts update old observations instead of piling up beside them. Python code for each run."
authors: [chrislatimer]
date: 2026-11-21
slug: "2026/11/21/always-on-agent-memory-observations"
tags: [hindsight, memory, agent-memory, always-on-agents, observations, mental-models, recall, retain]
hide_table_of_contents: false
image: /img/blog/always-on-agent-memory-observations/featured.png
---

import ThemedImage from '@theme/ThemedImage';

You run a Claude agent on a schedule. It answers the support queue at 6 a.m., or it triages issues every hour, or it drafts the weekly newsletter. Nobody watches it work. That is the point.

Then one morning it does something you told it to stop doing weeks ago. You open its notes file. The old rule is still in it, a few hundred lines above the new one. The agent read the old one.

<!-- truncate -->

## You are not the only one

On 13 November 2026, Anthropic published [Building always-on agents](https://claude.com/resources/articles/building-always-on-agents), a design guide by William Hu. A few days later, a [thread on r/ClaudeAI](https://www.reddit.com/r/ClaudeAI/comments/1xxdzcf/anthropic_published_a_guide_on_building_alwayson/) asked people which part of the guide gave them the most trouble. The person who started the thread wrote:

> "I run a daily scheduled Claude task, and memory is the part where mine is weakest."

The replies in that one discussion describe the same failure in three ways, from three different people.

**The file grows past what the agent reads.** [u/gptrv](https://www.reddit.com/r/ClaudeAI/comments/1xxdzcf/comment/19spt8wdx8r4y/) runs always-on Claude Code agents for support, code and marketing:

> "What failed first: one notes file the agent appended to. It grew past what the agent reliably read, old facts contradicted new ones, and it acted on a rule we had dropped weeks before."

**New notes and old notes fight.** [u/manojbadam](https://www.reddit.com/r/ClaudeAI/comments/1xxdzcf/comment/19sre8lnlgxup/) wrote:

> "Most common issue being new learnings overrides the old memory."

**The detail disappears.** [u/Sun77177](https://www.reddit.com/r/ClaudeAI/comments/1xxdzcf/comment/19spog1uswwgw/) wrote:

> "the agent kept rewriting its own summaries until the detail was gone, and then it would confidently do the step again."

<ThemedImage
  alt="Three reports describe a file the agent no longer reads reliably, new learning overriding old memory, and detail lost in rewrites."
  sources={{light: "/img/blog/always-on-agent-memory-observations/notes-file-failures-light.svg", dark: "/img/blog/always-on-agent-memory-observations/notes-file-failures-dark.svg"}}
/>

The cost is more than a wrong reply. An always-on agent is only useful if you can leave it alone. When its memory goes bad, you start to read every run by hand again. The agent stops saving you time.

## What the guide says about memory

- The guide names five core primitives: "an event inbox, timers, memory, permissions, and an outbox." On memory, it says:

  > "Always-on agents work across many disjointed runs and rely on memory to maintain continuity."

- It gives a rule of thumb:

  > "memory should only hold what the agent can't look up again."

- It describes the cycle of each run:

  > "The agent reads its memory to get a baseline, works within its permissions to take actions, and then updates its memory with anything it has learned."

The guide also says that in Anthropic's own products, memory "is implemented as a directory of files." It says you can also add dreaming, which "can replace outdated entries and find patterns that only show up across many runs."

The guide does not name a memory product. The rest of this post shows one way to fill the memory primitive: a memory store that, by default, reconciles outdated entries after each retain, and keeps the original facts.

## Why a notes file goes wrong

The people in the thread found the cause themselves. u/gptrv now gives every line "the date it was learned," and when two lines disagree, "the newer one wins." A separate pass "reads the session once it goes quiet and keeps the decisions." u/Sun77177 moved to "an append only log that nothing ever rewrites" plus a short index and one task file per job.

Those fixes work. Each one is also code that you now own. You wrote a reconciler, a dating scheme and a log format, and you maintain them next to the agent you meant to build.

A plain file has no way to know that two lines disagree. It also has no way to know that a summary lost a detail, because the original is gone.

| Question | One notes file | Dated lines or an indexed log | Hindsight observations |
|---|---|---|---|
| What happens to an old rule? | An old rule can stay beside a newer one. [u/gptrv](https://www.reddit.com/r/ClaudeAI/comments/1xxdzcf/comment/19spt8wdx8r4y/) | With dated lines, the newer line wins when two lines disagree. [u/gptrv](https://www.reddit.com/r/ClaudeAI/comments/1xxdzcf/comment/19spt8wdx8r4y/) | Consolidation can refine the observation and record what changed. [Observations](https://hindsight.vectorize.io/developer/observations) |
| Where does the detail go? | Additions stay in the file. It can grow past what the agent reliably reads. [u/gptrv](https://www.reddit.com/r/ClaudeAI/comments/1xxdzcf/comment/19spt8wdx8r4y/) | A separate pass keeps decisions, or an append-only log keeps entries. [u/gptrv](https://www.reddit.com/r/ClaudeAI/comments/1xxdzcf/comment/19spt8wdx8r4y/), [u/Sun77177](https://www.reddit.com/r/ClaudeAI/comments/1xxdzcf/comment/19spog1uswwgw/) | Observations reference supporting memories with quotes. Hindsight preserves raw facts. [Observations](https://hindsight.vectorize.io/developer/observations) |
| What code do you own? | The agent appends to the file. [u/gptrv](https://www.reddit.com/r/ClaudeAI/comments/1xxdzcf/comment/19spt8wdx8r4y/) | Add dates and a separate pass, or maintain a log, index, and task files. [u/gptrv](https://www.reddit.com/r/ClaudeAI/comments/1xxdzcf/comment/19spt8wdx8r4y/), [u/Sun77177](https://www.reddit.com/r/ClaudeAI/comments/1xxdzcf/comment/19spog1uswwgw/) | Wire recall and retain into your agent. By default, Hindsight consolidates the facts in the background. [Recall](https://hindsight.vectorize.io/developer/api/recall), [Retain](https://hindsight.vectorize.io/developer/api/retain), [Observations](https://hindsight.vectorize.io/developer/observations) |
| What does the agent read on each run? | The agent reads from the notes file and can miss parts. [u/gptrv](https://www.reddit.com/r/ClaudeAI/comments/1xxdzcf/comment/19spt8wdx8r4y/) | u/gptrv keeps "a short list of lines that is always loaded, plus topic files the agent opens only before related work." [u/gptrv](https://www.reddit.com/r/ClaudeAI/comments/1xxdzcf/comment/19spt8wdx8r4y/) | Recall returns relevant observations and raw facts if you request both. You can also load a stored mental model. [Recall](https://hindsight.vectorize.io/developer/api/recall), [Mental Models](https://hindsight.vectorize.io/developer/mental-models) |

## Where Hindsight fits

We build [Hindsight](https://github.com/vectorize-io/hindsight), an open-source memory system for agents. Hindsight fills the memory primitive only. You still need your own inbox, timers, permissions and outbox.

Hindsight does three things that map to the three failures in the thread. Each one comes from the docs.

### The agent recalls what matters, not the whole file

The agent does not read a growing file. It asks a question, and [recall](https://hindsight.vectorize.io/developer/api/recall) runs "four retrieval strategies in parallel — semantic similarity, keyword (BM25), graph traversal, and temporal — then fuses and reranks the results into a single ranked list."

You set a token budget with `max_tokens` (the default is 4096). The docs say: "facts are included in relevance order until this budget is exhausted — so you always get the most relevant memories that fit." The bank can grow for months. The memories in the agent's context per run stay within the token budget you set.

### A new fact refines an old belief. It does not pile up

By default, after each [retain](https://hindsight.vectorize.io/developer/api/retain), Hindsight consolidates related facts into [observations](https://hindsight.vectorize.io/developer/observations) in the background. The docs describe them as "deduplicated, evidence-grounded beliefs." Each observation "tracks its supporting evidence (with exact quotes) and a proof count, and is refined rather than overwritten when new evidence arrives."

The docs say what happens when a new fact contradicts an old one: "The consolidation engine doesn't blindly overwrite — it reconciles the contradiction by capturing the evolution." Here is one of the two examples on the Observations page:

| Time | Fact | Observation |
|---|---|---|
| Day 1 | "Alice works at Google" | "Alice is a Google employee" |
| Day 10 | "Alice actually works at Meta, not Google" | "Alice works at Meta (previously thought to work at Google)" |

Put a dropped rule in place of Alice's job. The observation says the rule changed. The old fact stays in the bank under it, so you can see when the rule changed.

By default, consolidation runs in the background after retain, so an observation can lag behind the newest facts. The docs cover that case. When newer memories are not consolidated yet, [reflect](https://hindsight.vectorize.io/developer/api/reflect) "treats the affected observations as stale and verifies them against raw facts." We wrote about this in [Staleness-Aware Memory](/blog/2026/06/17/freshness-aware-memory).

### The detail stays under the belief

An observation is not a summary that replaces its source. The docs say: "Every observation references the specific memories (with quotes) that support it." They also say: "The raw facts are always preserved, so you can trace back to see what was originally stated and when it was corrected."

That is the opposite of what u/Sun77177 saw. The agent reads the short belief, and it can still get the original facts when it needs them.

## The plan: three steps for a scheduled agent

<ThemedImage
  alt="At run start, recall memory and read a mental model if configured. The agent works, then retains the run record. Background consolidation can lag behind retain."
  sources={{light: "/img/blog/always-on-agent-memory-observations/scheduled-run-cycle-light.svg", dark: "/img/blog/always-on-agent-memory-observations/scheduled-run-cycle-dark.svg"}}
/>

The guide's cycle is "reads its memory", "works", "updates its memory". The plan follows that cycle. The code uses the Python client from the [Quick Start](https://hindsight.vectorize.io/developer/api/quickstart). The current release is 1.2.3.

### Step 1. Run Hindsight

Choose one of these.

**Hindsight Cloud.** Sign up with **Start free** on [hindsight.vectorize.io](https://hindsight.vectorize.io) and get an API key. Then:

```bash
pip install hindsight-client
export HINDSIGHT_API_URL=https://api.hindsight.vectorize.io
export HINDSIGHT_API_KEY=hsk_your-key-here
```

**The open-source server.** Run it with Docker. Hindsight needs an LLM with structured output to extract facts and build observations. This command uses an OpenAI key.

```bash
export OPENAI_API_KEY=sk-xxx

docker run -it --pull always --name hindsight --restart unless-stopped --shm-size=1g -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

```bash
pip install hindsight-client
export HINDSIGHT_API_URL=http://localhost:8888
```

The API runs on `http://localhost:8888`. The control plane runs on `http://localhost:9999`. In the control plane, open a bank and go to **Pages → Observations** to browse what the bank has learned.

Create one bank for the agent. Do this once. The `retain_mission` tells the bank what to pay attention to when it extracts facts.

```python
import os
from hindsight_client import Hindsight

client = Hindsight(
    base_url=os.environ["HINDSIGHT_API_URL"],
    api_key=os.environ.get("HINDSIGHT_API_KEY"),
)

client.create_bank(
    bank_id="support-agent",
    retain_mission="Track the rules, corrections and decisions this support agent must follow.",
)
```

### Step 2. At the start of each run, recall from the bank

When the scheduler wakes the agent, use the event as the query. Ask for all three memory types. Observations give the current beliefs. Raw facts cover anything that consolidation has not reached yet.

```python
def load_memory(event_text: str) -> str:
    response = client.recall(
        bank_id="support-agent",
        query=event_text,
        types=["world", "experience", "observation"],
        max_tokens=4096,
    )
    return "\n".join(f"- [{r.type}] {r.text}" for r in response.results)
```

Put the result in the agent's context with the event. That is the "baseline" the guide describes.

When the agent needs to check why it believes something, ask for the facts under each observation:

```python
response = client.recall(
    bank_id="support-agent",
    query="refund rules for old tickets",
    types=["observation"],
    include_source_facts=True,
    max_source_facts_tokens=4096,
)

for obs in response.results:
    print(f"Observation: {obs.text}")
    if obs.source_fact_ids and response.source_facts:
        print("  Derived from:")
        for fact_id in obs.source_fact_ids:
            fact = response.source_facts.get(fact_id)
            if fact:
                print(f"    - [{fact.type}] {fact.text}")
```

If you want one written answer in place of a list, use reflect. Reflect runs an agentic loop over the bank and returns a synthesized answer in `text`. With `include_facts=True`, the response also lists the memories it used, so you can see why the agent decided what it did.

```python
response = client.reflect(
    bank_id="support-agent",
    query="Can I offer a refund on ticket 4517? It is 45 days old.",
    include_facts=True,
)

print(response.text)
for fact in (response.based_on.memories if response.based_on else []):
    print(f"  - [{fact.type}] {fact.text}")
```

### Step 3. At the end of each run, retain what the run did

Send the run's record to the bank. Include what the user said, what the agent did, and any correction. Give it a `context` label and a `timestamp`, so each fact carries a source and a date. The docs say context "is injected directly into the LLM prompt, so it actively shapes how facts are extracted."

```python
from datetime import datetime, timezone

def save_run(run_id: str, run_log: str) -> None:
    client.retain(
        bank_id="support-agent",
        content=run_log,
        context="scheduled support run",
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        document_id=f"run-{run_id}",
    )

save_run(
    "2026-11-21-0600",
    "Dana (2026-11-21T06:02:00Z): Stop offering refunds on tickets older than 30 days. "
    "That rule is dropped.\n"
    "Agent (2026-11-21T06:40:00Z): Answered 14 tickets. Escalated ticket 4512 to billing.",
)
```

:::warning
Give each run its own `document_id`. The `save_run` code does not set `update_mode`, so the default `"replace"` applies. A newer retain with the same `document_id` then replaces that document, and the memories that came only from the earlier text are deleted.
:::

You do not edit an old note or delete the old rule. By default, consolidation reconciles the new fact with what the bank already holds. You do not write a reconciler or a dreaming pass.

## When every run needs the same answer

Some questions come up on every run: "Which rules apply now?" For those, Hindsight has [mental models](https://hindsight.vectorize.io/developer/mental-models).

The docs describe a mental model as "a standing answer to a question about a bank. You define the question once; Hindsight writes the answer, keeps it stored, and rewrites it in the background as the bank learns more." Reading one "is a database read. No retrieval, no synthesis, no LLM call, no waiting."

Mental models also answer the third failure in the thread. The docs say that a document an LLM rewrites many times "will still drift." The [Mental Models API](https://hindsight.vectorize.io/developer/api/mental-models) page says that in `delta` mode, a refresh "emits a list of typed operations" against the document, and "Sections that aren't targeted by any operation are copied through byte-identical." Each model also "keeps the previous version of its content every time it changes."

Create one when you set up the agent:

```python
client.create_mental_model(
    bank_id="support-agent",
    name="Current operating rules",
    source_query="Which rules must the support agent follow now, and which rules were dropped?",
    id="operating-rules",
    trigger={"refresh_after_consolidation": True, "mode": "delta"},
)
```

The call runs reflect in the background and returns an `operation_id`, so `content` can be empty until the first refresh completes. With `refresh_after_consolidation`, the model is rebuilt each time the bank's observations are consolidated. If you prefer a fixed time, use `refresh_cron` in its place, for example `"refresh_cron": "0 3 * * *"`. The [Mental Models API](https://hindsight.vectorize.io/developer/api/mental-models) says the two options are mutually exclusive.

At the start of each run, read the model and put it in the agent's prompt. This matches the guide's advice that user instructions be "loaded at the start of every run."

```python
rules = client.get_mental_model(
    bank_id="support-agent",
    mental_model_id="operating-rules",
)

system_prompt = BASE_PROMPT + "\n\nCurrent operating rules:\n" + rules.content
```

Two more things matter for an agent that nobody watches:

- **Know when the answer is fresh.** Since 1.2.2, the [changelog](https://hindsight.vectorize.io/changelog) says: "Webhooks now emit a mental_model.refreshed event when a mental-model refresh completes." Your inbox can take that event like any other.
- **Know when it stops.** The API docs say a failed refresh is retried by the worker (3 times by default) and then stops, and the automatic refreshes of that model pause until a refresh succeeds. Bank stats (`GET /v1/default/banks/{bank_id}/stats`) count these models as `paused_mental_models`. Alert when that number is above 0.

## What happens if nothing changes

The notes file keeps growing. The next rule you drop stays in it, next to the new rule, until an unattended run picks the wrong one.

## What it looks like after

The 6 a.m. run starts. The agent reads the current rules and recalls a short list for today's queue. One observation says the refund rule for old tickets was dropped on 21 November. Under it are the facts that say so, with Dana's words.

The agent follows the new rule. You read the outbox, not the agent's memory. You go from an agent you check after every run to an agent you check only when a result surprises you.

## Next steps

**Do this now:** run Hindsight with the [Quick Start](https://hindsight.vectorize.io/developer/api/quickstart), or sign up for Hindsight Cloud with **Start free** on [hindsight.vectorize.io](https://hindsight.vectorize.io). Then add the recall call from step 2 and the retain call from step 3 to one scheduled agent.

**Read more first:** the [Observations](https://hindsight.vectorize.io/developer/observations) page explains consolidation, contradictions and freshness in full. For the design behind it, read our post on [agent memory consolidation](/blog/2026/05/21/agent-memory-consolidation). If several tools need the same memory, see how [a Grok Bot reads what your other AI tools already learned](/blog/2026/10/09/grok-bot-agent-memory).
