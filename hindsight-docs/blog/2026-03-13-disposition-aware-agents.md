---
title: "How We Built Disposition-Aware Agents That Actually Think Differently"
authors: [chrislatimer]
date: 2026-03-13
tags: [disposition, personality, skepticism, empathy, reflect, agents]
image: /img/blog/disposition-aware-agents.png
---

When we started giving [Hindsight](https://ui.hindsight.vectorize.io/signup)-powered agents "personality," we weren't trying to invent another layer of prompt cosplay. The standard approach to AI agent personality — long blocks of "be skeptical," "be empathetic," "interpret literally" — is fragile, hard to maintain, and doesn't survive contact with real workloads. We wanted something smaller and architectural: a few traits baked into the system that quietly change how the agent thinks, not just how it phrases answers.

<!-- truncate -->

What we ended up with is a three-number disposition model: skepticism, literalism, and empathy. We use these to produce different agent reasoning styles without ever turning into paragraphs of extra prompt text.

## The problem: one AI agent reasoning style for every job

Most systems give you one internal reasoning style and ask you to reuse it for everything:

- A **fact-checker** that should aggressively hunt for contradictions.
- A **legal assistant** that must interpret language literally.
- A **counseling agent** that should care a lot about emotional context.
- A **creative partner** that should say "yes, and..." more than "actually..."

The standard workaround is to stack on instructions:

```text
"Be skeptical of all claims. Question sources. Look for contradictions..."
"Interpret everything literally. Don't make assumptions..."
"Be empathetic. Consider emotional context..."
```

That sort of prompt engineering:

- **Bloats the system prompt.**
- **Is easy to override** or conflict with specific task instructions.
- **Doesn't scale** when you want dozens of distinct agent "personas."

We wanted to be able to say "this bank's agent is skeptical, literal, and low-empathy" in three numbers and let the architecture do the rest.

## The three-trait disposition model

We ended up with three traits, each on a 1-5 scale:

### Skepticism (1-5)

- **1 – Trusting**: takes information at face value, assumes good faith.
- **3 – Balanced**: normal critical thinking.
- **5 – Highly skeptical**: looks for inconsistencies, prefers multiple sources, slow to accept claims.

### Literalism (1-5)

- **1 – Interpretive**: reads between the lines, comfortable inferring.
- **3 – Balanced**: mixes literal reading with inference.
- **5 – Highly literal**: sticks to exactly what is stated, avoids assumptions.

### Empathy (1-5)

- **1 – Detached**: optimizes for correctness, not feelings.
- **3 – Balanced**: considers both facts and emotional context.
- **5 – Highly empathetic**: weighs human impact and emotional state heavily.

These three numbers live in the [memory bank profile](https://docs.hindsight.vectorize.io/recall) and never show up as "do X, don't do Y" rules.

## Why these three traits and a 1-5 scale?

We experimented with richer personality spaces (including a full [Big Five](https://en.wikipedia.org/wiki/Big_Five_personality_traits) vector) and found that, for the kind of reasoning control we actually needed, almost everything collapsed onto three axes:

- **Trust vs. verification → skepticism**
  How quickly should the agent accept a claim vs. demand more evidence?
- **Literal vs. interpretive reading → literalism**
  How far is it allowed to move beyond the exact text into inference?
- **Abstract correctness vs. human impact → empathy**
  How much should it factor in people's feelings, history, and context?

Those dimensions are:

- **Orthogonal enough** to distinguish real roles (fact-checker vs. counselor vs. legal assistant vs. creative partner).
- **Minimal enough** that you can reason about them in your head instead of a spreadsheet.

We use a 1-5 integer scale instead of continuous floats because:

- **It's legible**: "skepticism=4" is easy to read and discuss.
- **It's tunable**: you can feel the difference between 2 and 4 without worrying about 0.73 vs. 0.76.
- **It's stable**: fewer degrees of freedom means fewer weird behavioral edge cases.

Earlier internal versions used a Big Five-style vector; we now project down into these three traits and expose only them. You keep most of the expressive power where it matters (how the agent reasons) with an interface that fits in a JSON bank config.

## How agent disposition shapes reasoning and memory retrieval

The key design choice: we don't turn these traits into more instructions. We surface them as metadata and let the rest of the stack respond.

### 1. Disposition as metadata, not a lecture

We include disposition in the system prompt as part of the bank context, not as a behavior block:

```python
def build_agent_prompt(query: str,
                       context_history: list[dict],
                       bank_profile: dict,
                       additional_context: str | None = None):
    parts = []

    name    = bank_profile.get("name", "Assistant")
    mission = bank_profile.get("mission", "")

    parts.append(f"## Memory Bank Context\nName: {name}")
    if mission:
        parts.append(f"Mission: {mission}")

    disposition = bank_profile.get("disposition", {})
    if disposition:
        traits = []
        if "skepticism" in disposition:
            traits.append(f"skepticism={disposition['skepticism']}")
        if "literalism" in disposition:
            traits.append(f"literalism={disposition['literalism']}")
        if "empathy" in disposition:
            traits.append(f"empathy={disposition['empathy']}")
        if traits:
            parts.append(f"Disposition: {', '.join(traits)}")

    # ... add tool instructions, query, etc. ...
```

Modern models are very good at picking up this kind of contextual cue. `Disposition: skepticism=5, literalism=4, empathy=2` is enough for the model to adjust how it searches, how cautious it is, and how it phrases conclusions.

### 2. Different navigation of the memory hierarchy

Hindsight exposes a [three-tier memory stack](/blog/spreading-activation-memory-graphs):

- **Mental models** — human-written or curated summaries.
- **Observations** — consolidated knowledge derived from raw memories.
- **Raw facts** — individual memories/events, the authoritative source of truth.

The prompt tells the agent what each layer is for; the disposition changes how aggressively it moves up and down the stack:

- **High skepticism** → frequent dives to raw facts, more cross-checking of mental models and observations.
- **High literalism** → heavier weighting on raw facts and precise observations, lighter use of interpretive mental models.
- **High empathy** → preference for mental models and observations that capture people, relationships, and emotional patterns.

Same tools, different "habits" in how they're used.

### 3. Tool-driven loops with different habits

When agents use reflection, they don't need to start with pre-stuffed context. They start with tools: `search_mental_models`, `search_observations`, `recall`, expansion functions, etc. Disposition comes in via the bank profile:

```python
async def run_reflect_agent(
    llm_config: LLMProvider,
    bank_id: str,
    query: str,
    bank_profile: dict[str, Any],  # includes disposition
    search_mental_models_fn,
    search_observations_fn,
    recall_fn,
    expand_fn,
    ...
):
    # The reasoning loop: decide which tools to call, in what order, how often.
```

In practice:

- **Skepticism↑** → more tool calls, more "verify" steps, longer reasoning chains.
- **Literalism↑** → narrower, more precise queries; fewer speculative expansions.
- **Empathy↑** → more use of user history, relationship memories, and "soft" context.

We don't hard-code that behavior; it emerges from combining the traits with a consistent tool surface.

## Agent personality profiles: same code, different reasoning

These are just JSON configs, but they produce noticeably different reasoning styles:

| Profile | Skepticism | Literalism | Empathy | Typical Behavior |
|---------|-----------|-----------|---------|-----------------|
| **Fact-checker** | 5 | 4 | 2 | Calls raw-fact tools frequently, flags stale observations, uses cautious language with explicit caveats |
| **Counselor** | 2 | 2 | 5 | Accepts user's framing, reads between the lines in conversation history, optimizes for emotional validation |
| **Legal assistant** | 3 | 5 | 2 | Quotes exact text from memories, avoids inferring unstated intent, focuses on precise document-like reasoning |
| **Creative partner** | 1 | 1 | 4 | Says "yes, and..." to ideas, pulls in loosely related memories for inspiration, cares about emotional resonance |

All of them run on the same code path. The only difference is three integers.

## Disposition in action: the same question, two different agents

To make this concrete, here's how two agents with different dispositions handle the same query against the same memory bank:

**Query**: "Is the team ready for the product launch next week?"

**Agent A** (skepticism=2, literalism=2, empathy=4):
> Based on the team's recent updates, things look on track. Sarah mentioned feeling stretched thin last Thursday — it might be worth checking in with her directly. The marketing assets are done and engineering closed their last blocker yesterday. The team seems cautiously optimistic.

**Agent B** (skepticism=5, literalism=4, empathy=2):
> Three of five launch checklist items are marked complete as of yesterday. Two remain open: QA sign-off (last updated 3 days ago, no status change since) and partner notification emails (assigned to Sarah, no completion logged). I couldn't find a confirmed QA completion in the raw facts. Recommend verifying before confirming readiness.

Same memories. Same tools. Same model. Different disposition, different answer — and both are useful depending on who's asking and why.

## Disposition-aware agents in multi-agent systems

Once disposition is first-class, it becomes easy to wire up agent ensembles:

- A **skeptical checker** can review outputs from a creative or optimistic agent.
- A **high-empathy agent** can handle user-facing interaction while a low-empathy, high-literalism agent handles internal reasoning.
- You can **fan a query out** to multiple dispositions and show their perspectives side by side.

You don't need separate prompts or models for each; you just vary the disposition and reuse the same infrastructure.

## Tuning guidance: picking the right disposition

If you're setting up a new memory bank agent, here are starting points:

- **Customer support**: skepticism=2, literalism=3, empathy=4 — trust the customer, be precise enough to be helpful, care about their experience.
- **Research assistant**: skepticism=4, literalism=3, empathy=2 — verify claims, allow some inference, prioritize accuracy over feelings.
- **Executive briefing**: skepticism=3, literalism=4, empathy=3 — balanced verification, stick close to facts, moderate awareness of organizational dynamics.

Start at the midpoint (3, 3, 3) and adjust one trait at a time. You'll feel the difference within a few queries.

## Why trait-based AI agent personality works

We didn't get distinct agent "personalities" by writing more clever instructions. We got them by:

1. **Picking three traits** that tightly control how an agent treats evidence, inference, and human context.
2. **Encoding those traits as compact metadata** instead of sprawling instructions.
3. **Letting our [memory hierarchy and tool-driven loops](https://docs.hindsight.vectorize.io/recall)** give those traits room to change behavior.

The result: agents that don't just sound different — they actually process memories differently, which helps agents behave the way you want.

---

*Hindsight gives your AI agents persistent, structured memory with disposition-aware reasoning. [Learn more about how Hindsight works](https://docs.hindsight.vectorize.io/recall), or [get started free](https://ui.hindsight.vectorize.io/signup).*