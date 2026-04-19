

# Observations: Knowledge Consolidation

After memories are retained, Hindsight automatically consolidates related facts into **observations** — deduplicated, evidence-grounded beliefs the bank has built up from multiple memories. Each observation tracks its supporting evidence (with exact quotes), a proof count, and a computed freshness trend, and is refined rather than overwritten when new evidence arrives.

```mermaid
graph LR
    A[New Facts] --> B[Consolidation Engine]
    B --> C{Existing Observation?}
    C -->|Yes| D[Refine Observation]
    C -->|No| E[Create Observation]
    D --> F[Observations]
    E --> F
```

---

## What Are Observations?

Observations are **consolidated knowledge** built from multiple facts. Unlike raw facts — which are individual pieces of information — observations represent deduplicated beliefs, preferences, and learnings grounded in accumulated evidence. They are not summaries the LLM invents on the fly: each observation is backed by specific source memories, carries a proof count, and evolves as new evidence supports, contradicts, or extends it.

| Raw Facts | Observation |
|-----------|--------------|
| "Alice prefers Python" | "Alice is a Python-focused developer who values readability and simplicity" |
| "Alice dislikes verbose code" | |
| "Alice recommends type hints" | |

Observations provide:
- **Deduplication**: One durable belief instead of many overlapping facts
- **Grounding**: Every observation references the specific memories (with quotes) that support it
- **Evolution**: Refined as evidence strengthens, weakens, or contradicts it — history is preserved
- **Freshness signal**: A computed trend (stable / strengthening / weakening / new / stale) tells you whether the belief still holds
- **Efficiency**: Condensed knowledge for faster retrieval

---

## How Consolidation Works

### Automatic Background Processing

After `retain()` completes, the consolidation engine runs automatically:

1. **New facts analyzed** — Each new fact is compared against existing observations
2. **Pattern detection** — Related facts are grouped and synthesized
3. **Observation creation/update** — New observations are created or existing ones refined
4. **Evidence tracking** — Each observation maintains references to supporting facts

### Evidence-Based Evolution

Observations evolve as new evidence arrives:

| Event | What the bank learns | Observation state |
|-------|---------------------|----------------|
| **Day 1** | "Redis is open source under BSD license" | "Redis is excellent for caching — fast, reliable, and OSS-friendly" (2 supporting facts) |
| **Day 2** | "Redis has great community support" | Observation reinforced (3 supporting facts) |
| **Day 30** | "Redis changed license to SSPL" | Observation refined: "Redis is technically strong, but has license concerns for cloud" |
| **Day 45** | "Valkey forked Redis under BSD" | New observation: "Consider Valkey for new projects requiring true OSS" |

### Handling Contradictory Evidence

What happens when a new fact contradicts an existing observation?

The consolidation engine doesn't blindly overwrite — it **reconciles** the contradiction by capturing the evolution:

**Example: User preference changes**

| Time | Fact | Observation |
|------|------|--------------|
| Week 1 | "User says they love React" | "User prefers React for frontend development" |
| Week 2 | "User praises React's component model" | "User is enthusiastic about React, particularly its component model" |
| Week 3 | "User says they've switched to Vue and won't use React anymore" | "User was previously a React enthusiast who appreciated its component model, but has now switched to Vue and no longer uses React" |

Notice how the final observation captures the **full journey** — not just "User prefers Vue" but the complete evolution of their preference. This nuanced understanding means:

- Your agent won't recommend React tutorials to someone who explicitly moved away from it
- Your agent understands *why* this matters (they were enthusiastic before, so this is a deliberate choice)
- Your agent can reference this history when relevant ("I know you used to work with React...")

The system:
1. **Detects the conflict** — New fact contradicts existing observation
2. **Preserves history** — Incorporates the previous understanding into the new observation
3. **Creates nuanced observation** — Synthesizes a richer understanding that captures the change
4. **Updates freshness** — Marks the observation as recently updated

**Example: Correcting misinformation**

| Time | Fact | Observation |
|------|------|--------------|
| Day 1 | "Alice works at Google" | "Alice is a Google employee" |
| Day 10 | "Alice actually works at Meta, not Google" | "Alice works at Meta (previously thought to work at Google)" |

When a fact explicitly corrects previous information, the observation is updated to reflect the correction while noting the previous understanding. The raw facts are always preserved, so you can trace back to see what was originally stated and when it was corrected.

---

## Observations in Retrieval

Observations are automatically included in both `recall()` and `reflect()` operations:

### In Recall

Observations are returned alongside raw facts, filtered by the `types` parameter:

```python
# Include observations in recall
results = client.recall(
    bank_id="my-bank",
    query="What programming languages does Alice prefer?",
    types=["world", "experience", "observation"]
)

# Observations only
observations = client.recall(
    bank_id="my-bank",
    query="What patterns have I learned?",
    types=["observation"]
)
```

### In Reflect

The reflect agent uses **hierarchical retrieval**:

1. **[Mental Models](api/mental-models.md)** — User-curated summaries (highest priority)
2. **Observations** — Consolidated knowledge with freshness awareness
3. **Raw Facts** — Ground truth for verification

The agent automatically queries observations and uses them to inform its reasoning.

---

## Freshness Awareness

Observations track when they were last updated. During reflect, the agent considers freshness:

- **Fresh observations**: Used directly for reasoning
- **Stale observations**: Agent verifies against current facts before relying on them

This ensures responses stay accurate even as the underlying data changes.

---

## Observation Scopes

By default, observations are scoped to all of a memory's tags combined. The `observation_scopes` retain parameter lets you control this — building separate observations per tag, per combination, or with a custom list of scopes. This is key when a single memory carries multiple tags and you want each tag to accumulate its own observations independently.

See [`observation_scopes` in the Retain API](./api/retain#observation_scopes) for the full explanation and options.

---

## Observations Mission

You can define exactly what this bank should synthesise by setting an **observations mission** (`observations_mission`). This replaces the built-in durable-knowledge rules with your own instructions, letting you control what shape observations take.

```
e.g. Observations are stable facts about people and projects.
     Always include preferences, skills, and recurring patterns.
     Ignore one-off events and ephemeral state.
```

Leave it blank to use the server default — durable, specific facts that stay true over time (preferences, skills, relationships, recurring patterns), with ephemeral state filtered out.

**Examples:**

| `observations_mission` | What gets synthesised |
|------------------------|----------------------|
| *(unset — default)* | Durable facts: preferences, skills, relationships, recurring patterns |
| *"Observations are weekly summaries of sprint outcomes and blockers"* | Broad event summaries grouped by time period |
| *"Observations are stable facts about named individuals only"* | Person-centric knowledge, tied to specific people |
| *"Observations are recurring patterns in customer support interactions"* | Failure modes, common requests, pain points |

Set `observations_mission` via the [bank config API](api/memory-banks.md#observations-configuration) or the [`HINDSIGHT_API_OBSERVATIONS_MISSION`](configuration.md#observations) environment variable.

---

## Observation Lifecycle & Invalidation

### When Memories Are Deleted

Observations are derived from source memories. When source memories are removed, Hindsight automatically keeps observations consistent:

| Action | Effect on observations |
|--------|----------------------|
| Delete a document | All observations derived from the document's memories are deleted |
| Delete individual memories (by type) | Observations sourced from those memories are deleted |
| Delete an entire bank | All observations are deleted along with everything else |

After deletion, the **remaining source memories** that fed the affected observations have their consolidation state reset, so they will be re-consolidated on the next consolidation run and produce fresh observations.

### Clearing Observations for a Specific Memory

You can clear all observations derived from a single memory without deleting the memory itself. This is useful when you want to force re-synthesis of a memory's contribution to consolidated knowledge.

Use the `DELETE /v1/default/banks/{bank_id}/memories/{memory_id}/observations` endpoint. This will:
1. Delete all observations that list the memory as a source
2. Reset `consolidated_at` on the memory itself and any other source memories that contributed to those observations
3. Trigger a consolidation job so fresh observations are produced automatically

### Resetting All Observations

To wipe all consolidated knowledge and start over:

```python
# Clear all observations for a bank
client.clear_observations(bank_id="my-bank")
```

This resets the consolidation state for all source memories in the bank, so the next consolidation run will re-derive all observations from scratch.

---

## Configuration

Observation consolidation runs automatically. You can monitor consolidation via the [Operations API](./api/operations).

---

## Next Steps

- [**Retain**](./retain) — How facts are stored and trigger consolidation
- [**Recall**](./retrieval) — How observations are retrieved
- [**Reflect**](./reflect) — How the agentic loop uses observations
- [**Mental Models**](./api/mental-models) — User-curated summaries for common queries
