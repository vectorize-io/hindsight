---
sidebar_position: 7
---

# Entities

Entities are the people, organizations, places, and concepts that Hindsight automatically extracts and tracks across your memory bank.

:::info Automatic Feature
You don't need to do anything to use entities—Hindsight extracts them automatically when you call `retain`. However, understanding how entities work is important because they power key features in [recall](./recall) and [reflect](./reflect).
:::

## Why Entities Matter

Entities improve recall quality in two ways:

1. **Co-occurrence tracking** — When entities appear together in facts, Hindsight builds a graph of relationships. This enables graph-based recall to find indirect connections.

2. **Observations** — Hindsight synthesizes high-level summaries about each entity from multiple facts. Including entity observations in recall provides richer context.

## What Gets Extracted?

When you retain information, the LLM extracts named entities from each fact:

- **People** — Names like "Alice", "Dr. Smith", "CEO John"
- **Organizations** — Companies, teams, institutions
- **Places** — Cities, countries, specific locations
- **Products/Objects** — Software, tools, significant items
- **Concepts** — Abstract themes like "career growth", "friendship"

**Example:**

```
Content: "Alice works at Google in Mountain View. She specializes in TensorFlow."

Entities extracted:
- Alice (person)
- Google (organization)
- Mountain View (location)
- TensorFlow (product)
```

## Entity Resolution

When the same entity is mentioned multiple times (possibly with different names), Hindsight resolves them to a single canonical entity using a scoring algorithm:

### Resolution Factors

1. **Name similarity (50%)** — How closely the text matches existing entity names. Handles variations like "Alice" vs "Alice Chen" or partial matches.

2. **Co-occurrence (30%)** — Entities that frequently appear together are more likely to be the same. If "Alice" always appears with "Google" and "TensorFlow", a new mention of "Alice" near those entities scores higher for matching.

3. **Temporal proximity (20%)** — Recent mentions are weighted more heavily. If an entity was seen in the last 7 days, new similar mentions are more likely to match.

### Resolution Threshold

A match requires a combined score above **0.6** (60%). Below this threshold, Hindsight creates a new entity rather than risk merging distinct entities.

This means:
- Exact name matches with recent co-occurring entities → strong match
- Partial name matches without context → likely creates new entity
- Same name in completely different contexts → may create separate entities

## Entity Observations

Observations are **derived state**—high-level summaries that Hindsight automatically synthesizes from the facts associated with an entity. They provide a condensed view of what the system knows about important entities.

**Example:**

Facts about Alice:
- "Alice works at Google"
- "Alice is a software engineer"
- "Alice specializes in ML"
- "Alice joined Google in 2020"
- "Alice leads the search team"

Observation created:
- "Alice is a software engineer at Google who joined in 2020, specializes in ML, and leads the search team"

### How Observations Work

Observations are **not generated for every entity**. When you retain new documents:

1. **Top entities selected** — Hindsight identifies the top 5 most-mentioned entities in the batch
2. **Threshold check** — Only entities with at least 5 facts get observations
3. **Regeneration** — Observations are regenerated using the entity's most recent 50 facts
4. **Old observations replaced** — Previous observations are deleted and new ones created

This means:
- Frequently mentioned entities get observations; rarely mentioned ones don't
- Observations stay up-to-date as new information is retained
- The system prioritizes entities that matter most to your memory bank

### Observations vs Opinions

Observations are **objective summaries**—they synthesize facts without any bias or perspective. This is different from [opinions](./opinions), which are influenced by the memory bank's disposition.

| | Observations | Opinions |
|---|---|---|
| **Purpose** | Summarize what's known about an entity | Express the bank's perspective on a topic |
| **Disposition influence** | No | Yes |
| **Scope** | Per-entity | Any topic |
| **Generation** | Automatic (top entities) | On-demand via reflect |

### Using Observations

Observations are included in recall results when you set `include_entities=True`. They provide quick context about key entities without retrieving all underlying facts.

## Next Steps

- [**Recall**](./recall) — Use entities in memory retrieval
- [**Reflect**](./reflect) — Get entity-aware responses
