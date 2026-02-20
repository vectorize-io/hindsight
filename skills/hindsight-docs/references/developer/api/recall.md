
# Recall Memories

Retrieve memories from a bank using multi-strategy recall.

When you **recall**, Hindsight runs four retrieval strategies in parallel — semantic similarity, keyword (BM25), graph traversal, and temporal — then fuses and reranks the results into a single ranked list. The response contains structured facts, not raw documents.

{/* Import raw source files */}

:::info How Recall Works
Learn about the four retrieval strategies (semantic, keyword, graph, temporal) and RRF fusion in the [Recall Architecture](/developer/retrieval) guide.
> **💡 Prerequisites**
> 
Make sure you've completed the [Quick Start](./quickstart) to install the client and start the server.
## Basic Recall

### Python

```python
response = client.recall(bank_id="my-bank", query="What does Alice do?")

# response.results is a list of RecallResult objects, each with:
# - id:             fact ID
# - text:           the extracted fact
# - type:           "world", "experience", or "observation"
# - context:        context label set during retain
# - metadata:       dict[str, str] set during retain
# - tags:           list of tags
# - entities:       list of entity name strings linked to this fact
# - occurred_start: ISO datetime of when the event started
# - occurred_end:   ISO datetime of when the event ended
# - mentioned_at:   ISO datetime of when the fact was retained
# - document_id:    document this fact belongs to
# - chunk_id:       chunk this fact was extracted from

# Example response.results:
# [
#   RecallResult(id="a1b2...", text="Alice works at Google as a software engineer", type="world", context="career", ...),
#   RecallResult(id="c3d4...", text="Alice got promoted to senior engineer", type="experience", occurred_start="2024-03-15T00:00:00Z", ...),
# ]
```

### Node.js

```javascript
const response = await client.recall('my-bank', 'What does Alice do?');

// response.results is an array of result objects, each with:
// - id:            fact ID
// - text:          the extracted fact
// - type:          "world", "experience", or "observation"
// - context:       context label set during retain
// - metadata:      Record<string, string> set during retain
// - tags:          string[] of tags
// - entities:      string[] of entity names linked to this fact
// - occurredStart: ISO datetime of when the event started
// - occurredEnd:   ISO datetime of when the event ended
// - mentionedAt:   ISO datetime of when the fact was retained
// - documentId:    document this fact belongs to
// - chunkId:       chunk this fact was extracted from

// Example response.results:
// [
//   { id: "a1b2...", text: "Alice works at Google as a software engineer", type: "world", context: "career", ... },
//   { id: "c3d4...", text: "Alice got promoted to senior engineer", type: "experience", occurredStart: "2024-03-15T00:00:00Z", ... },
// ]
```

### CLI

```bash
hindsight memory recall my-bank "What does Alice do?"
```

---

## Parameters

### query

The natural language question or statement to search for. This is the only required field. The query drives all four retrieval strategies simultaneously: it is embedded for semantic search, tokenized for BM25 keyword search, used to seed graph traversal, and parsed for temporal expressions. After retrieval, the raw query text is also passed to the cross-encoder reranker to re-score every candidate. Queries exceeding 500 tokens are rejected.

### types

Controls which categories of memory facts are searched. Accepted values are `world` (objective facts), `experience` (events and conversations), and `observation` (consolidated knowledge synthesized over time). When omitted, all three types are searched.

Each type runs the full four-strategy retrieval pipeline independently, so narrowing `types` reduces both the result set and query cost.

### Python

```python
# Only world facts (objective information)
world_facts = client.recall(
    bank_id="my-bank",
    query="Where does Alice work?",
    types=["world"]
)
```
```python
# Only experience (conversations and events)
experience = client.recall(
    bank_id="my-bank",
    query="What have I recommended?",
    types=["experience"]
)
```
```python
# Only observations (consolidated knowledge)
observations = client.recall(
    bank_id="my-bank",
    query="What patterns have I learned?",
    types=["observation"]
)
```

### CLI

```bash
hindsight memory recall my-bank "query" --fact-type world,observation
```

> **💡 About Observations**
> 
Observations are consolidated knowledge synthesized from multiple facts over time — patterns, preferences, and learnings the memory bank has built up. They are created automatically in the background after retain operations.
### budget

Controls retrieval depth and breadth. Accepted values are `low`, `mid` (default), and `high`. Use `low` for fast simple lookups, `mid` for balanced everyday queries, and `high` when you need to find indirect connections or exhaustive coverage.

### Python

```python
# Quick lookup
results = client.recall(bank_id="my-bank", query="Alice's email", budget="low")

# Deep exploration
results = client.recall(bank_id="my-bank", query="How are Alice and Bob connected?", budget="high")
```

### Node.js

```javascript
// Quick lookup
const quickResults = await client.recall('my-bank', "Alice's email", { budget: 'low' });

// Deep exploration
const deepResults = await client.recall('my-bank', 'How are Alice and Bob connected?', { budget: 'high' });
```

### max_tokens

The maximum number of tokens the returned facts can collectively occupy. Defaults to `4096`. Only the `text` field of each fact is counted toward this budget — metadata, tags, entities, and other fields are not included. After reranking, facts are included in relevance order until this budget is exhausted — so you always get the most relevant memories that fit. Hindsight is designed for agents, which think in tokens rather than result counts: set `max_tokens` to however much of your context window you want to allocate to memories.

### Python

```python
# Fill up to 4K tokens of context with relevant memories
results = client.recall(bank_id="my-bank", query="What do I know about Alice?", max_tokens=4096)

# Smaller budget for quick lookups
results = client.recall(bank_id="my-bank", query="Alice's email", max_tokens=500)
```

### query_timestamp

An ISO 8601 datetime representing when the query is being asked, from the user's perspective. When provided, it is used as the anchor for resolving relative temporal expressions in the query — for example, if the query says "last month" and `query_timestamp` is `2023-05-30`, the temporal search window becomes approximately April 2023. Without it, the server's current time is used as the anchor. This field matters most for replaying historical conversations or building agents that need time-anchored recall.

### include

An optional object controlling supplementary data returned alongside the main facts.

#### chunks

When enabled, the response includes the raw source text chunks from which each fact was extracted. Chunks are fetched before the `max_tokens` filter, so setting `max_tokens=0` returns no facts but can still return chunks. The `max_tokens` sub-option (default `8192`) controls the total chunk token budget independently of the main fact budget. This is useful when agents need surrounding context beyond the extracted fact text.

:::note
When `include_chunks` is enabled, chunks are fetched based on the top-scored reranked results before token filtering. The last chunk is truncated (not dropped) to fit exactly within the budget, and each chunk carries a `truncated` flag indicating whether it was cut.
#### source_facts

When enabled and `types` includes `observation`, each observation result is accompanied by the original contributing facts it was synthesized from. Source facts are returned in a top-level `source_facts` dict keyed by fact ID, and each observation result carries a `source_fact_ids` list for cross-referencing. Facts are deduplicated across observations. The `max_tokens` sub-option (default `4096`) limits the total token budget for source facts.

### Python

```python
# Recall observations and include their source facts
response = client.recall(
    bank_id="my-bank",
    query="What patterns have I learned about Alice?",
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

### Node.js

```javascript
// Recall observations and include their source facts
const obsResponse = await client.recall('my-bank', 'What patterns have I learned about Alice?', {
    types: ['observation'],
    includeSourceFacts: true,
    maxSourceFactsTokens: 4096,
});

for (const obs of obsResponse.results) {
    console.log(`Observation: ${obs.text}`);
    if (obs.source_fact_ids && obsResponse.source_facts) {
        console.log('  Derived from:');
        for (const factId of obs.source_fact_ids) {
            const fact = obsResponse.source_facts[factId];
            if (fact) console.log(`    - [${fact.type}] ${fact.text}`);
        }
    }
}
```

#### entities

Enabled by default. When active, each returned fact includes the canonical names of entities associated with it. Set to `null` to skip the entity JOIN query and reduce response size. The `max_tokens` sub-option (default `500`) is a future-facing guard for entity data.

### tags

Filters recall to only memories that match the specified tags. When omitted, all memories regardless of tags are eligible. Tag filtering is applied at the database level across all four retrieval strategies, not as a post-processing step.

The `tags_match` parameter controls the filtering logic:

- `any` (default) — memory matches if it has at least one of the specified tags, or has no tags at all. Use this for "user-specific + shared global" patterns.
- `any_strict` — memory matches if it has at least one of the specified tags, and untagged memories are excluded. Use this when you want only explicitly scoped memories.
- `all` — memory matches if it has every specified tag, or has no tags at all.
- `all_strict` — memory matches if it has every specified tag, and untagged memories are excluded.

### Python

```python
# Filter recall to only memories tagged for a specific user
response = client.recall(
    bank_id="my-bank",
    query="What feedback did the user give?",
    tags=["user:alice"],
    tags_match="any"  # OR matching, includes untagged (default)
)
```

### Python

```python
# Strict mode: only return memories that have matching tags (exclude untagged)
response = client.recall(
    bank_id="my-bank",
    query="What did the user say?",
    tags=["user:alice"],
    tags_match="any_strict"  # OR matching, excludes untagged memories
)
```

### Python

```python
# AND matching: require ALL specified tags to be present
response = client.recall(
    bank_id="my-bank",
    query="What bugs were reported?",
    tags=["user:alice", "bug-report"],
    tags_match="all_strict"  # Memory must have BOTH tags
)
```

### trace

When set to `true`, the response includes a detailed debug trace covering the query embedding, entry points, per-strategy retrieval results, RRF fusion candidates, reranked results, temporal constraints detected, and per-phase timings. Has no effect on the retrieval logic itself. Useful for understanding why specific memories were or were not returned.

---

## Response

### results

The main list of recalled facts, ordered by relevance. Relevance is computed by running four retrieval strategies in parallel — semantic similarity, BM25 keyword, graph traversal, and temporal — fusing their rankings with Reciprocal Rank Fusion (RRF), then re-scoring the merged candidates with a cross-encoder reranker against the original query.

Results do not include a numeric score. Raw retrieval scores are not meaningful on an absolute scale — a score of 0.8 from one query tells you nothing useful compared to a score of 0.8 from another. What matters is the relative ordering, which is already reflected in the list order. Agents should consume memories in order and let `max_tokens` determine how many fit, rather than filtering by score.

Each item in `results` has the following fields:

#### id

The unique identifier of this fact. Use it to cross-reference with `source_facts` or for application-level deduplication.

#### text

The extracted fact text as stored in the memory bank.

#### type

The fact category: `world` for objective information, `experience` for events and conversations, or `observation` for consolidated knowledge synthesized over time.

#### context

The context label provided when the fact was retained (e.g., `"team meeting"`, `"slack"`). `null` if none was set.

#### metadata

The key-value string pairs attached when the fact was retained. `null` if none were set.

#### tags

The visibility-scoping tags attached to this fact.

#### entities

A list of canonical entity name strings linked to this fact. Only populated when `include.entities` is enabled (the default). `null` otherwise.

#### occurred_start / occurred_end

ISO 8601 datetimes representing when the described event started and ended. Extracted by the LLM from the content during retain. `null` if the content had no temporal information.

#### mentioned_at

ISO 8601 datetime of when this fact was retained into the bank.

#### document_id

The document ID this fact belongs to, as set during retain.

#### chunk_id

The ID of the source text chunk this fact was extracted from. Used to cross-reference with `chunks` in the response when `include.chunks` is enabled.

#### source_fact_ids

For `observation`-type results only: the IDs of the original facts this observation was synthesized from. Cross-references with `source_facts` in the response. `null` for other types or when `include.source_facts` is not enabled.

---

### source_facts

A dict keyed by fact ID containing full `RecallResult` objects for the source facts that contributed to observation results. Only present when `include.source_facts` is enabled. Facts are deduplicated — if two observations share a source fact, it appears once.

### chunks

A dict keyed by chunk ID containing the raw source text chunks associated with the returned facts. Only present when `include.chunks` is enabled. Each chunk has `id`, `text`, `chunk_index`, and `truncated` (whether the text was cut to fit the token budget).

### entities

A dict keyed by canonical entity name containing entity state objects. Only present when `include.entities` is enabled. Each entry has `entity_id`, `canonical_name`, and `observations`.

### trace

A debug object present only when `trace: true` was set in the request. Contains per-phase timings, retrieval breakdowns, and RRF fusion details.
