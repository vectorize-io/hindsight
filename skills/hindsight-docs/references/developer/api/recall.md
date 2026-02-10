
# Recall Memories

Retrieve memories using multi-strategy recall.

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
for r in response.results:
    print(f"- {r.text}")
```

### Node.js

```javascript
const response = await client.recall('my-bank', 'What does Alice do?');
for (const r of response.results) {
    console.log(`${r.text} (score: ${r.weight})`);
}
```

### CLI

```bash
hindsight memory recall my-bank "What does Alice do?"
```

## Recall Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Natural language query |
| `types` | list | all | Filter: `world`, `experience`, `observation` |
| `budget` | string | "mid" | Budget level: `low`, `mid`, `high` |
| `max_tokens` | int | 4096 | Token budget for results |
| `trace` | bool | false | Enable trace output for debugging |
| `include_chunks` | bool | false | Include raw text chunks that generated the memories |
| `max_chunk_tokens` | int | 500 | Token budget for chunks |
| `tags` | list | None | Filter memories by tags (see [Tag Filtering](#filter-by-tags)) |
| `tags_match` | string | "any" | How to match tags: `any`, `all`, `any_strict`, `all_strict` |

### Python

```python
response = client.recall(
    bank_id="my-bank",
    query="What does Alice do?",
    types=["world", "experience"],
    budget="high",
    max_tokens=8000,
    trace=True,
)

# Access results
for r in response.results:
    print(f"- {r.text}")
```

### Node.js

```javascript
const detailedResponse = await client.recall('my-bank', 'What does Alice do?', {
    types: ['world', 'experience'],
    budget: 'high',
    maxTokens: 8000,
    trace: true
});

// Access results
for (const r of detailedResponse.results) {
    console.log(`${r.text} (score: ${r.weight})`);
}
```

## Filter by Fact Type

Recall specific memory types:

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
Observations are consolidated knowledge synthesized from multiple facts. They capture patterns, preferences, and learnings that the memory bank has built up over time. Observations are automatically created in the background after retain operations.
## Token Budget Management

Hindsight is built for AI agents, not humans. Traditional retrieval systems return "top-k" results, but agents don't think in terms of result counts—they think in tokens. An agent's context window is measured in tokens, and that's exactly how Hindsight measures results.

The `max_tokens` parameter lets you control how much of your agent's context budget to spend on memories:

### Python

```python
# Fill up to 4K tokens of context with relevant memories
results = client.recall(bank_id="my-bank", query="What do I know about Alice?", max_tokens=4096)

# Smaller budget for quick lookups
results = client.recall(bank_id="my-bank", query="Alice's email", max_tokens=500)
```

This design means you never have to guess whether 10 results or 50 results will fit your context. Just specify the token budget and Hindsight returns as many relevant memories as will fit.

## Budget Levels

The `budget` parameter controls graph traversal depth:

- **"low"**: Fast, shallow retrieval — good for simple lookups
- **"mid"**: Balanced — default for most queries
- **"high"**: Deep exploration — finds indirect connections

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

## Filter by Tags

Tags enable **visibility scoping**—filter memories based on tags assigned during [retain](./retain#tagging-memories). This is essential for multi-user agents where each user should only see their own memories.

### Basic Tag Filtering

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

### Tag Match Modes

The `tags_match` parameter controls how tags are matched:

| Mode | Behavior | Untagged Memories |
|------|----------|-------------------|
| `any` | OR: memory has ANY of the specified tags | **Included** |
| `all` | AND: memory has ALL of the specified tags | **Included** |
| `any_strict` | OR: memory has ANY of the specified tags | **Excluded** |
| `all_strict` | AND: memory has ALL of the specified tags | **Excluded** |

**Strict modes** are useful when you want to ensure only tagged memories are returned:

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

**AND matching** requires all specified tags to be present:

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

### Use Cases

| Scenario | Tags | Mode | Result |
|----------|------|------|--------|
| User A's memories only | `["user:alice"]` | `any_strict` | Only memories tagged `user:alice` |
| Support + feedback | `["support", "feedback"]` | `any` | Memories with either tag + untagged |
| Multi-user room | `["user:alice", "room:general"]` | `all_strict` | Only memories with both tags |
| Global + user-specific | `["user:alice"]` | `any` | Alice's memories + shared (untagged) |
