---
sidebar_position: 2
---

# Recall Memories

Retrieve memories using multi-strategy recall.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

:::info How Recall Works
Learn about the four retrieval strategies (semantic, keyword, graph, temporal) and RRF fusion in the [Recall Architecture](/developer/retrieval) guide.
:::

:::tip Prerequisites
Make sure you've completed the [Quick Start](./quickstart) to install the client and start the server.
:::

## Basic Recall

<Tabs>
<TabItem value="python" label="Python">

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")

response = client.recall(bank_id="my-bank", query="What does Alice do?")
for r in response.results:
    print(f"{r.text} (score: {r.weight:.2f})")
```

</TabItem>
<TabItem value="node" label="Node.js">

```typescript
import { HindsightClient } from '@vectorize-io/hindsight-client';

const client = new HindsightClient({ baseUrl: 'http://localhost:8888' });

const response = await client.recall('my-bank', 'What does Alice do?');
for (const r of response.results) {
    console.log(`${r.text} (score: ${r.weight})`);
}
```

</TabItem>
<TabItem value="cli" label="CLI">

```bash
hindsight recall my-bank "What does Alice do?"
```

</TabItem>
</Tabs>

## Recall Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Natural language query |
| `types` | list | all | Filter: `world`, `experience`, `opinion` |
| `budget` | string | "mid" | Budget level: "low", "mid", "high" |
| `max_tokens` | int | 4096 | Token budget for results |
| `trace` | bool | false | Enable trace output for debugging |
| `include_entities` | bool | false | Include entity observations |
| `max_entity_tokens` | int | 500 | Token budget for entity observations |

<Tabs>
<TabItem value="python" label="Python">

```python
response = client.recall(
    bank_id="my-bank",
    query="What does Alice do?",
    types=["world", "experience"],
    budget="high",
    max_tokens=8000,
    trace=True,
    include_entities=True,
    max_entity_tokens=500
)

# Access results
for r in response.results:
    print(f"{r.text} (score: {r.weight:.2f})")

# Access entity observations (if include_entities=True)
if response.entities:
    for entity in response.entities:
        print(f"Entity: {entity.name}")
```

</TabItem>
<TabItem value="node" label="Node.js">

```typescript
const response = await client.recall('my-bank', 'What does Alice do?', {
    types: ['world', 'experience'],
    budget: 'high',
    maxTokens: 8000,
    trace: true
});

// Access results
for (const r of response.results) {
    console.log(`${r.text} (score: ${r.weight})`);
}
```

</TabItem>
</Tabs>

## Filter by Fact Type

Recall specific memory types:

<Tabs>
<TabItem value="python" label="Python">

```python
# Only world facts (objective information)
world_facts = client.recall(
    bank_id="my-bank",
    query="Where does Alice work?",
    types=["world"]
)

# Only experience (conversations and events)
experience = client.recall(
    bank_id="my-bank",
    query="What have I recommended?",
    types=["experience"]
)

# Only opinions (formed beliefs)
opinions = client.recall(
    bank_id="my-bank",
    query="What do I think about Python?",
    types=["opinion"]
)

# World facts and experience (exclude opinions)
facts = client.recall(
    bank_id="my-bank",
    query="What happened?",
    types=["world", "experience"]
)
```

</TabItem>
<TabItem value="cli" label="CLI">

```bash
hindsight recall my-bank "Python" --fact-type opinion
hindsight recall my-bank "Alice" --fact-type world,experience
```

</TabItem>
</Tabs>

:::warning About Opinions
Opinions are beliefs formed during [reflect](/developer/api/reflect) operations. Unlike world facts and experience, opinions are subjective interpretations and may not represent objective truth. Depending on your use case:
- **Exclude opinions** (`types=["world", "experience"]`) when you need factual, verifiable information
- **Include opinions** when you want the agent's perspective or formed beliefs
- **Use opinions alone** (`types=["opinion"]`) only when specifically asking about the agent's views
:::

## Token Budget Management

Hindsight is built for AI agents, not humans. Traditional retrieval systems return "top-k" results, but agents don't think in terms of result counts—they think in tokens. An agent's context window is measured in tokens, and that's exactly how Hindsight measures results.

The `max_tokens` parameter lets you control how much of your agent's context budget to spend on memories:

```python
# Fill up to 4K tokens of context with relevant memories
results = client.recall(bank_id="my-bank", query="What do I know about Alice?", max_tokens=4096)

# Smaller budget for quick lookups
results = client.recall(bank_id="my-bank", query="Alice's email", max_tokens=500)
```

This design means you never have to guess whether 10 results or 50 results will fit your context. Just specify the token budget and Hindsight returns as many relevant memories as will fit.

## Include Related Context

Beyond the core memory results, you can optionally retrieve additional context—each with its own token budget:

| Option | Parameter | Description |
|--------|-----------|-------------|
| **Chunks** | `include_chunks`, `max_chunk_tokens` | Raw text chunks that generated the memories |
| **Entity Observations** | `include_entities`, `max_entity_tokens` | Related observations about entities mentioned in results |

```python
response = client.recall(
    bank_id="my-bank",
    query="What does Alice do?",
    max_tokens=4096,              # Budget for memories
    include_entities=True,
    max_entity_tokens=1000        # Budget for entity observations
)

# Access the additional context
entities = response.entities or []
```

This gives your agent richer context while maintaining precise control over total token consumption.

## Budget Levels

The `budget` parameter controls graph traversal depth:

- **"low"**: Fast, shallow retrieval — good for simple lookups
- **"mid"**: Balanced — default for most queries
- **"high"**: Deep exploration — finds indirect connections

<Tabs>
<TabItem value="python" label="Python">

```python
# Quick lookup
results = client.recall(bank_id="my-bank", query="Alice's email", budget="low")

# Deep exploration
results = client.recall(bank_id="my-bank", query="How are Alice and Bob connected?", budget="high")
```

</TabItem>
<TabItem value="node" label="Node.js">

```typescript
// Quick lookup
const results = await client.recall('my-bank', "Alice's email", { budget: 'low' });

// Deep exploration
const deep = await client.recall('my-bank', 'How are Alice and Bob connected?', { budget: 'high' });
```

</TabItem>
</Tabs>
