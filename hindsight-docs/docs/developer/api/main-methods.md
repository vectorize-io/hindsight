---
sidebar_position: 2
---

# Main Methods

Hindsight provides three core operations: **retain**, **recall**, and **reflect**.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

:::tip Prerequisites
Make sure you've [installed Hindsight](../installation) and completed the [Quick Start](./quickstart).
:::

## Retain: Store Information

Store conversations, documents, and facts into a memory bank.

<Tabs>
<TabItem value="python" label="Python">

```python
# Store a single fact
client.retain(
    bank_id="my-bank",
    content="Alice joined Google in March 2024 as a Senior ML Engineer"
)

# Store a conversation
conversation = """
User: What did you work on today?
Assistant: I reviewed the new ML pipeline architecture.
User: How did it look?
Assistant: Promising, but needs better error handling.
"""

client.retain(
    bank_id="my-bank",
    content=conversation,
    context="Daily standup conversation"
)

# Batch retain multiple items
client.retain_batch(
    bank_id="my-bank",
    contents=[
        {"content": "Bob prefers Python for data science"},
        {"content": "Alice recommends using pytest for testing"},
        {"content": "The team uses GitHub for code reviews"}
    ]
)
```

</TabItem>
<TabItem value="node" label="Node.js">

```javascript
// Store a single fact
await client.retain({
    bankId: 'my-bank',
    content: 'Alice joined Google in March 2024 as a Senior ML Engineer'
});

// Store a conversation
await client.retain({
    bankId: 'my-bank',
    content: `
User: What did you work on today?
Assistant: I reviewed the new ML pipeline architecture.
User: How did it look?
Assistant: Promising, but needs better error handling.
    `,
    context: 'Daily standup conversation'
});

// Batch retain
await client.retainBatch({
    bankId: 'my-bank',
    contents: [
        { content: 'Bob prefers Python for data science' },
        { content: 'Alice recommends using pytest for testing' },
        { content: 'The team uses GitHub for code reviews' }
    ]
});
```

</TabItem>
<TabItem value="cli" label="CLI">

```bash
# Store a single fact
hindsight retain my-bank "Alice joined Google in March 2024 as a Senior ML Engineer"

# Store from a file
hindsight retain my-bank --file conversation.txt --context "Daily standup"

# Store multiple files
hindsight retain my-bank --files docs/*.md
```

</TabItem>
</Tabs>

**What happens:** Content is processed by an LLM to extract rich facts, identify entities, and build connections in a knowledge graph.

**See:** [Retain Details](./retain) for advanced options and parameters.

---

## Recall: Search Memories

Search for relevant memories using multi-strategy retrieval.

<Tabs>
<TabItem value="python" label="Python">

```python
# Basic search
results = client.recall(
    bank_id="my-bank",
    query="What does Alice do at Google?"
)

for result in results:
    print(f"[{result['weight']:.2f}] {result['text']}")

# Search with options
results = client.recall(
    bank_id="my-bank",
    query="What happened last spring?",
    budget="high",  # More thorough graph traversal
    max_tokens=8192,  # Return more context
    fact_type="world"  # Only world facts
)

# Include entity information
results = client.recall(
    bank_id="my-bank",
    query="Tell me about Alice",
    include_entities=True,
    max_entity_tokens=500
)

# Check entity details
for entity in results["entities"]:
    print(f"Entity: {entity['name']}")
    print(f"Observations: {entity['observations']}")
```

</TabItem>
<TabItem value="node" label="Node.js">

```javascript
// Basic search
const results = await client.recall({
    bankId: 'my-bank',
    query: 'What does Alice do at Google?'
});

results.forEach(r => {
    console.log(`[${r.weight.toFixed(2)}] ${r.text}`);
});

// Search with options
const detailedResults = await client.recall({
    bankId: 'my-bank',
    query: 'What happened last spring?',
    budget: 'high',
    maxTokens: 8192,
    factType: 'world'
});

// Include entity information
const withEntities = await client.recall({
    bankId: 'my-bank',
    query: 'Tell me about Alice',
    includeEntities: true,
    maxEntityTokens: 500
});
```

</TabItem>
<TabItem value="cli" label="CLI">

```bash
# Basic search
hindsight recall my-bank "What does Alice do at Google?"

# Search with options
hindsight recall my-bank "What happened last spring?" \
    --budget high \
    --max-tokens 8192 \
    --fact-type world

# Verbose output (shows weights and sources)
hindsight recall my-bank "Tell me about Alice" -v
```

</TabItem>
</Tabs>

**What happens:** Four search strategies (semantic, keyword, graph, temporal) run in parallel, results are fused and reranked.

**See:** [Recall Details](./recall) for tuning quality vs latency.

---

## Reflect: Reason with Disposition

Generate disposition-aware responses that form opinions based on evidence.

<Tabs>
<TabItem value="python" label="Python">

```python
# Basic reflect
response = client.reflect(
    bank_id="my-bank",
    query="Should we adopt TypeScript for our backend?"
)

print(response["text"])
print("\nBased on:", len(response["based_on"]["world"]), "facts")
print("New opinions:", len(response["new_opinions"]))

# Reflect with options
response = client.reflect(
    bank_id="my-bank",
    query="What are Alice's strengths for the team lead role?",
    budget="high",  # More thorough reasoning
    include_entities=True
)

# Access formed opinions
for opinion in response["new_opinions"]:
    print(f"Opinion: {opinion['text']}")
    print(f"Confidence: {opinion['confidence']}")

# See which facts influenced the response
for fact in response["based_on"]["world"]:
    print(f"[{fact['weight']:.2f}] {fact['text']}")
```

</TabItem>
<TabItem value="node" label="Node.js">

```javascript
// Basic reflect
const response = await client.reflect({
    bankId: 'my-bank',
    query: 'Should we adopt TypeScript for our backend?'
});

console.log(response.text);
console.log(`\nBased on: ${response.basedOn.world.length} facts`);
console.log(`New opinions: ${response.newOpinions.length}`);

// Reflect with options
const detailed = await client.reflect({
    bankId: 'my-bank',
    query: "What are Alice's strengths for the team lead role?",
    budget: 'high',
    includeEntities: true
});

// Access formed opinions
detailed.newOpinions.forEach(op => {
    console.log(`Opinion: ${op.text}`);
    console.log(`Confidence: ${op.confidence}`);
});
```

</TabItem>
<TabItem value="cli" label="CLI">

```bash
# Basic reflect
hindsight reflect my-bank "Should we adopt TypeScript for our backend?"

# Verbose output (shows sources and opinions)
hindsight reflect my-bank "What are Alice's strengths for the team lead role?" -v

# With higher reasoning budget
hindsight reflect my-bank "Analyze our tech stack" --budget high
```

</TabItem>
</Tabs>

**What happens:** Memories are recalled, bank disposition is loaded, LLM reasons through evidence, new opinions are formed and stored.

**See:** [Reflect Details](./reflect) for disposition configuration.

---

## Comparison

| Feature | Retain | Recall | Reflect |
|---------|--------|--------|---------|
| **Purpose** | Store information | Find information | Reason about information |
| **Input** | Raw text/documents | Search query | Question/prompt |
| **Output** | Memory IDs | Ranked facts | Reasoned response + opinions |
| **Uses LLM** | Yes (extraction) | No | Yes (generation) |
| **Forms opinions** | No | No | Yes |
| **Disposition** | No | No | Yes |

---

## Next Steps

- [**Retain**](./retain) — Advanced options for storing memories
- [**Recall**](./recall) — Tuning search quality and performance
- [**Reflect**](./reflect) — Configuring disposition and opinions
- [**Memory Banks**](./memory-banks) — Managing memory bank disposition
