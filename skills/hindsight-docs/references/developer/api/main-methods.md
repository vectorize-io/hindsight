
# Main Methods

Hindsight provides three core operations: **retain**, **recall**, and **reflect**.

{/* Import raw source files */}

> **💡 Prerequisites**
> 
Make sure you've [installed Hindsight](../installation) and completed the [Quick Start](./quickstart).
## Retain: Store Information

Store conversations, documents, and facts into a memory bank.

### Python

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
    items=[
        {"content": "Bob prefers Python for data science"},
        {"content": "Alice recommends using pytest for testing"},
        {"content": "The team uses GitHub for code reviews"}
    ]
)
```

### Node.js

```javascript
// Store a single fact
await client.retain('my-bank', 'Alice joined Google in March 2024 as a Senior ML Engineer');

// Store a conversation
const conversation = `
User: What did you work on today?
Assistant: I reviewed the new ML pipeline architecture.
User: How did it look?
Assistant: Promising, but needs better error handling.
`;

await client.retain('my-bank', conversation, {
    context: 'Daily standup conversation'
});

// Batch retain multiple items
await client.retainBatch('my-bank', [
    { content: 'Bob prefers Python for data science' },
    { content: 'Alice recommends using pytest for testing' },
    { content: 'The team uses GitHub for code reviews' }
]);
```

### CLI

```bash
# Store a single fact
hindsight retain my-bank "Alice joined Google in March 2024 as a Senior ML Engineer"

# Store from a file
hindsight retain my-bank --file conversation.txt --context "Daily standup"

# Store multiple files
hindsight retain my-bank --files docs/*.md
```

**What happens:** Content is processed by an LLM to extract rich facts, identify entities, and build connections in a knowledge graph.

**See:** [Retain Details](./retain) for advanced options and parameters.

---

## Recall: Search Memories

Search for relevant memories using multi-strategy retrieval.

### Python

```python
# Basic search
results = client.recall(
    bank_id="my-bank",
    query="What does Alice do at Google?"
)

for result in results.results:
    print(f"- {result.text}")

# Search with options
results = client.recall(
    bank_id="my-bank",
    query="What happened last spring?",
    budget="high",  # More thorough graph traversal
    max_tokens=8192,  # Return more context
    types=["world"]  # Only world facts
)

# Include source chunks for more context
results = client.recall(
    bank_id="my-bank",
    query="Tell me about Alice",
    include_chunks=True,
    max_chunk_tokens=500
)

# Check chunk details (chunks are on response level, keyed by memory ID)
for result in results.results:
    print(f"Memory: {result.text}")
    if results.chunks and result.id in results.chunks:
        chunk = results.chunks[result.id]
        print(f"  Source: {chunk.text[:100]}...")
```

### Node.js

```javascript
// Basic search
const results = await client.recall('my-bank', 'What does Alice do at Google?');

for (const result of results.results) {
    console.log(`- ${result.text}`);
}

// Search with options
const filteredResults = await client.recall('my-bank', 'What happened last spring?', {
    budget: 'high',
    maxTokens: 8192,
    types: ['world']
});

// Include entity information
const entityResults = await client.recall('my-bank', 'Tell me about Alice', {
    includeEntities: true,
    maxEntityTokens: 500
});

// Check entity details
for (const [entityId, entity] of Object.entries(entityResults.entities || {})) {
    console.log(`Entity: ${entity.canonical_name}`);
    console.log(`Observations: ${entity.observations}`);
}
```

### CLI

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

**What happens:** Four search strategies (semantic, keyword, graph, temporal) run in parallel, results are fused and reranked.

**See:** [Recall Details](./recall) for tuning quality vs latency.

---

## Reflect: Reason with Disposition

Generate disposition-aware responses using memories and observations.

### Python

```python
# Basic reflect
response = client.reflect(
    bank_id="my-bank",
    query="Should we adopt TypeScript for our backend?"
)

print(response.text)
print("\nBased on:", len(response.based_on or []), "facts")

# Reflect with options
response = client.reflect(
    bank_id="my-bank",
    query="What are Alice's strengths for the team lead role?",
    budget="high"  # More thorough reasoning
)

# See which facts influenced the response
for fact in response.based_on or []:
    print(f"- {fact.text}")
```

### Node.js

```javascript
// Basic reflect
const response = await client.reflect('my-bank', 'Should we adopt TypeScript for our backend?');

console.log(response.text);
console.log('\nBased on:', (response.based_on || []).length, 'facts');

// Reflect with options
const detailedResponse = await client.reflect('my-bank', "What are Alice's strengths for the team lead role?", {
    budget: 'high'
});

// See which facts influenced the response
for (const fact of detailedResponse.based_on || []) {
    console.log(`- ${fact.text}`);
}
```

### CLI

```bash
# Basic reflect
hindsight reflect my-bank "Should we adopt TypeScript for our backend?"

# Verbose output (shows sources and observations)
hindsight reflect my-bank "What are Alice's strengths for the team lead role?" -v

# With higher reasoning budget
hindsight reflect my-bank "Analyze our tech stack" --budget high
```

**What happens:** Memories and observations are recalled, bank disposition is applied, and the LLM reasons through the evidence to generate a response.

**See:** [Reflect Details](./reflect) for disposition configuration.

---

## Comparison

| Feature | Retain | Recall | Reflect |
|---------|--------|--------|---------|
| **Purpose** | Store information | Find information | Reason about information |
| **Input** | Raw text/documents | Search query | Question/prompt |
| **Output** | Memory IDs | Ranked facts + observations | Reasoned response |
| **Uses LLM** | Yes (extraction) | No | Yes (generation) |
| **Uses observations** | No | Yes | Yes |
| **Disposition** | No | No | Yes |

---

## Next Steps

- [**Retain**](./retain) — Advanced options for storing memories
- [**Recall**](./recall) — Tuning search quality and performance
- [**Reflect**](./reflect) — Configuring disposition
- [**Memory Banks**](./memory-banks) — Managing memory bank disposition
