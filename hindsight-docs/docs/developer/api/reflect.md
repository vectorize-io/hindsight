---
sidebar_position: 3
---

# Reflect

Generate disposition-aware responses using retrieved memories.

When you call **reflect**, Hindsight performs a multi-step reasoning process:
1. **Recalls** relevant memories from the bank based on your query
2. **Applies** the bank's disposition traits to shape the reasoning style
3. **Generates** a contextual answer grounded in the retrieved facts
4. **Forms opinions** in the background based on the reasoning (available in subsequent calls)

The response includes the generated answer along with the facts that were used, providing full transparency into how the answer was derived.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

:::info How Reflect Works
Learn about disposition-driven reasoning and opinion formation in the [Reflect Architecture](/developer/reflect) guide.
:::

:::tip Prerequisites
Make sure you've completed the [Quick Start](./quickstart) to install the client and start the server.
:::

## Basic Usage

<Tabs>
<TabItem value="python" label="Python">

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")

client.reflect(bank_id="my-bank", query="What should I know about Alice?")
```

</TabItem>
<TabItem value="node" label="Node.js">

```typescript
import { HindsightClient } from '@vectorize-io/hindsight-client';

const client = new HindsightClient({ baseUrl: 'http://localhost:8888' });

await client.reflect('my-bank', 'What should I know about Alice?');
```

</TabItem>
<TabItem value="cli" label="CLI">

```bash
hindsight memory think my-bank "What should I know about Alice?"
```

</TabItem>
</Tabs>

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Question or prompt |
| `budget` | string | "low" | Budget level: "low", "mid", "high" |
| `context` | string | None | Additional context for the query |

<Tabs>
<TabItem value="python" label="Python">

```python
response = client.reflect(
    bank_id="my-bank",
    query="What do you think about remote work?",
    budget="mid",
    context="We're considering a hybrid work policy"
)
```

</TabItem>
<TabItem value="node" label="Node.js">

```typescript
const response = await client.reflect('my-bank', 'What do you think about remote work?', {
    budget: 'mid',
    context: "We're considering a hybrid work policy"
});
```

</TabItem>
</Tabs>

## The Role of Context

The `context` parameter steers how the reflection is performed without impacting the memory recall. It provides situational information that helps shape the reasoning and response.

**How context is used:**
- **Shapes reasoning**: Helps understand the situation when formulating an answer
- **Disambiguates intent**: Clarifies what aspect of the query matters most
- **Does not affect recall**: The same memories are retrieved regardless of context

<Tabs>
<TabItem value="python" label="Python">

```python
# Context is passed to the LLM to help it understand the situation
response = client.reflect(
    bank_id="my-bank",
    query="What do you think about the proposal?",
    context="We're in a budget review meeting discussing Q4 spending"
)
```

</TabItem>
<TabItem value="node" label="Node.js">

```typescript
// Context helps the LLM understand the current situation
const response = await client.reflect('my-bank', 'What do you think about the proposal?', {
    context: "We're in a budget review meeting discussing Q4 spending"
});
```

</TabItem>
</Tabs>

## Opinion Formation

When reflect reasons about a question, it may form new **opinions** based on the evidence in the memory bank. These opinions are created in the background and become available in subsequent `reflect` and `recall` calls.

**Why opinions matter:**
- **Consistent thinking**: Opinions ensure the memory bank maintains a coherent perspective over time
- **Evolving viewpoints**: As more information is retained, opinions can be refined or updated
- **Grounded reasoning**: Opinions are always derived from factual evidence in the memory bank

Opinions are stored as a special memory type and are automatically retrieved when relevant to future queries. This creates a natural evolution of the bank's perspective, similar to how humans form and refine their views based on accumulated experience.

## Disposition Influence

The bank's disposition affects reflect responses:

| Trait | Low (1) | High (5) |
|-------|---------|----------|
| **Skepticism** | Trusting, accepts claims | Questions and doubts claims |
| **Literalism** | Flexible interpretation | Exact, literal interpretation |
| **Empathy** | Detached, fact-focused | Considers emotional context |

<Tabs>
<TabItem value="python" label="Python">

```python
# Create a bank with specific disposition
client.create_bank(
    bank_id="cautious-advisor",
    background="I am a risk-aware financial advisor",
    disposition={
        "skepticism": 5,   # Very skeptical of claims
        "literalism": 4,   # Focuses on exact requirements
        "empathy": 2       # Prioritizes facts over feelings
    }
)

# Reflect responses will reflect this disposition
response = client.reflect(
    bank_id="cautious-advisor",
    query="Should I invest in crypto?"
)
# Response will likely emphasize risks and caution
```

</TabItem>
<TabItem value="node" label="Node.js">

```typescript
// Create a bank with specific disposition
await client.createBank('cautious-advisor', {
    background: 'I am a risk-aware financial advisor',
    disposition: {
        skepticism: 5,
        literalism: 4,
        empathy: 2
    }
});

// Reflect responses will reflect this disposition
const response = await client.reflect('cautious-advisor', 'Should I invest in crypto?');
```

</TabItem>
</Tabs>

## Using Sources

The `based_on` field shows which memories informed the response:

<Tabs>
<TabItem value="python" label="Python">

```python
response = client.reflect(bank_id="my-bank", query="Tell me about Alice")

print("Response:", response.text)
print("\nBased on:")
for fact in response.based_on or []:
    print(f"  - [{fact.type}] {fact.text}")
```

</TabItem>
<TabItem value="node" label="Node.js">

```typescript
const response = await client.reflect('my-bank', 'Tell me about Alice');

console.log('Response:', response.text);
console.log('\nBased on:');
for (const fact of response.based_on || []) {
    console.log(`  - [${fact.type}] ${fact.text}`);
}
```

</TabItem>
</Tabs>

This enables:
- **Transparency** — users see why the bank said something
- **Verification** — check if the response is grounded in facts
- **Debugging** — understand retrieval quality
