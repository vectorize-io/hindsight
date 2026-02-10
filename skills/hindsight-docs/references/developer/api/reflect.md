
# Reflect

Generate disposition-aware responses using an agentic reasoning loop.

When you call **reflect**, Hindsight runs an **agentic loop** that:
1. **Autonomously searches** for relevant information using multiple tools
2. **Applies** the bank's disposition traits to shape the reasoning style
3. **Generates** a grounded answer with citations to the sources used

The agent has access to hierarchical retrieval tools (mental models → observations → raw facts) and decides what information it needs to answer your query.

{/* Import raw source files */}

:::info How Reflect Works
Learn about disposition-driven reasoning in the [Reflect Architecture](/developer/reflect) guide.
> **💡 Prerequisites**
> 
Make sure you've completed the [Quick Start](./quickstart) to install the client and start the server.
## Basic Usage

### Python

```python
client.reflect(bank_id="my-bank", query="What should I know about Alice?")
```

### Node.js

```javascript
await client.reflect('my-bank', 'What should I know about Alice?');
```

### CLI

```bash
hindsight memory reflect my-bank "What do you know about Alice?"
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Question or prompt |
| `budget` | string | "low" | Budget level: `low`, `mid`, `high` (see below) |
| `max_tokens` | int | 4096 | Maximum tokens for the final response |
| `response_schema` | object | None | JSON Schema for [structured output](#structured-output) |
| `tags` | list | None | Filter memories by tags during reflection |
| `tags_match` | string | "any" | How to match tags: `any`, `all`, `any_strict`, `all_strict` |
| `trace` | bool | false | Include detailed agent trace in response |

### Budget

The `budget` parameter controls the research depth — how thoroughly the agent explores before answering:

| Budget | Research Depth | Use Case |
|--------|----------------|----------|
| `low` | Shallow | Quick answers, simple lookups. Prioritizes speed over completeness. |
| `mid` | Moderate | Balanced exploration. Checks multiple sources when warranted. |
| `high` | Deep | Comprehensive analysis. Explores all knowledge levels, uses multiple query variations. |

Use `high` for complex questions that require synthesizing information from multiple sources or verifying facts across different retrieval levels.

### Max Tokens

The `max_tokens` parameter limits the length of the final generated response. This does not affect how much the agent can retrieve during the agentic loop — only the final answer length.

### Python

```python
response = client.reflect(
    bank_id="my-bank",
    query="What do you think about remote work?",
    budget="mid",
    context="We're considering a hybrid work policy"
)
```

### Node.js

```javascript
const response = await client.reflect('my-bank', 'What do you think about remote work?', {
    budget: 'mid',
    context: "We're considering a hybrid work policy"
});
```

## Disposition Influence

The bank's disposition affects reflect responses:

| Trait | Low (1) | High (5) |
|-------|---------|----------|
| **Skepticism** | Trusting, accepts claims | Questions and doubts claims |
| **Literalism** | Flexible interpretation | Exact, literal interpretation |
| **Empathy** | Detached, fact-focused | Considers emotional context |

### Python

```python
# Create a bank with specific disposition
client.create_bank(
    bank_id="cautious-advisor",
    name="Cautious Advisor",
    mission="I am a risk-aware financial advisor",
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

### Node.js

```javascript
// Create a bank with specific disposition
await client.createBank('cautious-advisor', {
    name: 'Cautious Advisor',
    background: 'I am a risk-aware financial advisor',
    disposition: {
        skepticism: 5,
        literalism: 4,
        empathy: 2
    }
});

// Reflect responses will reflect this disposition
const advisorResponse = await client.reflect('cautious-advisor', 'Should I invest in crypto?');
```

## Citations

The response includes a `based_on` field that shows which sources were used:

- `based_on.memories` — Memory facts (world, experience) that were retrieved and cited
- `based_on.mental_models` — User-curated mental models that were used
- `based_on.directives` — Directives that were enforced

**Important:** Only IDs that were actually retrieved during the agent loop can be cited. The agent validates citations to prevent hallucinated references.

This enables:
- **Transparency** — users see exactly which sources informed the answer
- **Verification** — check if the response is grounded in actual memories
- **Debugging** — use `trace=True` for detailed tool call logs

## Structured Output

For applications that need to process responses programmatically, you can request structured output by providing a JSON Schema via `response_schema`. When provided, the response includes a `structured_output` field with the LLM response parsed according to the schema. The `text` field will be empty since only a single LLM call is made for efficiency.

The easiest way to define a schema is using **Pydantic models**:

### Python

```python
from pydantic import BaseModel

# Define your response structure with Pydantic
class HiringRecommendation(BaseModel):
    recommendation: str
    confidence: str  # "low", "medium", "high"
    key_factors: list[str]
    risks: list[str] = []

response = client.reflect(
    bank_id="hiring-team",
    query="Should we hire Alice for the ML team lead position?",
    response_schema=HiringRecommendation.model_json_schema(),
)

# Parse structured output into Pydantic model
result = HiringRecommendation.model_validate(response.structured_output)
print(f"Recommendation: {result.recommendation}")
print(f"Confidence: {result.confidence}")
print(f"Key factors: {result.key_factors}")
```

### Node.js

```javascript
// Define JSON schema directly
const responseSchema = {
    type: 'object',
    properties: {
        recommendation: { type: 'string' },
        confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
        key_factors: { type: 'array', items: { type: 'string' } },
        risks: { type: 'array', items: { type: 'string' } },
    },
    required: ['recommendation', 'confidence', 'key_factors'],
};

const structuredResponse = await client.reflect('my-bank', 'What do you know about Alice and her career?', {
    responseSchema: responseSchema,
});

// Structured output (if returned)
if (structuredResponse.structuredOutput) {
    console.log('Recommendation:', structuredResponse.structuredOutput.recommendation || 'N/A');
    console.log('Key factors:', structuredResponse.structuredOutput.key_factors || []);
}
```

### CLI

```bash
# First, create a JSON schema file schema.json:
cat > schema.json << 'EOF'
{
  "type": "object",
  "properties": {
    "recommendation": {"type": "string"},
    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    "key_factors": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["recommendation", "confidence", "key_factors"]
}
EOF

# Then use the --schema flag:
hindsight memory reflect hiring-team \
  "Should we hire Alice for the ML team lead position?" \
  --schema schema.json

# Cleanup the temporary schema file
rm -f schema.json
```

| Use Case | Why Structured Output Helps |
|----------|----------------------------|
| **Decision pipelines** | Parse recommendations into workflow systems |
| **Dashboards** | Extract confidence scores, risk factors for visualization |
| **Multi-agent systems** | Pass structured data between agents |
| **Auditing** | Log structured decisions with clear reasoning |

**Tips:**
- Use Pydantic's `model_json_schema()` for type-safe schema generation
- Use `model_validate()` to parse the response back into your Pydantic model
- Keep schemas focused — extract only what you need
- Use `Optional` fields for data that may not always be available

## Filter by Tags

Like [recall](./recall#filter-by-tags), reflect supports tag filtering to scope which memories are considered during reasoning. This is essential for multi-user scenarios where reflection should only consider memories relevant to a specific user.

### Python

```python
# Filter reflection to only consider memories for a specific user
response = client.reflect(
    bank_id="my-bank",
    query="What does this user think about our product?",
    tags=["user:alice"],
    tags_match="any_strict"  # Only use memories tagged for this user
)
```

The `tags_match` parameter works the same as in recall:

| Mode | Behavior |
|------|----------|
| `any` | OR matching, includes untagged memories |
| `all` | AND matching, includes untagged memories |
| `any_strict` | OR matching, excludes untagged memories |
| `all_strict` | AND matching, excludes untagged memories |

See [Retain API](./retain#tagging-memories) for how to tag memories and [Recall API](./recall#filter-by-tags) for more details on tag matching modes.
