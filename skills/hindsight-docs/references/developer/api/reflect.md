
# Reflect

Generate a grounded, disposition-aware response using an agentic reasoning loop.

When you call **reflect**, Hindsight runs an agentic loop that autonomously searches the memory bank using multiple retrieval tools, applies the bank's disposition traits to shape the reasoning style, and produces a final answer grounded in what it found. Unlike recall — which returns raw facts — reflect returns a synthesized response written by the LLM.

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

---

## Parameters

### query

The question or prompt to reflect on. This is the only required field. If you have situational context that should influence the answer, include it directly in the query rather than as a separate field.

### budget

Controls how thoroughly the agent explores the memory bank before answering. Accepted values are `low` (default), `mid`, and `high`. At `low`, the agent does a shallow search optimized for speed. At `mid`, it checks multiple sources when the question warrants it. At `high`, it performs deep exploration across all knowledge levels and may use multiple query variations to find indirect connections. Use `high` for complex questions that require synthesizing information from many sources.

### Python

```python
response = client.reflect(
    bank_id="my-bank",
    query="We're considering a hybrid work policy. What do you think about remote work?",
    budget="mid",
)
```

### Node.js

```javascript
const response = await client.reflect('my-bank', 'What do you think about remote work?', {
    budget: 'mid',
    context: "We're considering a hybrid work policy"
});
```

### max_tokens

Limits the length of the final generated response. Defaults to `4096`. This does not affect how much the agent can retrieve during the agentic loop — only the final answer length.

### response_schema

An optional JSON Schema object. When provided, the LLM generates a response that conforms to the schema and the response includes a `structured_output` field with the result parsed accordingly. The `text` field will be empty since only a single structured LLM call is made. Use this when you need to process the response programmatically rather than display it as prose.

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

### tags

Filters which memories the agent can access during reflection. Works identically to [recall tags](./recall#tags) — only memories matching the specified tags are considered. The `tags_match` parameter controls the matching logic (`any`, `all`, `any_strict`, `all_strict`) with the same semantics as recall.

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

### include

Controls optional supplementary data returned alongside the main response.

#### include.facts

When enabled, the response includes a `based_on` object listing the memories, mental models, and directives the agent actually used to construct the answer. Only sources retrieved during the agent loop can appear here — citations are validated to prevent hallucinated references. Useful for transparency and verification.

### Python

```python
# include_facts=True enables the based_on field in the response
response = client.reflect(
    bank_id="my-bank",
    query="Tell me about Alice",
    include_facts=True,
)

print("Response:", response.text)
print("\nBased on:")
for fact in (response.based_on.memories if response.based_on else []):
    print(f"  - [{fact.type}] {fact.text}")
```

#### include.tool_calls

When enabled, the response includes a `trace` object with the full execution log of every tool call and LLM call made during the agentic loop, including inputs, outputs, and durations. Set `output: false` to include only tool inputs for a smaller payload. Useful for debugging why the agent reached a particular conclusion.

---

## Response

### text

The synthesized answer as a well-formatted markdown string. This is the primary output of reflect. Empty when `response_schema` is provided (use `structured_output` instead in that case).

### structured_output

The LLM's response parsed according to the `response_schema` provided in the request. Only present when `response_schema` was set. `null` otherwise.

### based_on

The sources the agent used to construct the answer. Only present when `include.facts` was enabled. Contains three fields:

- `memories` — a list of memory facts (world, experience, observation) that were retrieved and cited. Each item has `id`, `text`, `type`, `context`, `occurred_start`, and `occurred_end`.
- `mental_models` — a list of mental models that were used. Each item has `id`, `text`, and `context`.
- `directives` — a list of directives that were enforced during reasoning. Each item has `id`, `name`, and `content`.

### usage

Token usage for all LLM calls made during the agentic loop: `input_tokens`, `output_tokens`, and `total_tokens`. Useful for cost tracking.

### trace

The full execution log of the agentic loop. Only present when `include.tool_calls` was enabled. Contains:

- `tool_calls` — each tool invocation with `tool` name (`lookup`, `recall`, `learn`, `expand`), `input`, `output` (if `output: true`), `duration_ms`, and `iteration` number.
- `llm_calls` — each LLM call with `scope` (e.g., `"agent_1"`, `"final"`) and `duration_ms`.
