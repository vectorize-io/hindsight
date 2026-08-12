
# Reflect

Generate a grounded, disposition-aware response using an agentic reasoning loop.

When you call **reflect**, Hindsight runs an agentic loop that autonomously searches the memory bank using multiple retrieval tools, applies the bank's disposition traits to shape the reasoning style, and produces a final answer grounded in what it found. Unlike recall — which returns raw facts — reflect returns a synthesized response written by the LLM.

{/* Import raw source files */}

> **ℹ️ How Reflect Works**
>
Learn about disposition-driven reasoning in the [Reflect Architecture](../reflect.md) guide.
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

### Go

```go
# Section 'reflect-basic' not found in api/reflect.go
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

### CLI

```bash
hindsight memory reflect my-bank "Summarize my week" --budget high --max-tokens 8192
```

### Go

```go
# Section 'reflect-with-params' not found in api/reflect.go
```

### max_tokens

Limits the length of the final generated response. Defaults to `4096`. This does not affect how much the agent can retrieve during the agentic loop — only the final answer length.

### response_schema

An optional JSON Schema **object** with a non-empty `properties` map (nested objects and arrays are supported). When provided, the response includes a `structured_output` field **in addition to** the markdown `text`: the agent reasons to its answer, then a second pass extracts that answer into JSON matching your schema. `structured_output` is therefore a faithful projection of `text` — you get the readable answer *and* a typed object to program against, never one instead of the other. Invalid schemas (not an object, or no properties) are rejected before the call runs.

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

### Go

```go
# Section 'reflect-structured-output' not found in api/reflect.go
```

### tags

Defines the visibility scope used throughout the reflect agent. It filters raw
facts, observations, and mental models that the agent can retrieve. The same
`tags` and `tags_match` values also select which tagged directives are injected
into the reflect prompt.

`tags` defaults to `null`, `tags_match` defaults to `any`, and `tag_groups`
defaults to `null`. For non-empty tags, raw facts, observations, and mental
models use the same matching modes as [recall tags](./recall#tags). Directives
have one additional rule: untagged directives are global and remain eligible
whenever a tag scope is supplied, including with a strict or exact match.

| Reflect configuration | Raw facts and observations | Mental models | Active directives |
|-----------------------|----------------------------|---------------|-------------------|
| Omit `tags`, `tags_match`, and `tag_groups` | All tagged and untagged data | All tagged and untagged models | Untagged/global directives only |
| `tags: []`, default `tags_match: "any"` | All tagged and untagged data | All tagged and untagged models | Untagged/global directives only |
| No tags, `tags_match: "exact"` | Untagged/global data only | Untagged/global models only | Untagged/global directives only |
| Non-empty `tags`, `any` or `all` | Matching tagged data plus untagged/global data | Matching models plus untagged/global models | Matching tagged directives plus untagged/global directives |
| Non-empty `tags`, `any_strict` or `all_strict` | Matching tagged data only | Matching tagged models only | Matching tagged directives plus untagged/global directives |
| Non-empty `tags`, `exact` | Data with exactly the requested tag set | Models with exactly the requested tag set | Exactly matching tagged directives plus untagged/global directives |
| Non-empty `tag_groups`, default top-level `tags_match` | Data matching the compound expression | Models matching the compound expression | Matching tagged directives plus untagged/global directives |

The first row is intentionally asymmetric: an unscoped reflect can search all
memories, but it does not load tagged directives. To create a directive that
applies to every reflect call, leave its `tags` empty. To create a scoped
directive, assign tags and pass a matching scope to `reflect`.

> **📝 `isolation_mode`**
>
`isolation_mode` is an internal `list_directives` option, not a public reflect
request parameter. Reflect always enables it. When neither `tags` nor
`tag_groups` is supplied, it limits directive loading to untagged directives.
There is currently no per-request switch to disable it.
> **📝 MCP omitted tags**
>
The MCP `reflect` tool forwards `tags_match` only when `tags` is present. To
request the empty exact scope through MCP, pass `tags: []` together with
`tags_match: "exact"`.
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

### Node.js

```javascript
// Filter reflect to only use memories tagged for a specific user
await client.reflect('my-bank', 'What feedback did the user give?', {
    tags: ['user:alice'],
    tagsMatch: 'any_strict'
});
```

### CLI

```bash
hindsight memory reflect my-bank "What feedback did the user give?" \
  --tags "user:alice" --tags-match any_strict
```

### Go

```go
# Section 'reflect-with-tags' not found in api/reflect.go
```

#### Common scope examples

Unscoped reflect searches all memories but applies only global directives:

```json
{
  "query": "Summarize the current project status"
}
```

A project scope includes global data and directives alongside matching
`project:a` data and directives:

```json
{
  "query": "Summarize the current project status",
  "tags": ["project:a"]
}
```

A strict project scope excludes untagged memories, observations, and mental
models. Global directives still apply:

```json
{
  "query": "Summarize the current project status",
  "tags": ["project:a"],
  "tags_match": "all_strict"
}
```

### tag_groups

Provides compound tag filtering with recursive `and`, `or`, and `not`
expressions. It affects the same reflect data sources and directive selection as
flat `tags`. `tag_groups` and `tags` are mutually exclusive in the public REST
request. Each leaf supplies its own matching mode and defaults to
`any_strict`; the top-level groups are AND-ed. Normally leave the top-level
`tags_match` at its default, `any`. Setting it to `exact` while using
`tag_groups` additionally constrains facts, observations, and mental models to
the global flat scope before applying the compound expression.

The MCP `reflect` tool currently exposes flat `tags` and `tags_match`, but not
`tag_groups`.

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

### Node.js

```javascript
const sourcesResponse = await client.reflect('my-bank', 'Tell me about Alice', {
    includeFacts: true
});

console.log('Response:', sourcesResponse.text);
console.log('\nBased on:');
for (const fact of (sourcesResponse.based_on?.memories || [])) {
    console.log(`  - [${fact.type}] ${fact.text}`);
}
```

### CLI

```bash
hindsight memory reflect my-bank "Tell me about Alice" --include-facts
```

### Go

```go
# Section 'reflect-sources' not found in api/reflect.go
```

#### include.tool_calls

When enabled, the response includes a `trace` object with the full execution log of every tool call and LLM call made during the agentic loop, including inputs, outputs, and durations. Set `output: false` to include only tool inputs for a smaller payload. Useful for debugging why the agent reached a particular conclusion.

---

## Response

### text

The synthesized answer as a well-formatted markdown string. This is the primary output of reflect. Still returned when `response_schema` is provided — `structured_output` is derived from it, not a replacement for it.

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
