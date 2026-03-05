
# Memory Banks

Memory banks are isolated containers that store all memory-related data for a specific context or use case.

{/* Import raw source files */}

## What is a Memory Bank?

A memory bank is a complete, isolated storage unit containing:

- **Memories** — Facts and information retained from conversations
- **Documents** — Files and content indexed for retrieval
- **Entities** — People, places, concepts extracted from memories
- **Relationships** — Connections between entities in the knowledge graph
- **Directives** — Hard rules the agent must follow during reflect operations

Banks are completely isolated from each other — memories stored in one bank are not visible to another.

You don't need to pre-create a bank. Hindsight will automatically create it with default settings when you first use it.

> **💡 Prerequisites**
> 
Make sure you've completed the [Quick Start](./quickstart) to install the client and start the server.
## Creating a Memory Bank

### Python

```python
client.create_bank(bank_id="my-bank")
```

### Node.js

```javascript
await client.createBank('my-bank');
```

### CLI

```bash
hindsight bank create my-bank
```

## Bank Configuration

Each memory bank can be configured independently per operation. Configuration can be set via the [bank config API](#updating-configuration), the [Control Plane UI](/), or [server-wide environment variables](/developer/configuration).

### retain_mission {#retain-configuration}

A plain-language description of what this bank should pay attention to during extraction. The mission is injected into the extraction prompt alongside the built-in rules — it steers focus without replacing the extraction logic.

```
e.g. Always include technical decisions, API design choices, and architectural trade-offs.
     Ignore meeting logistics, greetings, and social exchanges.
```

Works alongside any extraction mode. Leave blank for general-purpose extraction.

### retain_extraction_mode

Controls how aggressively facts are extracted:

| Mode | Description |
|------|-------------|
| `concise` *(default)* | Selective — only facts worth remembering long-term |
| `verbose` | Captures more detail per fact; slower and uses more tokens |
| `custom` | Write your own extraction rules via `retain_custom_instructions` |

### retain_custom_instructions

Only active when `retain_extraction_mode` is `custom`. Replaces the built-in extraction rules entirely with your own instructions.

See [Retain configuration](/developer/configuration#retain) for environment variable names and defaults.

### entity_labels {#entity-labels}

Defines a controlled vocabulary of `key:value` classification labels extracted at retain time and stored as entities. Because labels become entities, they automatically link memories in the knowledge graph (two memories with `pedagogy:scaffolding` are linked), improve semantic and BM25 retrieval, and optionally filter memories via the standard `tags`/`tags_match` API when `tag: true` is set on a group.

Each entry in `entity_labels` is a **label group** — one classification dimension:

```json
{
  "entity_labels": [
    {
      "key": "engagement",
      "description": "Student engagement level during the session",
      "type": "value",
      "optional": true,
      "values": [
        { "value": "active",  "description": "Student is actively participating" },
        { "value": "passive", "description": "Student is listening but not participating" }
      ]
    },
    {
      "key": "pedagogy",
      "description": "Teaching strategies used",
      "type": "multi-values",
      "values": [
        { "value": "scaffolding",          "description": "Breaking complex tasks into smaller steps" },
        { "value": "direct_instruction",   "description": "Explicit explanation by the teacher" },
        { "value": "socratic_questioning", "description": "Guiding through questions rather than answers" }
      ]
    }
  ]
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `key` | — | Label group identifier. Becomes the prefix in `key:value` entities. |
| `description` | `""` | Shown to the LLM to guide label assignment. |
| `type` | `"value"` | `"value"` → pick one enum value; `"multi-values"` → pick multiple; `"text"` → free-form string. |
| `values` | `[]` | Allowed values for `"value"` and `"multi-values"` types. Ignored for `"text"`. |
| `optional` | `true` | When `true` the LLM may skip the label if not applicable. When `false` the LLM must always assign a value. Has no effect on `"multi-values"` groups (always optional). |
| `tag` | `false` | When `true`, extracted `key:value` labels are also written as tags on the memory unit, enabling filtering via `tags`/`tags_match` in recall/reflect. |

**Enum groups** (`type: "value"` or `type: "multi-values"`): the LLM picks from the predefined `values` list; anything outside the list is silently dropped. Vocabulary stays stable and graph links stay tight. Use `"multi-values"` when a fact can belong to several values at once.

**Free-text groups** (`type: "text"`): the LLM writes any string. Use the `description` field to provide examples and guidance. Graph clustering is less reliable than with enum groups because the model may phrase the same concept differently across sessions.

```json
{
  "key": "topic",
  "description": "Specific subject being discussed. Examples: algebra, quadratic equations, geometry.",
  "type": "text",
  "optional": true,
  "values": []
}
```

### entities_allow_free_form

By default, entity labels are extracted **alongside** regular named entities (people, places, concepts). Set to `false` to disable free-form extraction so only label entities are stored:

```json
{
  "entity_labels": [...],
  "entities_allow_free_form": false
}
```

### enable_observations {#observations-configuration}

Toggles automatic observation consolidation on or off. Defaults to `true` when the observations feature is enabled on the server.

### observations_mission

Defines what this bank should synthesise into durable observations. Replaces the built-in consolidation rules entirely — leave blank to use the server default.

```
e.g. Observations are stable facts about people and projects.
     Always include preferences, skills, and recurring patterns.
     Ignore one-off events and ephemeral state.
```

See [Observations configuration](/developer/configuration#observations) for environment variable names and defaults.

### mission

A first-person narrative that provides identity and framing context for `reflect`. The agent uses this to ground its reasoning and apply a consistent perspective.

```
e.g. You are a senior engineering assistant.
     Always ground answers in documented decisions and rationale.
     Ignore speculation. Be direct and precise.
```

### disposition_skepticism

How skeptical vs trusting the bank is when evaluating claims during `reflect`. Scale 1–5.

### Python

```python
client.create_bank(bank_id="architect-bank")
client.update_bank_config(
    "architect-bank",
    reflect_mission="You're a senior software architect - keep track of system designs, "
            "technology decisions, and architectural patterns. Prefer simplicity over cutting-edge.",
    disposition_skepticism=4,   # Questions new technologies
    disposition_literalism=4,   # Focuses on concrete specs
    disposition_empathy=2,      # Prioritizes technical facts
)
```

### Node.js

```javascript
await client.createBank('architect-bank');
await client.updateBankConfig('architect-bank', {
    reflectMission: "You're a senior software architect - keep track of system designs, technology decisions, and architectural patterns.",
    dispositionSkepticism: 4,   // Questions new technologies
    dispositionLiteralism: 4,   // Focuses on concrete specs
    dispositionEmpathy: 2,      // Prioritizes technical facts
});
```

| Value | Behaviour |
|-------|-----------|
| `1` | Trusting — accepts information at face value |
| `3` *(default)* | Balanced |
| `5` | Skeptical — questions and doubts claims |

### disposition_literalism

How literally to interpret information during `reflect`. Scale 1–5.

| Value | Behaviour |
|-------|-----------|
| `1` | Flexible — reads between the lines, considers context |
| `3` *(default)* | Balanced |
| `5` | Literal — takes things exactly as stated |

### disposition_empathy

How much to weight emotional context when reasoning during `reflect`. Scale 1–5.

| Value | Behaviour |
|-------|-----------|
| `1` | Detached — focuses on facts and logic |
| `3` *(default)* | Balanced |
| `5` | Empathetic — considers emotional context |

:::info
Disposition traits and `mission` only affect the `reflect` operation. `retain_mission` and `observations_mission` are separate per-operation settings.
---

## Updating Configuration

Bank configuration fields (retain mission, extraction mode, observations mission, etc.) are managed via a **separate config API**, not the `create_bank` call. This lets you change operational settings independently from the bank's identity and disposition.

### Setting Configuration Overrides

### Python

```python
client.update_bank_config(
    "my-bank",
    retain_mission="Always include technical decisions, API design choices, and architectural trade-offs. Ignore meeting logistics and social exchanges.",
    retain_extraction_mode="verbose",
    observations_mission="Observations are stable facts about people and projects. Always include preferences, skills, and recurring patterns. Ignore one-off events.",
    disposition_skepticism=4,
    disposition_literalism=4,
    disposition_empathy=2,
)
```

### Node.js

```javascript
await client.updateBankConfig('my-bank', {
    retainMission: 'Always include technical decisions, API design choices, and architectural trade-offs. Ignore meeting logistics and social exchanges.',
    retainExtractionMode: 'verbose',
    observationsMission: 'Observations are stable facts about people and projects. Always include preferences, skills, and recurring patterns. Ignore one-off events.',
    dispositionSkepticism: 4,
    dispositionLiteralism: 4,
    dispositionEmpathy: 2,
});
```

You can update any subset of fields — only the keys you provide are changed.

### Reading the Current Configuration

### Python

```python
# Returns resolved config (server defaults merged with bank overrides) and the raw overrides
data = client.get_bank_config("my-bank")
# data["config"]     — full resolved configuration
# data["overrides"]  — only fields overridden at the bank level
```

### Node.js

```javascript
// Returns resolved config (server defaults merged with bank overrides) and the raw overrides
const { config, overrides } = await client.getBankConfig('my-bank');
// config    — full resolved configuration
// overrides — only fields overridden at the bank level
```

The response distinguishes:
- **`config`** — the fully resolved configuration (server defaults merged with bank overrides)
- **`overrides`** — only the fields explicitly overridden for this bank

### Resetting to Defaults

### Python

```python
# Remove all bank-level overrides, reverting to server defaults
client.reset_bank_config("my-bank")
```

### Node.js

```javascript
// Remove all bank-level overrides, reverting to server defaults
await client.resetBankConfig('my-bank');
```

This removes all bank-level overrides. The bank reverts to server-wide defaults (set via environment variables).

You can also update configuration directly from the [Control Plane UI](/) — navigate to a bank and open the **Configuration** tab.

---

## Directives

Directives are hard rules that the agent must follow during [reflect](./reflect) operations. Unlike disposition traits which influence *how* the agent reasons, directives are explicit instructions that are *always* enforced.

:::info
Directives only affect the `reflect` operation. They are injected into prompts and the agent is required to comply with them in all responses.
### When to Use Directives

Use directives for rules that must never be violated:

- **Language/style constraints**: "Always respond in formal English"
- **Privacy rules**: "Never share personal data with third parties"
- **Domain constraints**: "Prefer conservative investment recommendations"
- **Behavioral guardrails**: "Always cite sources when making claims"

### Creating Directives

### Python

```python
# Create a directive (hard rule for reflect)
directive = client.create_directive(
    bank_id=BANK_ID,
    name="Formal Language",
    content="Always respond in formal English, avoiding slang and colloquialisms."
)

print(f"Created directive: {directive.id}")
```

### Node.js

```javascript
// Create a directive (hard rule for reflect)
const directive = await client.createDirective(
    BANK_ID,
    'Formal Language',
    'Always respond in formal English, avoiding slang and colloquialisms.'
);

console.log(`Created directive: ${directive.id}`);
```

### Listing Directives

### Python

```python
# List all directives in a bank
directives = client.list_directives(bank_id=BANK_ID)

for d in directives.items:
    print(f"- {d.name}: {d.content[:50]}...")
```

### Node.js

```javascript
// List all directives in a bank
const directives = await client.listDirectives(BANK_ID);

for (const d of directives.items) {
    console.log(`- ${d.name}: ${d.content.slice(0, 50)}...`);
}
```

### Updating Directives

### Python

```python
# Update a directive (e.g., disable without deleting)
updated = client.update_directive(
    bank_id=BANK_ID,
    directive_id=directive_id,
    is_active=False
)

print(f"Directive active: {updated.is_active}")
```

### Node.js

```javascript
// Update a directive (e.g., disable without deleting)
const updated = await client.updateDirective(BANK_ID, directiveId, {
    isActive: false
});

console.log(`Directive active: ${updated.is_active}`);
```

### Deleting Directives

### Python

```python
# Delete a directive
client.delete_directive(
    bank_id=BANK_ID,
    directive_id=directive_id
)
```

### Node.js

```javascript
// Delete a directive
await client.deleteDirective(BANK_ID, directiveId);
```

### Directives vs Disposition

| Aspect | Directives | Disposition |
|--------|------------|-------------|
| **Nature** | Hard rules, must be followed | Soft influence on reasoning style |
| **Enforcement** | Strict — responses are rejected if violated | Flexible — shapes interpretation |
| **Use case** | Compliance, guardrails, constraints | Personality, character, tone |
| **Example** | "Never recommend specific stocks" | High skepticism: questions claims |
