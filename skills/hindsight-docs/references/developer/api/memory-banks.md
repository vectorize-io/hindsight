
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
client.create_bank(
    bank_id="my-bank",
    name="Research Assistant",
    mission="You're a research assistant specializing in machine learning - keep track of papers, methods, and findings.",
    disposition={
        "skepticism": 4,
        "literalism": 3,
        "empathy": 3
    }
)
```

### Node.js

```javascript
await client.createBank('my-bank', {
    name: 'Research Assistant',
    mission: 'I am a research assistant specializing in machine learning',
    disposition: {
        skepticism: 4,
        literalism: 3,
        empathy: 3
    }
});
```

### CLI

```bash
# Set mission
hindsight bank mission my-bank "I am a research assistant specializing in ML"

# Set disposition
hindsight bank disposition my-bank \
    --skepticism 4 \
    --literalism 3 \
    --empathy 3
```

## Mission and Disposition

Mission and disposition are optional settings that influence how the bank reasons during [reflect](./reflect) operations.

:::info
Mission and disposition only affect the `reflect` operation. They do not impact `retain`, `recall`, or other memory operations.
### Mission

The mission is a first-person narrative providing context for reasoning:

### Python

```python
client.create_bank(
    bank_id="financial-advisor",
    name="Financial Advisor",
    mission="""You're a conservative financial advisor - keep track of client risk tolerance,
    investment preferences, and market conditions. Prioritize capital preservation over growth."""
)
```

### Node.js

```javascript
await client.createBank('financial-advisor', {
    name: 'Financial Advisor',
    mission: `I am a conservative financial advisor with 20 years of experience.
    I prioritize capital preservation over aggressive growth.
    I have seen multiple market crashes and believe in diversification.`
});
```

### Disposition Traits

Disposition traits influence how reasoning is performed during reflection. Each trait is scored 1 to 5:

| Trait | Low (1) | High (5) |
|-------|---------|----------|
| **Skepticism** | Trusting, accepts information at face value | Skeptical, questions and doubts claims |
| **Literalism** | Flexible interpretation, reads between the lines | Literal interpretation, takes things exactly as stated |
| **Empathy** | Detached, focuses on facts and logic | Empathetic, considers emotional context |

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
