---
sidebar_position: 4
---

# Reflect: How Hindsight Reasons with Disposition

When you call `reflect()`, Hindsight doesn't just retrieve facts — it **reasons** about them through the lens of the bank's unique disposition, forming new opinions and generating contextual responses.

```mermaid
graph LR
    A[Query] --> B[Recall Memories]
    B --> C[Load Disposition]
    C --> D[Reason]
    D --> E[Form Opinions]
    E --> F[Response]
```

---

## Why Reflect?

Most AI systems can retrieve facts, but they can't **reason** about them in a consistent way. Every response is generated fresh without a stable perspective or evolving beliefs.

### The Problem

Without reflect:
- **No consistent character**: "Should we adopt remote work?" gets a different answer each time based on the LLM's randomness
- **No opinion formation**: The system never develops beliefs based on accumulated evidence
- **No reasoning context**: Responses don't reflect what the bank has learned or its perspective
- **Generic responses**: Every AI sounds the same — no disposition, no point of view

### The Value

With reflect:
- **Consistent character**: A bank configured as "detail-oriented, cautious" will consistently emphasize risks and thorough planning
- **Evolving opinions**: As the bank learns more about a topic, its opinions strengthen, weaken, or change — just like a real expert
- **Contextual reasoning**: Responses reflect the bank's accumulated knowledge and perspective: "Based on what I know about your team's remote work success..."
- **Differentiated behavior**: Customer support bots sound diplomatic, code reviewers sound direct, creative assistants sound open-minded

### When to Use Reflect

| Use `recall()` when... | Use `reflect()` when... |
|------------------------|-------------------------|
| You need raw facts | You need reasoned interpretation |
| You're building your own reasoning | You want disposition-consistent responses |
| You need maximum control | You want the bank to "think" for itself |
| Simple fact lookup | Forming recommendations or opinions |

**Example:**
- `recall("Alice")` → Returns all Alice facts
- `reflect("Should we hire Alice?")` → Reasons about Alice's fit based on accumulated knowledge, weighs evidence, forms opinion

---

## Disposition Traits

When you create a memory bank, you can configure its disposition using three traits. These traits influence how the bank interprets information and forms opinions during `reflect()`:

| Trait | Scale | Low (1) | High (5) |
|-------|-------|---------|----------|
| **Skepticism** | 1-5 | Trusting, accepts information at face value | Skeptical, questions and doubts claims |
| **Literalism** | 1-5 | Flexible interpretation, reads between the lines | Literal interpretation, takes things at face value |
| **Empathy** | 1-5 | Detached, focuses on facts | Empathetic, considers emotional context |

### Background: Natural Language Identity

Beyond numeric traits, you can provide a natural language **background** that describes the bank's identity:

```python
client.create_bank(
    bank_id="my-bank",
    background="I am a senior software architect with 15 years of distributed "
               "systems experience. I prefer simplicity over cutting-edge technology.",
    disposition={
        "skepticism": 4,   # Questions new technologies
        "literalism": 4,   # Focuses on concrete specs
        "empathy": 2       # Prioritizes technical facts
    }
)
```

The background provides context that shapes how disposition traits are applied:
- "I prefer simplicity" + high skepticism → questions complex solutions
- "15 years experience" → responses reference this expertise
- First-person perspective → creates consistent voice

---

## Opinion Formation

When `reflect()` encounters a question that warrants forming an opinion, disposition shapes the response.

### Same Facts, Different Opinions

Two banks with different dispositions, given identical facts about remote work:

**Bank A** (low skepticism, high empathy):
> "Remote work enables flexibility and work-life balance. The team seems happier and more productive when they can choose their environment."

**Bank B** (high skepticism, low empathy):
> "Remote work claims need verification. What are the actual productivity metrics? The anecdotal benefits may not translate to measurable outcomes."

**Same facts → Different conclusions** because disposition shapes interpretation.

---

## Opinion Evolution

Opinions aren't static — they evolve as new evidence arrives. Here's a real-world example with a database library:

| Event | What the bank learns | Opinion formed |
|-------|---------------------|----------------|
| **Day 1** | "Redis is open source under BSD license" | "Redis is excellent for caching — fast, reliable, and OSS-friendly" (confidence: 0.85) |
| **Day 2** | "Redis has great community support and documentation" | Opinion reinforced (confidence: 0.90) |
| **Day 30** | "Redis changed license to SSPL, restricting cloud usage" | "Redis is still technically strong, but license concerns for cloud deployments" (confidence: 0.65) |
| **Day 45** | "Valkey forked Redis under BSD license with Linux Foundation backing" | "Consider Valkey for new projects requiring true OSS; Redis for existing deployments" (confidence: 0.80) |

**Before the license change:**
> "Should we use Redis for our caching layer?"
> → "Yes, Redis is the industry standard — fast, battle-tested, and fully open source."

**After the license change:**
> "Should we use Redis for our caching layer?"
> → "It depends. For cloud deployments, consider Valkey (the BSD-licensed fork). For on-premise, Redis remains excellent technically."

This **continuous learning** ensures recommendations stay current with real-world changes.

---

## Disposition Presets by Use Case

Different use cases benefit from different disposition configurations:

| Use Case | Recommended Traits | Why |
|----------|-------------------|-----|
| **Customer Support** | skepticism: 2, literalism: 2, empathy: 5 | Trusting, flexible, understanding |
| **Code Review** | skepticism: 4, literalism: 5, empathy: 2 | Questions assumptions, precise, direct |
| **Legal Analysis** | skepticism: 5, literalism: 5, empathy: 2 | Highly skeptical, exact interpretation |
| **Therapist/Coach** | skepticism: 2, literalism: 2, empathy: 5 | Supportive, reads between lines |
| **Research Assistant** | skepticism: 4, literalism: 3, empathy: 3 | Questions claims, balanced interpretation |

---

## What You Get from Reflect

When you call `reflect()`:

**Returns:**
- **Response text** — Disposition-influenced answer
- **Based on** — Which memories were used (with relevance scores)

**Example:**
```json
{
  "text": "Based on Alice's ML expertise and her work at Google, she'd be an excellent fit for the research team lead position...",
  "based_on": {
    "world": [
      {"text": "Alice works at Google...", "weight": 0.95},
      {"text": "Alice specializes in ML...", "weight": 0.88}
    ]
  }
}
```

**Note:** New opinions are formed asynchronously in the background. They'll influence future `reflect()` calls but aren't returned directly.

---

## Why Disposition Matters

Without disposition, all AI assistants sound the same. With disposition:

- **Customer support bots** can be diplomatic and empathetic
- **Code review assistants** can be direct and thorough
- **Creative assistants** can be open to unconventional ideas
- **Risk analysts** can be appropriately cautious

Disposition creates **consistent character** across conversations while allowing opinions to **evolve with evidence**.

---

## Next Steps

- [**Retain**](./retain) — How rich facts are stored
- [**Recall**](./retrieval) — How multi-strategy search works
- [API Reference: Reflect](./api/reflect) — Code examples and usage
