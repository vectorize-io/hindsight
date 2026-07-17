# hindsight-tealtiger

Governance-aware agent memory for [TealTiger](https://github.com/agentguard-ai/tealtiger) — stores AI agent governance decisions with importance-weighted retention and contextual recall via Hindsight.

## What it does

TealTiger produces deterministic governance decisions (ALLOW/DENY/MONITOR) for every agent action. This integration stores those decisions in Hindsight with importance-based decay:

- **Critical DENYs** (PII, secrets) → high importance (0.90) → retained for months
- **Notable MONITORs** (flagged) → medium importance (0.70) → retained for weeks
- **Routine ALLOWs** → low importance (0.55) → natural decay within days

This enables contextual governance: before making a new decision, recall similar past decisions for the same agent/tool combination.

## Design Principle

**Storage = evidence, NOT authority.** A stored ALLOW cannot authorize a future action. Every new request gets a fresh deterministic evaluation. Memory informs; it doesn't permit.

## Installation

```bash
pip install hindsight-tealtiger
```

## Quick Start

```python
from hindsight_client import Hindsight
from hindsight_tealtiger import HindsightGovernanceMemory
from tealtiger import observe
from openai import OpenAI

# Connect to Hindsight
client = Hindsight(base_url="http://localhost:8888")

# Create governance memory
memory = HindsightGovernanceMemory(client=client, bank_id="governance")

# Wrap LLM client — every governance decision auto-stored
llm = observe(
    OpenAI(),
    guardrails={"pii_detection": True, "cost_limit": 5.00},
    on_decision=memory.store,
)

# Before making new decisions, recall context
past = memory.recall(agent_id="research-agent", context="tool:web_search", limit=5)

# Reflect on governance patterns
insights = memory.reflect(query="What denial patterns exist for this agent?")
```

## Importance Mapping

| Decision Type | Default Importance | Behavior |
|---------------|-------------------|----------|
| DENY | 0.90 | Retained months — compliance evidence |
| REFER/REQUIRE_APPROVAL | 0.85 | Retained weeks — approval tracking |
| MONITOR | 0.70 | Retained weeks — pattern detection |
| ALLOW | 0.55 | Days — natural decay, doesn't pollute recall |

Override with a custom function:

```python
def custom_importance(decision) -> float:
    base = {"DENY": 0.90, "MONITOR": 0.70, "ALLOW": 0.50}
    score = base.get(decision.get("action", "ALLOW"), 0.55)
    score += (decision.get("risk_score", 0) / 100) * 0.10
    return min(score, 1.0)

memory = HindsightGovernanceMemory(client=client, importance_fn=custom_importance)
```

## Use Cases

1. **Contextual governance** — recall past denials before evaluating new actions
2. **Anomaly detection** — detect agents whose denial rate spiked
3. **Compliance** — critical events persist for audit requirements
4. **Storage efficiency** — routine ALLOWs decay naturally

## Requirements

- Python >= 3.10
- hindsight-client >= 0.4.0
- A running Hindsight server (or [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup))

## Related

- Issue: [#2284](https://github.com/vectorize-io/hindsight/issues/2284)
- TealTiger: https://github.com/agentguard-ai/tealtiger
