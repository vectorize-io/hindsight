---
sidebar_position: 50
title: "TealTiger Governance Memory with Hindsight | Integration Guide"
description: "Store AI agent governance decisions with importance-weighted retention in Hindsight. Critical denials persist for compliance; routine approvals decay naturally."
---

# TealTiger

Importance-weighted governance decision memory for [TealTiger](https://tealtiger.ai) — store AI agent governance decisions in Hindsight with natural decay based on severity.

## Features

- **Importance-Weighted Retention** — Critical DENYs (PII, secrets) persist for months; routine ALLOWs decay within days
- **Contextual Recall** — Before making new governance decisions, recall similar past decisions for the same agent/tool
- **Governance Reflection** — Synthesize patterns from governance history (denial trends, anomaly detection)
- **Pluggable Importance** — Override the default importance mapping with custom logic
- **Storage ≠ Authority** — A stored ALLOW cannot authorize a future action; every request gets fresh deterministic evaluation

## Installation

```bash
pip install tealtiger-hindsight
```

## Quick Start

```python
from hindsight_client import Hindsight
from tealtiger_hindsight import HindsightGovernanceMemory
from tealtiger import observe
from openai import OpenAI

# Connect to Hindsight
client = Hindsight(base_url="http://localhost:8888")

# Create governance memory
memory = HindsightGovernanceMemory(client=client, bank_id="governance")

# Wrap LLM client — every governance decision auto-stored with importance
llm = observe(
    OpenAI(),
    guardrails={"pii_detection": True, "cost_limit": 5.00},
    on_decision=memory.store,
)

# Contextual recall before new decisions
past = memory.recall(agent_id="research-agent", context="tool:web_search", limit=5)

# Reflect on governance patterns
insights = memory.reflect(query="What denial patterns exist for this agent?")
```

## Importance Mapping

| Decision Type | Default Importance | Retention Behavior |
|---------------|-------------------|-------------------|
| DENY (PII, secrets) | 0.90 | Months — compliance evidence |
| REFER / REQUIRE_APPROVAL | 0.85 | Weeks — approval tracking |
| MONITOR (flagged) | 0.70 | Weeks — pattern detection |
| ALLOW (routine) | 0.55 | Days — natural decay |

### Custom Importance Function

```python
def custom_importance(decision) -> float:
    """Use risk_score + recurrence for richer decay."""
    base = {"DENY": 0.85, "MONITOR": 0.65, "ALLOW": 0.50}
    score = base.get(decision.get("action", "ALLOW"), 0.55)
    score += (decision.get("risk_score", 0) / 100) * 0.15
    if decision.get("is_recurring"):
        score += 0.10
    return min(score, 1.0)

memory = HindsightGovernanceMemory(client=client, importance_fn=custom_importance)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `client` | required | Hindsight client instance |
| `bank_id` | `"governance"` | Memory bank for governance decisions |
| `importance_fn` | `default_importance` | Maps decisions to importance scores (0.0–1.0) |
| `tags` | `["tealtiger", "governance"]` | Tags applied to all stored memories |
| `metadata_fields` | `["agent_id", "mode", "action", "tool_name", "risk_score"]` | Fields stored as Hindsight metadata |

## Use Cases

- **Contextual governance** — Recall past denials before evaluating new actions
- **Anomaly detection** — Detect agents whose denial rate spiked vs. historical baseline
- **Compliance audits** — Critical DENYs persist indefinitely for audit requirements
- **Storage efficiency** — Routine ALLOWs decay naturally without manual cleanup

## Design Principle

**Storage = evidence, NOT authority.** A stored ALLOW from yesterday cannot authorize today's action. Every new request gets a fresh deterministic evaluation. Memory informs; it doesn't permit.

## Links

- **PyPI:** [tealtiger-hindsight](https://pypi.org/project/tealtiger-hindsight/)
- **Source:** [github.com/agentguard-ai/tealtiger/packages/tealtiger-hindsight](https://github.com/agentguard-ai/tealtiger/tree/main/packages/tealtiger-hindsight)
- **TealTiger:** [tealtiger.ai](https://tealtiger.ai)
- **Issue:** [#2284](https://github.com/vectorize-io/hindsight/issues/2284)
