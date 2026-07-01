# Porter's 5 Forces — Analyst Prompt

You are filling the `## Porter's 5 Forces` section of an `oma market` brief.
This framework applies when the topic is competitive positioning of a
named subject (default for `intent=competitor`).

## Force-by-force classification rules

### Threat of new entrants

Cluster signals: how easy is it for new players to enter this market?

- Quote barriers (capital, regulatory licenses, network effects, brand,
  patents) → Low threat.
- Quote rising number of startup launches in this category, low entry
  cost commentary, easy-to-fork OSS alternatives → High threat.

### Bargaining power of suppliers

Cluster signals: dependency on upstream suppliers / vendors / partners.

- Single-source dependency, lock-in complaints, supplier price hikes
  → High supplier power.
- Multiple substitute suppliers, commoditized inputs → Low supplier power.

### Bargaining power of buyers

Cluster signals: how price-sensitive / switching-able are customers?

- Price complaints, churn intent, easy switching cost mentions
  → High buyer power.
- Locked-in customers, high switching cost (data migration, retraining)
  → Low buyer power.

### Threat of substitutes

Cluster signals: products from outside the category that satisfy the
same need.

- "I replaced X with Y" / "we don't need X anymore because Z"
  → High substitute threat.
- No alternative discussion / users defend why this category is unique
  → Low substitute threat.

### Industry rivalry

Cluster signals: intensity of competition within the category.

- Frequent head-to-head comparisons, price wars, public benchmark
  fights → High rivalry.
- Sleepy category, incumbents stable, little competitive marketing
  → Low rivalry.

## Output rules

1. Each force gets a verdict in `Low | Moderate | High` plus 1-3 cited
   bullets. Empty → `_(insufficient signal)_`.
2. Every bullet cites a cluster representative as `[name](url)` and tags
   the cluster ID `(C#)`.
3. Bottom line: one sentence summarizing the overall attractiveness of
   the market position for the subject.

## Output skeleton

```markdown
**Threat of new entrants** — Verdict: Low | Moderate | High
- [name](url) - signal summary. (C#)

**Bargaining power of suppliers** — Verdict: ...

**Bargaining power of buyers** — Verdict: ...

**Threat of substitutes** — Verdict: ...

**Industry rivalry** — Verdict: ...

**Strategic implication**: 1-2 sentences. Where does the subject have
the most room to defend or grow?
```

## Traps

- Don't conflate "many competitors mentioned" with "high rivalry" — the
  signal needs to be active competitive behavior (price, feature, switch).
- Substitute threat ≠ rivalry: Substitutes are OUT of category
  (Notion vs Markdown files), rivalry is IN category (Notion vs Coda).
- One cluster can speak to multiple forces; pick the dominant force or
  cite it twice with different framings.
