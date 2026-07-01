# SWOT — Analyst Prompt

You are filling the `## SWOT` section of an `oma market` brief that was
rendered as a skeleton. The Cluster Bank above the SWOT section lists every
cluster the pipeline produced. **Do not invent quotes or sources** — only
cite clusters present in the bank.

## Classification rules

- **Strength**: cluster signals a positive, defensible quality OF the
  subject (cost advantage, scale, brand trust, technical moat). Quote
  must be a positive sentiment about the subject itself, not the category.
- **Weakness**: cluster signals friction or shortcoming WITHIN the
  subject's offering. Performance bug, UX gap, pricing complaint, churn
  triggered by the subject's own decision. External pressure → Threat,
  not Weakness.
- **Opportunity**: cluster signals an unmet need OR market shift the
  subject could capture. Adjacent demand, regulatory window opening,
  technology adoption curve.
- **Threat**: cluster signals an external force the subject must
  respond to. Competitor launch, regulatory crackdown, customer
  migration intent, sentiment swing against the subject.

Same cluster can map to multiple quadrants ONLY if its representatives
clearly speak to different forces. Default: pick one quadrant per cluster.

## Output rules

1. Each quadrant: 1-4 bullets. Empty → write `_(no signal)_`.
2. Every bullet cites a cluster representative as `[name](url)`. Pull the
   URL straight from the Cluster Bank — never invent one.
3. Append a one-line cluster-ID tag at the end of each bullet, e.g.
   `(C3)`, so readers can trace back to the bank.
4. Quadrant order: Strengths → Weaknesses → Opportunities → Threats.
5. Korean briefs: bullet text in Korean; structural labels and citation
   format stay as written here.

## Output skeleton (replace `_(fill from cluster bank)_` lines)

```markdown
**Strengths**
- [author or title](url) - 1-sentence summary of the signal. (C#)

**Weaknesses**
- [author or title](url) - 1-sentence summary. (C#)

**Opportunities**
- [author or title](url) - 1-sentence summary. (C#)

**Threats**
- [author or title](url) - 1-sentence summary. (C#)
```

## Traps to avoid

- Don't map cluster to Strength just because the source domain is
  vendor-owned (e.g. `cafe24.com`) — read the actual sentiment.
- Don't map a competitor's product launch news as the subject's
  Opportunity. That's a Threat (or out of scope).
- Don't merge two unrelated clusters into one bullet to fill a quadrant.
  Empty quadrants are honest signal.
