# PESTEL — Analyst Prompt

You are filling the `## PESTEL` section of an `oma market` brief.
This framework applies when scanning a category for macro forces (default
for `intent=discovery`).

## Axis-by-axis classification rules

### Political

Government action, policy direction, geopolitical alignment, sanctions,
public-sector procurement signals.

- e.g. "EU mandates open-source documentation tools" → Political tailwind.
- e.g. "정부, 외국 SaaS에 정관계 로비 의혹" → Political risk.

### Economic

Macro demand, currency, recession, supply chain, capital availability.

- e.g. "Startup funding for the category up 3x YoY" → Economic tailwind.
- e.g. "merchants cutting subscription spend after rate hikes" → Economic
  headwind.

### Social

Demographics, generational behavior, lifestyle shift, sentiment trends,
cultural attitudes.

- e.g. "Gen Z prefers async over Zoom" → Social tailwind for async tools.
- e.g. "remote-work fatigue driving return-to-office" → Social headwind
  for collaboration SaaS.

### Technological

Platform shifts, AI capability waves, open-source adoption, dependency
trends.

- e.g. "Every product is bolting on MCP servers" → Technological tailwind
  for AI agent tooling.
- e.g. "WebAssembly maturity threatens native runtimes" → Technological
  disruption.

### Environmental

Sustainability, climate impact, ESG signals, regulatory carbon mandates.

- e.g. "Datacenter carbon costs cited in vendor selection" → Environmental
  pressure on cloud-heavy categories.
- Often absent in pure software categories — `_(no signal)_` is honest.

### Legal

IP / copyright / antitrust / GDPR-PIPA / litigation / compliance.

- e.g. "GDPR fine on Notion for data residency" → Legal headwind.
- e.g. "Antitrust case against incumbent opens window for entrants"
  → Legal tailwind.

## Output rules

1. Each axis: 1-3 cited bullets with direction
   (`tailwind | headwind | neutral`). Empty → `_(no signal)_`.
2. Every bullet cites a cluster representative as `[name](url)` with
   cluster ID tag `(C#)`.
3. End with a `**Net stance**` paragraph: which 2-3 axes are most active
   right now and what they imply for the subject.

## Output skeleton

```markdown
**Political** — _direction_
- [name](url) - signal summary. (C#)

**Economic** — _direction_
- [name](url) - signal summary. (C#)

**Social** — _direction_
- ...

**Technological** — _direction_
- ...

**Environmental** — _direction_
- ...

**Legal** — _direction_
- ...

**Net stance**: 2-3 sentences identifying the dominant axes and
implications.
```

## Traps

- Don't classify cluster as Political just because the source is news
  media — read the actual content.
- "AI" alone is not Technological signal — needs to show a shift the
  subject must respond to.
- Environmental axis is genuinely thin in most software briefs; do not
  fabricate signal. `_(no signal)_` is fine.
