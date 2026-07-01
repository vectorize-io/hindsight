# Onboarding Risk Signals (SCM)

Use these lightweight SCM metrics to quickly estimate maintenance risk in an unfamiliar repository.

## 1) High-Churn Files

Frequent edits often indicate maintenance hotspots or unstable boundaries.

```bash
git log --format=format: --name-only --since="1 year ago" | sort | uniq -c | sort -nr | head -20
```

## 2) Ownership / Bus Factor

Detect concentration risk and potential knowledge silos.

```bash
git shortlog -sn --no-merges
```

Heuristic:
- One author > 60% of commits can indicate bus-factor risk.
- If top contributors are inactive for 6+ months, flag continuity risk.
- With squash-merge-heavy teams, treat commit-count ownership as approximate.

## 3) Bug Hotspots

Approximate bug-prone files from bug-fix commit history.

```bash
git log -i -E --grep="fix|bug|broken" --name-only --format='' | sort | uniq -c | sort -nr | head -20
```

Cross-reference with high-churn list to identify high-risk files.

## 4) Development Velocity Trend

Inspect monthly activity and momentum shifts.

```bash
git log --format='%ad' --date=format:'%Y-%m' | sort | uniq -c
```

Heuristic:
- Stable/increasing trend suggests healthy delivery cadence.
- 50%+ sudden drop can indicate staffing or ownership disruption.
- 6-12 month steady decline can indicate project stagnation.

## 5) Revert / Hotfix / Firefighting Signal

Estimate operational reliability from emergency recovery patterns.

```bash
git log --oneline --since="1 year ago" | rg -i "revert|hotfix|emergency|rollback"
```

Heuristic:
- A few incidents per year can be normal.
- Incidents every ~2 weeks suggest unstable release/testing process.
- Zero incidents may mean stable delivery or poor commit labeling.

## Practical Output Template

Report these sections:
- Top churn files (top 10-20)
- Ownership concentration and inactive key contributors
- Bug hotspot overlaps with churn files
- Velocity trend summary (recent 6-12 months)
- Firefighting frequency and confidence level

Always include caveats: squash merges, inconsistent commit labels, and monorepo skew.
