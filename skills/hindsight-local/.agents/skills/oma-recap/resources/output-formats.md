# Recap Output Format Templates

Markdown templates for the recap files written to `.agents/results/recap/`.
Response language follows the `language` setting in `.agents/oma-config.yaml`.

## Daily format (1d or specific date)

```markdown
## {date} Recap

> **TL;DR**
> - {What I accomplished 1: project name + outcome}
> - {What I accomplished 2}
> - {What I accomplished 3}

### Overview
2-3 sentence summary of the day. Written from "I did X" perspective.
Focus on outcomes and progress, not tool ratios or technical details.

### {Theme 1} (AM 09:36~11:30)
- Core work performed
- Key decisions
- 2-4 bullets per theme

### {Theme 2} (PM 13:33~15:21)
- Core work performed
- Key decisions

### Miscellaneous
- Brief summary of sub-15-minute tasks

### Tool Usage Patterns
- Tool usage ratios and primary purposes
- Notable tool-switching patterns
```

## Multi-day format (3d, 7d, 2w, 30d)

For any multi-day window, use a **project-driven structure** like a sprint report.
Focus on what was accomplished per project, not day-by-day chronology.

```markdown
## {start} ~ {end} Monthly Recap

> **TL;DR**
> - {What I accomplished 1: project name + outcome}
> - {What I accomplished 2}
> - {What I accomplished 3}

### Overview
3-5 sentence narrative of the month. Major focus shifts week-by-week,
key milestones achieved, and overall direction. Written from "I did X" perspective.

### {Project A}
What this project is, what was accomplished during the period.
- Key milestone or deliverable 1
- Key milestone or deliverable 2
- Key decision made
- Current status (shipped / in progress / blocked)

### {Project B}
- ...

### Side Projects
Projects with <30 prompts, summarized briefly.
- {project}: one-line summary
- {project}: one-line summary

### Tool Usage Patterns
- Tool usage ratios and how they evolved over the month
- Notable shifts (e.g., "started using Codex mid-month")
```

## Multi-day grouping rules

- Group by **project**, not by date
- Order projects by activity volume (most active first)
- Each project section: what it is, what was accomplished, key decisions, current status
- Do NOT include prompt counts or date ranges in project headers (those are internal metrics)
- Small projects (<30 prompts) go into "Side Projects" as one-liners
- Overview should read like a sprint report narrative, not a log
