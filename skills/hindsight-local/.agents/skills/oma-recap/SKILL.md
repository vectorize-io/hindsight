---
name: oma-recap
description: Analyze conversation histories from multiple AI tools (Grok, Claude, Codex, Qwen, Cursor, Antigravity) and generate themed daily/period work summaries. Filter by date or time window.
---

# AI Tool Conversation History Summary

Analyze AI tool conversation histories for a given period and generate themed work summaries.

## Scheduling

### Goal
Collect AI tool conversation history for a date or window and synthesize it into a themed, project-oriented recap with saved Markdown output.

### Intent signature
- User asks for daily recap, weekly/monthly summary, standup notes, work log, tool usage pattern, or AI conversation history analysis.
- User wants conversation histories grouped by work content rather than raw chronological logs.

### When to use
- Summarizing a day or period of work activity
- Understanding the overall flow of work across multiple AI tools
- Analyzing tool-switching patterns between sessions
- Preparing daily standups, weekly retros, or work logs

### When NOT to use
- Git commit-based code change retrospective -> use `oma retro`
- Real-time agent monitoring -> use `oma dashboard`
- Productivity metrics -> use `oma stats`

### Expected inputs
- Date, relative date, time window, or tool filter
- Conversation history available through `oma recap --json` or fallback sources
- Desired daily or multi-day recap scope

### Expected outputs
- Markdown recap saved to `.agents/results/recap/{date}.md` or range filename
- TL;DR, overview, themes/projects, miscellaneous or side projects, and tool usage patterns
- User-facing summary in configured response language

### Dependencies
- `oma recap --json`
- Optional Claude fallback history at `~/.claude/history.jsonl`
- `.agents/oma-config.yaml` for language behavior

### Control-flow features
- Branches by date resolution, window length, available tool history, and daily vs multi-day output shape
- Reads local history data and writes Markdown recap files
- Groups by content, not by tool

## Structural Flow

### Entry
1. Resolve requested date or window.
2. Collect normalized conversation history.
3. Decide daily versus multi-day output structure.

### Scenes
1. **PREPARE**: Resolve time range and tool filters.
2. **ACQUIRE**: Collect history through CLI or fallback.
3. **REASON**: Group by content, infer themes/projects, decisions, artifacts, and tool-switching patterns.
4. **ACT**: Write recap Markdown in the required format.
5. **VERIFY**: Check TL;DR, grouping, language, and output path.
6. **FINALIZE**: Save and display summary.

### Transitions
- If no date is specified, use today.
- If window is 3 days or longer, group by project instead of day chronology.
- If CLI is unavailable, use Claude fallback only and report scope limits.
- If tasks are under threshold, group them into Miscellaneous or Side Projects.

### Failure and recovery
- If history is unavailable, report missing source and requested range.
- If timestamps are ambiguous, use configured timezone and state assumption.
- If extracted data is sparse, produce a concise recap and note limited coverage.

### Exit
- Success: recap file exists and summary is displayed.
- Partial success: missing tools/history or fallback-only coverage is explicit.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Resolve date/window | `INFER` | Natural-language date rules |
| Collect history | `CALL_TOOL` | `oma recap --json` or `jq` fallback |
| Read extracted records | `READ` | Conversation history |
| Group themes/projects | `INFER` | Time/content grouping rules |
| Validate output shape | `VALIDATE` | Daily or multi-day template |
| Write recap | `WRITE` | `.agents/results/recap/` |
| Report summary | `NOTIFY` | Displayed recap |

### Tools and instruments
- `oma recap --json`
- `jq` fallback for Claude history
- Markdown output templates

### Canonical command path
```bash
oma recap --json
oma recap --window 7d --json
oma recap --date YYYY-MM-DD --json
```

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `LOCAL_FS` | Conversation history and recap output files |
| `PROCESS` | `oma recap`, `jq`, date commands |
| `USER_DATA` | Conversation prompts and project activity |
| `MEMORY` | Theme grouping and summary notes |

### Preconditions
- Requested time range can be resolved.
- At least one history source is available.

### Effects and side effects
- Writes recap Markdown under `.agents/results/recap/`.
- Reads local conversation history data.

### Guardrails

1. **TL;DR required**: Top 3 lines of "what I accomplished". Project name + outcome. No tool names or technical details.
2. **Overview**: After TL;DR, describe the flow. Start with "I" as subject.
3. **Daily**: themes by time block (15+ min). Rest goes to "Miscellaneous".
4. **Multi-day (3d+)**: sections by project, ordered by activity. Read like a sprint report, not a daily log.
5. **2-4 bullets per theme/project**: Concise essentials only. Don't enumerate every step.
6. **Themes by content**: Group by actual work, not by tool.
7. **Time range (daily only)**: `(AM/PM/Evening HH:MM~HH:MM)`. AM: ~12:00, PM: 12:00~18:00, Evening: 18:00~.
8. **Save results**: Write markdown to `.agents/results/recap/`.
9. **Response language**: Follows `language` setting in `.agents/oma-config.yaml` if configured.
10. **No em dashes**: Use commas, periods, or parentheses instead of `—` (em dash).

### Process

### 1. Resolve Date

Determine the target date or window from the user's natural language input. Default is today.

**Resolution rules:**
- Relative day references (today, yesterday, day before yesterday, etc.) → calculate `--date YYYY-MM-DD`
- Specific date mentions (month + day, or full date) → convert to `--date YYYY-MM-DD`
- Relative weekday references (last Monday, this Friday, etc.) → calculate the date
- Period references (this week, last 3 days, past 2 weeks, etc.) → convert to `--window Nd`
- No date specified → today (`--window 1d`)

### 2. Collect Data

Extract normalized conversation history via CLI.

```bash
# Default (today, all tools)
oma recap --json

# Time window
oma recap --window 7d --json

# Specific date
oma recap --date 2026-04-10 --json

# Tool filter (supported: grok, claude, codex, qwen, cursor, antigravity)
oma recap --tool claude,codex --json
```

**Fallback when CLI is not installed**: process Claude history only via inline jq:

```bash
# Uses the system timezone; export TZ=<zone> first to override, and state the
# timezone assumption in the recap (per the Failure and recovery rules).
TARGET_DATE=$(date +%Y-%m-%d)
# macOS/BSD date:
start_ts=$(date -j -f "%Y-%m-%d %H:%M:%S" "${TARGET_DATE} 00:00:00" +%s)000
# Linux/GNU date alternative:
# start_ts=$(date -d "${TARGET_DATE} 00:00:00" +%s)000
end_ts=$((start_ts + 86400000))

jq -r --argjson start "$start_ts" --argjson end "$end_ts" '
  select(.timestamp >= $start and .timestamp < $end and .display != null and .display != "") |
  {
    time: (.timestamp / 1000 | localtime | strftime("%H:%M")),
    project: (.project | split("/") | .[-1]),
    prompt: (.display | gsub("\n"; " ") | if length > 150 then .[0:150] + "..." else . end)
  }
' ~/.claude/history.jsonl
```

### 3. Theme Analysis and Grouping

Read **all** extracted data and analyze with the following criteria:

**Grouping rules:**
- Only classify as a separate theme if the work spans **15+ minutes** (based on timestamp gaps and prompt count)
- Merge consecutive prompts on the same topic into one theme
- Collect sub-15-minute tasks into a "Miscellaneous" section
- Group by **work content**, not by tool

**Cross-tool analysis:**
- Track workflow when multiple tools are used in the same time window
- Example: "Designed in Gemini -> Implemented in Claude -> Reviewed in Codex"
- Derive insights from tool-switching patterns

**Extract from each theme:**
- Core work performed
- Key decisions made
- Tool combinations used
- Artifacts produced (docs, code, config, etc.)

### 4. Output Format

Save results to `.agents/results/recap/{date}.md` and display simultaneously.

Use the markdown templates in `resources/output-formats.md`:
- **Daily format** (1d or specific date): TL;DR → Overview → time-blocked themes → Miscellaneous → Tool Usage Patterns.
- **Multi-day format** (3d+): project-driven sprint-report structure with Side Projects for small (<30 prompts) work; follow the multi-day grouping rules in the same file.

**Response language follows `language` setting in `.agents/oma-config.yaml`.**

### 5. Save Results

Save to `.agents/results/recap/{date}.md`.
For window ranges, use `{start-date}~{end-date}.md` format.

```bash
# Example paths
.agents/results/recap/2026-04-12.md
.agents/results/recap/2026-04-06~2026-04-12.md
```

## References

- Output format templates: `resources/output-formats.md`
- Recap CLI: `oma recap --json`
- Output directory: `.agents/results/recap/`
- Language config: `.agents/oma-config.yaml`
- Claude fallback history: `~/.claude/history.jsonl`
