---
name: agent-memory
description: Build and maintain your own procedural memory as files in the workspace. Use whenever you learn something worth remembering across sessions — a user preference, a procedure that worked, a rule the user stated, a source to track, a decision and its rationale, or any knowledge you'd want available next time. Also use when the user says "remember this", "don't forget", "keep track of", or asks you to update something you previously noted.
---

# Agent Memory

You have a dedicated memory directory outside your workspace. It is yours — not shared with the harness, not inside any skill folder, not subject to workspace resets. Use it to build a growing wiki of everything you need to remember across sessions. Each file is one topic. You decide what's worth tracking.

## Where files live

```
~/.agent-memory/<agent-name>/
```

`<agent-name>` is your agent id (e.g. `news-feed`, `discord-watch`). If you don't know your agent name, check the session key or ask the user. Always use the absolute expanded path (`/Users/<user>/.agent-memory/<agent-name>/`) when reading/writing — never a relative path, since your working directory may vary.

Create the directory on first use if it doesn't exist.

## When to write

Two mandatory triggers. Both fire after the user's request is fulfilled, never mid-task.

### Trigger 1: You learned something (knowledge files)

After any turn where you learned something durable — something useful in a future session. Write or update a knowledge file. Examples:

- User stated a preference ("I want short bullets, no fluff")
- You discovered a procedure that works ("to fetch RSS, use exec + curl, not web_fetch")
- User corrected you ("no, I said 10 items, not 5")
- A decision was made ("we agreed to use per-agent bank routing")
- You learned a fact about the user's setup ("user runs Hindsight locally on port 8888")
- User explicitly asked you to remember something

### Trigger 2: You completed a task run (activity logs)

**After every completed task run** — a news feed delivery, a Discord sweep, a report, a draft, anything the user consumed as output — **always** append a summary entry to the activity log. This is not optional. No exceptions. The entry records:

- Date
- What was requested (query style / trigger)
- Sources used and any retrieval failures
- What was delivered (headlines / item titles — not full content)
- Notable user feedback on the output (if any in this session)

This is how you deduplicate across runs and answer "what did you show me yesterday". If you skip this step, the next run will re-surface the same items.

### Do NOT write memory for:

- Raw step-by-step execution logs or tool call traces
- Ephemeral task state (use the conversation for that)
- Things already in the skill files or BRIEF.md
- Trivial acknowledgements ("got it", "ok")

## Two kinds of memory files

### 1. Knowledge files — what you know

One file per topic. Tracks preferences, rules, procedures, setup facts, decisions.

```
~/.agent-memory/news-feed/preferences.md
~/.agent-memory/news-feed/rss-procedure.md
~/.agent-memory/news-feed/user-setup.md
~/.agent-memory/news-feed/source-list.md
```

These are **current-state** files — update in place when facts change. Keep them short and declarative.

### 2. Activity log — what you did

One file per recurring task type. Tracks what you produced and when, so you can deduplicate, reference prior output, and avoid repeating yourself.

```
~/.agent-memory/news-feed/feed-log.md
~/.agent-memory/discord-watch/sweep-log.md
```

These are **append-only** — add a new entry each time you complete a task run. Each entry is a short dated record of what you delivered: date, item count, the headlines or item titles (not full content), and any user reaction. Prune entries older than ~30 days to keep the file manageable.

When starting a new task run, read the log first. Skip items you already delivered in a recent run. If the user asks "what did you show me yesterday", the log is your answer.

Example:

```markdown
# Feed Log

## 2026-04-17

- 10 items delivered
- Headlines: "OpenAI ships GPT-5.5 API", "Hugging Face releases ...", ...
- User reaction: "good, but drop the Gemini item next time"

## 2026-04-16

- 8 items delivered
- Headlines: "Anthropic Claude 4.5 ...", "Google Gemini 2.5 ...", ...
- User reaction: none
```

## File naming

Lowercase with hyphens. Short, descriptive, greppable. Knowledge files are named by topic; activity logs are named by task + `-log` suffix.

If a topic file doesn't exist yet, create it. If it already exists, update it in place (for knowledge files) or append to it (for activity logs).

## File format

Every file follows this structure:

```markdown
# <Topic Name>

<Current state of knowledge on this topic. Written as if briefing a
colleague who has never seen this conversation. Concise, declarative,
no hedging.>

## Evidence

- [<date>] <what happened that established or changed this knowledge>
- [<date>] <another event>
```

The `## Evidence` section is mandatory. Every fact in the file must trace to at least one evidence entry. When you update a fact, add a new evidence line explaining what changed and why. When a fact is superseded, don't delete the old evidence — mark it as superseded so the history is visible.

Example:

```markdown
# News Feed Preferences

- Format: short bullets, one sentence each
- Item cap: 10 per run
- Sources: prefer RSS/Atom feeds; use web search as fallback only
- Topics in: developer-focused AI, memory systems, RAG, multimodal
- Topics out: academic papers, hardware benchmarks, product fluff
- Window: last 7 days
- Voice: concise, dev-centric, no marketing speak
- Always include at least one OpenClaw item when meaningful

## Evidence

- [2026-04-15] User said "use defaults, short summaries, numbered, 5 items, last 24h" during initial setup
- [2026-04-15] User corrected to 10 items: "I want 10 items"
- [2026-04-15] User said "no product PR, more papers" — later clarified "actually no papers either, just dev product news"
- [2026-04-15] User requested RSS-first sourcing after web_search returned stale results
- [2026-04-15] User asked to always include OpenClaw releases when available
- [2026-04-16] User changed window from 24h to 7 days
```

## How to use memory at the start of a turn

At the start of any task, check if relevant memory files exist:

```bash
ls ~/.agent-memory/<agent-name>/ 2>/dev/null
```

If files exist that might be relevant to the current request, read them before acting. They are your accumulated knowledge — treat them as ground truth unless the user contradicts them in this conversation (in which case, update the file).

## Rules

1. **One file per topic.** Don't dump everything into one file. If two concerns are distinct (e.g. "user preferences" vs "known procedures"), they get separate files.
2. **Evidence is mandatory.** Never write a fact without an evidence entry explaining where it came from. If you can't cite evidence, you're guessing — ask the user instead.
3. **Update, don't append.** When a fact changes, rewrite the fact in place and add a new evidence entry. The file should always read as the *current* state of knowledge, not a log.
4. **Supersede, don't delete.** When old evidence is contradicted, mark it `~~superseded~~` in the evidence section. Don't remove it — the history matters.
5. **Write after acting, not before.** Finish the user's task first, then update memory. Don't interrupt the flow to take notes.
6. **Keep files short.** A memory file that exceeds ~50 lines is probably covering too many topics. Split it.
7. **Date every evidence entry.** Use `[YYYY-MM-DD]` format. If you don't know today's date, ask the harness or omit the date rather than guess.
