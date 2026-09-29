---
name: hindsight-local
description: Store user preferences, learnings from tasks, and procedure outcomes. Use to remember what works and recall context before new tasks. (user)
---

# Hindsight Memory Skill (Local)

You have persistent memory via the `hindsight-embed` CLI. **Proactively store learnings and recall context** to provide better assistance.

## Setup Check (First-Time Only)

Before using memory commands, verify Hindsight is configured:

```bash
uvx hindsight-embed daemon status
```

**If this fails or shows "not configured"**, run the interactive setup:

```bash
uvx hindsight-embed configure
```

This will prompt for an LLM provider and API key. After setup, the commands below will work.

## How Hindsight Works

When you call `retain`, Hindsight does **not** store the string as-is. The server runs an internal pipeline that:

1. **Extracts structured facts** from the content using an LLM
2. **Identifies entities** (people, tools, concepts) and links related facts
3. **Builds temporal and causal relationships** between facts
4. **Generates embeddings** for semantic search

This means you should pass **rich, full-context content** — the server is better at extracting what matters than a pre-summarized string. Your job is to decide **when** to store, not **what** to extract.

## Commands

### Store a memory

Use `memory retain` to store what you learn. Pass the full context — raw observations, session notes, conversation excerpts, or detailed descriptions:

```bash
uvx hindsight-embed memory retain default "User is working on a TypeScript project. They enabled strict mode and prefer explicit type annotations over inference."
uvx hindsight-embed memory retain default "Ran the test suite with NODE_ENV=test. Tests pass. Without NODE_ENV=test, the suite fails with a missing config error." --context procedures
uvx hindsight-embed memory retain default "Build failed on Node 18 with error 'ERR_UNSUPPORTED_ESM_URL_SCHEME'. Switched to Node 20 and build succeeded." --context learnings
```

You can also pass a raw conversation transcript with timestamps:

```bash
uvx hindsight-embed memory retain default "[2026-03-16T10:12:03] User: The auth tests keep failing on CI but pass locally. Any idea?
[2026-03-16T10:12:45] Assistant: Let me check the CI logs. Looks like the tests are running without the TEST_DATABASE_URL env var set — they fall back to the production DB URL and hit a connection timeout.
[2026-03-16T10:13:20] User: Ah right, I never added that to the CI secrets. Adding it now.
[2026-03-16T10:15:02] User: That fixed it. All green now." --context learnings
```

### Recall memories

Use `memory recall` BEFORE starting tasks to get relevant context:

```bash
uvx hindsight-embed memory recall default "user preferences for this project"
uvx hindsight-embed memory recall default "what issues have we encountered before"
```

### Reflect on memories

Use `memory reflect` to synthesize context:

```bash
uvx hindsight-embed memory reflect default "How should I approach this task based on past experience?"
```

## IMPORTANT: When to Store Memories

**Always store** after you learn something valuable:

### User Preferences
- Coding style (indentation, naming conventions, language preferences)
- Tool preferences (editors, linters, formatters)
- Communication preferences
- Project conventions

### Procedure Outcomes
- Steps that successfully completed a task
- Commands that worked (or failed) and why
- Workarounds discovered
- Configuration that resolved issues

### Learnings from Tasks
- Bugs encountered and their solutions
- Performance optimizations that worked
- Architecture decisions and rationale
- Dependencies or version requirements

## IMPORTANT: When to Recall Memories

**Always recall** before:
- Starting any non-trivial task
- Making decisions about implementation
- Suggesting tools, libraries, or approaches
- Writing code in a new area of the project

## Best Practices

1. **Store immediately**: When you discover something, store it right away
2. **Pass rich context**: Include full observations, not pre-summarized strings — the server extracts facts automatically
3. **Include outcomes**: Store what happened AND why, including failures and workarounds
4. **Recall first**: Always check for relevant context before starting work
5. **Use `--context` for metadata**: The `--context` flag labels the type of memory (e.g., `procedures`, `learnings`, `preferences`), not a replacement for full content
