---
name: hindsight
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

## Commands

### Store a memory

Use `memory retain` to store what you learn:

```bash
uvx hindsight-embed memory retain default "User prefers TypeScript with strict mode"
uvx hindsight-embed memory retain default "Running tests requires NODE_ENV=test" --context procedures
uvx hindsight-embed memory retain default "Build failed when using Node 18, works with Node 20" --context learnings
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
2. **Be specific**: Store "npm test requires --experimental-vm-modules flag" not "tests need a flag"
3. **Include outcomes**: Store what worked AND what did not work
4. **Recall first**: Always check for relevant context before starting work
