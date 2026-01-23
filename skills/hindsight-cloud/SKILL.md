---
name: hindsight
description: Store team knowledge, project conventions, and learnings from tasks. Use to remember what works and recall context before new tasks. Connects to Hindsight Cloud. (user)
---

# Hindsight Memory Skill (Cloud)

You have persistent memory via **Hindsight Cloud**. This memory bank is **shared with the team**, so knowledge stored here benefits everyone working on this codebase.

**Proactively store team knowledge and recall context** to provide better assistance.

## Setup Check (First-Time Only)

Before using memory commands, verify the Hindsight CLI is configured:

```bash
cat ~/.hindsight/config
```

**If the file doesn't exist or is missing credentials**, help the user set it up:

1. **Install the CLI** (if `hindsight` command not found):
   ```bash
   curl -fsSL https://hindsight.vectorize.io/get-cli | bash
   ```

2. **Create the config file** - ask the user for their **API Key** (get it from https://ui.hindsight.vectorize.io):
   ```bash
   mkdir -p ~/.hindsight
   cat > ~/.hindsight/config << 'EOF'
   api_url = "https://api.hindsight.vectorize.io"
   api_key = "<user's API key>"
   EOF
   chmod 600 ~/.hindsight/config
   ```

3. **Get the bank ID** - ask the user for their team's bank ID (e.g., `team-myproject`)

After setup, use the bank ID in all commands below.

## Commands

Replace `<bank-id>` with the user's actual bank ID (e.g., `team-frontend`).

### Store a memory

Use `memory retain` to store what you learn:

```bash
hindsight memory retain <bank-id> "Project uses ESLint with Airbnb config and Prettier for formatting"
hindsight memory retain <bank-id> "Running tests requires NODE_ENV=test" --context procedures
hindsight memory retain <bank-id> "Build failed when using Node 18, works with Node 20" --context learnings
hindsight memory retain <bank-id> "Alice prefers verbose commit messages with context" --context preferences
```

### Recall memories

Use `memory recall` BEFORE starting tasks to get relevant context:

```bash
hindsight memory recall <bank-id> "project conventions and coding standards"
hindsight memory recall <bank-id> "Alice preferences for this project"
hindsight memory recall <bank-id> "what issues have we encountered before"
hindsight memory recall <bank-id> "how does the auth module work"
```

### Reflect on memories

Use `memory reflect` to synthesize context:

```bash
hindsight memory reflect <bank-id> "How should I approach this task based on past experience?"
```

## IMPORTANT: When to Store Memories

This is a **shared team bank**. Store knowledge that benefits the team. For individual preferences, include the person's name.

### Project/Team Conventions (shared)
- Coding standards ("Project uses 2-space indentation")
- Required tools and versions ("Project requires Node 20+, PostgreSQL 15+")
- Linting and formatting rules ("ESLint with Airbnb config")
- Testing conventions ("Integration tests require Docker running")
- Branch naming and PR conventions

### Individual Preferences (attribute to person)
- Personal coding style ("Alice prefers explicit type annotations")
- Communication preferences ("Bob prefers detailed PR descriptions")
- Tool preferences ("Carol uses vim keybindings")

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

### Team Knowledge
- Onboarding information for new team members
- Common pitfalls and how to avoid them
- Architecture decisions and their rationale
- Integration points with external systems
- Domain knowledge and business logic explanations

## IMPORTANT: When to Recall Memories

**Always recall** before:
- Starting any non-trivial task
- Making decisions about implementation
- Suggesting tools, libraries, or approaches
- Writing code in a new area of the project
- When answering questions about the codebase
- When a team member asks how something works

## Best Practices

1. **Store immediately**: When you discover something, store it right away
2. **Be specific**: Store "npm test requires --experimental-vm-modules flag" not "tests need a flag"
3. **Include outcomes**: Store what worked AND what did not work
4. **Recall first**: Always check for relevant context before starting work
5. **Think team-first**: Store knowledge that would help other team members
6. **Attribute individual preferences**: Store "Alice prefers X" not just "User prefers X"
7. **Distinguish project vs personal**: Project conventions apply to everyone; personal preferences are per-person
