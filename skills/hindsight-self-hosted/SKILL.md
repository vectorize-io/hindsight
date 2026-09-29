---
name: hindsight-self-hosted
description: Store team knowledge, project conventions, and learnings from tasks. Use to remember what works and recall context before new tasks. Connects to a self-hosted Hindsight server. (user)
---

# Hindsight Memory Skill (Self-Hosted)

You have persistent memory via a **self-hosted Hindsight server**. This memory bank can be **shared with the team**, so knowledge stored here benefits everyone working on this codebase.

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

2. **Create the config file** - ask the user for:
   - **API URL**: Their self-hosted Hindsight server URL (e.g., `https://hindsight.mycompany.com`)
   - **API Key**: Their authentication key

   ```bash
   mkdir -p ~/.hindsight
   cat > ~/.hindsight/config << 'EOF'
   api_url = "<user's server URL>"
   api_key = "<user's API key>"
   EOF
   chmod 600 ~/.hindsight/config
   ```

3. **Get the bank ID** - ask the user for their bank ID (e.g., `team-myproject`)

After setup, use the bank ID in all commands below.

## How Hindsight Works

When you call `retain`, Hindsight does **not** store the string as-is. The server runs an internal pipeline that:

1. **Extracts structured facts** from the content using an LLM
2. **Identifies entities** (people, tools, concepts) and links related facts
3. **Builds temporal and causal relationships** between facts
4. **Generates embeddings** for semantic search

This means you should pass **rich, full-context content** — the server is better at extracting what matters than a pre-summarized string. Your job is to decide **when** to store, not **what** to extract.

## Commands

Replace `<bank-id>` with the user's actual bank ID (e.g., `team-frontend`).

### Store a memory

Use `memory retain` to store what you learn. Pass full context — raw observations, session notes, or detailed descriptions:

```bash
hindsight memory retain <bank-id> "The project uses ESLint configured with the Airbnb rule set and Prettier for formatting. Auto-fix on save is enabled in the editor config."
hindsight memory retain <bank-id> "Ran the test suite with NODE_ENV=test. Tests pass. Without NODE_ENV=test, the suite fails with a missing config error." --context procedures
hindsight memory retain <bank-id> "Build failed on Node 18 with error 'ERR_UNSUPPORTED_ESM_URL_SCHEME'. Switched to Node 20 and build succeeded." --context learnings
hindsight memory retain <bank-id> "Alice reviewed the PR and asked for verbose commit messages that explain the motivation, not just what changed." --context preferences
```

You can also pass a raw conversation transcript with timestamps:

```bash
hindsight memory retain <bank-id> "[2026-03-16T10:12:03] User: The auth tests keep failing on CI but pass locally. Any idea?
[2026-03-16T10:12:45] Assistant: Let me check the CI logs. Looks like the tests are running without the TEST_DATABASE_URL env var set — they fall back to the production DB URL and hit a connection timeout.
[2026-03-16T10:13:20] User: Ah right, I never added that to the CI secrets. Adding it now.
[2026-03-16T10:15:02] User: That fixed it. All green now." --context learnings
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
2. **Pass rich context**: Include full observations, not pre-summarized strings — the server extracts facts automatically
3. **Include outcomes**: Store what happened AND why, including failures and workarounds
4. **Recall first**: Always check for relevant context before starting work
5. **Think team-first**: Store knowledge that would help other team members
6. **Attribute individual preferences**: Store "Alice reviewed the PR and asked for X" not just "User prefers X"
7. **Distinguish project vs personal**: Project conventions apply to everyone; personal preferences are per-person
8. **Use `--context` for metadata**: The `--context` flag labels the type of memory (e.g., `procedures`, `learnings`, `preferences`), not a replacement for full content
