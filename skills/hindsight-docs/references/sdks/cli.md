
# CLI Reference

The Hindsight CLI provides command-line access to memory operations and bank management. All commands follow the [OpenAPI specification](../openapi.json), so you can use `--help` on any command to see all available options.

## Installation

```bash
curl -fsSL https://hindsight.vectorize.io/get-cli | bash
```

## Configuration

Configure the API URL:

```bash
# Set directly
hindsight configure --api-url http://localhost:8888

# With API key for authentication
hindsight configure --api-url http://localhost:8888 --api-key your-api-key
```

You can also run `hindsight configure` with no flags to be prompted interactively, or use environment variables (highest priority):

```bash
export HINDSIGHT_API_URL=http://localhost:8888
export HINDSIGHT_API_KEY=your-api-key
```

### Named Profiles

When you need to switch between multiple Hindsight deployments (e.g. local,
staging, production) without constantly rewriting `~/.hindsight/config`, use
named profiles. Each profile is a TOML file at
`~/.hindsight/cli-profiles/<name>.toml` and is selected per-invocation with
`-p/--profile` (or by setting `$HINDSIGHT_PROFILE`).

```bash
# Create (or overwrite) a profile
hindsight profile create prod \
  --api-url https://api.hindsight.vectorize.io \
  --api-key hsk_...

# List and inspect profiles
hindsight profile list
hindsight profile show prod

# Use a profile for a single command
hindsight -p prod bank list

# Or make it sticky for the current shell
export HINDSIGHT_PROFILE=prod
hindsight bank list

# Remove a profile
hindsight profile delete prod -y
```

Profile files are written with `0600` permissions on Unix so the API key is
only readable by the owner.

**Configuration precedence** (highest first):

1. Environment variables (`HINDSIGHT_API_URL`, `HINDSIGHT_API_KEY`)
2. Named profile — explicit `-p <name>`, otherwise `$HINDSIGHT_PROFILE`
3. Shared config file (`~/.hindsight/config`, written by `hindsight configure`)
4. Default (`http://localhost:8888`)

`HINDSIGHT_API_URL` / `HINDSIGHT_API_KEY` always override profile values, which
makes it safe to use `-p` in scripts while letting CI inject credentials via
environment.

## Core Commands

### Retain (Store Memory)

Store a single memory:

```bash
hindsight memory retain my-cli-bank "Alice works at Google as a software engineer"

# With context
hindsight memory retain my-cli-bank "Bob loves hiking" --context "hobby discussion"

# Queue for background processing
hindsight memory retain my-cli-bank "Meeting notes" --async

# With an event date (ISO 8601 datetime or date)
hindsight memory retain my-cli-bank "Project launched" --timestamp 2024-01-15

# Store without a timestamp (overrides the default of "now")
hindsight memory retain my-cli-bank "Background fact" --timestamp unset
```

### Retain Files

Bulk import from files:

```bash
# Single file
hindsight memory retain-files my-cli-bank notes.txt

# Directory (recursive by default)
hindsight memory retain-files my-cli-bank ./documents/

# With context
hindsight memory retain-files my-cli-bank meeting-notes.txt --context "team meeting"

# With a named retain strategy (see retain_strategies in bank config)
hindsight memory retain-files my-cli-bank ./documents/ --strategy conversations

# Background processing
hindsight memory retain-files my-cli-bank ./data/ --async
```

### Recall (Search)

Search memories using semantic similarity:

```bash
hindsight memory recall my-cli-bank "What does Alice do?"

# With options
hindsight memory recall my-cli-bank "hiking recommendations" \
  --budget high \
  --max-tokens 8192

# Filter by fact type
hindsight memory recall my-cli-bank "query" --fact-type world,observation

# Filter by tags
hindsight memory recall my-cli-bank "query" --tags work,project \
  --tags-match all

# Pin results to a specific time
hindsight memory recall my-cli-bank "query" --query-timestamp "2026-01-15T00:00:00Z"

# Show trace information
hindsight memory recall my-cli-bank "query" --trace
```

### Reflect (Generate Response)

Generate a response using memories and bank disposition:

```bash
hindsight memory reflect my-cli-bank "What do you know about Alice?"

# With additional context
hindsight memory reflect my-cli-bank "Should I learn Python?" --context "career advice"

# Higher budget for complex questions
hindsight memory reflect my-cli-bank "Summarize my week" --budget high

# Filter by fact type
hindsight memory reflect my-cli-bank "query" \
  --fact-types world,experience \
  --exclude-mental-models
```

### Memory History

View the observation history for a specific memory unit:

```bash
hindsight memory history my-cli-bank "$MEMORY_ID"
```

### Clear Observations

Remove all observations for a memory unit, keeping the core fact:

```bash
hindsight memory clear-observations my-cli-bank "$MEMORY_ID"

# Skip confirmation prompt
hindsight memory clear-observations my-cli-bank "$MEMORY_ID" -y
```

## Bank Management

### List Banks

```bash
hindsight bank list
```

### View Disposition

```bash
hindsight bank disposition my-cli-bank
```

### Set Disposition

```bash
hindsight bank set-disposition my-cli-bank --skepticism 3 --literalism 4 --empathy 5
```

### View Statistics

```bash
hindsight bank stats my-cli-bank
```

### Set Bank Name

```bash
hindsight bank name my-cli-bank "My Assistant"
```

### Set Mission

```bash
hindsight bank mission my-cli-bank "I am a helpful AI assistant interested in technology"
```

### Clear Observations (Bank-wide)

Remove all observations across the entire bank:

```bash
hindsight bank clear-observations my-cli-bank

# Skip confirmation prompt
hindsight bank clear-observations my-cli-bank -y
```

### Recover Consolidation

Recover from a failed or stuck consolidation:

```bash
hindsight bank consolidation-recover my-cli-bank
```

## Document Management

```bash
# List documents
hindsight document list my-cli-bank

# Get document details
hindsight document get my-cli-bank "$DOCUMENT_ID"

# Replace a document's tags
hindsight document update my-cli-bank "$DOCUMENT_ID" --tags project-x,meetings

# Delete document and its memories
hindsight document delete my-cli-bank "$DOCUMENT_ID"
```

## Entity Management

```bash
# List entities
hindsight entity list my-cli-bank

# Get entity details
hindsight entity get my-cli-bank "$ENTITY_ID"
```

## Operation Management

Track and manage async operations (retain-files, consolidation, etc.):

```bash
# List operations
hindsight operation list my-cli-bank

# Get operation status
hindsight operation get my-cli-bank "$OPERATION_ID"

# Cancel a pending operation
hindsight operation cancel my-cli-bank "$OPERATION_ID"

# Retry a failed operation
hindsight operation retry my-cli-bank "$OPERATION_ID"
```

## Webhook Management

Configure event delivery hooks for bank activity:

```bash
# List webhooks
hindsight webhook list my-cli-bank

# Create a webhook (defaults to consolidation.completed events)
hindsight webhook create my-cli-bank https://example.com/hook

# Create with specific events and signing secret
hindsight webhook create my-cli-bank https://example.com/hook \
  --event-types retain.completed,consolidation.completed \
  --secret my-hmac-secret

# Update a webhook
hindsight webhook update my-cli-bank "$WEBHOOK_ID" --url https://new-url.com

# View delivery history
hindsight webhook deliveries my-cli-bank "$WEBHOOK_ID"

# Delete a webhook
hindsight webhook delete my-cli-bank "$WEBHOOK_ID" -y
```

## Knowledge Base

Manage a bank's knowledge pages — living documents organized in a folder tree. See [Knowledge Pages](../developer/api/knowledge-pages) for what they are and how they refresh.

```bash
# Show the folder/page tree (pages that have fallen behind are marked stale)
hindsight knowledge-base tree my-cli-bank
```

```bash
# Create a folder, optionally nested under another
hindsight knowledge-base create-folder my-cli-bank "Operations"
```

```bash
hindsight knowledge-base create-folder my-cli-bank "Runbooks" --parent-id "$FOLDER_ID"
```

```bash
# Create a page — content is generated in the background
hindsight knowledge-base create-page my-cli-bank \
  "Deploying the API" \
  "How is the API deployed?" \
  --parent-id "$FOLDER_ID" \
  --tags ops,type:runbook

# Build a page from raw facts instead of the observation-only default
hindsight knowledge-base create-page my-cli-bank "Recent Incidents" \
  "What incidents happened recently?" \
  --fact-types experience,world --mode full
```

```bash
# Read a page as a markdown document
hindsight knowledge-base get-page my-cli-bank "$PAGE_ID"

# Hybrid search (full-text + vector) over whole pages
hindsight knowledge-base search my-cli-bank "how do we deploy" --limit 5

# Rename, move, or reconfigure a node
hindsight knowledge-base update my-cli-bank "$NODE_ID" --name "New name"
hindsight knowledge-base update my-cli-bank "$PAGE_ID" --source-query "New question?"

# Export the whole knowledge base as a markdown bundle
hindsight knowledge-base export my-cli-bank

# Delete a folder or page and everything under it
hindsight knowledge-base delete my-cli-bank "$NODE_ID" -y
```

> **💡 Tip**
>
`hindsight fs mount --bank <bank_id>` mirrors the same knowledge base onto disk as
real markdown files, kept current by a background refresh loop — handy when you'd
rather use `grep`, `rg`, or your editor than the commands above.
## Audit Logs

Inspect the audit trail for a bank:

```bash
# List audit entries
hindsight audit list my-cli-bank

# Filter by action and transport
hindsight audit list my-cli-bank --action recall --transport mcp

# Filter by date range
hindsight audit list my-cli-bank \
  --start-date "2026-04-01T00:00:00Z" \
  --end-date "2026-04-10T00:00:00Z"

# Pagination
hindsight audit list my-cli-bank --limit 50 --offset 100
```

## Output Formats

```bash
# Pretty (default)
hindsight memory recall my-cli-bank "query"

# JSON
hindsight memory recall my-cli-bank "query" -o json

# YAML
hindsight memory recall my-cli-bank "query" -o yaml
```

## Global Options

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Show detailed output including request/response |
| `-o, --output <format>` | Output format: pretty, json, yaml |
| `-p, --profile <name>` | Use a named profile (see [Named Profiles](#named-profiles)) |
| `--help` | Show help |
| `--version` | Show version |

## Control Plane UI

Launch the web-based Control Plane UI directly from the CLI:

```bash
hindsight ui
```

This runs the Control Plane locally on port 9999 using the API URL from your configuration. The UI provides:

- **Memory bank management** — Browse and manage all your banks
- **Entity explorer** — Visualize the knowledge graph
- **Query testing** — Interactive recall and reflect testing
- **Operation history** — View ingestion and processing logs

> **💡 Tip**
>
The UI command requires Node.js to be installed. It automatically downloads and runs the `@vectorize-io/hindsight-control-plane` package via npx.
## Interactive Explorer

Launch the TUI explorer for visual navigation of your memory banks:

```bash
hindsight explore
```

The explorer provides an interactive terminal interface to:

- **Browse memory banks** — View all banks and their statistics
- **Search memories** — Run recall queries with real-time results
- **Inspect memory details** — Open a memory to view its text, metadata, entities, and full JSON fields
- **Inspect entities** — Explore the knowledge graph and entity relationships
- **View facts** — Browse world facts, experiences, and observations
- **Navigate documents** — See source documents and their extracted memories

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `↑/↓` | Navigate items |
| `Enter` | Select item / view details |
| `Tab` | Switch panels |
| `/` | Search |
| `q` | Quit |

{/* Screenshot placeholder: explore command TUI */}

## Example Workflow

```bash
# Configure API URL
hindsight configure --api-url http://localhost:8888

# Store some memories
hindsight memory retain cli-demo "Alice works at Google"
hindsight memory retain cli-demo "Bob is a data scientist"
hindsight memory retain cli-demo "Alice and Bob are colleagues"

# Search memories
hindsight memory recall cli-demo "Who works with Alice?"

# Generate a response
hindsight memory reflect cli-demo "What do you know about the team?"

# Check bank disposition
hindsight bank disposition cli-demo
```
