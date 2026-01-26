---
sidebar_position: 3
---

# CLI Reference

The Hindsight CLI provides command-line access to memory operations and bank management.

## Installation

```bash
curl -fsSL https://hindsight.vectorize.io/get-cli | bash
```

## Configuration

Configure the API URL:

```bash
# Interactive configuration
hindsight configure

# Or set directly
hindsight configure --api-url http://localhost:8888

# Or use environment variable (highest priority)
export HINDSIGHT_API_URL=http://localhost:8888
```

## Core Commands

### Retain (Store Memory)

Store a single memory:

```bash
hindsight memory retain <bank_id> "Alice works at Google as a software engineer"

# With context
hindsight memory retain <bank_id> "Bob loves hiking" --context "hobby discussion"

# Queue for background processing
hindsight memory retain <bank_id> "Meeting notes" --async
```

### Retain Files

Bulk import from files:

```bash
# Single file
hindsight memory retain-files <bank_id> notes.txt

# Directory (recursive by default)
hindsight memory retain-files <bank_id> ./documents/

# With context
hindsight memory retain-files <bank_id> meeting-notes.txt --context "team meeting"

# Background processing
hindsight memory retain-files <bank_id> ./data/ --async
```

### Recall (Search)

Search memories using semantic similarity:

```bash
hindsight memory recall <bank_id> "What does Alice do?"

# With options
hindsight memory recall <bank_id> "hiking recommendations" \
  --budget high \
  --max-tokens 8192

# Filter by fact type
hindsight memory recall <bank_id> "query" --fact-type world,observation

# Show trace information
hindsight memory recall <bank_id> "query" --trace
```

### Reflect (Generate Response)

Generate a response using memories and bank disposition:

```bash
hindsight memory reflect <bank_id> "What do you know about Alice?"

# With additional context
hindsight memory reflect <bank_id> "Should I learn Python?" --context "career advice"

# Higher budget for complex questions
hindsight memory reflect <bank_id> "Summarize my week" --budget high
```

## Bank Management

### List Banks

```bash
hindsight bank list
```

### View Disposition

```bash
hindsight bank disposition <bank_id>
```

### View Statistics

```bash
hindsight bank stats <bank_id>
```

### Set Bank Name

```bash
hindsight bank name <bank_id> "My Assistant"
```

### Set Mission

```bash
hindsight bank mission <bank_id> "I am a helpful AI assistant interested in technology"

# Skip automatic disposition inference
hindsight bank mission <bank_id> "Mission text" --no-update-disposition
```

## Document Management

```bash
# List documents
hindsight document list <bank_id>

# Get document details
hindsight document get <bank_id> <document_id>

# Delete document and its memories
hindsight document delete <bank_id> <document_id>
```

## Entity Management

```bash
# List entities
hindsight entity list <bank_id>

# Get entity details
hindsight entity get <bank_id> <entity_id>
```

## Output Formats

```bash
# Pretty (default)
hindsight memory recall <bank_id> "query"

# JSON
hindsight memory recall <bank_id> "query" -o json

# YAML
hindsight memory recall <bank_id> "query" -o yaml
```

## Global Options

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Show detailed output including request/response |
| `-o, --output <format>` | Output format: pretty, json, yaml |
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

:::tip
The UI command requires Node.js to be installed. It automatically downloads and runs the `@vectorize-io/hindsight-control-plane` package via npx.
:::

## Interactive Explorer

Launch the TUI explorer for visual navigation of your memory banks:

```bash
hindsight explore
```

The explorer provides an interactive terminal interface to:

- **Browse memory banks** — View all banks and their statistics
- **Search memories** — Run recall queries with real-time results
- **Inspect entities** — Explore the knowledge graph and entity relationships
- **View facts** — Browse world facts, experiences, and observations
- **Navigate documents** — See source documents and their extracted memories

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `↑/↓` | Navigate items |
| `Enter` | Select / Expand |
| `Tab` | Switch panels |
| `/` | Search |
| `q` | Quit |

<!-- Screenshot placeholder: explore command TUI -->

## Example Workflow

```bash
# Configure API URL
hindsight configure --api-url http://localhost:8888

# Store some memories
hindsight memory retain demo "Alice works at Google"
hindsight memory retain demo "Bob is a data scientist"
hindsight memory retain demo "Alice and Bob are colleagues"

# Search memories
hindsight memory recall demo "Who works with Alice?"

# Generate a response
hindsight memory reflect demo "What do you know about the team?"

# Check bank disposition
hindsight bank disposition demo
```
