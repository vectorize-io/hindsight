---
sidebar_position: 5
---

# Daemon CLI (hindsight-embed)

Zero-configuration local memory system with automatic daemon management. Perfect for development, prototyping, and single-user applications.

## Overview

`hindsight-embed` is a zero-configuration SDK that wraps the Hindsight API and PostgreSQL database into a single auto-managed local daemon. It's designed for development, prototyping, and single-user applications where you want memory capabilities without infrastructure overhead.

**How it works:**

1. **First command triggers startup**: When you run any `hindsight-embed` command, it checks if a local daemon is running
2. **Auto-daemon management**: If no daemon exists, it automatically spawns `hindsight-api --daemon` in the background
3. **Embedded database**: The daemon uses `pg0` (embedded PostgreSQL) — no separate database installation required
4. **Command forwarding**: Your command is forwarded to the local daemon via HTTP (localhost:8888)
5. **Auto-shutdown**: After 5 minutes of inactivity (configurable), the daemon gracefully shuts down to free resources

**Key features:**

- **Zero setup** — One `configure` command and you're ready
- **Automatic lifecycle** — Daemon starts on-demand, stops when idle
- **Isolated storage** — Each bank gets its own embedded PostgreSQL database
- **Local-only** — Binds to `127.0.0.1:8888`, not accessible from network
- **Production-grade engine** — Uses the same memory engine as the full API service

Think of it as SQLite for long-term memory — all the power of Hindsight without managing servers.

## Installation

Install via `uvx` (recommended - always latest version):

```bash
# Run directly without installation
uvx hindsight-embed@latest configure

# Or use pipx for persistent installation
pipx install hindsight-embed
```

## Quick Start

### 1. Configure

```bash
# Interactive configuration
hindsight-embed configure

# Or non-interactive via environment variables
export HINDSIGHT_API_LLM_PROVIDER=openai
export HINDSIGHT_API_LLM_API_KEY=sk-xxxxxxxxxxxx
export HINDSIGHT_API_LLM_MODEL=gpt-4o-mini
hindsight-embed configure
```

Configuration is saved to `~/.hindsight/embed`:

```bash
HINDSIGHT_API_LLM_PROVIDER=openai
HINDSIGHT_API_LLM_MODEL=gpt-4o-mini
HINDSIGHT_API_LLM_API_KEY=sk-xxxxxxxxxxxx

# Daemon settings (macOS: force CPU to avoid MPS/XPC issues)
HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU=1
HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU=1
```

### 2. Use Memory Operations

```bash
# Store a memory
hindsight-embed memory retain default "User prefers dark mode"

# Query memories
hindsight-embed memory recall default "user preferences"

# Reasoning with memory
hindsight-embed memory reflect default "What color scheme should I use?"
```

The daemon starts automatically on first use!

### 3. Open the Control Center (optional)

Use the local control center when you want a browser-based configuration wizard and daemon supervisor:

```bash
# Launch the control center and open the browser
hindsight-embed control start

# Or use the browser wizard instead of terminal prompts during setup
hindsight-embed configure --ui
```

The control center listens on localhost only (`http://localhost:7878` by default) and prints a tokenized URL. It stores the local access token at `~/.hindsight/control.token` and writes logs to `~/.hindsight/control.log`.

```bash
# Pick a different control-center port for this launch
hindsight-embed control start --port 7879

# Start without opening a browser automatically
hindsight-embed control start --no-open

# Check, inspect, or stop the control center
hindsight-embed control status
hindsight-embed control logs -f
hindsight-embed control stop
```

The control center runs as a separate process from the memory daemon. Stopping or restarting the control center does not stop a running daemon.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_LLM_API_KEY` | **Required**. API key for LLM provider | - |
| `HINDSIGHT_API_LLM_PROVIDER` | LLM provider: `openai`, `anthropic`, `gemini`, `groq`, `minimax`, `ollama` | `openai` |
| `HINDSIGHT_API_LLM_MODEL` | Model name | `gpt-4o-mini` |
| `HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT` | Seconds before daemon auto-exits when idle (0 = never) | `0` |
| `HINDSIGHT_EMBED_CONTROL_PORT` | Default port for `hindsight-embed control start` | `7878` |

**Provider Examples:**

```bash
# OpenAI
export HINDSIGHT_API_LLM_PROVIDER=openai
export HINDSIGHT_API_LLM_API_KEY=sk-xxxxxxxxxxxx
export HINDSIGHT_API_LLM_MODEL=gpt-4o

# Groq (fast inference)
export HINDSIGHT_API_LLM_PROVIDER=groq
export HINDSIGHT_API_LLM_API_KEY=gsk_xxxxxxxxxxxx
export HINDSIGHT_API_LLM_MODEL=llama-3.3-70b-versatile

# Anthropic
export HINDSIGHT_API_LLM_PROVIDER=anthropic
export HINDSIGHT_API_LLM_API_KEY=sk-ant-xxxxxxxxxxxx
export HINDSIGHT_API_LLM_MODEL=claude-sonnet-4-20250514
```

## Daemon Management

### Idle Timeout

Customize how long the daemon stays alive when idle:

```bash
# Never timeout (daemon runs until manually stopped)
export HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT=0

# Shorter timeout: 1 minute
export HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT=60

# Longer timeout: 30 minutes
export HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT=1800
```

### Daemon Commands

```bash
# Check daemon status
hindsight-embed daemon status

# View daemon logs in real-time
hindsight-embed daemon logs -f

# Stop daemon manually
hindsight-embed daemon stop
```

### Control Center Commands

```bash
# Start or reuse the local browser control center
hindsight-embed control start

# Check whether it is running
hindsight-embed control status

# View control-center logs
hindsight-embed control logs -f

# Stop the control-center process
hindsight-embed control stop
```

## Commands

All memory operations follow the same interface as the CLI:

### Retain (Store Memory)

```bash
hindsight-embed memory retain <bank_id> "content"

# With context
hindsight-embed memory retain <bank_id> "content" --context "source information"

# Background processing
hindsight-embed memory retain <bank_id> "content" --async
```

### Recall (Search)

```bash
hindsight-embed memory recall <bank_id> "query"

# With budget control
hindsight-embed memory recall <bank_id> "query" --budget high

# Show trace
hindsight-embed memory recall <bank_id> "query" --trace
```

### Reflect (Generate Response)

```bash
hindsight-embed memory reflect <bank_id> "prompt"

# With additional context
hindsight-embed memory reflect <bank_id> "prompt" --context "additional info"
```

### Bank Management

```bash
# List all banks
hindsight-embed bank list

# View bank stats
hindsight-embed bank stats <bank_id>

# Set bank name
hindsight-embed bank name <bank_id> "My Assistant"

# Set bank mission
hindsight-embed bank mission <bank_id> "I am a helpful AI assistant"
```

## Troubleshooting

### Daemon Won't Start

Check the daemon logs:

```bash
hindsight-embed daemon logs
# Or watch in real-time
hindsight-embed daemon logs -f
```

Common issues:
- **Missing API key**: Set `HINDSIGHT_API_LLM_API_KEY`
- **Port conflict**: Another service using port 8888
- **Permissions**: Check `~/.hindsight/` directory permissions

### Daemon Exits Immediately

Check if you have the idle timeout set too low:

```bash
# Disable idle timeout for debugging
export HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT=0
hindsight-embed daemon status
```

### Reset Configuration

```bash
# Remove config file and reconfigure
rm ~/.hindsight/embed
hindsight-embed configure
```

## When to Use

**Perfect for:**
- Development and prototyping
- Single-user applications
- Local-first tools
- Quick experiments with Hindsight

**Not suitable for:**
- Production multi-user deployments
- Network-accessible services
- High-availability requirements
- Multi-tenant applications

For production deployments, use the [API Service](../developer/services.md) with external PostgreSQL instead.
