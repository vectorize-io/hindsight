# hindsight-embed

Hindsight embedded CLI - local memory operations with automatic daemon management.

This package provides a simple CLI for storing and recalling memories using Hindsight's memory engine. It automatically manages a background daemon for fast operations - no manual server setup required.

## How It Works

`hindsight-embed` uses a background daemon architecture for optimal performance:

1. **First command**: Automatically starts a local daemon (first run downloads dependencies and loads ML models - can take 1-3 minutes)
2. **Subsequent commands**: Near-instant responses (~1-2s) since daemon is already running
3. **Auto-shutdown**: Daemon automatically exits after 5 minutes of inactivity

The daemon runs on `localhost:8889` and uses an embedded PostgreSQL database (pg0) - everything stays local on your machine.

## Installation

```bash
pip install hindsight-embed
# or with uvx (no install needed)
uvx hindsight-embed --help
```

## Quick Start

```bash
# Interactive setup (recommended)
hindsight-embed configure

# Or set your LLM API key manually
export OPENAI_API_KEY=sk-...

# Store a memory (bank_id = "default")
hindsight-embed memory retain default "User prefers dark mode"

# Recall memories
hindsight-embed memory recall default "What are user preferences?"
```

## Commands

### configure

Interactive setup wizard:

```bash
hindsight-embed configure
```

This will:
- Let you choose an LLM provider (OpenAI, Groq, Google, Ollama)
- Configure your API key
- Set the model and memory bank ID
- Start the daemon with your configuration

### memory retain

Store a memory:

```bash
hindsight-embed memory retain default "User prefers dark mode"
hindsight-embed memory retain default "Meeting on Monday" --context work
hindsight-embed memory retain myproject "API uses JWT authentication"
```

### memory recall

Search memories:

```bash
hindsight-embed memory recall default "user preferences"
hindsight-embed memory recall default "upcoming events"
```

Use `-o json` for JSON output:
```bash
hindsight-embed memory recall default "user preferences" -o json
```

### memory reflect

Get contextual answers that synthesize multiple memories:

```bash
hindsight-embed memory reflect default "How should I set up the dev environment?"
```

### bank list

List all memory banks:

```bash
hindsight-embed bank list
```

### daemon

Manage the background daemon:

```bash
hindsight-embed daemon status    # Check if daemon is running
hindsight-embed daemon start     # Start the daemon
hindsight-embed daemon stop      # Stop the daemon
hindsight-embed daemon logs      # View last 50 lines of logs
hindsight-embed daemon logs -f   # Follow logs in real-time
hindsight-embed daemon logs -n 100  # View last 100 lines
```

## Configuration

### Interactive Setup

Run `hindsight-embed configure` for a guided setup that saves to `~/.hindsight/embed`.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_EMBED_LLM_API_KEY` | LLM API key (or use `OPENAI_API_KEY`) | Required |
| `HINDSIGHT_EMBED_LLM_PROVIDER` | LLM provider (`openai`, `groq`, `google`, `ollama`) | `openai` |
| `HINDSIGHT_EMBED_LLM_MODEL` | LLM model | `gpt-4o-mini` |
| `HINDSIGHT_EMBED_BANK_ID` | Default memory bank ID (optional, used when not specified in CLI) | `default` |

**Note:** All banks share a single pg0 database (`pg0://hindsight-embed`). Bank isolation happens within the database via the `bank_id` parameter passed to CLI commands.

### Files

| Path | Description |
|------|-------------|
| `~/.hindsight/embed` | Configuration file |
| `~/.hindsight/config.env` | Alternative config file location |
| `~/.hindsight/daemon.log` | Daemon logs |
| `~/.hindsight/daemon.lock` | Daemon lock file (PID) |

## Use with AI Coding Assistants

This CLI is designed to work with AI coding assistants like Claude Code, Cursor, and Windsurf. Install the Hindsight skill:

```bash
curl -fsSL https://hindsight.vectorize.io/get-skill | bash
```

This will configure the LLM provider and install the skill to your assistant's skills directory.

## Troubleshooting

**Daemon won't start:**
```bash
# Check logs for errors
hindsight-embed daemon logs

# Stop any stuck daemon and restart
hindsight-embed daemon stop
hindsight-embed daemon start
```

**Slow first command:**
This is expected - the first command needs to download dependencies, start the daemon, and load ML models. First run can take 1-3 minutes depending on network speed. Subsequent commands will be fast (~1-2s).

**Change configuration:**
```bash
# Re-run configure (automatically restarts daemon)
hindsight-embed configure
```

## License

Apache 2.0
