# hindsight-embed

Hindsight embedded CLI - local memory operations with automatic daemon management.

This package provides a simple CLI for storing and recalling memories using Hindsight's memory engine. It automatically manages a background daemon for fast operations - no manual server setup required.

## How It Works

`hindsight-embed` uses a background daemon architecture for optimal performance:

1. **First command**: Automatically starts a local daemon (first run downloads dependencies and loads ML models - can take 1-3 minutes)
2. **Subsequent commands**: Near-instant responses (~1-2s) since daemon is already running
3. **Auto-shutdown**: Daemon automatically exits after 5 minutes of inactivity

The daemon runs on `localhost:8888` and uses an embedded PostgreSQL database (pg0) - everything stays local on your machine.

## Installation

```bash
pip install hindsight-embed
# or with uvx (no install needed)
uvx hindsight-embed --help
```

## Quick Start

```bash
# Interactive setup (configures default profile)
hindsight-embed configure

# Or set your LLM API key manually
export OPENAI_API_KEY=sk-...

# Store a memory (bank_id = "default")
hindsight-embed memory retain default "User prefers dark mode"

# Recall memories
hindsight-embed memory recall default "What are user preferences?"
```

All commands use the "default" profile unless you specify a different one with `--profile` or `HINDSIGHT_EMBED_PROFILE`.

## Commands

### configure

Configure the default profile or create/update named profiles:

```bash
# Interactive setup for default profile
hindsight-embed configure

# Create/update named profile with single command
hindsight-embed configure --profile my-app \
  --env HINDSIGHT_EMBED_LLM_PROVIDER=openai \
  --env HINDSIGHT_EMBED_LLM_API_KEY=sk-xxx

# Create/update named profile interactively
hindsight-embed configure --profile staging
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

### profile

Manage configuration profiles:

```bash
# List all profiles with status
hindsight-embed profile list

# Show current active profile
hindsight-embed profile show

# Set active profile (persists across commands)
hindsight-embed profile set-active my-app

# Clear active profile (revert to default)
hindsight-embed profile set-active --none

# Delete a profile
hindsight-embed profile delete my-app
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
| `HINDSIGHT_EMBED_PROFILE` | Profile name to use (overrides active profile) | None (uses default profile) |
| `HINDSIGHT_EMBED_LLM_API_KEY` | LLM API key (or use `OPENAI_API_KEY`) | Required |
| `HINDSIGHT_EMBED_LLM_PROVIDER` | LLM provider (`openai`, `groq`, `google`, `ollama`) | `openai` |
| `HINDSIGHT_EMBED_LLM_MODEL` | LLM model | `gpt-4o-mini` |
| `HINDSIGHT_EMBED_BANK_ID` | Default memory bank ID (optional, used when not specified in CLI) | `default` |
| `HINDSIGHT_EMBED_API_URL` | Use external API server instead of starting local daemon | None (starts local daemon) |
| `HINDSIGHT_EMBED_API_TOKEN` | Authentication token for external API (sent as Bearer token) | None |
| `HINDSIGHT_EMBED_API_DATABASE_URL` | Database URL for daemon | `pg0://hindsight-embed` |
| `HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT` | Seconds before daemon auto-exits when idle | `300` |

**Using an External API Server:**

To connect to an existing Hindsight API server instead of starting the local daemon:

```bash
export HINDSIGHT_EMBED_API_URL=http://your-server:8000
export HINDSIGHT_EMBED_API_TOKEN=your-api-token  # Optional, if API requires auth
hindsight-embed memory recall default "query"
```

**Custom Database:**

To use an external PostgreSQL database instead of the embedded pg0 database (useful when running as root or in containerized environments):

```bash
export HINDSIGHT_EMBED_API_DATABASE_URL=postgresql://user:password@localhost:5432/dbname
hindsight-embed daemon start
```

**Note:** All banks share a single database. Bank isolation happens within the database via the `bank_id` parameter passed to CLI commands.

### Configuration Profiles

Profiles let you maintain multiple independent configurations (e.g., different API endpoints, LLM providers, or projects). Each profile runs its own daemon on a unique port (8889-9888).

**The Default Profile:**

When you run `hindsight-embed configure` without specifying a profile, it configures the "default" profile. This uses the backward-compatible configuration at `~/.hindsight/embed` and runs on port 8888.

**Creating Named Profiles:**

```bash
# Create a profile with single command
hindsight-embed configure --profile my-app \
  --env HINDSIGHT_EMBED_LLM_PROVIDER=openai \
  --env HINDSIGHT_EMBED_LLM_API_KEY=sk-xxx \
  --env HINDSIGHT_EMBED_LLM_MODEL=gpt-4o-mini

# Create a profile interactively
hindsight-embed configure --profile staging
```

**Using Profiles:**

```bash
# Option 1: Environment variable (recommended for apps)
HINDSIGHT_EMBED_PROFILE=my-app hindsight-embed memory retain default "text"

# Option 2: CLI flag
hindsight-embed --profile my-app memory recall default "query"

# Option 3: Set as active (persists across commands)
hindsight-embed profile set-active my-app
hindsight-embed memory recall default "query"  # Uses my-app profile

# Clear active profile (revert to default)
hindsight-embed profile set-active --none
```

**Profile Management:**

```bash
# List all profiles with status
hindsight-embed profile list

# Show active profile
hindsight-embed profile show

# Delete a profile
hindsight-embed profile delete my-app
```

**Profile Resolution Priority:**
1. `HINDSIGHT_EMBED_PROFILE` environment variable (highest)
2. `--profile` CLI flag
3. Active profile from `~/.hindsight/active_profile` file
4. Default profile (lowest)

**Note:** If a profile is specified but doesn't exist, the command will fail with an error. Profiles must be explicitly created using `hindsight-embed configure --profile <name>`.

### Files

**Default Profile:**
| Path | Description |
|------|-------------|
| `~/.hindsight/embed` | Configuration file for default profile |
| `~/.hindsight/daemon.log` | Daemon logs for default profile |
| `~/.hindsight/daemon.lock` | Daemon lock file (PID) for default profile |

**Named Profiles:**
| Path | Description |
|------|-------------|
| `~/.hindsight/profiles/<name>.env` | Configuration file for profile |
| `~/.hindsight/profiles/<name>.log` | Daemon logs for profile |
| `~/.hindsight/profiles/<name>.lock` | Daemon lock file (PID) for profile |
| `~/.hindsight/profiles/metadata.json` | Profile metadata (ports, timestamps) |
| `~/.hindsight/active_profile` | Active profile name (when set with `profile set-active`) |

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
