# Hindsight Docker Setup

Hindsight is an agent memory system that gives AI agents persistent memory using biomimetic data structures.

---

## CRITICAL: Data Persistence

**All memory data is stored in the `hindsight-data` Docker volume.** This volume contains the PostgreSQL database with all memories, banks, and profiles.

### When Rebuilding the Docker Image

```bash
# Safe rebuild process:
docker-compose down                                                      # Stop container (volume preserved)
docker build -f docker/standalone/Dockerfile -t hindsight-standalone .   # Rebuild image
docker-compose up -d                                                     # Start with same volume
```

**DO NOT run these commands** (they will delete all data):
```bash
# DANGEROUS - destroys all memories:
docker volume rm hindsight-data
docker-compose down -v
```

### Verify Volume Exists

```bash
docker volume inspect hindsight-data
```

If the volume doesn't exist, create it before starting:
```bash
docker volume create hindsight-data
```

---

## Quick Start

```bash
# 1. Create persistent volume (first time only)
docker volume create hindsight-data

# 2. Copy and configure
cp docker-compose.example.yml docker-compose.yml
# Edit docker-compose.yml with your API keys

# 3. Start
docker-compose up -d

# 4. Verify
curl http://localhost:8888/health
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HINDSIGHT_API_LLM_PROVIDER` | Yes | LLM provider: `groq`, `openai`, `gemini`, `ollama` |
| `HINDSIGHT_API_LLM_MODEL` | Yes | Model name (e.g., `openai/gpt-oss-120b`) |
| `HINDSIGHT_API_LLM_API_KEY` | Yes | API key for the LLM provider |
| `HINDSIGHT_API_ANSWER_LLM_PROVIDER` | No | Separate provider for reflect operations |
| `HINDSIGHT_API_ANSWER_LLM_MODEL` | No | Separate model for reflect operations |
| `HINDSIGHT_API_ANSWER_LLM_API_KEY` | No | API key for answer LLM |
| `HINDSIGHT_MCP_BANK_ID` | No | Default memory bank ID (default: `default`) |
| `HINDSIGHT_API_MCP_ENABLED` | No | Enable MCP server (default: `true`) |

### Split LLM Configuration (Recommended)

Use a fast model for processing (retain/recall) and a powerful model for reflection:

```yaml
environment:
  # Fast processing (Groq)
  - HINDSIGHT_API_LLM_PROVIDER=groq
  - HINDSIGHT_API_LLM_MODEL=openai/gpt-oss-120b
  - HINDSIGHT_API_LLM_API_KEY=${GROQ_API_KEY}

  # Thoughtful reflection (Gemini)
  - HINDSIGHT_API_ANSWER_LLM_PROVIDER=gemini
  - HINDSIGHT_API_ANSWER_LLM_MODEL=gemini-3-pro-latest
  - HINDSIGHT_API_ANSWER_LLM_API_KEY=${GEMINI_API_KEY}
```

## MCP Tools

Once running, Hindsight exposes these MCP tools:

| Tool | Purpose |
|------|---------|
| `retain(content, context, bank_id?)` | Store a memory |
| `recall(query, max_results?, bank_id?)` | Search memories |
| `reflect(query, context?, budget?, bank_id?)` | Thoughtful analysis using memories |
| `list_banks()` | List all memory banks |
| `create_bank(bank_id, name?, background?)` | Create a new memory bank |

### Claude Code Integration

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "hindsight": {
      "type": "http",
      "url": "http://localhost:8888/mcp"
    }
  }
}
```

Or with a specific bank:

```json
{
  "mcpServers": {
    "hindsight": {
      "type": "http",
      "url": "http://localhost:8888/mcp",
      "headers": {
        "X-Bank-Id": "my-project-bank"
      }
    }
  }
}
```

## Multi-Bank Usage

Hindsight supports multiple isolated memory banks. Each bank has its own memories and personality.

### Creating Banks

```bash
# Via API
curl -X PUT "http://localhost:8888/v1/default/banks/my-new-bank" \
  -H "Content-Type: application/json" \
  -d '{"name": "My Project Memory", "background": "I am an AI working on..."}'
```

Or via MCP:
```python
create_bank(bank_id="my-new-bank", name="My Project Memory")
```

### Cross-Bank Operations

All tools accept an optional `bank_id` parameter:

```python
# Store in specific bank
retain(content="Important lesson", context="patterns", bank_id="orchestrator-memory")

# Search specific bank
recall(query="lessons learned", bank_id="orchestrator-memory")

# Reflect using specific bank's personality
reflect(query="What patterns have worked?", bank_id="orchestrator-memory")
```

### Setting Bank Background/Personality

```bash
curl -X POST "http://localhost:8888/v1/default/banks/{bank_id}/background" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "I am a Claude Code agent working on...",
    "update_disposition": true
  }'
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `http://localhost:8888` | REST API |
| `http://localhost:8888/mcp` | MCP Server |
| `http://localhost:9999` | Control Plane UI |
| `http://localhost:8888/health` | Health check |

## Common Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Check status
docker-compose ps
```

## Data Persistence

All data is stored in the `hindsight-data` Docker volume. This persists across container restarts and rebuilds.

```bash
# Backup volume
docker run --rm -v hindsight-data:/data -v $(pwd):/backup alpine \
  tar czf /backup/hindsight-backup.tar.gz -C /data .

# Restore volume
docker run --rm -v hindsight-data:/data -v $(pwd):/backup alpine \
  tar xzf /backup/hindsight-backup.tar.gz -C /data
```

## Troubleshooting

### Container won't start
```bash
docker-compose logs hindsight-mcp
```

### MCP not connecting
1. Check container is running: `docker-compose ps`
2. Verify health: `curl http://localhost:8888/health`
3. Check MCP endpoint: `curl http://localhost:8888/mcp`

### Reset everything
```bash
docker-compose down
docker volume rm hindsight-data
docker volume create hindsight-data
docker-compose up -d
```
