# Ollama Split Lane Configuration

## Overview

Hindsight is configured to use **split Ollama execution lanes** to prevent ReadTimeout and connection errors when both LLM extraction and embeddings share the same endpoint.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Hindsight API                        │
│                                                         │
│  ┌──────────────────┐         ┌──────────────────┐    │
│  │  LLM Extraction  │         │   Embeddings     │    │
│  │   (Reasoning)    │         │   (Vectors)      │    │
│  └────────┬─────────┘         └────────┬─────────┘    │
│           │                            │              │
└───────────┼────────────────────────────┼──────────────┘
            │                            │
            ▼                            ▼
   ┌────────────────┐          ┌────────────────┐
   │ Ollama :11435  │          │ Ollama :11434  │
   │ (LLM Lane)     │          │ (Embed Lane)   │
   │                │          │                │
   │ llama3.2       │          │ nomic-embed    │
   │ gemma3:12b     │          │ text           │
   │ + 20 others    │          │ + 20 others    │
   └────────────────┘          └────────────────┘
         ▲                            ▲
         └────────────────┬───────────┘
                          │
                  Shared Model Store
         /Volumes/Mac/Users/oliververmeulen/.ollama/models
```

## Configuration

**Location**: `/Users/oliververmeulen/hindsight/.env`

```bash
# LLM Extraction Lane (reasoning, generation)
HINDSIGHT_API_LLM_OLLAMA_API_BASE=http://localhost:11435
HINDSIGHT_API_LLM_OLLAMA_MODEL=ollama/llama3.2:latest

# Embeddings Lane (vector generation)
HINDSIGHT_API_EMBEDDINGS_OLLAMA_API_BASE=http://localhost:11434
HINDSIGHT_API_EMBEDDINGS_OLLAMA_MODEL=nomic-embed-text

# Worker Configuration
HINDSIGHT_API_WORKERS=4
```

## Available Scripts

### 1. Start Split Ollama Lanes

```bash
./scripts/dev/start-ollama-split.sh
```

**What it does**:
- Verifies primary Ollama (port 11434) is running
- Auto-starts secondary Ollama (port 11435) if not running
- Verifies required models exist (llama3.2, nomic-embed-text)
- Shows status of both lanes with model counts

**Output**:
```
✅ Primary Ollama (embeddings) running on :11434 (22 models)
✅ LLM Ollama lane running on :11435 (22 models)
✅ Required models verified
```

### 2. Stop LLM Lane

```bash
./scripts/dev/stop-ollama-split.sh
```

**What it does**:
- Gracefully stops the LLM lane (port 11435) with SIGTERM
- Leaves primary lane (port 11434) running
- Safe to run even if lane is already stopped

### 3. Scale Workers

```bash
# Start 4 workers
./scripts/dev/scale-workers.sh 4

# Check status
./scripts/dev/scale-workers.sh status

# Stop all workers
./scripts/dev/scale-workers.sh stop

# Restart with different count
./scripts/dev/scale-workers.sh restart 2
```

**What it does**:
- Starts N worker processes with unique IDs (worker-1, worker-2, ...)
- Each worker gets unique HTTP port for metrics (9001, 9002, ...)
- Logs written to `logs/worker-N.log`
- Tracks PIDs in `/tmp/hindsight-workers.state`
- Loads configuration from root `.env`

### 4. Start Full Stack

```bash
# Start everything (Ollama + API + Control Plane + 4 workers)
./scripts/dev/start-all.sh

# With monitoring stack
./scripts/dev/start-all.sh --monitoring

# Custom worker count
./scripts/dev/start-all.sh --workers 2

# No workers (API only)
./scripts/dev/start-all.sh --no-workers

# Random API port (avoid conflicts)
./scripts/dev/start-all.sh --random-port

# Help
./scripts/dev/start-all.sh --help
```

**What it does**:
- Orchestrates startup in correct order:
  1. Ollama split lanes
  2. (Optional) Monitoring stack (Grafana LGTM)
  3. API server
  4. Control Plane
  5. Workers (configurable count)
- Waits for API health check before proceeding
- Shows comprehensive status dashboard
- Graceful shutdown on Ctrl+C (cleanup function)

### 5. Stop Monitoring

```bash
./scripts/dev/stop-monitoring.sh
```

**What it does**:
- Stops Grafana LGTM Docker stack
- Removes containers (preserves volumes)

### 6. Integration Test

```bash
./scripts/dev/test-integration.sh
```

**What it does**:
- Tests all components independently:
  - Ollama split lanes (11434, 11435)
  - Worker scaling (start, verify, stop)
  - Monitoring stack (start, verify, stop)
  - Configuration verification
- Reports pass/fail for each test suite

## Deployment Strategy

### Current (Development)

- **Core Application**: Bare metal (API, workers, Ollama)
  - Reason: Hot-reload during development, avoid container rebuild cycles
- **Monitoring/Tracing**: Docker (optional, can toggle on/off)
  - Grafana LGTM stack (Grafana + Loki + Tempo + Mimir)

### Future Phases

- **Phase 2**: External LLM APIs (OpenAI, Anthropic, Gemini, Groq)
- **Phase 3**: Cloud deployment (Google Cloud, AWS)

## Troubleshooting

### Port Conflicts

If you see "Address already in use" errors:

```bash
# Check what's using the port
lsof -i :11434
lsof -i :11435
lsof -i :8000

# Use random port for API
./scripts/dev/start-all.sh --random-port
```

### Missing Models

If models are missing:

```bash
# Pull required models on primary lane
ollama pull llama3.2:latest
ollama pull nomic-embed-text

# Verify models exist
ollama list
```

The secondary lane (11435) shares the same model directory, so models only need to be pulled once.

### Worker Crashes

Check worker logs:

```bash
tail -f logs/worker-1.log
tail -f logs/worker-2.log
```

### Monitoring Stack Issues

```bash
# Check Docker status
docker ps

# View logs
docker logs hindsight-monitoring

# Restart monitoring
./scripts/dev/stop-monitoring.sh
./scripts/dev/start-all.sh --monitoring
```

## Model Inventory

Both Ollama lanes share the same model directory and have access to:

- llama3.2:latest (default LLM)
- nomic-embed-text (default embeddings)
- gemma3:12b
- llama3.1:latest
- llama3.3:latest
- qwen2.5:14b
- qwen2.5:32b
- qwen2.5:7b
- qwen2.5-coder:14b
- qwen2.5-coder:32b
- qwen2.5-coder:7b
- deepseek-r1:14b
- deepseek-r1:32b
- deepseek-r1:70b
- deepseek-r1:7b
- deepseek-r1:8b
- gemma2:27b
- gemma2:9b
- llama3:latest
- mistral:latest
- phi4:latest
- qwq:latest

## Performance Notes

- **Split lanes prevent timeout errors** by isolating LLM extraction (heavy, long-running) from embeddings (fast, frequent)
- **Worker scaling** allows parallel processing of extraction tasks
- **Shared model directory** avoids duplicating 100+ GB of model files
- **Monitoring stack** (optional) adds ~1GB RAM overhead but provides observability

## Maintenance

### R&D Mode

⚠️ **IMPORTANT**: The `.env` file contains real credentials and is configured for R&D mode.

- **DO NOT** modify `.env` programmatically
- **DO NOT** use placeholder/dummy credentials
- **DO NOT** commit `.env` to version control

### Updating Configuration

To change Ollama endpoints or models:

1. Manually edit `/Users/oliververmeulen/hindsight/.env`
2. Restart services: `./scripts/dev/start-all.sh`

### Cleaning Up

```bash
# Stop all workers
./scripts/dev/scale-workers.sh stop

# Stop LLM lane
./scripts/dev/stop-ollama-split.sh

# Stop monitoring
./scripts/dev/stop-monitoring.sh

# Clean worker logs
rm -rf logs/worker-*.log
```

## Quick Reference

| Task | Command |
|------|---------|
| Start everything | `./scripts/dev/start-all.sh` |
| Start with monitoring | `./scripts/dev/start-all.sh --monitoring` |
| Start 2 workers | `./scripts/dev/start-all.sh --workers 2` |
| Check worker status | `./scripts/dev/scale-workers.sh status` |
| Stop workers | `./scripts/dev/scale-workers.sh stop` |
| Test integration | `./scripts/dev/test-integration.sh` |
| View worker logs | `tail -f logs/worker-1.log` |

## Related Documentation

- [CLAUDE.md](./CLAUDE.md) - Project documentation and coding conventions
- [scripts/dev/README.md](./scripts/dev/README.md) - Development scripts overview
- [monitoring/README.md](./monitoring/README.md) - Monitoring stack setup

## Control Plane Access

The Hindsight Control Plane UI requires an access key for authentication.

**URL**: http://localhost:9999

**Configuration** (edit `.env` manually - R&D mode):

```bash
# Control Plane Access
HINDSIGHT_CP_ACCESS_KEY=your-secret-key-here  # Choose any secret for local development
```

**After adding the access key**:

```bash
# Restart control plane to pick up new config
pkill -f "next dev"
./scripts/dev/start-services.sh
```

**Login**:
1. Navigate to http://localhost:9999
2. Enter your `HINDSIGHT_CP_ACCESS_KEY` value
3. Session is valid for 24 hours

**Note**: Do NOT commit your access key to version control. The `.env` file is already in `.gitignore`.

