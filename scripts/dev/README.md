# Hindsight Development Scripts

Production-ready scripts for managing the Hindsight development environment.

## Quick Start

```bash
# Start everything (simplest)
./start-services.sh

# OR start with full control (monitoring, custom worker count)
./start-all.sh --monitoring --workers 8

# Check status
./status.sh --watch

# Stop everything
./stop-all.sh

# Stop everything including Docker monitoring
./stop-all.sh --monitoring

# Interactive dashboard
./dashboard.sh
```

## Production Scripts

### Core Management

#### `start-services.sh`
**Recommended** - Simple one-command startup for all services.

```bash
./start-services.sh
```

**What it does**:
1. Starts Ollama split lanes (11434, 11435)
2. Starts Hindsight API (port 8888) with health check
3. Starts Control Plane (port 9999) with health check
4. Starts 4 workers with unique IDs and ports
5. Shows final status and URLs

**Output**:
- All service URLs (API, Control Plane, Grafana, Workers)
- Log file locations
- Status monitoring command

**Note**: Services run in background. Use `./status.sh --watch` to monitor.

#### `start-all.sh`
Unified startup orchestration for the entire stack.

```bash
# Start everything with monitoring
./start-all.sh --monitoring --workers 4

# Start without monitoring
./start-all.sh --workers 4

# Start API only (no workers)
./start-all.sh --no-workers

# Use random port (avoid conflicts)
./start-all.sh --random-port

# Help
./start-all.sh --help
```

**What it does**:
1. Starts Ollama split lanes (11434, 11435)
2. (Optional) Starts monitoring stack (Grafana LGTM)
3. Starts Hindsight API (port 8888)
4. Starts Control Plane
5. Starts N workers with unique IDs and ports
6. Waits for health checks
7. Shows status dashboard
8. Graceful shutdown on Ctrl+C

#### `stop-all.sh`
**NEW** - Gracefully stop all Hindsight services.

```bash
# Stop all services (keep Docker monitoring running)
./stop-all.sh

# Stop everything including Docker monitoring
./stop-all.sh --monitoring

# Force kill stubborn processes
./stop-all.sh --force

# Help
./stop-all.sh --help
```

**What it does**:
1. Stops workers (graceful SIGTERM, 5s timeout)
2. Stops Control Plane (graceful SIGTERM, 10s timeout)
3. Stops Hindsight API (graceful SIGTERM, 10s timeout)
4. Stops Ollama LLM lane (port 11435)
5. (Optional) Stops Docker monitoring stack
6. Shows summary of stopped services
7. Lists any remaining processes

**Exit codes**:
- `0` - All services stopped successfully
- `1` - Some services failed to stop (use --force)

**Note**: Primary Ollama (port 11434) is kept running by default (it's the system default). Only the secondary LLM lane (11435) is stopped.

#### `status.sh`
Read-only status monitor (pure visibility, no actions).

```bash
# One-time status check
./status.sh

# Watch mode (auto-refresh every 5s)
./status.sh --watch

# Custom refresh interval
./status.sh --watch --interval 10

# Help
./status.sh --help
```

**Shows**:
- Service status (Ollama, API, workers, monitoring)
- Configuration from `.env`
- Recent errors (API and workers)
- Log file locations
- Management commands

#### `dashboard.sh`
Interactive management dashboard.

```bash
./dashboard.sh
```

**Features**:
- View API logs (live)
- View worker logs (live)
- View all errors
- Restart API
- Restart workers
- Scale workers
- Start/stop monitoring
- Start/stop Ollama LLM lane
- Refresh status

### Ollama Management

#### `start-ollama-split.sh`
Start and verify split Ollama lanes.

```bash
./start-ollama-split.sh
```

**What it does**:
- Verifies primary Ollama (port 11434) is running
- Auto-starts secondary Ollama (port 11435) if needed
- Verifies required models (llama3.2, nomic-embed-text)
- Shows status of both lanes with model counts

**Configuration**:
- Embeddings Lane: port 11434
- LLM Lane: port 11435
- Shared models: `/Volumes/Mac/Users/oliververmeulen/.ollama/models`

#### `stop-ollama-split.sh`
Stop LLM lane gracefully.

```bash
./stop-ollama-split.sh
```

**What it does**:
- Stops LLM lane (port 11435) with SIGTERM
- Leaves primary lane (port 11434) running
- Safe to run even if lane is already stopped

### Worker Management

#### `scale-workers.sh`
Worker lifecycle management.

```bash
# Start 4 workers
./scale-workers.sh 4

# Check status
./scale-workers.sh status

# Stop all workers
./scale-workers.sh stop

# Restart with different count
./scale-workers.sh restart 2
```

**What it does**:
- Starts N worker processes with unique IDs (worker-1, worker-2, ...)
- Each worker gets unique HTTP port for metrics (9001, 9002, ...)
- Logs written to `logs/worker-N.log`
- Tracks PIDs in `/tmp/hindsight-workers.state`
- Loads configuration from root `.env`

### Monitoring Management

#### `stop-monitoring.sh`
Stop Grafana LGTM Docker stack.

```bash
./stop-monitoring.sh
```

**What it does**:
- Stops Grafana LGTM Docker stack
- Removes containers (preserves volumes)

**Note**: To start monitoring, use `./start-all.sh --monitoring`

### Legacy Scripts (Still Available)

#### `start-api.sh`
Start API server directly.

```bash
./start-api.sh --port 8888
```

**Environment**: Loads `.env` from project root, uses `uv run hindsight-api`

#### `start-worker.sh`
Start a single worker directly.

```bash
./start-worker.sh
```

**Environment**: Loads `.env` from project root, uses `uv run hindsight-worker`

#### `start.sh`
Legacy startup script (use `start-all.sh` instead).

#### `start-monitoring.sh`
Legacy monitoring startup (use `start-all.sh --monitoring` instead).

## Archived Scripts

### `archive/test-integration.sh`
Integration test suite (archived after successful testing).

**Tests**:
- Ollama split lanes (11434, 11435)
- Worker scaling (start, verify, stop)
- Monitoring stack (start, verify, stop)
- Configuration verification

**Note**: All tests passed on 2026-06-26. Archived to keep production directory clean.

## Environment Isolation

**CRITICAL**: All scripts properly isolate environments:

1. **API and Workers**: Use `start-api.sh` and `start-worker.sh` which:
   - Load `.env` from project root
   - Use `uv run` to ensure correct Python environment
   - Export all environment variables before execution

2. **Never use raw `uv run` commands**: Always delegate to the proper scripts

3. **R&D Mode**: `.env` contains real credentials, NEVER modify programmatically

## File Locations

### Configuration
- Main config: `/Users/oliververmeulen/hindsight/.env`
- Monitoring: `scripts/dev/monitoring/docker-compose.yaml`
- Dashboards: `monitoring/grafana/dashboards/`

### Logs
- API: `/tmp/hindsight-api.log`
- Workers: `logs/worker-N.log`
- Monitoring: `docker logs hindsight-monitoring`

### State
- Worker state: `/tmp/hindsight-workers.state`
- Ollama models: `/Volumes/Mac/Users/oliververmeulen/.ollama/models`

## Common Workflows

### Development Session

```bash
# 1. Start everything
./start-all.sh --monitoring --workers 4

# 2. Monitor in another terminal
./status.sh --watch

# 3. Work on code (hot-reload enabled)

# 4. Check logs if needed
tail -f /tmp/hindsight-api.log
tail -f ../logs/worker-1.log

# 5. View traces in Grafana
open http://localhost:3000
```

### Debugging Issues

```bash
# 1. Check status
./status.sh

# 2. Review errors
./dashboard.sh
# Select option [3] View all errors

# 3. Check specific logs
tail -f /tmp/hindsight-api.log | grep -i error

# 4. Restart if needed
./dashboard.sh
# Select option [4] Restart API
```

### Scaling Workers

```bash
# Check current workers
./scale-workers.sh status

# Scale up
./scale-workers.sh 8

# Scale down
./scale-workers.sh 2

# Stop all
./scale-workers.sh stop
```

### Monitoring Stack

```bash
# Start with monitoring
./start-all.sh --monitoring --workers 4

# Access Grafana
open http://localhost:3000

# Stop monitoring when done
./stop-monitoring.sh
```

## Troubleshooting

### Port Conflicts

```bash
# Check what's using a port
lsof -i :8888
lsof -i :11434
lsof -i :11435

# Use random port for API
./start-all.sh --random-port
```

### Workers Not Starting

```bash
# Check worker state
cat /tmp/hindsight-workers.state

# Clean state and restart
rm /tmp/hindsight-workers.state
./scale-workers.sh 4
```

### Ollama Lane Issues

```bash
# Check both lanes
curl http://localhost:11434/api/tags
curl http://localhost:11435/api/tags

# Restart LLM lane
./stop-ollama-split.sh
./start-ollama-split.sh
```

### Monitoring Not Accessible

```bash
# Check container
docker ps --filter "name=hindsight-monitoring"

# Check logs
docker logs hindsight-monitoring

# Restart monitoring
./stop-monitoring.sh
./start-all.sh --monitoring
```

## Best Practices

1. **Always use `start-all.sh`** for unified startup
2. **Use `status.sh --watch`** for continuous monitoring
3. **Check status before actions** to avoid conflicts
4. **Review logs regularly** during development
5. **Use monitoring stack** for observability
6. **Never modify `.env` programmatically** (R&D mode)
7. **Archive test scripts** after successful testing

## Documentation

- **OLLAMA_SPLIT_SETUP.md** - Complete split Ollama guide
- **MONITORING_GUIDE.md** - Grafana, tracing, metrics, logs
- **AGENTS.md** - Critical information for AI agents
- **scripts/dev/monitoring/README.md** - Monitoring stack details

## Quick Reference

| Task | Command |
|------|---------|
| Start everything | `./start-all.sh --monitoring --workers 4` |
| Check status | `./status.sh --watch` |
| Interactive dashboard | `./dashboard.sh` |
| Scale workers | `./scale-workers.sh 4` |
| Stop workers | `./scale-workers.sh stop` |
| Stop monitoring | `./stop-monitoring.sh` |
| View API logs | `tail -f /tmp/hindsight-api.log` |
| View worker logs | `tail -f ../logs/worker-*.log` |
| Access Grafana | `open http://localhost:3000` |

---

**Last Updated**: 2026-06-26
**Status**: Production Ready ✅
