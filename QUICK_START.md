# Hindsight Quick Start Guide

**Last Updated**: 2026-06-26

## TL;DR - Just Get It Running

```bash
# Start everything
./scripts/dev/start-services.sh

# Check status
./scripts/dev/status.sh

# Access services
open http://localhost:9999  # Control Plane (key: your-collabminds-access-key)
open http://localhost:8888/health  # API health check
```

## Complete Workflow

### 1. Start All Services

```bash
cd /Users/oliververmeulen/hindsight
./scripts/dev/start-services.sh
```

**What starts**:
- ✅ Ollama Embeddings Lane (port 11434)
- ✅ Ollama LLM Lane (port 11435)
- ✅ Hindsight API (port 8888)
- ✅ Control Plane UI (port 9999)
- ✅ 4 Workers (ports 9001-9004)

**Startup time**: ~30 seconds (workers load ML models)

---

### 2. Verify Everything is Running

```bash
./scripts/dev/status.sh
```

**Expected output**:
```
✓ Ollama Embeddings (port 11434) - 22 models
✓ Ollama LLM (port 11435) - 22 models
✓ Hindsight API (port 8888) - RUNNING
✓ Workers - 4 healthy
```

**Live monitoring**:
```bash
./scripts/dev/status.sh --watch
```

---

### 3. Access Services

#### Control Plane UI
- **URL**: http://localhost:9999
- **Access Key**: `your-collabminds-access-key` (from `.env`)
- **Use**: Manage tenants, banks, configuration

#### API Endpoints
- **Health**: http://localhost:8888/health
- **Docs**: http://localhost:8888/docs
- **Metrics**: http://localhost:8888/metrics

#### Grafana Monitoring (Optional)
- **URL**: http://localhost:3000
- **Login**: admin / admin
- **Dashboards**: Hindsight Operations, LLM, API Service

---

### 4. Stop All Services

```bash
# Stop everything (keep Docker monitoring)
./scripts/dev/stop-all.sh

# Stop everything including Docker monitoring
./scripts/dev/stop-all.sh --monitoring
```

**Graceful shutdown**:
- Workers: 5 second timeout
- API/Control Plane: 10 second timeout
- Ollama LLM lane: 10 second timeout

**Force kill stubborn processes**:
```bash
./scripts/dev/stop-all.sh --force
```

---

## Common Tasks

### Scale Workers

```bash
# Increase to 8 workers
./scripts/dev/scale-workers.sh 8

# Decrease to 2 workers
./scripts/dev/scale-workers.sh 2

# Stop all workers
./scripts/dev/scale-workers.sh stop

# Check worker status
./scripts/dev/scale-workers.sh status
```

### View Logs

```bash
# API logs
tail -f /tmp/hindsight-api.log

# Control Plane logs
tail -f /tmp/hindsight-control-plane.log

# Worker logs
tail -f logs/worker-1.log

# All worker logs
tail -f logs/worker-*.log

# Ollama LLM lane logs
tail -f /tmp/ollama-llm-lane.log
```

### Restart Services

```bash
# Restart API
./scripts/dev/stop-all.sh
./scripts/dev/start-services.sh

# Restart just workers
./scripts/dev/scale-workers.sh 4
```

### Start with Monitoring

```bash
# Start with Grafana LGTM stack
./scripts/dev/start-all.sh --monitoring --workers 4

# Access Grafana
open http://localhost:3000
```

---

## Troubleshooting

### Workers show as "stopped" but processes are running

**Cause**: Workers are still loading ML models (takes 10-15 seconds).

**Solution**: Wait a bit, then check again:
```bash
sleep 15 && ./scripts/dev/status.sh
```

### API won't start - port 8888 already in use

**Cause**: Previous instance still running.

**Solution**:
```bash
# Kill orphaned processes
./scripts/dev/stop-all.sh --force

# Start fresh
./scripts/dev/start-services.sh
```

### Docker Desktop keeps crashing

**Cause**: Monitoring container volume mounts or resource limits.

**Solution**: Skip Docker monitoring (core Hindsight works without it):
```bash
# Start without monitoring
./scripts/dev/start-services.sh

# If monitoring crashes, stop it
./scripts/dev/stop-all.sh --monitoring
```

### Workers using wrong Ollama endpoint

**Cause**: Database bank configuration overrides `.env` settings.

**Impact**: This is NORMAL for multi-tenant setups. Different tenants can use different models.

**Verify split lanes are working**:
```bash
# Check both Ollama instances are running
lsof -i :11434 -i :11435 | grep LISTEN

# Should show TWO ollama processes
```

### Control Plane asks for access key

**Key location**: `.env` file, line 247:
```bash
HINDSIGHT_CP_ACCESS_KEY=your-collabminds-access-key
```

**To change**:
1. Edit `.env` line 247
2. Restart Control Plane:
   ```bash
   pkill -f "next dev"
   ./scripts/dev/start-services.sh
   ```

---

## Configuration Files

### Environment Variables (`.env`)

**Location**: `/Users/oliververmeulen/hindsight/.env`

**Critical settings**:
```bash
# Split Ollama lanes (DON'T CHANGE - prevents ReadTimeout errors)
HINDSIGHT_API_LLM_OLLAMA_API_BASE=http://localhost:11435      # LLM lane
HINDSIGHT_API_EMBEDDINGS_OLLAMA_API_BASE=http://localhost:11434  # Embeddings lane

# Worker count
HINDSIGHT_API_WORKERS=4

# Control Plane access
HINDSIGHT_CP_ACCESS_KEY=your-collabminds-access-key
```

**Rule**: NEVER modify `.env` without documenting WHY you changed it (see AGENTS.md).

### Worker State

**Location**: `/tmp/hindsight-workers.state`

**Format**:
```
worker-1|<PID>|9001
worker-2|<PID>|9002
worker-3|<PID>|9003
worker-4|<PID>|9004
```

**Note**: Auto-managed by scripts. Don't delete while workers are running.

---

## Port Reference

| Service | Port | Purpose |
|---------|------|---------|
| Ollama (Embeddings) | 11434 | Fast vector generation (nomic-embed-text) |
| Ollama (LLM) | 11435 | Heavy extraction/reasoning (llama3.2, gemma3) |
| Hindsight API | 8888 | Main API server |
| Control Plane | 9999 | Management UI |
| Worker-1 | 9001 | Async task processor + metrics |
| Worker-2 | 9002 | Async task processor + metrics |
| Worker-3 | 9003 | Async task processor + metrics |
| Worker-4 | 9004 | Async task processor + metrics |
| Grafana | 3000 | Monitoring UI (Docker, optional) |
| Tempo | 3200 | Distributed tracing (Docker, optional) |
| OTLP gRPC | 4317 | Trace ingestion (Docker, optional) |
| OTLP HTTP | 4318 | Trace ingestion (Docker, optional) |
| Pyroscope | 4040 | Continuous profiling (Docker, optional) |
| Prometheus | 9090 | Metrics storage (Docker, optional) |

---

## Service Dependencies

```
PostgreSQL (pg0:5433)
    ↑
    |
Ollama (11434, 11435)
    ↑
    |
Hindsight API (8888)
    ↑
    |
    ├── Control Plane (9999)
    └── Workers (9001-9004)
```

**Startup order**:
1. PostgreSQL (auto-started by pg0)
2. Ollama lanes (11434, 11435)
3. API (waits for DB + Ollama)
4. Control Plane (waits for API)
5. Workers (connect to DB + Ollama)

**Shutdown order** (reverse):
1. Workers
2. Control Plane
3. API
4. Ollama LLM lane (11435)
5. Ollama Embeddings lane (11434) - kept running

---

## What's Running Where

### Bare Metal (Hot-Reload Development)
- Ollama (both lanes)
- Hindsight API
- Control Plane
- Workers
- PostgreSQL (pg0)

### Docker (Optional Monitoring)
- Grafana LGTM stack
- Tempo (traces)
- Loki (logs)
- Prometheus (metrics)
- Pyroscope (profiling)

**Why split**: Core services restart fast for development. Monitoring is isolated (failures don't break core functionality).

---

## Next Steps

### Enable Distributed Tracing

Uncomment in `.env`:
```bash
HINDSIGHT_API_OTEL_TRACES_ENABLED=true
HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
HINDSIGHT_API_OTEL_SERVICE_NAME=hindsight-dev
```

Restart services:
```bash
./scripts/dev/stop-all.sh
./scripts/dev/start-all.sh --monitoring
```

Access traces:
- Grafana: http://localhost:3000/explore → Tempo
- Query: `{service.name="hindsight-dev"}`

### Collect Dataset from Traces

See `TRACING_AND_DATASETS.md` for complete guide.

### Understand WHY Things Work

See `ARCHITECTURE_WHY.md` for deep dive into:
- WHY split Ollama lanes prevent ReadTimeout errors
- WHY workers failed (and how we fixed it)
- WHY Docker keeps stopping (and how to prevent it)
- Configuration hierarchy (env → bank → runtime)
- How to make it better (Phases 2-6)

---

## Help & Documentation

- **Full Architecture**: `ARCHITECTURE_WHY.md`
- **Monitoring Setup**: `MONITORING_GUIDE.md`
- **Ollama Split Setup**: `OLLAMA_SPLIT_SETUP.md`
- **Tracing & Datasets**: `TRACING_AND_DATASETS.md`
- **Services Dashboard**: `SERVICES_DASHBOARD.md`
- **Scripts Reference**: `scripts/dev/README.md`
- **Agent Instructions**: `AGENTS.md`

---

**Questions?** Check the troubleshooting section in `ARCHITECTURE_WHY.md` or run:
```bash
./scripts/dev/dashboard.sh  # Interactive help menu
```
