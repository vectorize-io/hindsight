# Hindsight Quick Start Guide

## 🚀 Essential Commands

### Start All Services
```bash
cd /Users/oliververmeulen/hindsight
./scripts/dev/start-services.sh
```

**What it does:**
1. Kills zombie processes
2. Starts Ollama split lanes (11434 embeddings, 11435 LLM)
3. Starts Hindsight API (8888)
4. Starts Control Plane UI (9998)
5. Starts workers (count from .env)

### Stop All Services
```bash
./scripts/dev/stop-all.sh
```

**What it does:**
- Graceful shutdown (SIGTERM → SIGINT → SIGKILL)
- Port-based cleanup
- Worker state cleanup
- Optional Ollama stopping

### Check Status
```bash
./scripts/dev/status.sh --watch
```

### View Logs
```bash
# All logs
tail -f logs/*.log

# Specific service
tail -f logs/hindsight-api.log
tail -f logs/hindsight-control-plane.log
tail -f logs/worker-0.log
```

---

## 🎛️ Control Plane UI

### Access
```bash
# Local development
open http://localhost:9998

# Production (via CLI)
hindsight ui
# → http://localhost:10001

# Production (via tunnel)
open https://neuron-ai-controller.collabmind.dev
```

### Features
- **Services Tab:** Start/Stop/Restart services, worker scaling
- **Operations Tab:** View worker queue, stuck operations
- **Logs Tab:** Live log viewer for all services

### System Page
```bash
open http://localhost:9998/system
```

**Available Controls:**
- Start/Stop/Restart: Hindsight API, Ollama lanes, Workers, Memlord
- Scale workers: 0-10 (click +/- buttons)
- View logs: Select service from dropdown
- Auto-refresh: Toggle on/off (10s interval)

---

## 📊 Service Ports

| Service | Port | Health Check |
|---------|------|--------------|
| Hindsight API | 8888 | http://localhost:8888/health |
| Control Plane (dev) | 9998 | http://localhost:9998 |
| Control Plane (prod) | 10001 | http://localhost:10001 |
| Ollama Embeddings | 11434 | http://localhost:11434/api/tags |
| Ollama LLM | 11435 | http://localhost:11435/api/tags |
| PostgreSQL (pg0) | 5433 | psql -h localhost -p 5433 -U oliververmeulen hindsight17 |
| Workers | 9001-9010 | Per-worker metrics ports |
| Memlord MCP | 8005 | http://localhost:8005/health |
| Grafana | 3000 | http://localhost:3000 |

---

## 🔧 Common Tasks

### Restart API Only
```bash
pkill -9 -f "hindsight-api"
lsof -ti:8888 | xargs kill -9 2>/dev/null
./scripts/dev/start-api.sh
```

### Restart Control Plane Only
```bash
pkill -9 -f "next dev.*hindsight-control-plane"
lsof -ti:9998 | xargs kill -9 2>/dev/null
./scripts/dev/start-control-plane.sh
```

### Scale Workers
```bash
# Via script
./scripts/dev/scale-workers.sh 4

# Via Control Plane UI
open http://localhost:9998/system
# → Click Workers card → Use +/- buttons

# Via API
curl -X POST http://localhost:9998/api/system/services/workers \
  -H "Content-Type: application/json" \
  -d '{"action": "scale", "count": 4}'
```

### View Worker Status
```bash
# Worker state file
cat /tmp/hindsight-workers.state

# Worker logs
tail -f logs/worker-*.log

# Active operations
curl http://localhost:8888/v1/default/operations?limit=20
```

---

## 🐛 Troubleshooting

### Service Won't Start
```bash
# Check for zombies
lsof -ti:8888,9998,10001

# Kill everything
./scripts/dev/stop-all.sh
pkill -9 -f "hindsight|next|ollama"

# Clean state
rm -f /tmp/hindsight-workers.state

# Restart
./scripts/dev/start-services.sh
```

### Logs Not Showing
```bash
# Check directory exists
mkdir -p /Users/oliververmeulen/hindsight/logs

# Check permissions
chmod 755 /Users/oliververmeulen/hindsight/logs

# Test log API
curl http://localhost:9998/api/system/logs?service=api&lines=10
```

### Worker Stuck
```bash
# View operations
curl http://localhost:8888/v1/default/operations?status=processing

# Cancel operation
curl -X POST http://localhost:8888/v1/default/operations/{operation_id}/cancel

# Restart workers
./scripts/dev/scale-workers.sh 0
sleep 2
./scripts/dev/scale-workers.sh 4
```

### Control Plane 401 Unauthorized
```bash
# Check access key
grep HINDSIGHT_CP_ACCESS_KEY /Users/oliververmeulen/hindsight/.env

# Add if missing
echo "HINDSIGHT_CP_ACCESS_KEY=your-secret-key" >> .env

# Restart control plane
pkill -9 -f "next dev.*hindsight-control-plane"
./scripts/dev/start-control-plane.sh
```

---

## 📂 Important Files

### Configuration
- `/Users/oliververmeulen/hindsight/.env` - Main config (API, embeddings, LLM)
- `/tmp/hindsight-workers.state` - Worker registry
- `/Users/oliververmeulen/hindsight/logs/` - All service logs

### Scripts
- `scripts/dev/start-services.sh` - Start everything
- `scripts/dev/stop-all.sh` - Stop everything
- `scripts/dev/status.sh` - Check status
- `scripts/dev/scale-workers.sh` - Manage workers

### Logs
- `logs/hindsight-api.log` - API server
- `logs/hindsight-control-plane.log` - Control Plane UI
- `logs/worker-N.log` - Worker N logs
- `/tmp/ollama-embeddings.log` - Ollama embeddings lane
- `/tmp/ollama-llm.log` - Ollama LLM lane

---

## 🔐 Authentication

### Control Plane Access
```bash
# Set access key in .env
HINDSIGHT_CP_ACCESS_KEY=your-secret-key

# Login via browser
open http://localhost:9998/login
# Enter access key → Dashboard
```

### API Access (Optional)
```bash
# Public access (default)
curl http://localhost:8888/health

# With auth (if HINDSIGHT_API_KEY set)
curl -H "Authorization: Bearer your-api-key" http://localhost:8888/v1/default/banks
```

---

## 📈 Monitoring

### Grafana
```bash
open http://localhost:3000
# Default: no login required
```

**Dashboards:**
- Hindsight Operations
- LLM Metrics
- API Service Health

### Prometheus Metrics
```bash
curl http://localhost:8888/metrics
```

### Traces (Tempo)
```bash
# Enable in .env
HINDSIGHT_API_OTEL_ENABLED=true
HINDSIGHT_API_OTEL_ENDPOINT=http://localhost:4317

# View in Grafana
open http://localhost:3000/explore
# → Select Tempo → Query traces
```

---

## 🎯 Quick Health Check

```bash
# One-liner status
curl -s http://localhost:8888/health && \
curl -s http://localhost:9998/api/health && \
echo "✅ All systems operational" || \
echo "❌ Some services down"
```

---

## 📚 Documentation

- **Full Guide:** `STARTUP_IMPROVEMENTS.md`
- **Agent Instructions:** `AGENTS.md`
- **Monitoring Setup:** `MONITORING_GUIDE.md`
- **Ollama Split Setup:** `OLLAMA_SPLIT_SETUP.md`
