# Hindsight Services Dashboard

## 🚀 Quick Access

| Service | URL | Credentials | Purpose |
|---------|-----|-------------|---------|
| **Control Plane** | http://localhost:9999 | Access key in `.env` | Main UI, memory management |
| **Grafana** | http://localhost:3000 | admin / admin | Dashboards, metrics, traces |
| **API** | http://localhost:8888 | None | REST API, MCP endpoints |
| **Prometheus** | http://localhost:9090 | None | Metrics collection |
| **Tempo** | http://localhost:3200 | None | Distributed tracing |
| **Pyroscope** | http://localhost:4040 | None | Continuous profiling |

## 📊 Monitoring Stack (Grafana LGTM)

### Grafana Dashboards
**URL**: http://localhost:3000  
**Login**: `admin` / `admin`

Available Dashboards:
- **Hindsight Operations** - Overall system health
- **Hindsight LLM** - LLM performance and usage
- **Hindsight API Service** - API metrics and performance

### OpenTelemetry Endpoints

**OTLP gRPC**: http://localhost:4317  
**OTLP HTTP**: http://localhost:4318

Send traces from your application:
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

### Tempo (Distributed Tracing)

**API Endpoint**: http://localhost:3200  
**MCP Endpoint**: http://localhost:3200/api/mcp

Query traces:
```bash
# Search traces
curl http://localhost:3200/api/search

# Get trace by ID
curl http://localhost:3200/api/traces/<trace-id>
```

### Prometheus (Metrics)

**URL**: http://localhost:9090

Query metrics:
```bash
# API request rate
curl 'http://localhost:9090/api/v1/query?query=rate(http_requests_total[5m])'

# Worker metrics
curl 'http://localhost:9090/api/v1/query?query=hindsight_worker_tasks_total'
```

**Scrape Targets**:
- API: http://host.docker.internal:8888/metrics
- Worker 1: http://host.docker.internal:9001/metrics
- Worker 2: http://host.docker.internal:9002/metrics
- Worker 3: http://host.docker.internal:9003/metrics
- Worker 4: http://host.docker.internal:9004/metrics

### Pyroscope (Continuous Profiling)

**URL**: http://localhost:4040

Continuous profiling for performance analysis.

## 🤖 AI Tool Integration (MCP)

### Claude Desktop Integration

```bash
# Setup MCP for Claude Desktop
bash <(docker exec hindsight-monitoring cat /etc/lgtm/claude-mcp-setup.sh)
```

### Other AI Tools

```bash
# Get MCP configuration
docker exec hindsight-monitoring cat /etc/lgtm/mcp.json
```

**Documentation**: https://github.com/grafana/docker-otel-lgtm/blob/main/docs/mcp-integration.md

## 🔧 Hindsight Services

### API Server
**URL**: http://localhost:8888  
**Health**: http://localhost:8888/health  
**Metrics**: http://localhost:8888/metrics  
**MCP**: http://localhost:8888/mcp/

**API Documentation**: http://localhost:8888/docs (OpenAPI/Swagger)

### Control Plane
**URL**: http://localhost:9999  
**Auth**: Requires `HINDSIGHT_CP_ACCESS_KEY` from `.env`

**Features**:
- Memory management
- Agent configuration
- Task monitoring
- System status

### Workers

| Worker | Metrics | Logs |
|--------|---------|------|
| worker-1 | http://localhost:9001/metrics | logs/worker-1.log |
| worker-2 | http://localhost:9002/metrics | logs/worker-2.log |
| worker-3 | http://localhost:9003/metrics | logs/worker-3.log |
| worker-4 | http://localhost:9004/metrics | logs/worker-4.log |

### Ollama Lanes

| Lane | URL | Purpose | Models |
|------|-----|---------|--------|
| **Embeddings** | http://localhost:11434 | Vector generation | nomic-embed-text + 21 others |
| **LLM** | http://localhost:11435 | Reasoning, extraction | llama3.2, gemma3:12b + 20 others |

**Check models**:
```bash
# Embeddings lane
curl http://localhost:11434/api/tags

# LLM lane
curl http://localhost:11435/api/tags
```

## 📋 Service Status

### Quick Status Check

```bash
# Read-only status monitor
cd /Users/oliververmeulen/hindsight
./scripts/dev/status.sh

# Watch mode (auto-refresh)
./scripts/dev/status.sh --watch
```

### Interactive Dashboard

```bash
cd /Users/oliververmeulen/hindsight
./scripts/dev/dashboard.sh
```

Features:
- View live logs (API, workers)
- View all errors
- Restart services
- Scale workers
- Start/stop monitoring

## 🔍 Debugging & Logs

### Log Locations

| Component | Log File |
|-----------|----------|
| API | `/tmp/hindsight-api.log` |
| Control Plane | `/tmp/hindsight-control-plane.log` |
| Workers | `logs/worker-N.log` |
| Ollama LLM | `/tmp/ollama-llm-lane.log` |
| Monitoring | `docker logs hindsight-monitoring` |

### View Logs

```bash
# API logs (live)
tail -f /tmp/hindsight-api.log

# Control Plane logs (live)
tail -f /tmp/hindsight-control-plane.log

# All worker logs (live)
tail -f logs/worker-*.log

# Monitoring stack logs
docker logs -f hindsight-monitoring
```

### Error Filtering

```bash
# API errors only
tail -f /tmp/hindsight-api.log | grep -i error

# Worker errors only
tail -f logs/worker-*.log | grep -i error
```

## 🎯 Common Tasks

### Start Everything

```bash
cd /Users/oliververmeulen/hindsight
./scripts/dev/start-services.sh
```

### Check Status

```bash
./scripts/dev/status.sh --watch
```

### Scale Workers

```bash
# Start 8 workers
./scripts/dev/scale-workers.sh 8

# Check worker status
./scripts/dev/scale-workers.sh status

# Stop all workers
./scripts/dev/scale-workers.sh stop
```

### Restart Services

```bash
# Restart API
pkill -f hindsight-api
nohup ./scripts/dev/start-api.sh --port 8888 > /tmp/hindsight-api.log 2>&1 &

# Restart Control Plane
pkill -f "next dev"
nohup ./scripts/dev/start-control-plane.sh > /tmp/hindsight-control-plane.log 2>&1 &

# Restart workers
./scripts/dev/scale-workers.sh restart 4
```

### Stop Everything

```bash
# Stop API and Control Plane
pkill -f hindsight-api
pkill -f "next dev"

# Stop workers
./scripts/dev/scale-workers.sh stop

# Stop monitoring (optional)
./scripts/dev/stop-monitoring.sh

# Stop Ollama LLM lane (optional)
./scripts/dev/stop-ollama-split.sh
```

## 📊 Monitoring Queries

### Prometheus Queries

Access Prometheus at http://localhost:9090, then try these queries:

```promql
# API request rate (requests per second)
rate(http_requests_total[5m])

# API error rate
rate(http_requests_total{status=~"5.."}[5m])

# API latency (95th percentile)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Worker task queue depth
hindsight_worker_queue_depth

# LLM request latency
histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))

# Memory usage
process_resident_memory_bytes
```

### Grafana Explore

1. Open http://localhost:3000
2. Click **Explore** (compass icon)
3. Select data source:
   - **Prometheus**: Metrics
   - **Loki**: Logs
   - **Tempo**: Traces

### Tempo Trace Queries

```bash
# Search traces by service
curl 'http://localhost:3200/api/search?tags=service.name%3Dhindsight-dev'

# Get trace by ID
curl 'http://localhost:3200/api/traces/<trace-id>'
```

## 🔐 Configuration

### Environment File

**Location**: `/Users/oliververmeulen/hindsight/.env`

**Critical Settings**:
```bash
# Split Ollama Configuration
HINDSIGHT_API_LLM_OLLAMA_API_BASE=http://localhost:11435
HINDSIGHT_API_EMBEDDINGS_OLLAMA_API_BASE=http://localhost:11434

# Control Plane Access
HINDSIGHT_CP_ACCESS_KEY=your-secret-key-here

# OpenTelemetry (optional - uncomment to enable tracing)
# HINDSIGHT_API_OTEL_TRACES_ENABLED=true
# HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
# HINDSIGHT_API_OTEL_SERVICE_NAME=hindsight-dev
```

**⚠️ NEVER modify `.env` programmatically - R&D mode with real credentials**

### Enable Tracing

To send traces to Tempo, uncomment in `.env`:

```bash
HINDSIGHT_API_OTEL_TRACES_ENABLED=true
HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
HINDSIGHT_API_OTEL_SERVICE_NAME=hindsight-dev
HINDSIGHT_API_OTEL_DEPLOYMENT_ENVIRONMENT=development
```

Then restart API:
```bash
pkill -f hindsight-api
./scripts/dev/start-services.sh
```

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| `OLLAMA_SPLIT_SETUP.md` | Complete Ollama split lanes guide |
| `MONITORING_GUIDE.md` | Grafana, tracing, metrics, logs |
| `scripts/dev/README.md` | Development scripts reference |
| `AGENTS.md` | Critical info for AI agents |

## 🆘 Troubleshooting

### Service Not Responding

```bash
# Check if service is running
ps aux | grep hindsight

# Check port availability
lsof -i :8888  # API
lsof -i :9999  # Control Plane
lsof -i :3000  # Grafana

# Check Docker containers
docker ps
```

### Cannot Access Control Plane

1. Ensure `HINDSIGHT_CP_ACCESS_KEY` is set in `.env`
2. Restart Control Plane:
   ```bash
   pkill -f "next dev"
   ./scripts/dev/start-services.sh
   ```
3. Check logs: `tail -f /tmp/hindsight-control-plane.log`

### No Traces in Tempo

1. Check OTEL settings are uncommented in `.env`
2. Verify API is sending traces:
   ```bash
   curl http://localhost:8888/metrics | grep otel
   ```
3. Check Tempo is receiving data:
   ```bash
   curl http://localhost:3200/api/search
   ```

### Workers Stopped After Docker Restart

```bash
# Restart workers
./scripts/dev/scale-workers.sh restart 4
```

## 🎨 Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User / Browser                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
      ┌──────────┐ ┌──────────┐ ┌──────────┐
      │ Control  │ │ Grafana  │ │   API    │
      │  Plane   │ │   LGTM   │ │  :8888   │
      │  :9999   │ │  :3000   │ └────┬─────┘
      └──────────┘ └────┬─────┘      │
                        │            │
                        ▼            ▼
              ┌─────────────────────────┐
              │  Monitoring Stack       │
              │  • Prometheus (:9090)   │
              │  • Tempo (:3200)        │
              │  • Pyroscope (:4040)    │
              │  • OTLP (:4317, :4318)  │
              └─────────────────────────┘
                        
                   API connects to:
                        
        ┌───────┬────────┬────────┬───────┐
        ▼       ▼        ▼        ▼       ▼
    ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐
    │ W1 │  │ W2 │  │ W3 │  │ W4 │  │ DB │
    │9001│  │9002│  │9003│  │9004│  │    │
    └────┘  └────┘  └────┘  └────┘  └────┘
    
              ┌──────────┬──────────┐
              ▼          ▼          
        ┌─────────┐  ┌─────────┐
        │ Ollama  │  │ Ollama  │
        │ Embed   │  │  LLM    │
        │ :11434  │  │ :11435  │
        └─────────┘  └─────────┘
```

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Start everything | `./scripts/dev/start-services.sh` |
| Check status | `./scripts/dev/status.sh --watch` |
| Interactive dashboard | `./scripts/dev/dashboard.sh` |
| View API logs | `tail -f /tmp/hindsight-api.log` |
| View traces | Open http://localhost:3000 → Explore → Tempo |
| Query metrics | Open http://localhost:9090 |
| Scale workers | `./scripts/dev/scale-workers.sh 8` |
| Restart API | `pkill -f hindsight-api && ./scripts/dev/start-services.sh` |

---

**All services are running and ready!** 🚀

For detailed guides, see:
- `OLLAMA_SPLIT_SETUP.md` - Ollama configuration
- `MONITORING_GUIDE.md` - Complete monitoring setup
- `scripts/dev/README.md` - All scripts reference
