# Hindsight Monitoring & Tracing Guide

## Overview

Hindsight uses **Grafana LGTM** (Loki + Grafana + Tempo + Mimir) for complete observability:
- **Grafana**: Visualization and dashboards
- **Loki**: Log aggregation
- **Tempo**: Distributed tracing
- **Mimir**: Metrics storage
- **Prometheus**: Metrics scraping

## Quick Access

### Grafana UI
**URL**: http://localhost:3000

**Login**: No login required (anonymous admin access enabled for local dev)

### Available Dashboards

1. **Hindsight Operations** - Overall system health
2. **Hindsight LLM** - LLM performance and usage
3. **Hindsight API Service** - API metrics and performance

### OTLP Endpoints (for sending traces)

- **gRPC**: http://localhost:4317
- **HTTP**: http://localhost:4318

## Current Configuration Status

### ✅ Enabled
- Grafana LGTM stack running
- Anonymous admin access (no login needed)
- Custom Hindsight dashboards mounted
- Prometheus scraping configured
- OTLP receivers ready (ports 4317, 4318)

### ⚠️ Partially Enabled
Your `.env` has:
```bash
HINDSIGHT_API_LLM_TRACE_ENABLED=true  # ✅ LLM tracing enabled

# ❌ OTEL exporter commented out (traces not being sent)
# HINDSIGHT_API_OTEL_TRACES_ENABLED=true
# HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

## Enable Full Tracing

To send traces from Hindsight API to Grafana Tempo, you need to uncomment and configure the OTEL settings in `.env`:

### Step 1: Edit `.env`

Open `/Users/oliververmeulen/hindsight/.env` and find these lines (around line 60-65):

**BEFORE** (commented out):
```bash
# HINDSIGHT_API_OTEL_TRACES_ENABLED=true
# HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
# HINDSIGHT_API_OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer your-token"
# HINDSIGHT_API_OTEL_SERVICE_NAME=hindsight-production
# HINDSIGHT_API_OTEL_DEPLOYMENT_ENVIRONMENT=production
```

**AFTER** (enabled for local dev):
```bash
HINDSIGHT_API_OTEL_TRACES_ENABLED=true
HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
# HINDSIGHT_API_OTEL_EXPORTER_OTLP_HEADERS=""  # No auth needed for local
HINDSIGHT_API_OTEL_SERVICE_NAME=hindsight-dev
HINDSIGHT_API_OTEL_DEPLOYMENT_ENVIRONMENT=development
```

### Step 2: Restart API and Workers

```bash
# Stop current API and workers
pkill -f hindsight-api
./scripts/dev/scale-workers.sh stop

# Restart with new config
./scripts/dev/start-all.sh --monitoring --workers 4
```

### Step 3: Verify Traces

1. Open Grafana: http://localhost:3000
2. Click **Explore** (compass icon on left sidebar)
3. Select **Tempo** from the data source dropdown
4. Click **Search** tab
5. Set filters:
   - Service Name: `hindsight-dev`
   - Click **Run query**

You should see traces appearing!

## Using Grafana

### Navigate to Dashboards

1. Open http://localhost:3000
2. Click **Dashboards** (four squares icon on left sidebar)
3. You'll see:
   - Hindsight Operations
   - Hindsight LLM
   - Hindsight API Service

### Explore Traces (Tempo)

1. Click **Explore** (compass icon)
2. Select **Tempo** data source
3. Search by:
   - **Service Name**: `hindsight-dev`
   - **Operation**: Specific API endpoints
   - **Tags**: Custom trace attributes
   - **Duration**: Find slow requests

### View Logs (Loki)

1. Click **Explore**
2. Select **Loki** data source
3. Query examples:
   ```
   {service_name="hindsight-dev"}
   {service_name="hindsight-dev"} |= "error"
   {service_name="hindsight-dev"} |= "extraction"
   ```

### Query Metrics (Prometheus/Mimir)

1. Click **Explore**
2. Select **Prometheus** data source
3. Query examples:
   ```
   # Request rate
   rate(http_requests_total[5m])
   
   # Error rate
   rate(http_requests_total{status=~"5.."}[5m])
   
   # LLM latency
   histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))
   ```

## Monitoring Stack Management

### Start Monitoring

```bash
# Start with everything
./scripts/dev/start-all.sh --monitoring --workers 4

# Or start monitoring separately
cd /Users/oliververmeulen/hindsight/scripts/dev/monitoring
docker compose up -d
```

### Stop Monitoring

```bash
./scripts/dev/stop-monitoring.sh
```

### Check Status

```bash
./scripts/dev/status.sh

# Or watch mode
./scripts/dev/status.sh --watch
```

### View Logs

```bash
# Monitoring stack logs
docker logs hindsight-monitoring -f

# API logs
tail -f /tmp/hindsight-api.log

# Worker logs
tail -f /Users/oliververmeulen/hindsight/logs/worker-*.log
```

## What to Monitor

### Key Metrics

1. **API Performance**
   - Request rate
   - Response time (p50, p95, p99)
   - Error rate
   - Active connections

2. **LLM Performance**
   - Ollama request latency (embeddings vs LLM lane)
   - Token throughput
   - Model load time
   - Queue depth

3. **Worker Health**
   - Active workers
   - Task queue length
   - Task processing time
   - Worker crashes/restarts

4. **Resource Usage**
   - CPU usage (API, workers, Ollama)
   - Memory usage
   - Disk I/O
   - Network throughput

### Key Traces to Review

1. **Extraction Pipeline**
   - End-to-end extraction time
   - LLM call duration
   - Embedding generation time
   - Database write time

2. **Search Queries**
   - Query parsing time
   - Vector search latency
   - Reranking time
   - Result assembly time

3. **Error Traces**
   - Failed LLM calls
   - Timeout errors
   - Database errors
   - Validation failures

## Troubleshooting

### No Traces Appearing

**Check 1**: Is OTEL enabled in `.env`?
```bash
grep OTEL_TRACES_ENABLED /Users/oliververmeulen/hindsight/.env
# Should show: HINDSIGHT_API_OTEL_TRACES_ENABLED=true (uncommented)
```

**Check 2**: Is the API using the new config?
```bash
# Restart API to pick up .env changes
pkill -f hindsight-api
./scripts/dev/start-all.sh --workers 4
```

**Check 3**: Is Tempo receiving data?
```bash
# Check OTLP endpoint
curl -v http://localhost:4318/v1/traces
# Should return 405 Method Not Allowed (means endpoint is alive)
```

### Dashboards Not Showing Data

**Check 1**: Are metrics being scraped?
```bash
# Check Prometheus targets
curl -s http://localhost:3000/api/datasources/proxy/1/api/v1/targets | jq
```

**Check 2**: Is the API exposing metrics?
```bash
# Check API metrics endpoint
curl http://localhost:8888/metrics
```

**Check 3**: Are workers running?
```bash
./scripts/dev/scale-workers.sh status
```

### Grafana Not Accessible

**Check 1**: Is the container running?
```bash
docker ps --filter "name=hindsight-monitoring"
```

**Check 2**: Is port 3000 available?
```bash
lsof -i :3000
```

**Check 3**: Check container logs
```bash
docker logs hindsight-monitoring
```

## Advanced: Custom Dashboards

### Create New Dashboard

1. Open Grafana: http://localhost:3000
2. Click **Dashboards** → **New** → **New Dashboard**
3. Add panels with queries
4. Save dashboard

### Export Dashboard

1. Open dashboard
2. Click **Share** (top right)
3. Click **Export** tab
4. Click **Save to file**
5. Save to `/Users/oliververmeulen/hindsight/monitoring/grafana/dashboards/`

### Import Dashboard

1. Click **Dashboards** → **New** → **Import**
2. Upload JSON file or paste JSON
3. Select data sources
4. Click **Import**

## Monitoring Best Practices

### 1. Set Up Alerts

Create alerts for:
- API error rate > 5%
- LLM latency > 10s (p95)
- Worker queue depth > 100
- Disk usage > 80%

### 2. Regular Review

- Check dashboards daily during development
- Review error traces immediately
- Monitor resource trends weekly

### 3. Trace Sampling

For production, consider sampling:
```bash
# In .env
HINDSIGHT_API_OTEL_TRACE_SAMPLE_RATE=0.1  # 10% of requests
```

### 4. Log Retention

Configure retention in Loki:
- Development: 7 days
- Production: 30 days

## Integration with Development Workflow

### During Development

1. **Start monitoring with your stack**:
   ```bash
   ./scripts/dev/start-all.sh --monitoring --workers 4
   ```

2. **Keep status monitor open**:
   ```bash
   ./scripts/dev/status.sh --watch
   ```

3. **Review traces after testing**:
   - Open Grafana
   - Go to Explore → Tempo
   - Search for recent traces

### Debugging Issues

1. **Check status first**:
   ```bash
   ./scripts/dev/status.sh
   ```

2. **Review error logs**:
   ```bash
   tail -f /tmp/hindsight-api.log | grep -i error
   ```

3. **Find slow traces**:
   - Grafana → Explore → Tempo
   - Filter by duration > 5s

4. **Check metrics**:
   - Open relevant dashboard
   - Look for anomalies

## Resources

### Documentation
- Grafana LGTM: https://grafana.com/docs/grafana-cloud/monitor-applications/application-observability/setup/quickstart/lgtm/
- OpenTelemetry: https://opentelemetry.io/docs/
- Tempo: https://grafana.com/docs/tempo/
- Loki: https://grafana.com/docs/loki/

### Local Files
- Docker Compose: `scripts/dev/monitoring/docker-compose.yaml`
- Prometheus Config: `scripts/dev/monitoring/prometheus.yml`
- Dashboards: `monitoring/grafana/dashboards/`

### Quick Commands
```bash
# Start everything with monitoring
./scripts/dev/start-all.sh --monitoring --workers 4

# Check status
./scripts/dev/status.sh --watch

# View logs
tail -f /tmp/hindsight-api.log

# Stop monitoring
./scripts/dev/stop-monitoring.sh

# Restart with tracing enabled
pkill -f hindsight-api && ./scripts/dev/start-all.sh --monitoring --workers 4
```

## Next Steps

1. ✅ Monitoring stack is running
2. ⏳ Enable OTEL tracing in `.env` (see "Enable Full Tracing" above)
3. ⏳ Restart API and workers
4. ⏳ Open Grafana and explore dashboards
5. ⏳ Run some test queries and review traces
6. ⏳ Set up alerts for critical metrics

---

**Need help?** Check `./scripts/dev/status.sh` for current system status or `./scripts/dev/dashboard.sh` for interactive management.
