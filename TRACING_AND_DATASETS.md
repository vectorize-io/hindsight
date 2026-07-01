# Hindsight Tracing & Datasets Collection Guide

## 📊 What Collects Traces and Logs?

### Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Hindsight API                            │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ LLM Tracing  │  │ OTEL Tracer  │  │   Metrics    │     │
│  │   (enabled)  │  │  (disabled)  │  │   (enabled)  │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │              │
└─────────┼─────────────────┼─────────────────┼──────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │   Logs   │    │  Tempo   │    │Prometheus│
    │  Files   │    │  :3200   │    │  :9090   │
    └──────────┘    └──────────┘    └──────────┘
          │                 │                 │
          └─────────────────┴─────────────────┘
                          │
                          ▼
                  ┌────────────────┐
                  │    Grafana     │
                  │     :3000      │
                  │                │
                  │  • Dashboards  │
                  │  • Traces      │
                  │  • Logs        │
                  │  • Metrics     │
                  └────────────────┘
```

## 🔍 Trace & Log Collection Services

### 1. **LLM Tracing** (Currently Enabled)

**Status**: ✅ ENABLED  
**Config**: `HINDSIGHT_API_LLM_TRACE_ENABLED=true`  
**Storage**: Log files

**What it captures**:
- LLM request/response pairs
- Prompt templates
- Token counts
- Latency
- Model used
- Errors

**Where to find it**:
```bash
# API logs contain LLM traces
tail -f /tmp/hindsight-api.log | grep "LLM\|llm\|extraction"

# Worker logs contain extraction traces
tail -f logs/worker-*.log | grep "extraction\|llm"
```

### 2. **OpenTelemetry (OTEL) Tracing** (Currently Disabled)

**Status**: ⚠️ DISABLED (commented out in `.env`)  
**Target**: Tempo (port 3200)  
**Protocol**: OTLP HTTP (port 4318)

**What it would capture**:
- Distributed traces across API → Workers → LLM
- Span hierarchy (parent-child relationships)
- Timing data for each operation
- Custom attributes (user_id, memory_id, etc.)
- Error spans

**To enable**, uncomment in `.env`:
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

### 3. **Prometheus Metrics** (Currently Enabled)

**Status**: ✅ ENABLED  
**Endpoint**: http://localhost:9090  
**Scraped from**:
- API: http://localhost:8888/metrics
- Workers: http://localhost:9001-9004/metrics

**What it captures**:
- Request counts
- Request durations (histograms)
- Error rates
- Active connections
- Queue depths
- Memory usage
- CPU usage

### 4. **Loki (Log Aggregation)** (Available in LGTM Stack)

**Status**: ✅ AVAILABLE  
**Part of**: Grafana LGTM Docker container  
**Query UI**: Grafana Explore → Loki

**What it can capture**:
- Application logs from API
- Worker logs
- Structured log queries
- Log streaming

**Current setup**: Logs go to files, not Loki yet

## 📁 Custom Dashboards

**Location**: `/Users/oliververmeulen/hindsight/monitoring/grafana/dashboards/`

### Available Dashboards

1. **hindsight-operations.json** (16KB)
   - Overall system health
   - Service status
   - Resource usage
   - Error rates

2. **hindsight-llm.json** (14KB)
   - LLM request rates
   - Token usage
   - Latency by model
   - LLM errors

3. **hindsight-api-service.json** (34KB)
   - API request metrics
   - Endpoint performance
   - HTTP status codes
   - Response times

### How to Access Dashboards

1. Open Grafana: http://localhost:3000
2. Login: `admin` / `admin`
3. Click **Dashboards** (four squares icon on left)
4. Select one of the Hindsight dashboards

### How to Create New Dashboards

1. Open Grafana: http://localhost:3000
2. Click **Dashboards** → **New** → **New Dashboard**
3. Add panels with queries
4. Save dashboard
5. Export as JSON:
   - Click **Share** (top right)
   - Click **Export** tab
   - Click **Save to file**
6. Save to: `/Users/oliververmeulen/hindsight/monitoring/grafana/dashboards/your-dashboard.json`
7. Add to provisioning config (see below)

## 🎯 Dataset Collection Strategy

### What Data to Collect for AI Training

For building datasets from Hindsight usage, you want:

1. **Input/Output Pairs**
   - User queries
   - System responses
   - Retrieved context
   - LLM prompts and completions

2. **Performance Metrics**
   - Retrieval quality scores
   - Response times
   - Token usage

3. **User Feedback**
   - Relevance ratings
   - Corrections
   - Follow-up queries

### Recommended Collection Services

#### Option 1: **OpenTelemetry → Tempo** (Recommended)

**Pros**:
- Distributed tracing shows full request flow
- Custom span attributes for datasets
- Query by trace ID
- Export to analysis tools

**Setup**:
```bash
# 1. Enable OTEL in .env
HINDSIGHT_API_OTEL_TRACES_ENABLED=true
HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
HINDSIGHT_API_OTEL_SERVICE_NAME=hindsight-dev

# 2. Restart API
pkill -f hindsight-api
./scripts/dev/start-services.sh

# 3. Query traces
curl 'http://localhost:3200/api/search?tags=service.name=hindsight-dev'
```

**Custom Attributes to Add**:
```python
# In your extraction code
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("memory_extraction") as span:
    span.set_attribute("user_id", user_id)
    span.set_attribute("memory_id", memory_id)
    span.set_attribute("query", query_text)
    span.set_attribute("retrieved_docs", len(docs))
    span.set_attribute("llm_tokens", token_count)
    span.set_attribute("quality_score", score)
    
    # Your extraction logic here
```

#### Option 2: **Structured Logs → Loki**

**Pros**:
- Simple to implement
- Rich querying
- Low overhead

**Setup**:
```python
import logging
import json

logger = logging.getLogger(__name__)

# Log structured data
logger.info(json.dumps({
    "event": "memory_extraction",
    "user_id": user_id,
    "query": query_text,
    "retrieved_docs": doc_count,
    "llm_model": "llama3.2",
    "tokens": token_count,
    "latency_ms": latency
}))
```

**Query in Grafana**:
```
{service_name="hindsight-dev"} | json | event="memory_extraction"
```

#### Option 3: **Direct Database Logging**

**Pros**:
- Structured storage
- Easy to query
- Can export to CSV/JSON

**Setup**:
```sql
CREATE TABLE extraction_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id TEXT,
    query TEXT,
    response TEXT,
    retrieved_docs JSONB,
    llm_model TEXT,
    tokens INTEGER,
    latency_ms INTEGER,
    quality_score FLOAT
);
```

```python
# Log to database
db.execute("""
    INSERT INTO extraction_logs 
    (user_id, query, response, retrieved_docs, llm_model, tokens, latency_ms)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
""", user_id, query, response, docs, model, tokens, latency)
```

## 🚀 Recommended Setup for Dataset Collection

### Step 1: Enable OTEL Tracing

Edit `/Users/oliververmeulen/hindsight/.env`:

```bash
# OpenTelemetry Tracing
HINDSIGHT_API_OTEL_TRACES_ENABLED=true
HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
HINDSIGHT_API_OTEL_SERVICE_NAME=hindsight-dev
HINDSIGHT_API_OTEL_DEPLOYMENT_ENVIRONMENT=development

# Optional: Add custom attributes
HINDSIGHT_API_OTEL_RESOURCE_ATTRIBUTES=deployment.environment=dev,service.version=0.8.3
```

### Step 2: Restart Services

```bash
pkill -f hindsight-api
./scripts/dev/start-services.sh
```

### Step 3: Verify Traces Arriving

```bash
# Check if traces are being exported
curl http://localhost:8888/metrics | grep otel

# Query Tempo for traces
curl 'http://localhost:3200/api/search?tags=service.name=hindsight-dev'

# View in Grafana
open http://localhost:3000
# Navigate to: Explore → Tempo → Search
```

### Step 4: Create Dataset Export Script

```bash
#!/bin/bash
# export-traces-for-dataset.sh

# Export traces from Tempo for last 24 hours
START_TIME=$(date -u -v-24H +%s)
END_TIME=$(date -u +%s)

curl "http://localhost:3200/api/search?start=${START_TIME}&end=${END_TIME}&tags=service.name=hindsight-dev" \
  | jq '.traces[] | {
      trace_id: .traceID,
      spans: .spanSets[].spans[] | {
        operation: .name,
        duration_ms: .duration / 1000000,
        attributes: .attributes
      }
    }' > dataset_traces_$(date +%Y%m%d).json
```

### Step 5: Create Custom Dashboard for Dataset Metrics

Create `/Users/oliververmeulen/hindsight/monitoring/grafana/dashboards/hindsight-datasets.json`:

```json
{
  "dashboard": {
    "title": "Hindsight Dataset Collection",
    "panels": [
      {
        "title": "Extraction Requests",
        "targets": [
          {
            "expr": "rate(hindsight_extraction_total[5m])"
          }
        ]
      },
      {
        "title": "Dataset Size",
        "targets": [
          {
            "expr": "hindsight_dataset_samples_total"
          }
        ]
      },
      {
        "title": "Quality Scores",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(hindsight_quality_score_bucket[5m]))"
          }
        ]
      }
    ]
  }
}
```

## 📊 Querying Collected Data

### Tempo (Traces)

```bash
# Search by service
curl 'http://localhost:3200/api/search?tags=service.name=hindsight-dev'

# Get specific trace
curl 'http://localhost:3200/api/traces/<trace-id>'

# Search by custom attribute
curl 'http://localhost:3200/api/search?tags=user_id=user123'
```

### Loki (Logs)

In Grafana Explore:
```logql
# All extraction logs
{service_name="hindsight-dev"} |= "extraction"

# Filter by user
{service_name="hindsight-dev"} | json | user_id="user123"

# High latency extractions
{service_name="hindsight-dev"} | json | latency_ms > 5000
```

### Prometheus (Metrics)

In Grafana Explore:
```promql
# Extraction rate
rate(hindsight_extraction_total[5m])

# Average tokens per extraction
rate(hindsight_extraction_tokens_total[5m]) 
/ 
rate(hindsight_extraction_total[5m])

# Error rate
rate(hindsight_extraction_errors_total[5m])
```

## 🔧 Configuration Files

### Dashboard Provisioning

**Location**: `/Users/oliververmeulen/hindsight/scripts/dev/monitoring/grafana-dashboards.yaml`

**Current config**:
```yaml
apiVersion: 1

providers:
  - name: 'Hindsight Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /otel-lgtm
      foldersFromFilesStructure: true
```

### Prometheus Scrape Config

**Location**: `/Users/oliververmeulen/hindsight/scripts/dev/monitoring/prometheus.yml`

Check current targets:
```bash
cat scripts/dev/monitoring/prometheus.yml
```

## 🎯 Best Practices for Dataset Collection

1. **Use Trace Sampling in Production**
   ```bash
   # Sample 10% of requests
   HINDSIGHT_API_OTEL_TRACE_SAMPLE_RATE=0.1
   ```

2. **Add Rich Span Attributes**
   - user_id
   - memory_id
   - query_text
   - response_text
   - retrieved_doc_ids
   - llm_model
   - token_count
   - quality_score

3. **Export Regularly**
   - Daily exports to prevent data loss
   - Store in versioned datasets
   - Include metadata (timestamp, version, config)

4. **Privacy & Compliance**
   - Anonymize user_ids if needed
   - Filter sensitive data from traces
   - Set retention policies

5. **Quality Metrics**
   - Track retrieval precision/recall
   - Log user feedback
   - Measure response relevance

## 📚 Summary

**Currently Collecting**:
- ✅ LLM traces (in logs)
- ✅ API metrics (Prometheus)
- ✅ Worker metrics (Prometheus)

**Not Yet Collecting** (but available):
- ⏳ Distributed traces (OTEL → Tempo)
- ⏳ Structured logs (Loki)
- ⏳ Dataset-specific metrics

**Custom Dashboards**:
- ✅ `hindsight-operations.json` - System health
- ✅ `hindsight-llm.json` - LLM metrics
- ✅ `hindsight-api-service.json` - API performance
- ⏳ `hindsight-datasets.json` - Dataset collection (create this)

**Next Steps**:
1. Enable OTEL tracing in `.env`
2. Add custom span attributes for dataset collection
3. Create dataset export scripts
4. Build custom dashboard for dataset metrics

---

**See Also**:
- `MONITORING_GUIDE.md` - Complete monitoring setup
- `SERVICES_DASHBOARD.md` - All service endpoints
- `OLLAMA_SPLIT_SETUP.md` - Ollama configuration
