# Hindsight Architecture - WHY Things Work This Way

**Last Updated**: 2026-06-26  
**Purpose**: Explain the REAL architecture, not just what's running, but WHY it works this way and HOW to make it better.

---

## Table of Contents

- [The Core Problem We Solved](#the-core-problem-we-solved)
- [WHY Split Ollama Lanes Work](#why-split-ollama-lanes-work)
- [WHY Workers Were Failing](#why-workers-were-failing)
- [WHY Docker Keeps Stopping](#why-docker-keeps-stopping)
- [The REAL Architecture](#the-real-architecture)
- [Configuration Hierarchy](#configuration-hierarchy)
- [How to Make It Better](#how-to-make-it-better)
- [Common Mistakes to Avoid](#common-mistakes-to-avoid)

---

## The Core Problem We Solved

### Original Issue: ReadTimeout Errors

**Symptom**: Random `ReadTimeout` and connection errors during LLM extraction and embeddings generation.

**Root Cause**: Both LLM extraction (heavy, long-running) and embeddings (fast, frequent) were sharing the **same Ollama endpoint** (localhost:11434). This caused:

1. **Connection pool exhaustion** - Embeddings requests would flood the connection pool
2. **Request queueing** - LLM extraction requests would timeout waiting for available connections
3. **Resource contention** - Both services competed for the same Ollama server threads

### The Solution: Split Ollama Execution Lanes

We created two **independent Ollama instances**:

- **Port 11434 (Embeddings Lane)**: Fast, frequent vector generation
  - Model: `nomic-embed-text` (768-dim)
  - Traffic: High frequency, short duration
  - Purpose: Convert text → vectors for semantic search

- **Port 11435 (LLM Lane)**: Heavy, long-running extraction/reasoning
  - Model: `llama3.2:latest`, `gemma3:12b`, etc.
  - Traffic: Low frequency, long duration
  - Purpose: Extract facts, answer questions, summarize

### WHY This Works

**Isolation**: Each lane has its own:
- Process (separate `ollama serve` instances)
- Connection pool (independent HTTP clients)
- Thread pool (no resource contention)
- Request queue (no cross-service blocking)

**Shared Models**: Both instances point to the same model directory:
```bash
/Volumes/Mac/Users/oliververmeulen/.ollama/models
```

This means:
- No duplicate model downloads (save disk space)
- No model version conflicts
- Easy model management (one place to update)

---

## WHY Workers Were Failing

### The Worker Environment Problem

**Discovery**: Workers were logging:
```
OpenAI-compatible client initialized: provider=ollama, model=gemma3:12b, base_url=http://localhost:11434/v1
```

But the `.env` file said:
```bash
HINDSIGHT_API_LLM_OLLAMA_API_BASE=http://localhost:11435
HINDSIGHT_API_LLM_OLLAMA_MODEL=ollama/llama3.2:latest
```

**Root Cause #1**: The `scale-workers.sh` script was NOT passing environment variables to worker processes correctly.

**Original Code** (lines 176-183):
```bash
cd "$ROOT_DIR"
HINDSIGHT_API_WORKER_ID="$worker_id" \
HINDSIGHT_API_WORKER_HTTP_PORT="$http_port" \
nohup uv run hindsight-worker \
    --worker-id "$worker_id" \
    --http-port "$http_port" \
    > "$log_file" 2>&1 &
```

**Problem**: Only two env vars (`WORKER_ID`, `HTTP_PORT`) were passed. All other config (LLM endpoints, models, etc.) was missing.

**Fixed Code**:
```bash
cd "$ROOT_DIR"
set -a                          # Export all variables
source "$ROOT_DIR/.env"        # Load full .env
set +a                         # Stop auto-export
HINDSIGHT_API_WORKER_ID="$worker_id" \
HINDSIGHT_API_WORKER_HTTP_PORT="$http_port" \
nohup uv run hindsight-worker \
    --worker-id "$worker_id" \
    --http-port "$http_port" \
    > "$log_file" 2>&1 &
```

**WHY This Works**: `set -a` exports ALL variables before running `uv run`, so the Python process sees the full environment.

---

### Root Cause #2: Database Configuration Overrides

**Discovery**: Even after fixing env loading, workers logged `gemma3:12b` instead of `llama3.2:latest`.

**Reason**: Hindsight has a **hierarchical configuration system**:

1. **Static Config** (`.env` file) - Server-level defaults
2. **Bank Config** (database `bank_configurations` table) - Per-tenant overrides
3. **Runtime Config** (API parameters) - Per-request overrides

**WHY This Exists**: Multi-tenant deployments need different LLM settings per customer:
- Customer A uses `gpt-4` (high-cost, high-quality)
- Customer B uses `gemma3:12b` (low-cost, local)
- Customer C uses `deepseek-v4` (medium-cost, fast)

**Impact**: Workers can have different models than the API, which is CORRECT behavior. The critical part is they use the **split Ollama endpoints** (11434 vs 11435), not the model choice.

---

## WHY Docker Keeps Stopping

### The Monitoring Container Issue

**Symptom**: Docker Desktop stops unexpectedly, causing monitoring stack (Grafana LGTM) to go offline.

**Root Cause**: The monitoring container `hindsight-monitoring` has **volume mounts** that can cause issues:

```yaml
volumes:
  # Mount Prometheus config
  - ./prometheus.yml:/otel-lgtm/prometheus.yaml:ro
  # Mount Hindsight dashboards
  - ../../../monitoring/grafana/dashboards/hindsight-operations.json:/otel-lgtm/hindsight-operations.json:ro
  - ../../../monitoring/grafana/dashboards/hindsight-llm.json:/otel-lgtm/hindsight-llm.json:ro
  - ../../../monitoring/grafana/dashboards/hindsight-api-service.json:/otel-lgtm/hindsight-api-service.json:ro
  # Mount custom dashboard provisioning
  - ./grafana-dashboards.yaml:/otel-lgtm/grafana/conf/provisioning/dashboards/grafana-dashboards.yaml:ro
```

**Potential Issues**:
1. **File watchers** - macOS file system events can overwhelm Docker
2. **Permission conflicts** - Volume mounts across user/container boundaries
3. **Resource limits** - Docker Desktop default limits (2 GB RAM, 2 CPUs)

**Current Status**: Docker is running fine now. Monitoring is OPTIONAL for local development.

**WHY Monitoring is Separate**: 
- Core Hindsight (API, Workers, Ollama) runs **bare metal** for hot-reload during development
- Monitoring (Grafana, Tempo, Loki, Prometheus) runs in **Docker** for easy setup/teardown
- This separation means monitoring failures DON'T break core functionality

---

## The REAL Architecture

### Current State (2026-06-26)

```
┌─────────────────────────────────────────────────────────────┐
│                    BARE METAL (Development)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐          ┌──────────────────┐         │
│  │  Ollama (11434)  │          │  Ollama (11435)  │         │
│  │  EMBEDDINGS LANE │          │     LLM LANE     │         │
│  ├──────────────────┤          ├──────────────────┤         │
│  │ nomic-embed-text │          │ llama3.2:latest  │         │
│  │                  │          │ gemma3:12b       │         │
│  │ Fast, frequent   │          │ Heavy, slow      │         │
│  │ 768-dim vectors  │          │ Extraction       │         │
│  └──────────────────┘          └──────────────────┘         │
│         ▲                               ▲                    │
│         │                               │                    │
│         │       ┌───────────────────────┴──────┐            │
│         │       │   Hindsight API (8888)       │            │
│         │       ├──────────────────────────────┤            │
│         │       │ FastAPI + uvicorn            │            │
│         └───────┤ Coordinates all services     │            │
│                 │ Health: /health              │            │
│                 │ Metrics: /metrics (Prom)     │            │
│                 └──────────────────────────────┘            │
│                          │                                   │
│         ┌────────────────┼────────────────┐                 │
│         │                │                │                 │
│         ▼                ▼                ▼                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │ Worker-1 │    │ Worker-2 │    │ Worker-3 │   Worker-4  │
│  │ Port:9001│    │ Port:9002│    │ Port:9003│   Port:9004 │
│  ├──────────┤    ├──────────┤    ├──────────┤             │
│  │ Process  │    │ Process  │    │ Process  │   Process   │
│  │ async    │    │ async    │    │ async    │   async     │
│  │ tasks    │    │ tasks    │    │ tasks    │   tasks     │
│  └──────────┘    └──────────┘    └──────────┘             │
│         │                │                │                 │
│         └────────────────┴────────────────┘                 │
│                          │                                   │
│                          ▼                                   │
│                 ┌────────────────┐                          │
│                 │   PostgreSQL   │                          │
│                 │   (pg0:5433)   │                          │
│                 ├────────────────┤                          │
│                 │ Schema: public │                          │
│                 │ Migrations ✓   │                          │
│                 └────────────────┘                          │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          Control Plane (Next.js) :9999              │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ UI for managing tenants, banks, config              │   │
│  │ Requires: HINDSIGHT_CP_ACCESS_KEY in .env           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    DOCKER (Optional Monitoring)              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        Grafana LGTM (hindsight-monitoring)           │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  • Grafana UI (:3000) - admin/admin                  │  │
│  │  • Tempo (:3200) - Distributed tracing               │  │
│  │  • Loki - Log aggregation                            │  │
│  │  • Prometheus (:9090) - Metrics storage              │  │
│  │  • Pyroscope (:4040) - Continuous profiling          │  │
│  │  • OTLP gRPC (:4317) - Trace ingestion               │  │
│  │  • OTLP HTTP (:4318) - Trace ingestion               │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ▲                                   │
│                          │ (scrapes metrics)                │
│                          │                                   │
│              (OTEL traces - currently DISABLED)             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Communication Flow

1. **User Request** → API (port 8888)
2. **API** → Creates async task in PostgreSQL `async_operations` table
3. **Workers** poll database every 500ms for new tasks
4. **Worker picks task** → Executes using:
   - **Embeddings** via Ollama:11434 (fast lane)
   - **LLM extraction** via Ollama:11435 (heavy lane)
5. **Worker updates task status** → API returns result to user

### WHY This Design?

**Separation of Concerns**:
- API: HTTP handling, authentication, task creation
- Workers: Heavy computation, LLM calls, embeddings
- Ollama: Model serving, inference
- Database: State management, task queue

**Horizontal Scaling**:
- Add more workers → increase throughput
- Add more Ollama instances → increase inference capacity
- API remains lightweight → handles more concurrent requests

**Fault Tolerance**:
- Worker crashes → Task remains in database, picked up by another worker
- Ollama crashes → Workers retry with exponential backoff
- API crashes → Workers continue processing existing tasks

---

## Configuration Hierarchy

### Level 1: Static Configuration (`.env`)

**Purpose**: Server-level defaults for the entire Hindsight instance.

**Who Sets It**: DevOps, system administrators

**Scope**: ALL tenants, ALL banks, ALL users

**Examples**:
```bash
HINDSIGHT_API_HOST=0.0.0.0
HINDSIGHT_API_PORT=8888
HINDSIGHT_API_WORKERS=4
HINDSIGHT_API_LLM_OLLAMA_API_BASE=http://localhost:11435  # LLM lane
HINDSIGHT_API_EMBEDDINGS_OLLAMA_API_BASE=http://localhost:11434  # Embeddings lane
```

**Characteristics**:
- Cannot be changed at runtime
- Requires service restart to apply
- Affects ALL workloads

---

### Level 2: Bank Configuration (Database)

**Purpose**: Per-tenant or per-bank customizations.

**Who Sets It**: Tenant admins via Control Plane UI

**Scope**: Specific bank/tenant only

**Storage**: `bank_configurations` table in PostgreSQL

**Examples**:
```sql
-- Customer A uses expensive GPT-4
INSERT INTO bank_configurations (bank_id, config_key, config_value)
VALUES (1, 'llm_model', 'gpt-4');

-- Customer B uses local Gemma
INSERT INTO bank_configurations (bank_id, config_key, config_value)
VALUES (2, 'llm_model', 'gemma3:12b');
```

**Characteristics**:
- Overrides `.env` defaults
- Can be changed without restart
- Applies immediately to new requests
- Stored in database (survives deployments)

---

### Level 3: Runtime Configuration (API Parameters)

**Purpose**: Per-request overrides for testing, debugging, or special cases.

**Who Sets It**: API users, developers

**Scope**: Single API request only

**Examples**:
```bash
# Use a different model for this one request
curl -X POST http://localhost:8888/api/extract \
  -H "X-LLM-Model: llama3.2:latest" \
  -d '{"text": "..."}'
```

**Characteristics**:
- Highest priority (overrides all)
- Does not persist
- Useful for A/B testing

---

### Configuration Resolution Order

```
Runtime API params  >  Bank config (DB)  >  Static .env
```

**Example**:

`.env` says: `llama3.2:latest`  
Bank config says: `gemma3:12b`  
API request says: `gpt-4`  

**Result**: Request uses `gpt-4`

---

## How to Make It Better

### Phase 2: External LLM APIs (Planned)

**Goal**: Support cloud-based LLMs (OpenAI, Anthropic, Gemini, Groq) in addition to local Ollama.

**Why**: 
- Higher quality (GPT-4, Claude Opus)
- Faster inference (dedicated cloud GPUs)
- No local GPU required

**Configuration**:
```bash
# Uncomment to use Anthropic Claude
HINDSIGHT_API_LLM_PROVIDER=anthropic
HINDSIGHT_API_LLM_API_KEY=sk-ant-...
HINDSIGHT_API_LLM_MODEL=claude-sonnet-4-20250514
```

**Trade-offs**:
- Cost: $0.003/1K tokens (Claude Sonnet) vs free (Ollama)
- Latency: Network roundtrip vs local inference
- Privacy: Data sent to third party vs stays local

---

### Phase 3: Multi-LLM Failover

**Goal**: Automatically fall back to alternative LLMs if primary fails.

**Configuration**:
```bash
# Primary: Anthropic Claude
HINDSIGHT_API_LLM_PROVIDER=anthropic
HINDSIGHT_API_LLM_MODEL=claude-sonnet-4

# Failover 1: Local Ollama (if Claude API down)
HINDSIGHT_API_LLM_1_PROVIDER=ollama
HINDSIGHT_API_LLM_1_BASE_URL=http://localhost:11435
HINDSIGHT_API_LLM_1_MODEL=gemma3:12b

# Failover strategy
HINDSIGHT_API_LLM_STRATEGY='{"mode": "failover", "retry_attempts": 3}'
```

**Benefits**:
- Higher availability (no single point of failure)
- Cost optimization (use cheap local models when possible)
- Quality fallback (try expensive model first, fall back if budget exceeded)

---

### Phase 4: Distributed Tracing (Enable OTEL)

**Goal**: Collect end-to-end traces for dataset generation and ML training.

**Current State**: OTEL is DISABLED (commented out in `.env`)

**To Enable**:
```bash
# Uncomment these lines in .env
HINDSIGHT_API_OTEL_TRACES_ENABLED=true
HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
HINDSIGHT_API_OTEL_SERVICE_NAME=hindsight-dev
HINDSIGHT_API_OTEL_DEPLOYMENT_ENVIRONMENT=development
```

**What This Gives You**:
- **Tempo integration**: All traces flow to Grafana Tempo
- **Span attributes**: LLM inputs, outputs, tokens, latency
- **Trace visualization**: See full request flow (API → Worker → Ollama → DB)
- **Dataset export**: Query Tempo API for ML training data

**Use Cases**:
- Fine-tune local models on production traces
- Analyze LLM performance (which prompts work best)
- Debug slow requests (where is the bottleneck?)

---

### Phase 5: Worker Auto-Scaling

**Goal**: Automatically add/remove workers based on queue depth.

**Current**: Fixed 4 workers (`HINDSIGHT_API_WORKERS=4`)

**Proposed**:
```bash
# Auto-scaling config
HINDSIGHT_API_WORKER_MIN=2
HINDSIGHT_API_WORKER_MAX=16
HINDSIGHT_API_WORKER_SCALE_UP_THRESHOLD=10   # Queue depth > 10 → add worker
HINDSIGHT_API_WORKER_SCALE_DOWN_THRESHOLD=2  # Queue depth < 2 → remove worker
HINDSIGHT_API_WORKER_SCALE_COOLDOWN=60s      # Wait 60s between scaling actions
```

**Benefits**:
- Cost: Use fewer resources during low traffic
- Performance: Handle traffic spikes automatically
- Simplicity: No manual scaling decisions

---

### Phase 6: Model Hot-Swapping

**Goal**: Switch LLM models without restarting services.

**Current**: Changing model requires worker restart (because Python imports cache model)

**Proposed**:
1. API endpoint: `POST /admin/reload-model`
2. Workers watch config changes via database triggers
3. Graceful model swap:
   - Finish current tasks with old model
   - Load new model in parallel
   - Switch atomically
   - Unload old model

**Benefits**:
- Zero downtime model updates
- A/B test different models live
- Respond to quality issues faster

---

## Common Mistakes to Avoid

### ❌ DON'T: Run API in foreground with long timeout

**Why**: Blocks the shell, can cause Docker Desktop to crash if interrupted.

**Bad**:
```bash
uv run hindsight-api --port 8888
# Ctrl+C → Docker Desktop might restart
```

**Good**:
```bash
nohup uv run hindsight-api --port 8888 > /tmp/hindsight-api.log 2>&1 &
```

---

### ❌ DON'T: Use same Ollama endpoint for both LLM and embeddings

**Why**: Connection pool exhaustion, ReadTimeout errors.

**Bad**:
```bash
HINDSIGHT_API_LLM_OLLAMA_API_BASE=http://localhost:11434
HINDSIGHT_API_EMBEDDINGS_OLLAMA_API_BASE=http://localhost:11434  # SAME!
```

**Good**:
```bash
HINDSIGHT_API_LLM_OLLAMA_API_BASE=http://localhost:11435          # LLM lane
HINDSIGHT_API_EMBEDDINGS_OLLAMA_API_BASE=http://localhost:11434  # Embeddings lane
```

---

### ❌ DON'T: Modify `.env` programmatically without documentation

**Why**: Future developers (or yourself in 3 months) won't understand WHY the config changed.

**Bad**:
```bash
# Just change the value
HINDSIGHT_API_WORKERS=8
```

**Good**:
```bash
# ═══════════════════════════════════════════════════════════════════
# CHANGED 2026-06-26: Increased workers from 4 to 8
# Reason: Production load increased 2x after new customer onboarding
# Tested: ./scripts/dev/scale-workers.sh 8 && verified queue depth < 5
# Rollback: ./scripts/dev/scale-workers.sh 4
# ═══════════════════════════════════════════════════════════════════
HINDSIGHT_API_WORKERS=8
```

---

### ❌ DON'T: Delete `tmp/hindsight-workers.state` while workers are running

**Why**: Breaks worker management scripts (can't stop workers gracefully).

**Bad**:
```bash
rm /tmp/hindsight-workers.state  # Workers still running!
./scripts/dev/scale-workers.sh stop  # Can't find PIDs, workers orphaned
```

**Good**:
```bash
./scripts/dev/scale-workers.sh stop  # Stop workers first
# State file auto-deleted by script
```

---

### ❌ DON'T: Start multiple `start-all.sh` instances

**Why**: Port conflicts, duplicate processes, database connection exhaustion.

**Bad**:
```bash
./scripts/dev/start-all.sh --workers 4 &
./scripts/dev/start-all.sh --workers 4 &  # PORT CONFLICT!
```

**Good**:
```bash
# Check status first
./scripts/dev/status.sh

# If already running, scale instead of restart
./scripts/dev/scale-workers.sh 8
```

---

### ❌ DON'T: Assume bank config matches .env

**Why**: Database overrides can change config per-tenant.

**Bad**:
```python
# Assume llama3.2 because that's what .env says
model = "llama3.2:latest"
```

**Good**:
```python
from hindsight_api.config_resolver import get_resolved_config

config = await get_resolved_config(bank_id=1, context="extract")
model = config.llm_model  # Might be gemma3:12b for this bank!
```

---

## Verification Checklist

Use this to verify the system is working correctly:

### ✅ Ollama Split Lanes
```bash
# Check both instances running
lsof -i :11434 -i :11435 | grep LISTEN
# Should show TWO ollama processes

# Test embeddings lane
curl http://localhost:11434/api/embeddings -d '{"model":"nomic-embed-text","prompt":"test"}'

# Test LLM lane
curl http://localhost:11435/api/generate -d '{"model":"llama3.2:latest","prompt":"Say OK"}'
```

### ✅ Workers Running
```bash
./scripts/dev/status.sh
# Should show:
# ✓ Ollama Embeddings (port 11434) - 22 models
# ✓ Ollama LLM (port 11435) - 22 models
# ✓ Hindsight API (port 8888) - RUNNING
# ✓ Workers - 4 healthy
```

### ✅ API Health
```bash
curl http://localhost:8888/health
# Should return: {"status":"healthy","database":"connected"}
```

### ✅ Worker Health
```bash
for port in 9001 9002 9003 9004; do
  curl -s http://localhost:$port/health | jq -r '.status'
done
# Should print "healthy" 4 times
```

### ✅ Monitoring Stack
```bash
docker ps | grep hindsight-monitoring
# Should show container running

curl -s http://localhost:3000 | grep -q "Grafana"
echo $?  # Should print 0 (found)
```

### ✅ End-to-End Test
```bash
# TODO: Add actual extraction test here
# This would verify no ReadTimeout errors occur
```

---

## Resources

- **Monitoring Guide**: `MONITORING_GUIDE.md` - Grafana LGTM setup, tracing, dashboards
- **Ollama Setup**: `OLLAMA_SPLIT_SETUP.md` - Complete split lane configuration guide
- **Tracing Guide**: `TRACING_AND_DATASETS.md` - Dataset collection from traces
- **Services Dashboard**: `SERVICES_DASHBOARD.md` - Quick reference for all endpoints
- **Scripts README**: `scripts/dev/README.md` - Development script documentation
- **Agent Instructions**: `AGENTS.md` - Rules for AI coding agents

---

## Questions & Troubleshooting

### Q: Workers log a different model than .env - is this wrong?

**A**: No! This is correct. Workers use **bank-level config** from the database, which overrides `.env` defaults. Check `bank_configurations` table to see per-tenant settings.

### Q: Docker Desktop keeps stopping - is this breaking Hindsight?

**A**: No. Monitoring (Grafana LGTM) is OPTIONAL. Core Hindsight (API, Workers, Ollama) runs bare metal and works fine without Docker.

### Q: Should I use the same model for LLM and embeddings?

**A**: No. Embeddings require specialized models like `nomic-embed-text`. LLMs like `llama3.2` are for text generation, not embedding.

### Q: Can I use multiple LLM models simultaneously?

**A**: Yes! Use bank-level config to assign different models per tenant. Or use the failover system to chain multiple models.

### Q: How do I enable tracing for dataset collection?

**A**: Uncomment the `OTEL` variables in `.env` (see Phase 4 above), then restart API and workers.

---

**End of ARCHITECTURE_WHY.md**
