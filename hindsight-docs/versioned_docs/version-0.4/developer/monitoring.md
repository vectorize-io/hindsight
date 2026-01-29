# Monitoring

Hindsight provides comprehensive monitoring through Prometheus metrics and pre-built Grafana dashboards.

## Local Development

For local metrics visualization, a convenience script downloads and runs Prometheus and Grafana:

```bash
./scripts/dev/start-monitoring.sh
```

This will start:
- **Grafana**: http://localhost:8890 (anonymous access enabled)
- **Prometheus**: http://localhost:8889
- **API Metrics**: http://localhost:8888/metrics

:::note Production Deployment
The local monitoring script is for development only. In production, you need to install and configure Prometheus and Grafana separately, then point Prometheus to scrape your Hindsight API's `/metrics` endpoint.
:::

## Grafana Dashboards

Pre-built dashboards are available in [`monitoring/grafana/dashboards/`](https://github.com/anthropics/hindsight/tree/main/monitoring/grafana/dashboards). Import these JSON files into your Grafana instance:

| Dashboard | Description |
|-----------|-------------|
| **Hindsight Operations** | Operation rates, latency percentiles, per-bank metrics |
| **Hindsight LLM Metrics** | LLM calls, token usage, latency by scope/provider |
| **Hindsight API Service** | HTTP requests, error rates, DB pool, process metrics |

The dashboards are automatically provisioned when using the monitoring stack script.

## Metrics Endpoint

Hindsight exposes Prometheus metrics at `/metrics`:

```bash
curl http://localhost:8888/metrics
```

## Available Metrics

### Operation Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `hindsight.operation.duration` | Histogram | operation, bank_id, source, budget, max_tokens, success | Duration of operations in seconds |
| `hindsight.operation.total` | Counter | operation, bank_id, source, budget, max_tokens, success | Total number of operations executed |

**Labels:**
- `operation`: Operation type (`retain`, `recall`, `reflect`)
- `bank_id`: Memory bank identifier
- `source`: Where the operation was triggered from (`api`, `reflect`, `internal`)
- `budget`: Budget level if specified (`low`, `mid`, `high`)
- `max_tokens`: Max tokens if specified
- `success`: Whether the operation succeeded (`true`, `false`)

The `source` label allows distinguishing between:
- `api`: Direct API calls from clients
- `reflect`: Internal recall calls made during reflect operations
- `internal`: Other internal operations

### LLM Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `hindsight.llm.duration` | Histogram | provider, model, scope, success | Duration of LLM API calls in seconds |
| `hindsight.llm.calls.total` | Counter | provider, model, scope, success | Total number of LLM API calls |
| `hindsight.llm.tokens.input` | Counter | provider, model, scope, success, token_bucket | Input tokens for LLM calls |
| `hindsight.llm.tokens.output` | Counter | provider, model, scope, success, token_bucket | Output tokens from LLM calls |

**Labels:**
- `provider`: LLM provider (`openai`, `anthropic`, `gemini`, `groq`, `ollama`, `lmstudio`)
- `model`: Model name (e.g., `gpt-4`, `claude-3-sonnet`)
- `scope`: What the LLM call is for (`memory`, `reflect`, `consolidation`, `answer`)
- `success`: Whether the call succeeded (`true`, `false`)
- `token_bucket`: Token count bucket for cardinality control (`0-100`, `100-500`, `500-1k`, `1k-5k`, `5k-10k`, `10k-50k`, `50k+`)

### HTTP Request Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `hindsight.http.duration` | Histogram | method, endpoint, status_code, status_class | Duration of HTTP requests in seconds |
| `hindsight.http.requests.total` | Counter | method, endpoint, status_code, status_class | Total number of HTTP requests |
| `hindsight.http.requests.in_progress` | UpDownCounter | method, endpoint | Number of HTTP requests currently being processed |

**Labels:**
- `method`: HTTP method (`GET`, `POST`, `PUT`, `DELETE`)
- `endpoint`: Request path (normalized to reduce cardinality - UUIDs replaced with `{id}`)
- `status_code`: HTTP status code (`200`, `400`, `500`, etc.)
- `status_class`: Status code class (`2xx`, `4xx`, `5xx`)

### Database Pool Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `hindsight.db.pool.size` | Gauge | - | Current number of connections in the pool |
| `hindsight.db.pool.idle` | Gauge | - | Number of idle connections in the pool |
| `hindsight.db.pool.min` | Gauge | - | Minimum pool size |
| `hindsight.db.pool.max` | Gauge | - | Maximum pool size |

### Process Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `hindsight.process.cpu.seconds` | Gauge | type | Process CPU time in seconds |
| `hindsight.process.memory.bytes` | Gauge | type | Process memory usage in bytes |
| `hindsight.process.open_fds` | Gauge | - | Number of open file descriptors |
| `hindsight.process.threads` | Gauge | - | Number of active threads |

**Labels:**
- `type` (CPU): `user` or `system`
- `type` (Memory): `rss_max` (maximum resident set size)

### Histogram Buckets

Custom bucket boundaries are configured for better percentile accuracy:

**Operation Duration Buckets (seconds):**
```
0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 60.0, 120.0
```

**LLM Duration Buckets (seconds):**
```
0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0
```

**HTTP Duration Buckets (seconds):**
```
0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0
```

## Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'hindsight'
    static_configs:
      - targets: ['localhost:8888']
```

## Example Queries

### Average operation latency by type
```promql
rate(hindsight_operation_duration_sum[5m]) / rate(hindsight_operation_duration_count[5m])
```

### LLM calls per minute by provider
```promql
rate(hindsight_llm_calls_total[1m]) * 60
```

### P95 LLM latency
```promql
histogram_quantile(0.95, rate(hindsight_llm_duration_bucket[5m]))
```

### Total tokens consumed by model
```promql
sum by (model) (hindsight_llm_tokens_input_total + hindsight_llm_tokens_output_total)
```

### Internal vs API recall operations
```promql
sum by (source) (rate(hindsight_operation_total{operation="recall"}[5m]))
```

### HTTP requests per second by endpoint
```promql
sum by (endpoint) (rate(hindsight_http_requests_total[1m]))
```

### HTTP error rate (5xx)
```promql
sum(rate(hindsight_http_requests_total{status_class="5xx"}[5m])) / sum(rate(hindsight_http_requests_total[5m]))
```

### P95 HTTP latency
```promql
histogram_quantile(0.95, sum by (le) (rate(hindsight_http_duration_seconds_bucket[5m])))
```

### Database pool utilization
```promql
hindsight_db_pool_size / hindsight_db_pool_max
```

### Active database connections
```promql
hindsight_db_pool_size - hindsight_db_pool_idle
```

### CPU usage rate
```promql
rate(hindsight_process_cpu_seconds{type="user"}[1m])
```
