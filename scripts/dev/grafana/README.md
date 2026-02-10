# Grafana LGTM Stack for Local Development

Single-container observability stack with **Loki** (logs), **Grafana** (visualization), **Tempo** (traces), and **Mimir** (metrics).

## Quick Start

```bash
# Start the stack
./scripts/dev/start-grafana.sh

# Or manually with docker-compose
cd scripts/dev/grafana && docker-compose up -d
```

## Access

- **Grafana UI**: http://localhost:3000
  - No login required (anonymous admin enabled for dev)
  - Username/Password (if needed): `admin` / `admin`

## Configure Hindsight API

Set these environment variables in your `.env`:

```bash
# Enable tracing
HINDSIGHT_API_OTEL_TRACES_ENABLED=true

# Grafana Tempo OTLP endpoint (HTTP)
HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

# Optional: Custom service name
HINDSIGHT_API_OTEL_SERVICE_NAME=hindsight-api

# Optional: Deployment environment
HINDSIGHT_API_OTEL_DEPLOYMENT_ENVIRONMENT=development
```

## View Traces

1. Open Grafana: http://localhost:3000
2. Go to **Explore** (compass icon in sidebar)
3. Select **Tempo** as the data source
4. Run a Hindsight operation (retain, recall, reflect)
5. Click "Search" to see recent traces
6. Click on a trace to see the full span hierarchy

## View Metrics & Dashboards

1. Open Grafana: http://localhost:3000
2. Go to **Dashboards** (dashboard icon in sidebar)
3. Open **Hindsight GenAI Metrics** dashboard
4. View metrics:
   - LLM call rates and durations
   - Token usage (input/output)
   - Operation rates and durations (retain, recall, reflect, consolidation)

**Note**: Metrics require the Hindsight API to be running on `localhost:8888`. Prometheus scrapes the `/metrics` endpoint every 10 seconds.

### Explore Raw Metrics

1. Go to **Explore** (compass icon)
2. Select **Prometheus** as data source
3. Query examples:
   ```promql
   # LLM call rate by scope
   rate(hindsight_llm_calls_total[5m])

   # P95 operation latency
   histogram_quantile(0.95, rate(hindsight_operation_duration_seconds_bucket[5m]))

   # Token usage rate
   rate(hindsight_llm_input_tokens_total[5m])
   ```

## Features

- **Traces**: Full OpenTelemetry trace support with GenAI semantic conventions
- **Metrics**: Prometheus scraping of Hindsight API `/metrics` endpoint
- **Dashboards**: Pre-configured GenAI dashboard with LLM metrics, token usage, and latencies
- **Logs**: Loki log aggregation (future)
- **Single Container**: Everything in one Docker image (~515MB)

## Ports

| Port | Service |
|------|---------|
| 3000 | Grafana UI |
| 4317 | OTLP gRPC endpoint |
| 4318 | OTLP HTTP endpoint |

## Stop

```bash
cd scripts/dev/grafana && docker-compose down
```

## Learn More

- [Grafana docker-otel-lgtm](https://github.com/grafana/docker-otel-lgtm)
- [Grafana Tempo Documentation](https://grafana.com/docs/tempo/latest/)
- [OpenTelemetry with Grafana](https://grafana.com/docs/opentelemetry/)
