# Hindsight Monitoring Stack

Docker-based monitoring stack using **Grafana LGTM** (Loki, Grafana, Tempo, Mimir) for complete observability.

## Quick Start

```bash
# Start the monitoring stack
./scripts/dev/start-monitoring.sh

# Or manually with docker-compose
cd scripts/dev/monitoring && docker-compose up -d
```

## Access

- **Grafana UI**: http://localhost:3000
  - No login required (anonymous admin enabled for dev)

## Features

- **Traces**: OpenTelemetry traces with GenAI semantic conventions (Tempo)
- **Metrics**: Prometheus scraping of Hindsight API `/metrics` endpoint
- **Logs**: Loki log aggregation (future)
- **Dashboards**: Pre-configured dashboards from `monitoring/grafana/dashboards/`:
  - Hindsight Operations
  - Hindsight LLM Metrics
  - Hindsight API Service

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

## View Data

### Traces
1. Open http://localhost:3000
2. Go to **Explore** (compass icon)
3. Select **Tempo** as data source
4. Click "Search" to see recent traces

### Metrics & Dashboards
1. Open http://localhost:3000
2. Go to **Dashboards** (dashboard icon)
3. Browse the Hindsight folder

### Raw Metrics
- Prometheus metrics: http://localhost:8888/metrics
- PromQL queries: Explore → Prometheus

## Ports

| Port | Service |
|------|---------|
| 3000 | Grafana UI |
| 4317 | OTLP gRPC endpoint |
| 4318 | OTLP HTTP endpoint |

## Stop

```bash
cd scripts/dev/monitoring && docker-compose down
```

## Architecture

- **Single Container**: Grafana LGTM (~515MB) provides all observability components
- **Auto-provisioned Dashboards**: Dashboards from `monitoring/grafana/dashboards/` are automatically loaded
- **Prometheus Scraping**: Configured to scrape Hindsight API at `host.docker.internal:8888/metrics` every 5 seconds
- **Network**: Uses `hindsight-network` (shared with API for future service-to-service tracing)
