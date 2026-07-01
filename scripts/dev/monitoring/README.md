# CollabMind Dev Observability Stack

Docker-based observability engines that feed the Central API and Operator Panel.
You never open these directly — the CollabMind Control Panel queries their APIs.

## Quick Start

```bash
# Core only — Grafana LGTM (metrics, logs, traces)
./scripts/dev/monitoring/start.sh core

# Recommended dev stack — core + traces + LLM observability
./scripts/dev/monitoring/start.sh full

# Full AI observability — adds Phoenix evaluation lab
./scripts/dev/monitoring/start.sh llm-obs
```

## Profiles

| Profile | Services | Use Case |
|---------|----------|----------|
| `core` | Grafana LGTM | Minimal — just metrics/logs/traces |
| `full` | core + Jaeger + Langfuse V2 | **Recommended dev stack** |
| `llm-obs` | full + Phoenix | Full AI evaluation stack |

## Services

### Grafana LGTM (always on)
- **Ports**: `:3000` (Grafana UI), `:4317` (OTLP gRPC), `:4318` (OTLP HTTP)
- Traces via Tempo, metrics via Prometheus/Mimir, logs via Loki
- Pre-configured dashboards in `monitoring/grafana/dashboards/`

### Trace Explorer (profile: full)
- **Ports**: `:16686` (dev UI), `:14317` (OTLP gRPC), `:14318` (OTLP HTTP)
- Container: `collabmind-traces`
- OTLP endpoints mapped to avoid conflict with LGTM

### LLM Observability (profile: full)
- **Port**: `:3002`
- Container: `collabmind-llm-obs`
- Uses host PostgreSQL (`langfuse` database on `:5432`)
- Langfuse V2 (no ClickHouse required)

### AI Evaluation Lab (profile: llm-obs)
- **Port**: `:6006`
- Container: `collabmind-ai-eval`
- Arize AI Phoenix for LLM evaluation and debugging

## Architecture

```
Operator Panel ──→ Central API ──→ Observability Engines (Docker)
```
The Operator Panel never talks to these engines directly — all routing goes through the Central API (Verify-Once-at-Edge pattern).

## Configure Hindsight API for Tracing

```bash
HINDSIGHT_API_OTEL_TRACES_ENABLED=true
HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

## Stop

```bash
cd scripts/dev/monitoring && docker compose down
# Or with profile:
cd scripts/dev/monitoring && docker compose --profile full down
```

## Ports

| Port | Service | Profile |
|------|---------|---------|
| 3000 | Grafana LGTM | core+ |
| 3002 | LLM Observability | full+ |
| 4317 | OTLP gRPC | core+ |
| 4318 | OTLP HTTP | core+ |
| 6006 | AI Evaluation Lab | llm-obs+ |
| 16686 | Trace Explorer | full+ |
| 14317 | OTLP gRPC (traces) | full+ |
| 14318 | OTLP HTTP (traces) | full+ |
