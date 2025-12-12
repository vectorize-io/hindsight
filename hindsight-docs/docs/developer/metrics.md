# Metrics

Hindsight exposes Prometheus metrics at `/metrics` for monitoring.

```bash
curl http://localhost:8888/metrics
```

## Available Metrics

### Operation Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `hindsight.operation.duration` | Histogram | operation, bank_id, budget, max_tokens, success | Duration of operations in seconds |
| `hindsight.operation.total` | Counter | operation, bank_id, budget, max_tokens, success | Total number of operations executed |

The `operation` label values are: `retain`, `recall`, `reflect`.

### Token Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `hindsight.tokens.input` | Counter | operation, bank_id, budget, max_tokens | Input tokens consumed |
| `hindsight.tokens.output` | Counter | operation, bank_id, budget, max_tokens | Output tokens generated |

## Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'hindsight'
    static_configs:
      - targets: ['localhost:8888']
```
