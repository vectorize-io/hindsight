# Hindsight Helm Chart

Helm chart for deploying Hindsight - a temporal-semantic-entity memory system for AI agents.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- PostgreSQL database (external or bundled)

## Quick Start

```bash
# Update dependencies first
helm dependency update ./helm/hindsight

# Install (PostgreSQL included by default)
export OPENAI_API_KEY="sk-your-openai-key"
helm upgrade hindsight --install ./helm/hindsight -n hindsight --create-namespace \
  --set api.secrets.HINDSIGHT_API_LLM_API_KEY="$OPENAI_API_KEY"
```

To use an external database instead:

```bash
helm install hindsight ./helm/hindsight -n hindsight --create-namespace \
  --set api.secrets.HINDSIGHT_API_LLM_API_KEY="sk-your-openai-key" \
  --set postgresql.enabled=false \
  --set postgresql.external.host=my-postgres.example.com \
  --set postgresql.external.password=mypassword
```

## Installation

### Add the repository (if published)

```bash
helm repo add hindsight https://your-helm-repo.com
helm repo update
```

### Install with custom values file

Create a `values-override.yaml`:

```yaml
api:
  secrets:
    HINDSIGHT_API_LLM_API_KEY: "sk-your-openai-key"

postgresql:
  external:
    host: "my-postgres.example.com"
    password: "mypassword"
```

Then install:

```bash
helm install hindsight ./helm/hindsight -n hindsight --create-namespace -f values-override.yaml
```

## Configuration

### Key Values

| Parameter | Description | Default |
|-----------|-------------|---------|
| `version` | Default image tag for all components | Chart `appVersion` |
| `api.enabled` | Enable the API component | `true` |
| `api.image.repository` | API image repository | `ghcr.io/vectorize-io/hindsight-api` |
| `api.image.tag` | API image tag (defaults to `version`) | - |
| `api.service.port` | API service port | `8888` |
| `controlPlane.enabled` | Enable the control plane | `true` |
| `controlPlane.image.repository` | Control plane image repository | `ghcr.io/vectorize-io/hindsight-control-plane` |
| `controlPlane.image.tag` | Control plane image tag (defaults to `version`) | - |
| `controlPlane.service.port` | Control plane service port | `3000` |
| `postgresql.enabled` | Deploy PostgreSQL as subchart | `true` |
| `postgresql.external.host` | External PostgreSQL host | `postgresql` |
| `postgresql.external.port` | External PostgreSQL port | `5432` |
| `postgresql.external.database` | Database name | `hindsight` |
| `postgresql.external.username` | Database username | `hindsight` |
| `ingress.enabled` | Enable ingress | `false` |
| `autoscaling.enabled` | Enable HPA | `false` |
| `metrics.serviceMonitor.enabled` | Create ServiceMonitors for api/worker (needs Prometheus operator) | `false` |
| `metrics.serviceMonitor.labels` | Labels for Prometheus operator selection, e.g. `release: kube-prometheus-stack` | `{}` |

### Environment Variables

All environment variables in `api.env` and `controlPlane.env` are automatically added to the respective pods. Sensitive values should go in `api.secrets` or `controlPlane.secrets`.

```yaml
api:
  env:
    HINDSIGHT_API_LLM_PROVIDER: "openai"
    HINDSIGHT_API_LLM_MODEL: "gpt-4"
  secrets:
    HINDSIGHT_API_LLM_API_KEY: "your-api-key"
    HINDSIGHT_API_LLM_BASE_URL: "https://api.openai.com/v1"

controlPlane:
  env:
    NODE_ENV: "production"
  secrets: {}
```

### External Database

To connect to an external PostgreSQL database:

```yaml
postgresql:
  enabled: false
  external:
    host: "my-postgres.example.com"
    port: 5432
    database: "hindsight"
    username: "hindsight"
    password: "your-password"
```

### Sidecars and Init Containers

`api`, `worker`, and `controlPlane` each take `extraContainers` and `extraInitContainers`. Both default to `[]` and render nothing when empty. Use them for containers that have to share the pod, such as a database auth proxy reached over localhost:

```yaml
api:
  extraContainers:
    - name: cloud-sql-proxy
      image: gcr.io/cloud-sql-connectors/cloud-sql-proxy:2.24.1
      args:
        - --port=5432
        - my-project:us-west1:my-instance
      securityContext:
        runAsNonRoot: true

postgresql:
  enabled: false
  external:
    host: "127.0.0.1"
```

### Ingress

To expose the services via ingress:

```yaml
ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: hindsight.example.com
      paths:
        - path: /
          pathType: Prefix
          service: controlPlane
        - path: /api
          pathType: Prefix
          service: api
  tls:
    - secretName: hindsight-tls
      hosts:
        - hindsight.example.com
```

### Prometheus metrics

The api (port 8888) and worker (port 8889) containers expose Prometheus
format metrics at `/metrics`. The control plane does not expose metrics.
On clusters running the Prometheus operator (e.g. kube-prometheus-stack),
enable ServiceMonitor discovery:

```yaml
metrics:
  serviceMonitor:
    enabled: true
    labels:
      release: kube-prometheus-stack  # must match the stack's serviceMonitorSelector
```

The worker monitor requires `worker.enabled: true`. Without the operator,
scrape `svc/<release>-api:8888/metrics` and `svc/<release>-worker:8889/metrics`
directly with annotation-based discovery or static targets.

## Upgrading

```bash
helm upgrade hindsight ./helm/hindsight -n hindsight
```

## Uninstalling

```bash
helm uninstall hindsight -n hindsight
```

## Components

The chart deploys:

- **API**: The main Hindsight API server for memory operations
- **Control Plane**: Web UI for managing agents and viewing memories

## Development

### Lint the chart

```bash
helm lint ./helm/hindsight
```

### Template locally

```bash
helm template hindsight ./helm/hindsight --debug
```

### Dry run installation

```bash
helm install hindsight ./helm/hindsight --dry-run --debug
```
