# Hindsight Production Deployment Architecture (2026-06-27)

## Problem Statement

Development scripts (`start-services.sh`) are **tightly coupled**: stopping one service crashes the others. Production requires:
- ✅ Independent service lifecycle management
- ✅ Redundancy & automatic failover
- ✅ Horizontal scaling
- ✅ Health checks & self-healing
- ✅ Worker pool with failure isolation

---

## Solution: Kubernetes (Native)

Hindsight ships with **production-ready Helm charts** supporting:

### Architecture (Kubernetes-Native)

```
┌─────────────────────────────────────────────────────┐
│ Kubernetes Cluster                                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  API Deployment (Replicas: N)                      │
│  ├─ Pod 1: hindsight-api (Port 8888)              │
│  ├─ Pod 2: hindsight-api (Port 8888)              │
│  └─ Pod N: hindsight-api (Port 8888)              │
│      └─ Behind: Service (Stable IP, Load Balanced)
│                                                     │
│  Worker StatefulSet (Replicas: M)                  │
│  ├─ Pod 0: hindsight-worker-0                     │
│  ├─ Pod 1: hindsight-worker-1                     │
│  └─ Pod M: hindsight-worker-M                     │
│      └─ Behind: Headless Service (DNS: worker-0, worker-1, ...)
│                                                     │
│  Control Plane Deployment (Replicas: K)           │
│  ├─ Pod 1: hindsight-control-plane                │
│  ├─ Pod 2: hindsight-control-plane                │
│  └─ Pod K: hindsight-control-plane                │
│      └─ Behind: Service (Stable IP, Load Balanced)
│                                                     │
│  PostgreSQL StatefulSet (Replicas: 1+)            │
│  ├─ Primary (Read/Write)                          │
│  └─ Replicas (Read-Only failover)                 │
│                                                     │
│  Optional: TEI (Text Embedding Infrastructure)    │
│  ├─ Embeddings Deployment (Replicas: N)           │
│  └─ Reranker Deployment (Replicas: N)             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Key Features

| Feature | Benefit |
|---------|---------|
| **Deployment** | API scales independently (add/remove replicas without downtime) |
| **StatefulSet** | Workers have stable identities + persistent storage (task recovery) |
| **Service** | Automatic load balancing + DNS discovery |
| **Probes** | Liveness (restart dead pods) + Readiness (remove slow pods from LB) |
| **PDB** | Pod Disruption Budget protects quorum during maintenance |
| **Ingress** | Single entry point, TLS termination, path-based routing |
| **HPA** | Horizontal Pod Autoscaling based on CPU/memory/custom metrics |

---

## Deployment Option 1: Helm (Recommended for Kubernetes)

### Prerequisites

```bash
# Kubernetes cluster (local minikube, GKE, EKS, AKS, etc.)
kubectl version --client

# Helm 3
helm version
```

### Install with Defaults (PostgreSQL Included)

```bash
cd /Users/oliververmeulen/hindsight

# Update Helm dependencies (PostgreSQL subchart, etc.)
helm dependency update ./helm/hindsight

# Install (all defaults, PostgreSQL inside cluster)
export OPENAI_API_KEY="sk-xxx"
helm upgrade hindsight --install ./helm/hindsight \
  -n hindsight --create-namespace \
  --set api.secrets.HINDSIGHT_API_LLM_API_KEY="$OPENAI_API_KEY"

# Verify
kubectl get pods -n hindsight
kubectl port-forward -n hindsight svc/hindsight-api 8888:8888 &
curl http://localhost:8888/health
```

### Install with External PostgreSQL

```bash
# Use existing database (e.g., Cloud SQL, RDS, managed Postgres)
helm upgrade hindsight --install ./helm/hindsight \
  -n hindsight --create-namespace \
  --set api.secrets.HINDSIGHT_API_LLM_API_KEY="$OPENAI_API_KEY" \
  --set postgresql.enabled=false \
  --set postgresql.external.host="my-postgres.example.com" \
  --set postgresql.external.port=5432 \
  --set postgresql.external.database="hindsight" \
  --set postgresql.external.username="hindsight" \
  --set postgresql.external.password="my-db-password"
```

### Scale API (Add Replicas)

API pods are stateless — scale horizontally:

```bash
# Change API replicas from 1 to 3 (zero downtime)
helm upgrade hindsight ./helm/hindsight \
  -n hindsight \
  --reuse-values \
  --set api.replicaCount=3

# Watch rollout
kubectl rollout status deployment/hindsight-api -n hindsight

# Verify
kubectl get pods -n hindsight -l app.kubernetes.io/component=api
```

### Scale Workers (Add More Processing Capacity)

Workers are stateful (task state) but use shared database:

```bash
# Change worker replicas from 2 to 8
helm upgrade hindsight ./helm/hindsight \
  -n hindsight \
  --reuse-values \
  --set worker.replicaCount=8

# Watch rollout (StatefulSet updates sequentially)
kubectl rollout status statefulset/hindsight-worker -n hindsight

# Verify
kubectl get statefulset hindsight-worker -n hindsight
kubectl get pods -n hindsight -l app.kubernetes.io/component=worker
```

### Monitor Services

```bash
# Real-time pod status
kubectl get pods -n hindsight --watch

# Logs from specific pod
kubectl logs -n hindsight hindsight-api-0

# Stream logs from all API pods
kubectl logs -n hindsight -l app.kubernetes.io/component=api -f --all-containers=true

# Port-forward to access locally
kubectl port-forward -n hindsight svc/hindsight-api 8888:8888
kubectl port-forward -n hindsight svc/hindsight-control-plane 3000:3000

# Open in browser
open http://localhost:3000
```

### Upgrade Services (Rolling Update)

```bash
# Update to new version (e.g., from v1.0.0 to v1.1.0)
helm upgrade hindsight ./helm/hindsight \
  -n hindsight \
  --reuse-values \
  --set version=v1.1.0

# Monitor rollout (API pods update one at a time, zero downtime)
kubectl rollout status deployment/hindsight-api -n hindsight
```

### Rollback if Needed

```bash
# Check previous releases
helm history hindsight -n hindsight

# Rollback to previous release
helm rollback hindsight 1 -n hindsight

# Verify
kubectl get pods -n hindsight
```

---

## Deployment Option 2: Docker Compose (Multi-Machine Orchestration)

For environments without Kubernetes, use Docker Compose with `docker-compose-swarm` or `docker stack deploy`.

### Multi-Service Compose (External Database)

```yaml
# docker-compose-prod.yaml
version: '3.8'

services:
  api:
    image: ghcr.io/vectorize-io/hindsight-api:latest
    container_name: hindsight-api
    restart: unless-stopped
    ports:
      - "8888:8888"
    environment:
      HINDSIGHT_API_LLM_PROVIDER: openai
      HINDSIGHT_API_LLM_API_KEY: ${OPENAI_API_KEY}
      HINDSIGHT_API_DATABASE_URL: postgresql://hindsight:${DB_PASSWORD}@postgres:5432/hindsight
      HINDSIGHT_API_WORKER_ENABLED: "false"  # Use dedicated workers
    depends_on:
      - postgres
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8888/health"]
      interval: 10s
      timeout: 5s
      retries: 3

  worker-1:
    image: ghcr.io/vectorize-io/hindsight-api:latest
    container_name: hindsight-worker-1
    restart: unless-stopped
    command: hindsight-worker
    environment:
      HINDSIGHT_API_WORKER_ID: worker-1
      HINDSIGHT_API_DATABASE_URL: postgresql://hindsight:${DB_PASSWORD}@postgres:5432/hindsight
    depends_on:
      - postgres
      - api
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9001/metrics"]
      interval: 10s
      timeout: 5s
      retries: 3

  worker-2:
    image: ghcr.io/vectorize-io/hindsight-api:latest
    container_name: hindsight-worker-2
    restart: unless-stopped
    command: hindsight-worker
    environment:
      HINDSIGHT_API_WORKER_ID: worker-2
      HINDSIGHT_API_DATABASE_URL: postgresql://hindsight:${DB_PASSWORD}@postgres:5432/hindsight
    depends_on:
      - postgres
      - api

  control-plane:
    image: ghcr.io/vectorize-io/hindsight-control-plane:latest
    container_name: hindsight-control-plane
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      HINDSIGHT_CP_DATAPLANE_API_URL: http://api:8888
      NODE_ENV: production
    depends_on:
      - api
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000"]
      interval: 10s
      timeout: 5s
      retries: 3

  postgres:
    image: postgres:17-alpine
    container_name: hindsight-postgres
    restart: unless-stopped
    ports:
      - "5432:5432"
    volumes:
      - hindsight_postgres_data:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: hindsight
      POSTGRES_USER: hindsight
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "hindsight"]
      interval: 10s
      timeout: 5s
      retries: 3

volumes:
  hindsight_postgres_data:

networks:
  default:
    name: hindsight-network
```

**Usage**:

```bash
# Set environment
export OPENAI_API_KEY="sk-xxx"
export DB_PASSWORD="your-secure-password"

# Start services (all in background, independent lifecycle)
docker-compose -f docker-compose-prod.yaml up -d

# Scale workers (add worker-3)
docker-compose -f docker-compose-prod.yaml up -d --scale worker=3

# Monitor
docker-compose -f docker-compose-prod.yaml ps
docker-compose -f docker-compose-prod.yaml logs -f api

# Stop individual service
docker-compose -f docker-compose-prod.yaml stop api
# (workers + control-plane still running)

# Stop all
docker-compose -f docker-compose-prod.yaml down
```

---

## Deployment Option 3: Standalone Docker (Single Container, Embedded DB)

For the simplest production setup without external dependencies:

```bash
export OPENAI_API_KEY="sk-xxx"

docker run -d --name hindsight --restart unless-stopped \
  -p 8888:8888 -p 3000:3000 \
  -e HINDSIGHT_API_LLM_API_KEY="$OPENAI_API_KEY" \
  -v hindsight-data:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest

# Verify
curl http://localhost:8888/health
open http://localhost:3000
```

**Tradeoffs**:
- ✅ Simplest (one container)
- ❌ No redundancy (single failure point)
- ❌ Limited scaling (one API, one embedded worker)
- ✅ Good for: Local development, small deployments, testing

---

## Comparison: Dev vs Production

| Aspect | Development (`start-services.sh`) | Production (Kubernetes) | Production (Docker Compose) |
|--------|------|-------------|-------------------|
| **Lifecycle** | Tightly coupled (all or nothing) | Independent (scale each service) | Independent (scale each service) |
| **Failures** | One crash → all down | Pod dies → auto-restart, other pods live | Container dies → auto-restart, other containers live |
| **Scaling** | Manual script editing | `helm upgrade --set replicaCount=N` | `docker-compose up -d --scale worker=N` |
| **Monitoring** | `./status.sh --watch` | `kubectl get pods --watch` | `docker-compose ps` |
| **Logs** | Tail local files | `kubectl logs -f` | `docker-compose logs -f` |
| **Rolling Update** | Manual restart | `kubectl rollout` (zero downtime) | Manual restart or overlay |
| **Redundancy** | None | Full (replicas, quorum, failover) | Partial (needs load balancer) |

---

## Full Example: Kubernetes on Local Minikube

```bash
# Install minikube (local Kubernetes)
brew install minikube

# Start cluster
minikube start --cpus=4 --memory=8192

# Verify
kubectl cluster-info

# Install Hindsight
cd /Users/oliververmeulen/hindsight
helm dependency update ./helm/hindsight

helm install hindsight ./helm/hindsight \
  -n hindsight --create-namespace \
  --set api.secrets.HINDSIGHT_API_LLM_API_KEY="sk-xxx"

# Wait for all pods ready
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/instance=hindsight \
  -n hindsight --timeout=300s

# Access
kubectl port-forward -n hindsight svc/hindsight-api 8888:8888 &
kubectl port-forward -n hindsight svc/hindsight-control-plane 3000:3000 &

curl http://localhost:8888/health
open http://localhost:3000

# Scale to 3 API replicas
helm upgrade hindsight ./helm/hindsight -n hindsight \
  --reuse-values --set api.replicaCount=3

# Watch scaling
kubectl get pods -n hindsight --watch

# Cleanup
helm uninstall hindsight -n hindsight
minikube stop
```

---

## Key Differences from Development

| Scenario | How Kubernetes Handles It | How Docker Compose Handles It |
|----------|--------------------------|----------------------------|
| API crashes | New pod auto-spawns (readiness probe triggers) | Container restarts (restart policy) |
| Worker dies | StatefulSet recovers task state from DB | Container restarts, reloads task from DB |
| Need 10 more API pods | `helm upgrade --set api.replicaCount=10` (instant DNS update) | Docker Swarm or manual composition |
| New LLM API key | Update secret → pods auto-reload via checksum annotation | `docker-compose up --force-recreate` |
| Database migration | Pods drain gracefully (PDB) before update | Manual drain logic needed |
| Multi-zone failover | Kubernetes spreads pods across zones | Manual multi-host setup needed |

---

## Next Steps

1. **Choose deployment**:
   - Kubernetes (Helm): Enterprise, auto-scaling, multi-zone
   - Docker Compose: Simpler, multi-machine without orchestration
   - Standalone: Minimal dependencies, single-container

2. **Run starter example**:
   ```bash
   cd /Users/oliververmeulen/hindsight
   
   # Option A: Kubernetes
   helm dependency update ./helm/hindsight
   helm install hindsight ./helm/hindsight -n hindsight --create-namespace \
     --set api.secrets.HINDSIGHT_API_LLM_API_KEY="sk-xxx"
   
   # Option B: Docker Compose
   docker-compose -f docker/docker-compose/external-pg/docker-compose.yaml up
   
   # Option C: Standalone
   docker run -d -p 8888:8888 -p 3000:3000 \
     -e HINDSIGHT_API_LLM_API_KEY="sk-xxx" \
     ghcr.io/vectorize-io/hindsight:latest
   ```

3. **Monitor & verify**:
   - Kubernetes: `kubectl get pods --watch`
   - Docker: `docker-compose ps`

4. **Scale independently**:
   - Kubernetes: `helm upgrade --set worker.replicaCount=8`
   - Docker: `docker-compose up -d --scale worker=3`

---

**Key Insight**: Hindsight's **architecture already separates API, workers, and control plane**. Production just chooses the right orchestration layer (Kubernetes, Docker Compose, or standalone) that matches your infrastructure and redundancy needs.
