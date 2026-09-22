
# Services

Hindsight consists of three services that can run together or separately depending on your deployment needs.

**Figure: Services.** An animated diagram on the docs site; its narration, step by step:

- **retain · async**
  1. A retain is slow work: an LLM has to read it. With async=true the API does not do it in the request.
  2. It writes a task row into async_operations — the queue is just a table in the same database.
  3. And answers right away with the operation id.
  4. Every worker polls the table (every 500 ms). The claim locks the row, so exactly one worker gets it — the others skip it.
  5. The worker calls the LLM to extract facts and embeds them.
  6. Then writes them to the bank.
  7. The task is marked completed. New facts queue their own follow-up: consolidation (when auto-consolidation is on), plus graph and index upkeep.
  8. The Control Plane talks to the same API, so you can watch the operation there.
- **background**
  1. Background work goes through the same queue. This time worker-2 wins the claim.
  2. Consolidation asks the LLM to merge new facts into observations.
  3. Mental models that refresh after consolidation — and are now stale — get a refresh task of their own.
  4. The API’s internal worker claims from the same table. Set HINDSIGHT_API_WORKER_ENABLED=false and only dedicated workers do.
  5. Any worker can run any task: they share the package, the image and the database, so you add more to scale.
- **recall · sync**
  1. Later, a recall. Like reflect and any read, it is answered inside the API process. Nothing is queued.
  2. The query is embedded — by a model loaded in the process itself, or a remote embeddings service.
  3. The four searches run as queries against PostgreSQL, where every bank lives.
  4. Candidates are reranked by the cross-encoder, local or remote like the embeddings.
  5. The API keeps no state of its own, so you can run as many copies as you like behind a load balancer.

## API Service

The core memory engine. Handles all memory operations:

- **Retain**: Ingests content, extracts facts, builds knowledge graph
- **Recall**: Semantic search across memories
- **Reflect**: Disposition-aware answer generation

```bash
hindsight-api        # Default port: 8888
```

The API service is stateless and can be horizontally scaled behind a load balancer. All state is stored in PostgreSQL.

By default, the API also processes background tasks (mental model consolidation) internally. For high-throughput deployments, you can disable this and run dedicated workers instead.

## Worker Service

Dedicated task processor for background operations. Uses the **same package and Docker image** as the API service, just with a different entry point.

```bash
hindsight-worker     # Default metrics port: 8889
```

Workers use PostgreSQL as a task broker, polling for pending tasks. Multiple workers can run simultaneously without conflicts.

| Deployment | Internal Worker | Dedicated Workers |
|------------|-----------------|-------------------|
| **Development** | ✅ Simple, all-in-one | ❌ Overkill |
| **Small production** | ✅ Less infrastructure | ❌ Overkill |
| **High throughput** | ❌ API bottleneck | ✅ Scale independently |
| **Long-running tasks** | ❌ Blocks API resources | ✅ Isolated processing |

To use dedicated workers, disable the internal worker in the API and start worker processes:

```bash
# Disable internal worker in API
HINDSIGHT_API_WORKER_ENABLED=false hindsight-api

# Start dedicated workers (run multiple instances)
hindsight-worker --worker-id worker-1
hindsight-worker --worker-id worker-2
```

Each worker exposes `/health/live` (liveness, no database access), `/health` and
`/health/ready` (readiness, checks the database), and `/metrics` for monitoring.
See [Monitoring - Health Endpoints](./monitoring#health-endpoints).

Before scaling down or removing workers, release their tasks with `hindsight-admin decommission-worker <worker-id>`.

See [Configuration - Distributed Workers](./configuration#distributed-workers) for all worker settings and [Installation - Helm](./installation#distributed-workers) for Kubernetes deployment.

## Control Plane

Web UI for managing and exploring your memory banks:

- Browse agents and memory banks
- Explore entities and relationships
- View ingestion history and operations
- Test recall queries interactively

The Control Plane connects to the API service and provides a visual interface for development and debugging.

For bare metal deployments, you can run the Control Plane standalone using npx. See [Installation - Bare Metal](./installation#control-plane) for details.
