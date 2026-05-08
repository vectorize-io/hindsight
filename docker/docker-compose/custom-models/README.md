# Hindsight with Custom Local Models

Example Docker Compose setup that builds a Hindsight image with **non-default
local embedder and reranker models baked in at build time**.

This is the recommended pattern for production when you use a non-default
local model: the container registry caches model layers per node, pod
startup is deterministic, and you don't need a model-cache PVC (or any
runtime dependency on HuggingFace).

## When to use this

- You override `HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL` or
  `HINDSIGHT_API_RERANKER_LOCAL_MODEL` to a non-default model.
- You want pod startup to be deterministic and offline-capable.
- You'd otherwise reach for a Helm `modelCache` PVC just to avoid
  re-downloading models.

If you're using the **default** local models, the published full image
(`ghcr.io/vectorize-io/hindsight:latest`) already bakes them in — you don't
need this example.

If you're using **external** providers (TEI, OpenAI, Cohere, ...) for
embeddings and reranking, use the slim image directly — no models are
needed in the image.

## Quick start

```bash
export OPENAI_API_KEY=sk-xxx

docker compose -f docker/docker-compose/custom-models/docker-compose.yaml up --build
```

- API: http://localhost:8888
- Control Plane: http://localhost:9999

## Using your own models

Override the build args to bake different models:

```bash
docker compose -f docker/docker-compose/custom-models/docker-compose.yaml build \
  --build-arg EMBEDDER=your-org/your-embedder \
  --build-arg RERANKER=your-org/your-reranker
```

Then update the matching `HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL` and
`HINDSIGHT_API_RERANKER_LOCAL_MODEL` values in `docker-compose.yaml` so the
runtime points at the same model IDs.

## Verifying the models are baked in

`docker-compose.yaml` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
so that any attempt to download a model at runtime fails loudly instead of
silently re-downloading. If the container starts and serves recall queries
with these set, the models are correctly baked in.

You can also inspect the image directly:

```bash
docker run --rm --entrypoint sh hindsight-custom-models-hindsight \
  -c 'ls ~/.cache/huggingface/hub/'
```

## Why not a model-cache PVC?

The Helm chart exposes an optional `api.persistence.modelCache` PVC for
caching downloaded models across pod restarts. Compared to baking models
into the image:

- A PVC adds storage cost — one PVC per worker replica with
  `volumeClaimTemplates`.
- `ReadWriteOnce` (the default) pins pods to a node.
- The PVC needs lifecycle management on `helm uninstall` / `helm upgrade`
  — without `helm.sh/resource-policy: keep` it is deleted on uninstall;
  with it, storage keeps billing forever until manually cleaned up.
- Pod startup still depends on HuggingFace being reachable on first run.

Image layers, by contrast, are pulled once per node and cached for free by
the container runtime, with no orphaned-storage cleanup story.
