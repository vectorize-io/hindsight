# Hindsight with TEI embeddings + reranker

Example Docker Compose setup that serves **embeddings and reranking from two
[HuggingFace Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference)
sidecars** instead of the in-process local models.

Because embeddings and reranking run outside the API, Hindsight itself needs
no baked-in models, so this uses the **slim** image
(`ghcr.io/vectorize-io/hindsight:latest-slim`). Only the LLM — used for
retain/recall/reflect — still needs a provider and API key.

## When to use this

- You want embeddings/reranking on a dedicated, independently scalable
  inference server (e.g. a GPU node) rather than in the API process.
- You run the **slim** image and pull embeddings/reranking from an external
  service.
- You want a self-hosted, offline-capable alternative to a cloud embeddings
  provider (OpenAI, Cohere, ...).

If you just want local models in-process, use the default full image — no
sidecars required.

## What it runs

| Service         | Image                                                  | Model                                    |
| --------------- | ------------------------------------------------------ | ---------------------------------------- |
| `tei-embedding` | `ghcr.io/huggingface/text-embeddings-inference:cpu-1.8.3` | `BAAI/bge-small-en-v1.5` (384-dim)       |
| `tei-reranker`  | `ghcr.io/huggingface/text-embeddings-inference:cpu-1.8.3` | `BAAI/bge-reranker-base`                 |
| `hindsight`     | `ghcr.io/vectorize-io/hindsight:latest-slim`           | — (slim; talks to the sidecars)          |

This is a prod-like configuration: the embedding model is Hindsight's default
(`bge-small-en-v1.5`), the reranker is the `bge-reranker-base` cross-encoder
commonly paired with it on dedicated inference servers, and both services carry
throughput flags (`--max-concurrent-requests`, `--max-batch-tokens`,
`--max-client-batch-size`) tuned for sustained multi-client load instead of
TEI's bare defaults. The API points at the sidecars with:

```
HINDSIGHT_API_EMBEDDINGS_PROVIDER=tei
HINDSIGHT_API_EMBEDDINGS_TEI_URL=http://tei-embedding:80
HINDSIGHT_API_RERANKER_PROVIDER=tei
HINDSIGHT_API_RERANKER_TEI_URL=http://tei-reranker:80
```

## Quick start

```bash
export HINDSIGHT_API_LLM_API_KEY=sk-xxx

docker compose -f docker/docker-compose/tei/docker-compose.yaml up
```

- API: http://localhost:8888
- Control Plane: http://localhost:9999
- TEI embedding server: http://localhost:8080 (exposed for debugging)
- TEI reranker server: http://localhost:8081 (exposed for debugging)

`hindsight` waits (via `depends_on: service_healthy`) until both TEI servers
report healthy, so the first boot pauses while each model downloads into its
`tei_*_cache` volume. Subsequent boots reuse the cached models.

To use an LLM provider other than the default `openai`:

```bash
export HINDSIGHT_API_LLM_PROVIDER=gemini
export HINDSIGHT_API_LLM_API_KEY=...
docker compose -f docker/docker-compose/tei/docker-compose.yaml up
```

## Using your own models

Change the `--model-id` in each service's `command` to any TEI-supported
model. The embedding dimension is **auto-detected from the server** and the
pgvector schema is adjusted to match on first boot — no dimension env var to
set. (If you switch the embedding model after data already exists, start from
a fresh `pg_data` volume, since the stored vectors were built for the old
dimension.)

## Verifying the servers

```bash
# Health
curl 127.0.0.1:8080/health && curl 127.0.0.1:8081/health

# Embedding (returns a 384-length vector for the default model)
curl 127.0.0.1:8080/embed -H 'content-type: application/json' \
     -d '{"inputs":"hello world"}'

# Rerank
curl 127.0.0.1:8081/rerank -H 'content-type: application/json' \
     -d '{"query":"what is the capital of France?","texts":["Paris is the capital of France.","Bananas are yellow."]}'
```

## Apple Silicon / arm64

The `cpu-1.8.3` TEI images are published for `linux/amd64` only. On an
Apple Silicon Mac, run under emulation:

```bash
export DOCKER_DEFAULT_PLATFORM=linux/amd64
docker compose -f docker/docker-compose/tei/docker-compose.yaml up
```

Emulated startup is slow (model load takes a few minutes). For production,
run on `amd64` hosts — or a GPU node with the CUDA-tagged TEI image and a GPU
reservation.
