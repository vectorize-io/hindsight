# Hindsight with NVIDIA CUDA GPU Acceleration

Example setup that builds a custom Hindsight image with **CUDA-enabled PyTorch**
for NVIDIA GPU-accelerated local embeddings and reranking.

## When to use this

- You want to use in-process local embeddings (`HINDSIGHT_API_EMBEDDINGS_PROVIDER: local`)
  and reranking (`HINDSIGHT_API_RERANKER_PROVIDER: local`) with NVIDIA GPU acceleration.
- You want lower latency and higher throughput for local embedding and reranker inference.
- You have an NVIDIA GPU and want to run Hindsight locally without external TEI sidecars.

> [!NOTE]
> This accelerates Hindsight's in-process PyTorch embedding and reranker models.
> The LLM (used for retain/recall/reflect) is external by default (e.g. OpenAI, Anthropic, Ollama, vLLM).

The CUDA runtime is added on top of the full image, whose CPU PyTorch wheel stays
in the base layers. Expect the result to be **roughly 11 GB on disk**, against
~9 GB for the base image it builds on.

## Prerequisites

1. **NVIDIA GPU** with compatible driver (driver version `>= 525.60.13` recommended for CUDA 12.x).
2. **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**
   installed and configured on the host Docker daemon.
3. PyTorch ships CUDA wheels for both `x86_64` and `aarch64`, so either architecture
   works. Build on the machine that will run the image — an emulated
   cross-architecture build cannot reach the GPU.

Verify GPU access in Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

## Quick start

```bash
export HINDSIGHT_API_LLM_API_KEY=sk-xxx

docker compose -f docker/docker-compose/cuda/docker-compose.yaml up --build
```

- API: http://localhost:8888
- Control Plane: http://localhost:9999

To build against a pinned release rather than `latest`, set `HINDSIGHT_VERSION`:

```bash
HINDSIGHT_VERSION=0.9.2 docker compose -f docker/docker-compose/cuda/docker-compose.yaml up --build
```

## Building manually

You can also build the image directly using `docker build`:

```bash
docker build -t hindsight:cuda docker/docker-compose/cuda/
```

Then run the container with GPU passthrough:

```bash
docker run --gpus all \
  --name hindsight-cuda \
  -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=sk-xxx \
  hindsight:cuda
```

## Verifying CUDA GPU Acceleration

The build itself fails if the CUDA wheel did not land, so a successful build
already proves PyTorch has a CUDA runtime. To confirm the models actually loaded
onto the GPU, check the container logs:

```bash
docker logs hindsight-cuda | grep -i "device:"
```

Both the embedding and the reranker provider report their device on startup, and
both should read `device: cuda` rather than `device: cpu`:

```
Embeddings: local provider initialized (dim: 384, device: cuda)
Reranker: local provider initialized (device: cuda, max_concurrent=4)
```

You can also query PyTorch directly inside the running container:

```bash
docker exec hindsight-cuda python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## Tuning

- `HINDSIGHT_API_RERANKER_LOCAL_FP16` — half-precision reranking, enabled in the
  compose file above. Measurably faster on GPU and quality-identical; it is off by
  default only because some CPUs lack native FP16 support.
- `HINDSIGHT_API_RERANKER_LOCAL_BATCH_SIZE` — optimal batch size varies by GPU and
  model; worth tuning if reranking dominates your recall latency.

See [Configuration](https://hindsight.vectorize.io/developer/configuration) for the full set of knobs.
