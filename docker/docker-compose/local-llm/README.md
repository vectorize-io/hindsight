# Hindsight with a local llama.cpp server sidecar

Example Docker Compose setup that runs Hindsight against a **local
llama.cpp server**, fully offline, with no external API key required.

## Architecture

```
┌────────────┐  HTTP /v1/chat/completions  ┌──────────────────────────────┐
│ hindsight  │ ──────────────────────────▶ │ llama.cpp server (sidecar)   │
│ (API + CP) │                             │ ghcr.io/ggml-org/llama.cpp   │
└────────────┘                             └──────────────────────────────┘
```

`llama.cpp` runs as its own container and exposes an OpenAI-compatible
HTTP API. Hindsight talks to it via the standard `openai` LLM provider
with `HINDSIGHT_API_LLM_BASE_URL` pointed at the sidecar.

This pattern follows
[*Hosting llama-server with Docker* (ServiceStack)](https://servicestack.net/posts/hosting-llama-server).

### Why a sidecar and not the in-process `llamacpp` provider?

Hindsight does ship an in-process `llamacpp` provider that spawns
`llama-cpp-python`, but the **published `ghcr.io/vectorize-io/hindsight`
image deliberately omits `llama-cpp-python`** to keep the image small and
avoid bundling native inference libraries that most users don't need.
Trying to set `HINDSIGHT_API_LLM_PROVIDER=llamacpp` against the published
image fails with `ModuleNotFoundError: No module named 'llama_cpp'`.

The sidecar approach side-steps that entirely: the official llama.cpp
image is used as-is for inference, Hindsight is used as-is for memory.
Clean separation, no derived images.

## Quick start

```bash
docker compose -f docker/docker-compose/local-llm/docker-compose.yaml up
```

- API: http://localhost:8888
- Control Plane: http://localhost:9999

**First boot downloads ~3.5 GB** (Gemma 4 E2B Q4_K_M GGUF) into the
`llama_models` named volume. Subsequent boots reuse it.

Hindsight only starts after llama.cpp's `/health` endpoint reports
healthy, so the API will appear "stuck" for a few minutes on the first
run while the model downloads.

## Using a different model

Override the HuggingFace repo / file in `docker-compose.yaml`:

```yaml
environment:
  LLAMA_ARG_HF_REPO: bartowski/Qwen2.5-7B-Instruct-GGUF
  LLAMA_ARG_HF_FILE: Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

Also update `HINDSIGHT_API_LLM_MODEL` on the `hindsight` service to a
matching alias (the value is sent to llama-server as the OpenAI `model`
field — llama-server is lenient about this but it shows up in logs).

## GPU acceleration

The default compose file targets CPU because not everyone has a GPU. On
CPU, Gemma 4 E2B runs at ~2-3 tokens/sec — fine for a smoke test, but the
retain pipeline (which makes several multi-hundred-token LLM calls per
memory) will time out against Hindsight's default LLM timeout. **For any
real use, run on a GPU.**

### NVIDIA

1. Switch the `llama` service image from `:server` to `:server-cuda`.
2. Uncomment the `LLAMA_ARG_N_GPU_LAYERS: "999"` env var (offload all
   layers to GPU).
3. Uncomment the `deploy.resources.reservations.devices` block.
4. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   on the host.

The compose file has all four spots marked with inline comments.

### Apple Silicon / ROCm / Vulkan

The official `ghcr.io/ggml-org/llama.cpp` image only ships CPU and CUDA
variants. For Metal (Apple Silicon), ROCm (AMD), or Vulkan backends,
build llama.cpp yourself with the appropriate flags and reference the
image you build instead. Docker Desktop on macOS cannot pass through the
host GPU to a Linux container in any case — for Apple Silicon, run
llama-server directly on the host and only put Hindsight in Docker.

## Caveats

- llama.cpp's HTTP API is OpenAI-compatible but not 100% feature-parity.
  Function/tool calling support depends on the chat template baked into
  the GGUF; some retain/reflect flows may behave differently than against
  a hosted OpenAI model.
- Small GGUFs (~3 B params) are useful for smoke testing but will
  underperform a hosted frontier model on retain quality. Use a larger
  GGUF (7-13 B params) for production-quality memory.
- The `llama_models` named volume persists the GGUF across `docker
  compose down`/`up` so the model is downloaded once, not every restart.
