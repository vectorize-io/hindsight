# Hindsight with CUDA ONNX embeddings

This recipe builds a private image on top of the published slim image. It installs the optional ONNX embedding dependencies plus `onnxruntime-gpu` only in that image and enables `CUDAExecutionProvider` through the opt-in ONNX settings. Official CPU and slim images are unchanged.

The host needs an NVIDIA GPU, a compatible driver, and the NVIDIA Container Toolkit. Build natively on the host that will run the container:

```bash
docker compose -f docker/docker-compose/cuda-onnx/docker-compose.yaml up --build
```

Pin `BASE_IMAGE` to a release tag or digest for repeatable deployments. The base must contain this feature. For an unreleased checkout, first build its API-only base and use that tag:

```bash
docker build --target api-only --build-arg INCLUDE_LOCAL_MODELS=false --build-arg PRELOAD_ML_MODELS=false -f docker/standalone/Dockerfile -t hindsight-onnx-base:dev .
BASE_IMAGE=hindsight-onnx-base:dev docker compose -f docker/docker-compose/cuda-onnx/docker-compose.yaml up --build
```

The development base serves the API on port 8888; published standalone slim images also serve the control plane on port 9999. Set `HINDSIGHT_API_LLM_API_KEY` for your LLM provider before starting. The recipe disables local reranking to avoid installing PyTorch; configure a supported remote reranker if needed.

The recipe targets Linux x86_64 and defaults to ONNX Runtime 1.26.0 with CUDA 12 and cuDNN 9. This pin retains CUDA 12 compatibility: the default PyPI GPU wheels switch to CUDA 13 starting with ORT 1.27. See the [ORT CUDA compatibility documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html). CUDA libraries are installed inside the image; the host needs a compatible NVIDIA driver and GPU passthrough. Set `ONNXRUNTIME_GPU_VERSION` to select another compatible wheel. Transitive NVIDIA dependency versions may change between builds. Set `HINDSIGHT_API_EMBEDDINGS_ONNX_CUDA_DEVICE_ID` to a non-negative GPU index within the container's visible devices (default: 0).

The slim image's document detection dependency (`magika`) requires the CPU `onnxruntime` distribution. CPU and GPU wheels share the same module name and should not overwrite each other in one site-packages directory. The Dockerfile retains the base virtual environment and installs `onnxruntime-gpu` and the NVIDIA dependencies declared by its `cuda`/`cudnn` extras in `/opt/hindsight-onnx-gpu`. `PYTHONPATH` selects that private GPU module, and `LD_LIBRARY_PATH` selects its NVIDIA libraries. The overlay contains only GPU ONNX Runtime and NVIDIA libraries. Shared Python dependencies (including NumPy, protobuf, packaging and flatbuffers) stay in the base virtual environment; the build fails if they do not satisfy the GPU wheel requirements. The build checks their import locations, the overlay package list, Hindsight configuration and embedding imports, and Magika initialization. No PyTorch is installed. The GPU runtime is still a substantial download; this recipe makes no fixed image-size or speedup claim.

At startup, Hindsight checks the provider list, initialized session and warmup inference. If CUDA was explicitly requested but cannot be activated, startup fails. Whole-session fallback is disabled for CUDA sessions; individual unsupported operators may still run on CPU through normal graph partitioning. Use a GPU-compatible ONNX model; quantized graphs can contain operators unsupported by CUDA. A build-time provider check alone does not prove GPU execution.

`HINDSIGHT_API_EMBEDDINGS_ONNX_DEVICE=cpu|cuda` defaults to `cpu`. These settings are process-level configuration, not bank overrides. Restart the API after changing them. Official images and default dependencies are unchanged.

## Tests

Unit tests need neither a GPU nor model downloads:

```bash
uv sync --frozen --package hindsight-api-slim --extra local-onnx --extra embedded-db
.venv/bin/pytest -c hindsight-api-slim/pyproject.toml hindsight-api-slim/tests/test_onnx_embeddings.py hindsight-api-slim/tests/test_onnx_cuda.py -n 0
```

For real inference, download an `intfloat/multilingual-e5-small` snapshot including `onnx/model.onnx` and tokenizer files. Use a compatible GPU runtime and libraries, for example a separate installation directory as in the Dockerfile. Do not install CPU and GPU wheels over each other. Export `PYTHONPATH` and `LD_LIBRARY_PATH` when using a private installation so API subprocesses inherit them.

```bash
export HINDSIGHT_TEST_ONNX_MODEL_DIR=/absolute/path/to/e5-snapshot
export HINDSIGHT_TEST_ONNX_CUDA=1
.venv/bin/pytest -c hindsight-api-slim/pyproject.toml hindsight-api-slim/tests/test_onnx_runtime.py -n 0 -s
uv sync --project hindsight-system-tests --frozen
hindsight-system-tests/.venv/bin/pytest hindsight-system-tests/tests/test_76_onnx_embeddings.py -v
```

Keep the API environment's optional dependencies installed; the system-test API process uses `UV_NO_SYNC=1`. Omit `HINDSIGHT_TEST_ONNX_CUDA` for real CPU-only tests. Explicit GPU tests fail rather than skip if CUDA is unusable. Without a local model directory, real-model tests skip and ordinary unit tests still run.

Runtime tests cover multilingual batching, ordering, normalization, concurrent calls, CPU/CUDA numerical agreement, actual CUDA kernel events from profiling, and invalid device rejection. The public-client system story starts separate CPU/CUDA API processes with a real database, retains multiple facts, waits for background consolidation, recalls semantic results, and replaces a document. LLM and reranker responses are scripted through HTTP; no live LLM service is required.
