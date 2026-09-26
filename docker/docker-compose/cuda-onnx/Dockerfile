# Optional CUDA ONNX embeddings, outside the official release pipeline.
# BASE_IMAGE must contain the opt-in ONNX device configuration (issue #4780).
ARG BASE_IMAGE=ghcr.io/vectorize-io/hindsight:latest-slim
FROM $BASE_IMAGE

USER root
# Keep CUDA 12 compatibility; PyPI GPU wheels switch to CUDA 13 in ORT 1.27.
ARG ONNXRUNTIME_GPU_VERSION=1.26.0

# Use the base release's extra to retain its tokenizer/version constraints.
RUN API_VERSION=$(/app/api/.venv/bin/python -c 'from importlib.metadata import version; print(version("hindsight-api-slim"))') \
    && uv pip install --python /app/api/.venv/bin/python --no-cache \
        "hindsight-api-slim[local-onnx]==$API_VERSION" \
    && uv pip check --python /app/api/.venv/bin/python

# magika (document conversion) requires the CPU distribution by package name.
# A GPU wheel in the SAME site-packages would overwrite its Python/native files.
# Only GPU ORT and NVIDIA libraries belong in the overlay. Shared Python
# dependencies must satisfy the wheel requirements in the base virtual environment.
RUN uv pip install --python /app/api/.venv/bin/python --no-cache \
    --no-deps --target /opt/hindsight-onnx-gpu "onnxruntime-gpu==$ONNXRUNTIME_GPU_VERSION"

# Read dependencies from this wheel, checking shared requirements without upgrading
# or shadowing the application environment. Resolve only NVIDIA libraries separately.
COPY runtime_dependencies.py /tmp/onnx-runtime-dependencies.py
RUN /app/api/.venv/bin/python /tmp/onnx-runtime-dependencies.py \
    && rm /tmp/onnx-runtime-dependencies.py
RUN uv pip install --python /app/api/.venv/bin/python --no-cache \
    --target /opt/hindsight-onnx-gpu -r /tmp/onnx-nvidia.txt \
    && rm /tmp/onnx-nvidia.txt

ENV PYTHONPATH=/opt/hindsight-onnx-gpu
# Dynamic linking is explicit: no torch import or host CUDA toolkit is required.
ENV LD_LIBRARY_PATH=/opt/hindsight-onnx-gpu/nvidia/cuda_runtime/lib:/opt/hindsight-onnx-gpu/nvidia/cuda_nvrtc/lib:/opt/hindsight-onnx-gpu/nvidia/cublas/lib:/opt/hindsight-onnx-gpu/nvidia/cudnn/lib:/opt/hindsight-onnx-gpu/nvidia/cufft/lib:/opt/hindsight-onnx-gpu/nvidia/curand/lib:/opt/hindsight-onnx-gpu/nvidia/nvjitlink/lib

# This proves the wheel includes CUDA, not that a GPU is usable during the build.
# Hindsight's startup warm-up separately verifies device/library initialization.
COPY verify_runtime.py /tmp/onnx-verify-runtime.py
RUN /app/api/.venv/bin/python /tmp/onnx-verify-runtime.py \
    && rm /tmp/onnx-verify-runtime.py

USER hindsight
