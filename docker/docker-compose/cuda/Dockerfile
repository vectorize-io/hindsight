# Example: build a custom Hindsight image with CUDA-enabled PyTorch
# for NVIDIA GPU-accelerated local embedding and reranker models.
#
# Based on the full standalone image (which already bakes in the default
# BAAI/bge-small-en-v1.5 and cross-encoder/ms-marco-MiniLM-L-6-v2 models).
# The CPU torch wheel stays in the base layers, so the CUDA runtime is purely
# additive: expect roughly 11 GB on disk against ~9 GB for the base image.
#
# Requirements on the host:
# - NVIDIA GPU with driver >= 525.60.13
# - NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
# - Docker daemon configured with nvidia runtime (or run with --gpus all)
#
# PyTorch publishes CUDA wheels for both x86_64 and aarch64, so this builds on
# either architecture. Build natively on the host that will run it — an emulated
# cross-architecture image cannot reach the GPU.

ARG HINDSIGHT_VERSION=latest
FROM ghcr.io/vectorize-io/hindsight:${HINDSIGHT_VERSION}

ARG PYTORCH_CUDA_FLAVOR=cu126
ARG PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu126

# Upgrade PyTorch inside the venv to the CUDA runtime matching the installed base
# version. uv pip install against the image's venv replaces the CPU torch wheel
# with torch+cu126 and adds the NVIDIA CUDA 12 runtime libraries (cuDNN, cuBLAS,
# etc.). Everything already installed that still satisfies the resolution is kept,
# so torch is the only package replaced.
RUN TORCH_BASE_VERSION=$(/app/api/.venv/bin/python -c \
        'import torch; print(torch.__version__.split("+", 1)[0])') \
    && uv pip install \
        --python /app/api/.venv/bin/python \
        --no-cache \
        --index "$PYTORCH_CUDA_INDEX" \
        "torch==${TORCH_BASE_VERSION}+${PYTORCH_CUDA_FLAVOR}" \
    && uv pip check --python /app/api/.venv/bin/python \
    && /app/api/.venv/bin/python -c \
        'import torch; assert torch.version.cuda, "CUDA-enabled PyTorch wheel was not installed"; print(f"PyTorch CUDA runtime: {torch.version.cuda}")'
