# Example: custom Hindsight image with non-default local models baked in.
#
# Use this pattern in production when you run a non-default embedder or
# reranker. Baking models into the image removes the runtime dependency on
# HuggingFace and lets the container registry handle caching per node, so
# you don't need a model-cache PVC.
#
# Built on top of the slim image so only the deps and models you actually
# use end up in the final image.
FROM ghcr.io/vectorize-io/hindsight:latest-slim

# Install the local-ml deps required to load sentence-transformers /
# cross-encoder models at runtime. Pinned ranges mirror hindsight-api-slim's
# `local-ml` extra in hindsight-api-slim/pyproject.toml. Use `uv pip
# install` against the image's venv explicitly: the slim image's venv was
# created by `uv sync` and does not ship its own `pip`, so a bare
# `pip install` would fall back to user site-packages and not be visible
# to the runtime python.
RUN uv pip install --python /app/api/.venv/bin/python --no-cache \
    'sentence-transformers>=3.3.0' \
    'transformers>=4.53.0' \
    'torch>=2.6.0'

# Pre-download the models you want to use. Replace these with your own.
# The defaults bundled in the full image are BAAI/bge-small-en-v1.5 and
# cross-encoder/ms-marco-MiniLM-L-6-v2; here we pick multilingual variants
# as a concrete non-default example.
ARG EMBEDDER=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ARG RERANKER=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
ENV HF_HUB_DOWNLOAD_TIMEOUT=600
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDER}'); \
CrossEncoder('${RERANKER}')"
