# A Hindsight image with the bank LLM routing extension in it.
#
# Extensions are not bundled with the server and are not published to PyPI:
# you ship one by copying it into an image built on top of Hindsight.
#
# Build from the repository root so the extension sources are in context:
#   docker build -f hindsight-extensions/bank-llm-routing/Dockerfile \
#     -t hindsight-with-bank-llm-routing .
#
# Run:
#   docker run -p 8888:8888 \
#     -e HINDSIGHT_API_LLM_REQUEST_EXTENSION=hindsight_ext_bank_llm_routing:BankLLMRoutingExtension \
#     -e HINDSIGHT_API_LLM_REQUEST_ROUTES='[{"bank": "acme-*", "api_key": "sk-acme"}]' \
#     hindsight-with-bank-llm-routing
#
# Override the base with `--build-arg HINDSIGHT_IMAGE=...:latest-slim` if you
# don't need the bundled local embedding/reranking models.
ARG HINDSIGHT_IMAGE=ghcr.io/vectorize-io/hindsight:latest
FROM ${HINDSIGHT_IMAGE}

# This extension has no third-party dependencies beyond the server
# itself — nothing to pip install here.

# Put the extension on the server's import path. /app/extensions is ours — the
# image does not use it — so this cannot shadow anything the server ships.
COPY hindsight-extensions/bank-llm-routing/hindsight_ext_bank_llm_routing \
     /app/extensions/hindsight_ext_bank_llm_routing
ENV PYTHONPATH=/app/extensions

# Fail the build, rather than the first LLM call, if the extension
# is not importable from the interpreter the server actually runs.
RUN /app/api/.venv/bin/python -c "import hindsight_ext_bank_llm_routing"
