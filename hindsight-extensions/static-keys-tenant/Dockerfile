# A Hindsight image with the static-keys tenant extension in it.
#
# Extensions are not bundled with the server and are not published to PyPI:
# you ship one by copying it into an image built on top of Hindsight.
#
# Build from the repository root so the extension sources are in context:
#   docker build -f hindsight-extensions/static-keys-tenant/Dockerfile \
#     -t hindsight-with-static-keys .
#
# Run:
#   docker run -p 8888:8888 \
#     -e HINDSIGHT_API_TENANT_EXTENSION=hindsight_ext_static_keys_tenant:StaticKeysTenantExtension \
#     -e HINDSIGHT_API_TENANT_USERS=user1:key1,user2:key2 \
#     hindsight-with-static-keys
#
# Override the base with `--build-arg HINDSIGHT_IMAGE=...:latest-slim` if you
# don't need the bundled local embedding/reranking models.
ARG HINDSIGHT_IMAGE=ghcr.io/vectorize-io/hindsight:latest
FROM ${HINDSIGHT_IMAGE}

# The static-keys extension has no third-party dependencies beyond the server
# itself — nothing to pip install here.

# Put the extension on the server's import path. /app/extensions is ours — the
# image does not use it — so this cannot shadow anything the server ships.
COPY hindsight-extensions/static-keys-tenant/hindsight_ext_static_keys_tenant \
     /app/extensions/hindsight_ext_static_keys_tenant
ENV PYTHONPATH=/app/extensions

# Fail the build, rather than the first authenticated request, if the extension
# is not importable from the interpreter the server actually runs.
RUN /app/api/.venv/bin/python -c "import hindsight_ext_static_keys_tenant"
