# A Hindsight image with the Supabase tenant extension in it.
#
# Extensions are not bundled with the server and are not published to PyPI:
# you ship one by copying it into an image built on top of Hindsight.
#
# Build from the repository root so the extension sources are in context:
#   docker build -f hindsight-extensions/supabase-tenant/Dockerfile \
#     -t hindsight-with-supabase .
#
# Run:
#   docker run -p 8888:8888 \
#     -e HINDSIGHT_API_TENANT_EXTENSION=hindsight_ext_supabase_tenant:SupabaseTenantExtension \
#     -e HINDSIGHT_API_TENANT_SUPABASE_URL=https://xxx.supabase.co \
#     hindsight-with-supabase
#
# Override the base with `--build-arg HINDSIGHT_IMAGE=...:latest-slim` if you
# don't need the bundled local embedding/reranking models.
ARG HINDSIGHT_IMAGE=ghcr.io/vectorize-io/hindsight:latest
FROM ${HINDSIGHT_IMAGE}

# Install into the server's virtualenv explicitly. It was created by `uv sync`
# and ships no `pip` of its own, so a bare `pip install` would land in user
# site-packages and be invisible to the running server.
RUN uv pip install --python /app/api/.venv/bin/python --no-cache \
      'PyJWT[crypto]>=2.12.0' \
      'httpx>=0.27.0'

# Put the extension on the server's import path. /app/extensions is ours — the
# image does not use it — so this cannot shadow anything the server ships.
COPY hindsight-extensions/supabase-tenant/hindsight_ext_supabase_tenant \
     /app/extensions/hindsight_ext_supabase_tenant
ENV PYTHONPATH=/app/extensions

# Fail the build, rather than the first authenticated request, if the extension
# is not importable from the interpreter the server actually runs.
RUN /app/api/.venv/bin/python -c "import hindsight_ext_supabase_tenant"
