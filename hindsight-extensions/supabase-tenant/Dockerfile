# Example: a Hindsight image with the Supabase tenant extension installed.
#
# Extensions are not bundled with the server, so adding one means deriving an
# image that installs it into the environment the server actually runs from.
#
# Build and run:
#   docker build -t hindsight-with-supabase .
#   docker run -p 8000:8000 \
#     -e HINDSIGHT_API_TENANT_EXTENSION=hindsight_ext_supabase_tenant:SupabaseTenantExtension \
#     -e HINDSIGHT_API_TENANT_SUPABASE_URL=https://xxx.supabase.co \
#     hindsight-with-supabase
#
# Use `:latest-slim` instead if you don't need the bundled local
# embedding/reranking models.
FROM ghcr.io/vectorize-io/hindsight:latest

# Two things matter here:
#
# 1. `uv pip install --python /app/api/.venv/bin/python`. The image's venv was
#    created by `uv sync` and ships no `pip` of its own, so a bare `pip install`
#    would land in user site-packages and be invisible to the running server.
#
# 2. The extension is installed with `--no-deps`, its own dependencies pinned
#    explicitly on the line above. The extension does not declare
#    hindsight-api-slim as a dependency (the server is its host, not its
#    dependency), and --no-deps keeps it that way even if a transitive
#    resolution ever tried to pull a different server version into this image.
RUN uv pip install --python /app/api/.venv/bin/python --no-cache \
      'PyJWT[crypto]>=2.12.0' \
      'httpx>=0.27.0' \
 && uv pip install --python /app/api/.venv/bin/python --no-cache --no-deps \
      hindsight-ext-supabase-tenant \
 && /app/api/.venv/bin/python -c "import hindsight_ext_supabase_tenant"
