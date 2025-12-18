# Hindsight Docker Image
# Supports building API-only, Control Plane-only, or both
#
# Build args:
#   INCLUDE_API=true/false       - Include API (default: true)
#   INCLUDE_CP=true/false        - Include Control Plane (default: true)
#   PRELOAD_ML_MODELS=true/false - Pre-download ML models during build (default: true)
#
# Examples:
#   docker build -t hindsight .                                          # Both (standalone)
#   docker build -t hindsight-api --build-arg INCLUDE_CP=false .         # API only
#   docker build -t hindsight-cp --build-arg INCLUDE_API=false .         # Control Plane only
#   docker build -t hindsight --build-arg PRELOAD_ML_MODELS=false .      # Skip ML model preload

ARG INCLUDE_API=true
ARG INCLUDE_CP=true
ARG PRELOAD_ML_MODELS=true

# =============================================================================
# Stage: API Builder
# =============================================================================
FROM python:3.11-slim AS api-builder

ARG INCLUDE_API
RUN if [ "$INCLUDE_API" != "true" ]; then echo "Skipping API build" && exit 0; fi

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Copy dependency files and README (required by pyproject.toml)
COPY hindsight-api/pyproject.toml ./api/
COPY hindsight-api/README.md ./api/

WORKDIR /app/api

# Sync dependencies (will create lock file if needed)
RUN uv sync

# Copy source code (alembic migrations are inside hindsight_api/)
COPY hindsight-api/hindsight_api ./hindsight_api

# Install the local package (uv sync only installed dependencies, not the package itself)
RUN uv pip install -e .

# =============================================================================
# Stage: SDK Builder (needed for Control Plane)
# =============================================================================
FROM node:20-slim AS sdk-builder

ARG INCLUDE_CP
RUN if [ "$INCLUDE_CP" != "true" ]; then echo "Skipping SDK build" && exit 0; fi

WORKDIR /app

# Copy root package files for npm workspaces
COPY package.json package-lock.json ./
COPY hindsight-clients/typescript/ ./hindsight-clients/typescript/

# Install and build SDK using workspace (--ignore-scripts skips git hooks setup)
RUN npm ci --ignore-scripts -w @vectorize-io/hindsight-client
RUN npm run build -w @vectorize-io/hindsight-client

# =============================================================================
# Stage: Control Plane Builder
# =============================================================================
FROM node:20-slim AS cp-builder

ARG INCLUDE_CP
RUN if [ "$INCLUDE_CP" != "true" ]; then echo "Skipping CP build" && exit 0; fi

# Create directory structure matching the monorepo layout
# This is required because build:standalone script expects .next/standalone/memory-poc/hindsight-control-plane
WORKDIR /app/memory-poc/hindsight-control-plane

# Install Control Plane dependencies
# Only copy package.json (not package-lock.json) to ensure npm installs
# correct platform-specific native bindings for lightningcss/tailwindcss
COPY hindsight-control-plane/package.json ./
# Remove the file: dependency on SDK (we'll copy it directly later)
RUN sed -i '/"@vectorize-io\/hindsight-client":/d' package.json
RUN npm install

# Copy Control Plane source (excluding node_modules via .dockerignore)
COPY hindsight-control-plane/ ./
# Remove package-lock.json to avoid conflicts with installed native bindings
# Also remove the file: dependency from package.json (restored by COPY above)
RUN rm -f package-lock.json && sed -i '/"@vectorize-io\/hindsight-client":/d' package.json

# Copy built SDK directly into node_modules (more reliable than npm link in Docker)
COPY --from=sdk-builder /app/hindsight-clients/typescript ./node_modules/@vectorize-io/hindsight-client

# Build Control Plane - run next build first, then custom standalone copy
# (The build:standalone script expects a specific path structure that differs in Docker)
RUN npm exec -- next build

# Create standalone directory structure manually
# Note: Must exclude node_modules from find to avoid wrong server.js from next/dist/experimental/testmode/
# Note: Must explicitly copy .next since glob * doesn't match hidden directories
RUN STANDALONE_ROOT=$(find .next/standalone -path '*/node_modules' -prune -o -name 'server.js' -print | head -1 | xargs dirname) && \
    mkdir -p standalone && \
    cp -r "$STANDALONE_ROOT"/* standalone/ && \
    cp -r "$STANDALONE_ROOT"/.next standalone/.next && \
    # Copy node_modules if separate from app dir (monorepo structure)
    if [ -d ".next/standalone/node_modules" ] && [ "$STANDALONE_ROOT" != ".next/standalone" ]; then \
      cp -r .next/standalone/node_modules standalone/node_modules; \
    fi && \
    cp -r .next/static standalone/.next/static && \
    mkdir -p standalone/public && \
    cp -r public/* standalone/public/ 2>/dev/null || true && \
    # Verify required files exist
    test -f standalone/server.js || (echo "ERROR: server.js missing!" && exit 1) && \
    test -f standalone/.next/BUILD_ID || (echo "ERROR: BUILD_ID missing!" && exit 1)

# =============================================================================
# Stage: Final Image - API Only
# =============================================================================
FROM python:3.11-slim AS api-only

WORKDIR /app

# Install pg0 dependencies (procps provides 'kill' command needed by pg0)
# Note: libicu version varies by Debian version - try common versions in order
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    libxml2 \
    libssl3 \
    libgssapi-krb5-2 \
    libossp-uuid16 \
    && (apt-get install -y libicu72 2>/dev/null || apt-get install -y libicu74 2>/dev/null || apt-get install -y libicu76 2>/dev/null || true) \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Create non-root user (PostgreSQL cannot run as root)
RUN useradd -m -s /bin/bash hindsight

# Copy API with virtual environment from builder
COPY --from=api-builder /app/api /app/api

# Copy startup script
COPY docker/standalone/start-all.sh /app/start-all.sh
RUN chmod +x /app/start-all.sh

# Create data directory for pg0 and set ownership
RUN mkdir -p /app/data && chown -R hindsight:hindsight /app

# Switch to non-root user
USER hindsight

# Set PATH for hindsight user
ENV PATH="/app/api/.venv/bin:${PATH}"

# Pre-cache PostgreSQL binaries by starting/stopping pg0-embedded
ENV PG0_HOME=/home/hindsight/.pg0-cache

ENV PG0_HOME=/home/hindsight/.pg0

# Pre-download ML models to avoid runtime download (conditional)
ARG PRELOAD_ML_MODELS
RUN if [ "$PRELOAD_ML_MODELS" = "true" ]; then \
    /app/api/.venv/bin/python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Downloading embedding model...'); \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
print('Downloading cross-encoder model...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Models cached successfully')"; \
    else echo "Skipping ML model preload"; fi

EXPOSE 8888

ENV HINDSIGHT_API_HOST=0.0.0.0
ENV HINDSIGHT_API_PORT=8888
ENV HINDSIGHT_API_LOG_LEVEL=info
ENV HINDSIGHT_ENABLE_API=true
ENV HINDSIGHT_ENABLE_CP=false
ENV PYTHONUNBUFFERED=1

CMD ["/app/start-all.sh"]

# =============================================================================
# Stage: Final Image - Control Plane Only
# =============================================================================
FROM node:20-alpine AS cp-only

WORKDIR /app

# Copy built SDK
COPY --from=sdk-builder /app/hindsight-clients/typescript /app/sdk

# Copy Control Plane standalone build
WORKDIR /app/control-plane
COPY --from=cp-builder /app/memory-poc/hindsight-control-plane/standalone ./
COPY --from=cp-builder /app/memory-poc/hindsight-control-plane/.next/static ./.next/static
COPY --from=cp-builder /app/memory-poc/hindsight-control-plane/public ./public

WORKDIR /app

# Copy startup script
COPY docker/standalone/start-all.sh /app/start-all.sh
RUN chmod +x /app/start-all.sh

# Install curl for health checks
RUN apk add --no-cache curl bash

EXPOSE 9999

ENV NODE_ENV=production
ENV HINDSIGHT_CP_DATAPLANE_API_URL=http://localhost:8888
ENV HINDSIGHT_ENABLE_API=false
ENV HINDSIGHT_ENABLE_CP=true

CMD ["/app/start-all.sh"]

# =============================================================================
# Stage: Final Image - Standalone (both API and Control Plane)
# =============================================================================
FROM python:3.11-slim AS standalone

WORKDIR /app

# Install Node.js, curl, uv, and pg0 dependencies (procps provides 'kill' command needed by pg0)
# Note: libicu version varies by Debian version - try common versions in order
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    libxml2 \
    libssl3 \
    libgssapi-krb5-2 \
    libossp-uuid16 \
    && (apt-get install -y libicu72 2>/dev/null || apt-get install -y libicu74 2>/dev/null || apt-get install -y libicu76 2>/dev/null || true) \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Create non-root user (PostgreSQL cannot run as root)
RUN useradd -m -s /bin/bash hindsight

# Copy API with virtual environment from builder
COPY --from=api-builder /app/api /app/api

# Copy built SDK
COPY --from=sdk-builder /app/hindsight-clients/typescript /app/sdk

# Copy Control Plane standalone build
WORKDIR /app/control-plane
COPY --from=cp-builder /app/memory-poc/hindsight-control-plane/standalone ./
COPY --from=cp-builder /app/memory-poc/hindsight-control-plane/.next/static ./.next/static
COPY --from=cp-builder /app/memory-poc/hindsight-control-plane/public ./public

WORKDIR /app

# Copy startup script
COPY docker/standalone/start-all.sh /app/start-all.sh
RUN chmod +x /app/start-all.sh

# Create data directory for pg0 and set ownership
RUN mkdir -p /app/data && chown -R hindsight:hindsight /app

# Switch to non-root user
USER hindsight

# Set PATH for hindsight user
ENV PATH="/app/api/.venv/bin:${PATH}"

# Pre-cache PostgreSQL binaries by starting/stopping pg0-embedded
ENV PG0_HOME=/home/hindsight/.pg0-cache
RUN /app/api/.venv/bin/python -c "\
from pg0 import Pg0; \
print('Pre-caching PostgreSQL binaries...'); \
pg = Pg0(name='hindsight', port=5555, username='hindsight', password='hindsight', database='hindsight'); \
pg.start(); \
pg.stop(); \
print('PostgreSQL pre-cached to PG0_HOME')" || echo "Pre-download skipped"

ENV PG0_HOME=/home/hindsight/.pg0

# Pre-download ML models to avoid runtime download (conditional)
ARG PRELOAD_ML_MODELS
RUN if [ "$PRELOAD_ML_MODELS" = "true" ]; then \
    /app/api/.venv/bin/python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Downloading embedding model...'); \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
print('Downloading cross-encoder model...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Models cached successfully')"; \
    else echo "Skipping ML model preload"; fi

EXPOSE 8888 9999

ENV HINDSIGHT_API_HOST=0.0.0.0
ENV HINDSIGHT_API_PORT=8888
ENV HINDSIGHT_API_LOG_LEVEL=info
ENV NODE_ENV=production
ENV HINDSIGHT_CP_DATAPLANE_API_URL=http://localhost:8888
ENV HINDSIGHT_ENABLE_API=true
ENV HINDSIGHT_ENABLE_CP=true
ENV PYTHONUNBUFFERED=1

CMD ["/app/start-all.sh"]

# =============================================================================
# Default target selection based on build args
# =============================================================================
FROM standalone AS default-both
FROM api-only AS default-api
FROM cp-only AS default-cp

# This selects the final stage based on INCLUDE_API and INCLUDE_CP
# Use --target to override: docker build --target api-only .
FROM standalone
