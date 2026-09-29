# PostgreSQL with pgvector, pgvectorscale, and pg_textsearch extensions
# All three extensions from Timescale/pgvector for high-performance vector and text search
# Note: Requires PostgreSQL 16+
FROM postgres:17

# Install build dependencies and Rust toolchain
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-17 \
    libpq-dev \
    cmake \
    curl \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain (required for pgvectorscale)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install pgvector (required by pgvectorscale)
RUN cd /tmp && \
    git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install && \
    rm -rf /tmp/pgvector

# Install cargo-pgrx (PostgreSQL extension framework for Rust)
RUN cargo install cargo-pgrx --version 0.12.5 --locked && \
    cargo pgrx init --pg17 /usr/bin/pg_config

# Install pgvectorscale (DiskANN index support)
RUN cd /tmp && \
    git clone --branch 0.5.1 https://github.com/timescale/pgvectorscale.git && \
    cd pgvectorscale/pgvectorscale && \
    cargo pgrx install --release && \
    rm -rf /tmp/pgvectorscale

# Install pg_textsearch (BM25 text search)
RUN cd /tmp && \
    git clone https://github.com/timescale/pg_textsearch.git && \
    cd pg_textsearch && \
    make && \
    make install && \
    rm -rf /tmp/pg_textsearch

# Clean up build dependencies (keep runtime dependencies)
RUN apt-get purge -y --auto-remove git cmake curl && \
    rm -rf /root/.cargo/registry /root/.cargo/git

# Ensure extensions are preloaded (pg_textsearch requires preloading)
RUN echo "shared_preload_libraries = 'pg_textsearch'" >> /usr/share/postgresql/postgresql.conf.sample

