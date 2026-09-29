# PostgreSQL with pgvector and pg_textsearch extensions
# Note: pg_textsearch requires PostgreSQL 17+
FROM postgres:17

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-17 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pgvector
RUN cd /tmp && \
    git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install

# Install pg_textsearch
RUN cd /tmp && \
    git clone https://github.com/timescale/pg_textsearch.git && \
    cd pg_textsearch && \
    make && \
    make install

# Clean up source files and build dependencies
RUN rm -rf /tmp/pgvector /tmp/pg_textsearch && \
    apt-get purge -y --auto-remove build-essential git postgresql-server-dev-17

# Ensure extensions are preloaded
RUN echo "shared_preload_libraries = 'pg_textsearch'" >> /usr/share/postgresql/postgresql.conf.sample
