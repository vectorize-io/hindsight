# PostgreSQL with pgvector and pgroonga extensions.
#
# pgroonga is a multilingual full-text search extension built on Groonga.
# It works out of the box for CJK (Chinese, Japanese, Korean) and other
# non-whitespace-segmented languages via the TokenBigram tokenizer.
FROM groonga/pgroonga:latest-debian-pg17

# Install pgvector on top of the pgroonga base image (which already provides
# pgroonga and the Groonga library).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    postgresql-server-dev-17 \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install

RUN rm -rf /tmp/pgvector && \
    apt-get purge -y --auto-remove build-essential git postgresql-server-dev-17
