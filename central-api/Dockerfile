FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libpq-dev openssh-client && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv pip install --system -e ".[dev]"

COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY migrations/ ./migrations/

EXPOSE 8000

# Clean startup: uvicorn (no reload). DB tables ensured via lifespan init_models() in dev
# or alembic upgrade head in prod (run manually or via entrypoint as needed).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]