# Hindsight with Timescale Extensions

This Docker Compose setup provides a complete Hindsight deployment with **Timescale extensions**:
- **pgvectorscale** - DiskANN algorithm for disk-based scalable vector search  
- **pg_textsearch** - High-performance BM25 text search

Both extensions are from [Timescale](https://github.com/timescale) and provide production-grade performance.

## Prerequisites

- Docker and Docker Compose installed
- An OpenAI API key (or a key for another LLM provider)

## Quick Start

```bash
# Set environment variables
export HINDSIGHT_DB_PASSWORD="your-secure-password"
export HINDSIGHT_API_LLM_API_KEY="your-openai-api-key"

# Build and start
docker compose -f docker/docker-compose/timescale/docker-compose.yaml up -d --build

# Check logs

docker compose -f docker/docker-compose/timescale/docker-compose.yaml logs -f
```

**Access:**
- API: http://localhost:8888
- Control Plane: http://localhost:9999

## Stop and Clean Up

```bash
# Stop services
docker compose -f docker/docker-compose/timescale/docker-compose.yaml down

# Remove volumes (deletes all data)
docker compose -f docker/docker-compose/timescale/docker-compose.yaml down -v
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_DB_PASSWORD` | PostgreSQL password | `hindsight_password` |
| `HINDSIGHT_DB_USER` | PostgreSQL username | `hindsight_user` |
| `HINDSIGHT_DB_NAME` | Database name | `hindsight_db` |
| `HINDSIGHT_VERSION` | Hindsight Docker image version | `latest` |
| `HINDSIGHT_API_LLM_API_KEY` | API key for the LLM provider | (required) |
| `HINDSIGHT_API_LLM_PROVIDER` | LLM provider | `openai` |

### Why Timescale Extensions?

**pgvectorscale (DiskANN):**
- 28x lower p95 latency vs dedicated vector databases
- 16x higher query throughput at 99% recall
- 60-75% cost reduction (disk is cheaper than RAM)
- Best for large datasets (10M+ vectors)

**pg_textsearch (BM25):**
- High-performance keyword retrieval
- Native BM25 ranking algorithm
- Optimized for full-text search

## Troubleshooting

### Extensions not installed

Check if extensions are available:

```bash
docker exec -it hindsight-db-timescale psql -U hindsight_user -d hindsight_db -c "\dx"
```

You should see:
- `vector` (pgvector)
- `vectorscale` (pgvectorscale/DiskANN)
- `pg_textsearch` (BM25 search)

### Build fails

If the Docker build fails during pgvectorscale compilation:

1. Ensure you have sufficient memory (recommended: 4GB+)
2. Check Docker build logs for Rust compilation errors
3. Try building with more resources: `docker compose build --no-cache --memory 4g`

### Port conflicts

If port 5438 is already in use, modify the `ports` section in docker-compose.yaml.

## Learn More

- [pgvectorscale GitHub](https://github.com/timescale/pgvectorscale)
- [pg_textsearch GitHub](https://github.com/timescale/pg_textsearch)
- [HNSW vs DiskANN](https://www.tigerdata.com/learn/hnsw-vs-diskann)
- [Hindsight Documentation](https://hindsight.dev)
