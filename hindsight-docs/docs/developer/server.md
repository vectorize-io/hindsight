---
sidebar_position: 6
---

# Server Deployment

Guide to deploying the Hindsight server in production.

## Deployment Options

### Docker Compose (Recommended)

The simplest way to deploy Hindsight with all dependencies:

```bash
# Clone the repository
git clone https://github.com/vectorize-io/hindsight.git
cd hindsight

# Create environment file
cp .env.example .env
# Edit .env with your LLM API key

# Start all services
cd docker
./start.sh
```

Services will be available at:
- **API Server**: http://localhost:8888
- **Control Plane**: http://localhost:3000
- **Swagger UI**: http://localhost:8888/docs

#### Docker Compose Commands

```bash
# Start all services
cd docker && ./start.sh

# Stop services
cd docker && ./stop.sh

# Clean all data
cd docker && ./clean.sh

# View logs
docker-compose logs -f api
docker-compose logs -f control-plane
```

### Helm Chart (Kubernetes)

For Kubernetes deployments, use the Hindsight Helm chart:

```bash
helm repo add hindsight https://vectorize-io.github.io/hindsight
helm install hindsight hindsight/hindsight \
  --set api.llm.provider=groq \
  --set api.llm.apiKey=gsk_xxxxxxxxxxxx
```

See the [Helm chart documentation](https://github.com/vectorize-io/hindsight/tree/main/deploy/helm) for configuration options.

### Bare Metal / pip install

For custom deployments, install the API server directly:

```bash
pip install hindsight-api
```

Run the server:

```bash
python -m hindsight_api.web.server --host 0.0.0.0 --port 8888
```

You'll need to:
1. Provision PostgreSQL with pgvector extension
2. Set environment variables (see [Configuration](#environment-variables))
3. Handle process management (systemd, supervisor, etc.)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENTS                                         │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│  Python Client  │  Node.js Client │      CLI        │   AI Assistants       │
│  hindsight-client  │  @hindsight/client │   hindsight-cli    │  (Claude, etc.)       │
└────────┬────────┴────────┬────────┴────────┬────────┴───────────┬───────────┘
         │                 │                 │                     │
         ▼                 ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HINDSIGHT API SERVER                               │
│                          (localhost:8888)                                    │
├─────────────────────────────────┬───────────────────────────────────────────┤
│          HTTP API               │              MCP API                       │
│    /api/memories/*              │         MCP Server (stdio)                 │
│    /api/agents/*                │      hindsight_search, hindsight_think,    │
│    /api/search, /api/think      │      hindsight_store, hindsight_agents     │
└─────────────────────────────────┴───────────────────────────────────────────┘
         │                                           │
         ▼                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PROCESSING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   INGESTION  │    │  RETRIEVAL   │    │  REASONING   │                   │
│  │              │    │   (TEMPR)    │    │   (CARA)     │                   │
│  │  LLM Extract │    │              │    │              │                   │
│  │  Entity Res. │    │  4-way Search│    │  Personality │                   │
│  │  Graph Build │    │  RRF Fusion  │    │  Opinion Gen │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ML MODELS                                       │
├───────────────────┬───────────────────┬─────────────────────────────────────┤
│    Embeddings     │   Cross-Encoder   │         LLM Provider                 │
│  all-MiniLM-L6-v2 │ ms-marco-MiniLM   │   OpenAI / Groq / Ollama            │
│    (384-dim)      │   (reranking)     │   (extraction, reasoning)           │
└───────────────────┴───────────────────┴─────────────────────────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         POSTGRESQL + PGVECTOR                                │
│                          (localhost:5432)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Memory Units (facts, opinions)    • HNSW Vector Index                    │
│  • Entity Graph (nodes, edges)       • GIN Full-Text Index                  │
│  • Agent Profiles                    • Temporal Indexes                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      CONTROL PLANE (Optional)                                │
│                         (localhost:3000)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Web UI for administration         • Agent management                     │
│  • Memory visualization              • Graph explorer                       │
│  • Connects to API Server            • Monitoring dashboard                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Environment Variables

### API Server (`HINDSIGHT_API_*`)

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_DATABASE_URL` | PostgreSQL connection string | Required |
| `HINDSIGHT_API_LLM_PROVIDER` | LLM provider: `openai`, `groq`, `ollama` | `groq` |
| `HINDSIGHT_API_LLM_API_KEY` | API key for LLM provider | Required (except ollama) |
| `HINDSIGHT_API_LLM_MODEL` | Model name | `llama-3.1-70b-versatile` |
| `HINDSIGHT_API_LLM_BASE_URL` | Custom LLM endpoint | Provider default |
| `HINDSIGHT_API_HOST` | Server bind address | `0.0.0.0` |
| `HINDSIGHT_API_PORT` | Server port | `8888` |
| `HINDSIGHT_API_MCP_ENABLED` | Enable MCP server | `true` |

### Control Plane (`HINDSIGHT_CP_*`)

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_CP_DATAPLANE_API_URL` | API server URL | `http://localhost:8888` |
| `HINDSIGHT_CP_HOSTNAME` | Server bind address | `0.0.0.0` |
| `HINDSIGHT_CP_PORT` | Server port | `3000` |

### Example Configuration

```bash
# .env file

# Database
HINDSIGHT_API_DATABASE_URL=postgresql://hindsight:hindsight_dev@localhost:5432/hindsight

# LLM - Using Groq (fast inference)
HINDSIGHT_API_LLM_PROVIDER=groq
HINDSIGHT_API_LLM_API_KEY=gsk_xxxxxxxxxxxx
HINDSIGHT_API_LLM_MODEL=llama-3.1-70b-versatile

# LLM - Using OpenAI
# HINDSIGHT_API_LLM_PROVIDER=openai
# HINDSIGHT_API_LLM_API_KEY=sk-xxxxxxxxxxxx
# HINDSIGHT_API_LLM_MODEL=gpt-4o

# LLM - Using Ollama (local, no API key)
# HINDSIGHT_API_LLM_PROVIDER=ollama
# HINDSIGHT_API_LLM_BASE_URL=http://localhost:11434/v1
# HINDSIGHT_API_LLM_MODEL=llama3.1

# Control Plane
HINDSIGHT_CP_DATAPLANE_API_URL=http://localhost:8888
```

## ML Models

Hindsight uses several ML models that are downloaded automatically on first run:

### Embedding Model

| Model | Dimensions | Purpose |
|-------|------------|---------|
| `all-MiniLM-L6-v2` | 384 | Semantic vector embeddings |

Used for memory vectorization, query embedding, and semantic similarity search.

### Cross-Encoder (Reranking)

| Model | Purpose |
|-------|---------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Neural reranking |

Reranks search results for precision after initial retrieval.

### Temporal Parser

| Model | Purpose |
|-------|---------|
| `t5-small` | Temporal expression parsing |

Parses natural language time expressions like "last spring" or "in June 2024".

### LLM (Configurable)

Used for fact extraction, entity resolution, opinion generation, and think responses.

Supported providers:
- **OpenAI**: GPT-4, GPT-4o, GPT-3.5-turbo
- **Groq**: Llama 3.1, Mixtral (fast inference)
- **Ollama**: Any local model

## Database Schema

PostgreSQL with pgvector extension:

```sql
-- Memory units (facts and opinions)
CREATE TABLE memory_units (
    id UUID PRIMARY KEY,
    agent_id VARCHAR NOT NULL,
    text TEXT NOT NULL,
    fact_type VARCHAR NOT NULL,  -- 'world', 'agent', 'opinion'
    confidence_score FLOAT,
    embedding VECTOR(384),
    occurred_start DATE,
    occurred_end DATE,
    mentioned_at TIMESTAMP,
    context VARCHAR,
    document_id VARCHAR
);

-- Entity graph
CREATE TABLE entities (
    id UUID PRIMARY KEY,
    agent_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    entity_type VARCHAR NOT NULL,
    canonical_name VARCHAR
);

CREATE TABLE entity_links (
    memory_id UUID REFERENCES memory_units(id),
    entity_id UUID REFERENCES entities(id),
    PRIMARY KEY (memory_id, entity_id)
);

-- Agent profiles
CREATE TABLE agent_profiles (
    agent_id VARCHAR PRIMARY KEY,
    background TEXT,
    openness FLOAT DEFAULT 0.5,
    conscientiousness FLOAT DEFAULT 0.5,
    extraversion FLOAT DEFAULT 0.5,
    agreeableness FLOAT DEFAULT 0.5,
    neuroticism FLOAT DEFAULT 0.5,
    bias_strength FLOAT DEFAULT 0.5
);
```

### Indexes

- **HNSW Index**: Fast approximate nearest neighbor for vector search
- **GIN Index**: Full-text search with BM25 ranking
- **B-tree Indexes**: Agent ID, timestamps, entity lookups

## Health Checks

### API Server

```bash
curl http://localhost:8888/api/v1/agents
```

### Control Plane

```bash
curl http://localhost:3000/
```

## Production Considerations

For production deployments:

1. **Use managed PostgreSQL** with pgvector extension (AWS RDS, Google Cloud SQL, Supabase)
2. **Set proper secrets** via environment variables or secrets manager
3. **Configure resource limits** for ML model inference
4. **Set up monitoring** for API latency and error rates
5. **Use HTTPS** with proper TLS certificates
6. **Configure rate limiting** at load balancer level

### Resource Requirements

| Component | CPU | Memory | Notes |
|-----------|-----|--------|-------|
| API Server | 2+ cores | 4GB+ | ML models loaded in memory |
| PostgreSQL | 2+ cores | 4GB+ | Depends on data size |
| Control Plane | 1 core | 512MB | Lightweight Next.js app |
