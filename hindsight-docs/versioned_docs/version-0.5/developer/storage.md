# Storage

Hindsight uses PostgreSQL as its sole storage backend.

## Why PostgreSQL?

PostgreSQL provides all capabilities required for a semantic memory system in a single database:

| Capability | Implementation |
|------------|----------------|
| Vector search | pgvector extension with HNSW indexes |
| Full-text search | Built-in tsvector with GIN indexes |
| Relational data | Native PostgreSQL |
| JSON documents | JSONB with indexing |
| Graph queries | Recursive CTEs |

### Reduced System Dependencies

Building exclusively for PostgreSQL simplifies deployment and operations:

- Single connection string to configure
- Single backup and restore strategy
- Single monitoring target
- ACID transactions across all data types
- Single upgrade path

### No Storage Abstraction

Hindsight does not abstract storage behind a generic interface. This is a deliberate trade-off.

We believe PostgreSQL is becoming the standard database API. Its popularity, extension ecosystem, and modularity mean that PostgreSQL-compatible interfaces are appearing everywhere—from serverless offerings to distributed databases. Building for PostgreSQL today means compatibility with a growing ecosystem tomorrow.

Supporting multiple databases would increase flexibility but conflict with our core goals: Hindsight is fully open source and designed to be as simple as possible to run and use. Adding database abstractions introduces complexity in code, testing, documentation, and operations—complexity that we pass on to users.

By committing to PostgreSQL, we keep the system simple:
- One set of deployment instructions
- One set of performance characteristics to understand
- One codebase optimized for one backend
- No configuration decisions about which database to use

## Development with pg0

For local development, Hindsight uses **[pg0](https://github.com/vectorize-io/pg0)**—an embedded PostgreSQL distribution.

### What is pg0?

pg0 is a single binary containing:
- PostgreSQL server
- pgvector extension (pre-installed)
- Automatic initialization

### Behavior

When no `DATABASE_URL` is configured, Hindsight:
1. Starts an embedded PostgreSQL instance on port 5555
2. Initializes the schema
3. Stores data in `~/.hindsight/pg0/`

### Environments

| Environment | Database | Configuration |
|-------------|----------|---------------|
| Development | pg0 (embedded) | Automatic |
| Production | PostgreSQL 15+ | `DATABASE_URL` environment variable |

## Requirements

- PostgreSQL 15 or later
- pgvector 0.5.0 or later

Any PostgreSQL instance that satisfies these requirements should work. If you encounter issues with a specific setup, [open a GitHub issue](https://github.com/hindsight-ai/hindsight/issues).

### Tested Managed Services

- AWS RDS (PostgreSQL 15+)
- Google Cloud SQL
- Azure Database for PostgreSQL
- Supabase
- Neon
