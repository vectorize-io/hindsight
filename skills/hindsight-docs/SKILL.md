---
name: hindsight-docs
description: Complete Hindsight documentation for AI agents. Use this to learn about Hindsight architecture, APIs, configuration, and best practices.
---

# Hindsight Documentation Skill

Complete technical documentation for Hindsight - a biomimetic memory system for AI agents.

## When to Use This Skill

Use this skill when you need to:
- Understand Hindsight architecture and core concepts
- Learn about retain/recall/reflect operations
- Configure memory banks and dispositions
- Set up the Hindsight API server (Docker, Kubernetes, pip)
- Integrate with Python/Node.js/Rust SDKs
- Understand retrieval strategies (semantic, BM25, graph, temporal)
- Debug issues or optimize performance
- Review API endpoints and parameters
- Find cookbook examples and recipes

## Documentation Categories

Documentation is organized in `references/` by category:

### Core API Operations (`references/developer/api/`)
- **quickstart.md** - Get started in 60 seconds
- **retain.md** - Store memories (fact extraction, entity linking)
- **recall.md** - Retrieve memories (multi-strategy search)
- **reflect.md** - Disposition-aware reasoning
- **memory-banks.md** - Configure banks and dispositions
- **mental-models.md** - Consolidated knowledge structures
- **documents.md** - Document-level memory management
- **operations.md** - Core operation details

### Developer Guides (`references/developer/`)
- **installation.md** - Docker, Kubernetes, pip deployment
- **configuration.md** - Environment variables and settings
- **models.md** - LLM providers and ML models
- **retrieval.md** - Search strategies and fusion
- **retain.md** - Memory ingestion pipeline
- **reflect.md** - Reflection and disposition
- **storage.md** - Database schema and pgvector
- **performance.md** - Optimization and benchmarks
- **monitoring.md** - Metrics and observability
- **services.md** - Worker service architecture
- **mcp-server.md** - Model Context Protocol server
- **admin-cli.md** - Admin CLI tool
- **development.md** - Local development setup
- **extensions.md** - Extending Hindsight
- **multilingual.md** - Multi-language support
- **rag-vs-hindsight.md** - RAG comparison

### SDK Guides (`references/sdks/`)
- **python.md** - Python SDK
- **nodejs.md** - Node.js/TypeScript SDK
- **cli.md** - CLI tool
- **embed.md** - Embedded Hindsight
- **integrations/litellm.md** - LiteLLM integration
- **integrations/ai-sdk.md** - Vercel AI SDK integration
- **integrations/openclaw.md** - OpenClaw integration
- **integrations/local-mcp.md** - Local MCP setup
- **integrations/skills.md** - Skills system

### Cookbook (`references/cookbook/`)

**Recipes** (`references/cookbook/recipes/`)
- **quickstart.md** - Basic usage patterns
- **per-user-memory.md** - One bank per user
- **support-agent-shared-knowledge.md** - Shared knowledge bases
- **personal_assistant.md** - Personal assistant pattern
- **fitness_tracker.md** - Fitness tracking example
- **healthcare_assistant.md** - Healthcare assistant pattern
- **movie_recommendation.md** - Recommendation system
- **personalized_search.md** - Personalized search
- **study_buddy.md** - Study assistant pattern
- **litellm-memory-demo.md** - LiteLLM integration demo
- **tool-learning-demo.md** - Tool learning example

**Applications** (`references/cookbook/applications/`)
- **chat-memory.md** - Chat memory implementation
- **deliveryman-demo.md** - Deliveryman agent demo
- **hindsight-litellm-demo.md** - LiteLLM integration
- **hindsight-tool-learning-demo.md** - Tool learning agent
- **openai-fitness-coach.md** - OpenAI fitness coach
- **sanity-blog-memory.md** - Blog memory system
- **stancetracker.md** - Stance tracking system
- **taste-ai.md** - Taste preference system

## How to Search the Documentation

### 1. Find Files by Pattern (use Glob tool)

```bash
# API documentation
references/developer/api/*.md

# SDK guides
references/sdks/*.md

# Cookbook recipes
references/cookbook/recipes/*.md

# All configuration docs
references/**/configuration.md

# All Python-related docs
references/**/*python*.md
```

### 2. Search for Content (use Grep tool)

```bash
# Search for specific concepts
pattern: "disposition" → understand memory bank configuration
pattern: "graph retrieval" → learn about graph-based search
pattern: "helm install" → find Kubernetes deployment
pattern: "document_id" → learn about document management
pattern: "HINDSIGHT_API_" → find environment variables
pattern: "retain.*async" → find async retain examples

# Search in specific directories
path: references/developer/api/
pattern: "POST /v1"  → find API endpoints

path: references/cookbook/
pattern: "def|async def"  → find Python code examples
```

### 3. Read Full Documentation (use Read tool)

Once you find the relevant file, read it:
```
references/developer/api/retain.md
references/sdks/python.md
references/cookbook/recipes/per-user-memory.md
```

## Key Concepts Quick Reference

- **Memory Banks**: Isolated memory stores (one per user/agent)
- **Retain**: Store memories (auto-extracts facts/entities/relationships)
- **Recall**: Retrieve memories (4 parallel strategies: semantic, BM25, graph, temporal)
- **Reflect**: Disposition-aware reasoning using memories
- **document_id**: Groups messages in a conversation (upsert on same ID)
- **Dispositions**: Skepticism, literalism, empathy traits (1-5) affecting reflect
- **Mental Models**: Consolidated knowledge synthesized from facts

## Important Notes

- All code examples in this skill are inlined from actual working examples
- Configuration uses environment variables prefixed with `HINDSIGHT_API_`
- Database migrations run automatically on API startup
- Multi-bank queries require client-side orchestration (no cross-bank API)
- Use `document_id` for conversation evolution (same ID = replace/upsert)

## Version

This documentation skill is generated from the source docs in `hindsight-docs/docs/`.
Run `./scripts/generate-docs-skill.sh` to regenerate after documentation changes.
