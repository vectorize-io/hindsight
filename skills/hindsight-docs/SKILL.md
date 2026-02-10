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

## Documentation Structure

All documentation is in `references/` organized by category:

```
references/
├── developer/
│   ├── api/          # Core operations: retain, recall, reflect, memory banks
│   └── *.md          # Architecture, configuration, deployment, performance
├── sdks/
│   ├── *.md          # Python, Node.js, CLI, embedded
│   └── integrations/ # LiteLLM, AI SDK, OpenClaw, MCP, skills
└── cookbook/
    ├── recipes/      # Usage patterns and examples
    └── applications/ # Full application demos
```

## How to Find Documentation

### 1. Find Files by Pattern (use Glob tool)

```bash
# Core API operations
references/developer/api/*.md

# SDK documentation
references/sdks/*.md
references/sdks/integrations/*.md

# Cookbook examples
references/cookbook/recipes/*.md
references/cookbook/applications/*.md

# Find specific topics
references/**/configuration.md
references/**/*python*.md
references/**/*deployment*.md
```

### 2. Search Content (use Grep tool)

```bash
# Search for concepts
pattern: "disposition"        # Memory bank configuration
pattern: "graph retrieval"    # Graph-based search
pattern: "helm install"       # Kubernetes deployment
pattern: "document_id"        # Document management
pattern: "HINDSIGHT_API_"     # Environment variables

# Search in specific areas
path: references/developer/api/
pattern: "POST /v1"           # Find API endpoints

path: references/cookbook/
pattern: "def |async def "    # Find Python examples
```

### 3. Read Full Documentation (use Read tool)

```
references/developer/api/retain.md
references/sdks/python.md
references/cookbook/recipes/per-user-memory.md
```

## Key Concepts

- **Memory Banks**: Isolated memory stores (one per user/agent)
- **Retain**: Store memories (auto-extracts facts/entities/relationships)
- **Recall**: Retrieve memories (4 parallel strategies: semantic, BM25, graph, temporal)
- **Reflect**: Disposition-aware reasoning using memories
- **document_id**: Groups messages in a conversation (upsert on same ID)
- **Dispositions**: Skepticism, literalism, empathy traits (1-5) affecting reflect
- **Mental Models**: Consolidated knowledge synthesized from facts

## Notes

- Code examples are inlined from working examples
- Configuration uses `HINDSIGHT_API_*` environment variables
- Database migrations run automatically on startup
- Multi-bank queries require client-side orchestration
- Use `document_id` for conversation evolution (same ID = upsert)

---

**Auto-generated** from `hindsight-docs/docs/`. Run `./scripts/generate-docs-skill.sh` to update.
