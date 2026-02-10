#!/bin/bash
set -e

# Generate agent skill from Hindsight documentation
# Converts docs/ to skills/hindsight-docs/ for AI agent consumption

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$ROOT_DIR/hindsight-docs/docs"
EXAMPLES_DIR="$ROOT_DIR/hindsight-docs/examples"
SKILL_DIR="$ROOT_DIR/skills/hindsight-docs"
REFS_DIR="$SKILL_DIR/references"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_info "Generating Hindsight documentation skill..."

# Clean and recreate skill directory
rm -rf "$SKILL_DIR"
mkdir -p "$REFS_DIR"

# Process markdown files
process_file() {
    local src_file="$1"
    local rel_path="${src_file#$DOCS_DIR/}"
    local dest_file="$REFS_DIR/$rel_path"

    # Create destination directory
    mkdir -p "$(dirname "$dest_file")"

    # Process the file
    if [[ "$src_file" == *.mdx ]]; then
        # Change .mdx to .md
        dest_file="${dest_file%.mdx}.md"
        print_info "Converting: $rel_path"
        convert_mdx_to_md "$src_file" "$dest_file"
    else
        print_info "Copying: $rel_path"
        cp "$src_file" "$dest_file"
    fi
}

# Convert MDX to Markdown by:
# 1. Removing import statements
# 2. Replacing JSX components with markdown equivalents
# 3. Inlining code examples from example files
convert_mdx_to_md() {
    local src="$1"
    local dest="$2"

    # Use Python for more robust processing
    python3 - "$src" "$dest" "$EXAMPLES_DIR" <<'PYTHON'
import sys
import re
from pathlib import Path

src_file = Path(sys.argv[1])
dest_file = Path(sys.argv[2])
examples_dir = Path(sys.argv[3])

content = src_file.read_text()
original_content = content  # Keep original for import searches

# Remove frontmatter
content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)

# Remove import statements
content = re.sub(r'^import .*?;?\n', '', content, flags=re.MULTILINE)

# Extract code example inlining: <CodeSnippet code={varName} section="..." language="..." />
# Replace with actual code content from examples directory
def inline_code_snippet(match):
    var_name = match.group(1)
    section = match.group(2)
    language = match.group(3)

    # Find the import that loaded this variable - search in original content
    import_match = re.search(rf"import {var_name} from '!!raw-loader!@site/(.+?)';", original_content)
    if not import_match:
        return f"```{language}\n# Could not find import for: {var_name}\n```"

    # Load the example file
    # The import path is like "examples/api/quickstart.py", but examples_dir already points to examples/
    example_rel_path = import_match.group(1)
    # Strip "examples/" prefix if present since examples_dir already includes it
    if example_rel_path.startswith("examples/"):
        example_rel_path = example_rel_path[len("examples/"):]
    example_path = examples_dir / example_rel_path

    if not example_path.exists():
        return f"```{language}\n# Example file not found: {example_path}\n```"

    example_content = example_path.read_text()

    # Extract section if specified - examples use comment markers like # [docs:section] or // [docs:section]
    if section:
        # Try various comment formats: #, //, etc.
        # Pattern: (comment) [docs:section] ... (comment) [/docs:section]
        section_pattern = rf"(?:^|\n)(?:#|//)\s*\[docs:{re.escape(section)}\]\n(.*?)\n(?:#|//)\s*\[/docs:{re.escape(section)}\]"
        section_match = re.search(section_pattern, example_content, re.DOTALL | re.MULTILINE)

        if not section_match:
            # Try alternative # section-start / # section-end format
            section_pattern = rf"(?:^|\n)#\s*{re.escape(section)}-start\n(.*?)\n#\s*{re.escape(section)}-end"
            section_match = re.search(section_pattern, example_content, re.DOTALL | re.MULTILINE)

        if section_match:
            example_content = section_match.group(1).strip()
        else:
            return f"```{language}\n# Section '{section}' not found in {example_rel_path}\n```"

    return f"```{language}\n{example_content}\n```"

content = re.sub(
    r'<CodeSnippet code=\{(\w+)\} section="([^"]+)" language="([^"]+)" />',
    inline_code_snippet,
    content
)

# Convert <Tabs> to markdown sections
# Replace <Tabs> ... </Tabs> with markdown headers
content = re.sub(r'<Tabs>\s*', '', content)
content = re.sub(r'</Tabs>\s*', '', content)

# Convert <TabItem value="x" label="Y"> to ### Y
content = re.sub(r'<TabItem value="[^"]*" label="([^"]+)">', r'### \1\n', content)
content = re.sub(r'</TabItem>', '', content)

# Convert :::tip, :::warning, :::note to markdown blockquotes
content = re.sub(r':::tip (.+?)\n', r'> **💡 \1**\n> \n', content)
content = re.sub(r':::warning (.+?)\n', r'> **⚠️ \1**\n> \n', content)
content = re.sub(r':::note (.+?)\n', r'> **📝 \1**\n> \n', content)
content = re.sub(r':::\s*\n', '', content)

# Clean up extra blank lines
content = re.sub(r'\n{3,}', '\n\n', content)

dest_file.write_text(content)
PYTHON
}

# Find and process all markdown files
print_info "Processing documentation files..."
find "$DOCS_DIR" -type f \( -name "*.md" -o -name "*.mdx" \) | while read -r file; do
    process_file "$file"
done

# Generate SKILL.md
print_info "Generating SKILL.md..."
cat > "$SKILL_DIR/SKILL.md" <<'EOF'
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
EOF

print_info "✓ Generated skill at: $SKILL_DIR"
print_info "✓ Documentation files: $(find "$REFS_DIR" -type f | wc -l | tr -d ' ')"
print_info "✓ SKILL.md created with search guidance"

echo ""
print_info "Usage:"
echo "  - Agents can use Glob to find files: references/developer/api/*.md"
echo "  - Agents can use Grep to search content: pattern='disposition'"
echo "  - Agents can use Read to view full docs"
