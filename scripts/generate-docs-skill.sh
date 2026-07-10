#!/bin/bash
set -e

# Generate agent skill from Hindsight documentation
# Converts docs/ to skills/hindsight-docs/ for AI agent consumption

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$ROOT_DIR/hindsight-docs/docs"
PAGES_DIR="$ROOT_DIR/hindsight-docs/src/pages"
INTEGRATIONS_DIR="$ROOT_DIR/hindsight-docs/docs-integrations"
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
    python3 - "$src" "$dest" "$EXAMPLES_DIR" "$ROOT_DIR/hindsight-docs/src/data/llmProviders.json" <<'PYTHON'
import json
import sys
import re
from pathlib import Path

src_file = Path(sys.argv[1])
dest_file = Path(sys.argv[2])
examples_dir = Path(sys.argv[3])
llm_providers_json = Path(sys.argv[4])

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

# Render <LLMProvidersTable /> as a markdown table sourced from
# hindsight-docs/src/data/llmProviders.json (single source of truth shared
# with the React grid + table components).
def render_llm_providers_table(_match):
    providers = json.loads(llm_providers_json.read_text())
    rows = ["| Provider | Default Model |", "|----------|--------------|"]
    for p in providers:
        if not p.get("id") or not p.get("defaultModel"):
            continue
        model_cell = f"`{p['defaultModel']}`"
        if p.get("defaultModelNote"):
            model_cell += f" ({p['defaultModelNote']})"
        rows.append(f"| `{p['id']}` | {model_cell} |")
    return "\n".join(rows)

content = re.sub(r'<LLMProvidersTable\s*/>', render_llm_providers_table, content)

# Render <LLMProvidersGrid /> as a bullet list of supported provider labels.
def render_llm_providers_grid(_match):
    providers = json.loads(llm_providers_json.read_text())
    return "\n".join(f"- {p['label']}" for p in providers)

content = re.sub(r'<LLMProvidersGrid\s*/>', render_llm_providers_grid, content)

# Render <LLMProviderCapabilities /> as a markdown capability table, sourced from
# the same single-source-of-truth provider list (batchApi / promptCaching flags).
def render_llm_capabilities_table(_match):
    providers = json.loads(llm_providers_json.read_text())
    rows = [
        "| Provider | Batch API | Explicit prompt caching |",
        "|----------|:---------:|:-----------------------:|",
    ]
    for p in providers:
        if not p.get("id"):
            continue
        batch = "✅" if p.get("batchApi") else "—"
        cache = "✅" if p.get("promptCaching") else "—"
        rows.append(f"| {p['label']} (`{p['id']}`) | {batch} | {cache} |")
    return "\n".join(rows)

content = re.sub(r'<LLMProviderCapabilities\s*/>', render_llm_capabilities_table, content)

def render_cookbook_grid(match):
    items = match.group(1)
    rows = []
    for item in re.finditer(
        r'title:\s*"([^"]+)".*?href:\s*"([^"]+)".*?description:\s*"([^"]+)"',
        items,
        re.DOTALL,
    ):
        title, href, description = item.groups()
        rows.append(f"- [{title}]({href}) — {description}")
    return "\n".join(rows) if rows else ""

content = re.sub(r'<PageHero\s+title="([^"]+)"\s+subtitle="([^"]+)"\s*/>', r'# \1\n\n\2', content)
content = re.sub(r'<CookbookGrid\s+items=\{\[(.*?)\]\}\s*/>', render_cookbook_grid, content, flags=re.DOTALL)
content = re.sub(r'</?div>\s*', '', content)

# Convert <Tabs> to markdown sections
# Replace <Tabs> ... </Tabs> with markdown headers
content = re.sub(r'<Tabs>\s*', '', content)
content = re.sub(r'</Tabs>\s*', '', content)

# Convert <TabItem value="x" label="Y"> to ### Y
content = re.sub(r'<TabItem value="[^"]*" label="([^"]+)">', r'### \1\n', content)
content = re.sub(r'</TabItem>', '', content)

# Convert Docusaurus admonitions (:::tip/:::note/:::warning/:::info/:::caution,
# with or without an inline title) to markdown blockquotes. The title is
# optional so title-less openers like a bare ':::note' are handled too, and
# every supported keyword is covered so none leaks through as raw markdown.
_ADMONITION_EMOJI = {
    'tip': '💡',
    'note': '📝',
    'info': 'ℹ️',
    'warning': '⚠️',
    'caution': '🚨',
}
def _convert_admonition(m):
    kind = m.group(1)
    title = m.group(2)
    emoji = _ADMONITION_EMOJI[kind]
    heading = f'{emoji} {title}' if title else f'{emoji} {kind.capitalize()}'
    return f'> **{heading}**\n>\n'
content = re.sub(r':{3,4}(tip|note|warning|info|caution)(?: (.+?))?\n', _convert_admonition, content)
content = re.sub(r':{3,4}\s*\n', '', content)

# Clean up extra blank lines
content = re.sub(r'\n{3,}', '\n\n', content)
content = "\n".join(line.rstrip() for line in content.splitlines()).rstrip() + "\n"

dest_file.write_text(content)
PYTHON
}

process_reference_tree() {
    local src_root="$1"
    local dest_root="$2"
    local label="$3"

    if [ ! -d "$src_root" ]; then
        print_warn "$label not found at $src_root — skipping"
        return
    fi

    find "$src_root" -type f \( -name "*.md" -o -name "*.mdx" \) | while read -r file; do
        rel="${file#$src_root/}"
        dest="$dest_root/$rel"
        if [[ "$file" == *.mdx ]]; then
            dest="${dest%.mdx}.md"
        fi
        mkdir -p "$(dirname "$dest")"
        if [[ "$file" == *.mdx ]]; then
            convert_mdx_to_md "$file" "$dest"
        else
            cp "$file" "$dest"
            normalize_markdown_file "$dest"
        fi
        print_info "Included $label: $rel"
    done
}

normalize_markdown_file() {
    local file="$1"

    python3 - "$file" <<'PYTHON'
import re
import sys
from pathlib import Path

md_file = Path(sys.argv[1])
content = md_file.read_text()

content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)

admonition_emoji = {
    'tip': '💡',
    'note': '📝',
    'info': 'ℹ️',
    'warning': '⚠️',
    'caution': '🚨',
}

def convert_admonition(match):
    kind = match.group(1)
    title = match.group(2)
    emoji = admonition_emoji[kind]
    heading = f'{emoji} {title}' if title else f'{emoji} {kind.capitalize()}'
    return f'> **{heading}**\n>\n'

content = re.sub(
    r'(?m)^:{3,4}(tip|note|warning|info|caution)(?: (.+?))?\n',
    convert_admonition,
    content,
)
content = re.sub(r'(?m)^:{3,4}\s*\n', '', content)
content = re.sub(r'\n{3,}', '\n\n', content)
content = "\n".join(line.rstrip() for line in content.splitlines()).rstrip() + "\n"

md_file.write_text(content)
PYTHON
}

# Find and process all markdown files
print_info "Processing documentation files..."
find "$DOCS_DIR" -type f \( -name "*.md" -o -name "*.mdx" \) | while read -r file; do
    process_file "$file"
done

# Process standalone pages (e.g. best-practices, faq) from src/pages/
print_info "Processing standalone pages..."
for page in best-practices faq; do
    for ext in md mdx; do
        src="$PAGES_DIR/$page.$ext"
        if [ -f "$src" ]; then
            dest="$REFS_DIR/$page.md"
            mkdir -p "$(dirname "$dest")"
            if [[ "$src" == *.mdx ]]; then
                convert_mdx_to_md "$src" "$dest"
            else
                cp "$src" "$dest"
            fi
            print_info "Included page: $page.$ext"
        fi
    done
done

# Process changelog — may be a single file or a directory
if [ -f "$PAGES_DIR/changelog.md" ] || [ -f "$PAGES_DIR/changelog.mdx" ]; then
    for ext in md mdx; do
        src="$PAGES_DIR/changelog.$ext"
        if [ -f "$src" ]; then
            dest="$REFS_DIR/changelog.md"
            mkdir -p "$(dirname "$dest")"
            if [[ "$src" == *.mdx ]]; then
                convert_mdx_to_md "$src" "$dest"
            else
                cp "$src" "$dest"
            fi
            print_info "Included page: changelog.$ext"
        fi
    done
elif [ -d "$PAGES_DIR/changelog" ]; then
    find "$PAGES_DIR/changelog" -type f \( -name "*.md" -o -name "*.mdx" \) | while read -r file; do
        rel="${file#$PAGES_DIR/}"
        dest="$REFS_DIR/$rel"
        if [[ "$file" == *.mdx ]]; then
            dest="${dest%.mdx}.md"
        fi
        mkdir -p "$(dirname "$dest")"
        if [[ "$file" == *.mdx ]]; then
            convert_mdx_to_md "$file" "$dest"
        else
            cp "$file" "$dest"
        fi
        print_info "Included changelog: ${file#$PAGES_DIR/changelog/}"
    done
fi

print_info "Processing cookbook pages..."
process_reference_tree "$PAGES_DIR/cookbook" "$REFS_DIR/cookbook" "cookbook"

print_info "Processing integration docs..."
process_reference_tree "$INTEGRATIONS_DIR" "$REFS_DIR/sdks/integrations" "integration"

# Copy OpenAPI spec into the skill
OPENAPI_SRC="$ROOT_DIR/hindsight-docs/static/openapi.json"
if [ -f "$OPENAPI_SRC" ]; then
    cp "$OPENAPI_SRC" "$REFS_DIR/openapi.json"
    print_info "Included: openapi.json"
else
    print_warn "openapi.json not found at $OPENAPI_SRC — skipping"
fi

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

## Documentation Structure

All documentation is in `references/` organized by category:

```
references/
├── best-practices.md # START HERE — missions, tags, formats, anti-patterns
├── faq.md            # Common questions and decisions
├── changelog/        # Release history and version changes (index.md + integrations/)
├── openapi.json      # Full OpenAPI spec — endpoint schemas, request/response models
├── developer/
│   ├── api/          # Core operations: retain, recall, reflect, memory banks
│   └── *.md          # Architecture, configuration, deployment, performance
├── sdks/
│   ├── *.md          # Python, Node.js, CLI, embedded
│   └── integrations/ # Framework and tool integrations
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

## Start Here: Best Practices

Before reading API docs, read the best practices guide. It covers practical rules for missions, tags, content format, observation scopes, and anti-patterns — the fastest way to integrate correctly.

```
references/best-practices.md
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

**Auto-generated** from `hindsight-docs/docs/`, `hindsight-docs/src/pages/cookbook/`, and `hindsight-docs/docs-integrations/`. Run `./scripts/generate-docs-skill.sh` to update.
EOF

print_info "✓ Generated skill at: $SKILL_DIR"
print_info "✓ Documentation files: $(find "$REFS_DIR" -type f | wc -l | tr -d ' ')"
print_info "✓ SKILL.md created with search guidance"

# Rewrite Docusaurus absolute paths (e.g. /developer/foo) to relative paths
print_info "Rewriting Docusaurus absolute paths to relative paths..."
python3 - "$REFS_DIR" <<'PYTHON'
import sys
import re
import os
from pathlib import Path

refs_dir = Path(sys.argv[1]).resolve()
link_pattern = re.compile(r'\[([^\]]*)\]\((/[^)]*)\)')

SPECIAL_MAPPINGS = {
    '/api-reference': 'openapi.json',
}

def try_resolve(url_path, refs_dir):
    """Try to find the file in refs_dir for a Docusaurus absolute path like /developer/foo."""
    if url_path in SPECIAL_MAPPINGS:
        candidate = refs_dir / SPECIAL_MAPPINGS[url_path]
        return candidate if candidate.exists() else None
    doc_path = url_path.lstrip('/')
    for candidate in [
        refs_dir / (doc_path + '.md'),
        refs_dir / doc_path / 'index.md',
        refs_dir / doc_path,
    ]:
        if candidate.exists():
            return candidate
    return None

image_pattern = re.compile(r'!\[[^\]]*\]\([^)]*\)')
html_img_pattern = re.compile(r'<img\b[^>]*/?>',  re.IGNORECASE)

changed = 0
for md_file in refs_dir.rglob("*.md"):
    original_content = md_file.read_text()

    # Strip images (markdown and HTML)
    content = image_pattern.sub('', original_content)
    content = html_img_pattern.sub('', content)

    def rewrite(match):
        text = match.group(1)
        url = match.group(2)
        anchor = ''
        if '#' in url:
            url, frag = url.split('#', 1)
            anchor = '#' + frag
        if not url or url == '/':
            return text  # strip link, keep text
        resolved = try_resolve(url, refs_dir)
        if resolved is None:
            return text  # strip unresolvable link, keep text
        rel = os.path.relpath(resolved, md_file.parent)
        return f'[{text}]({rel}{anchor})'

    new_content = link_pattern.sub(rewrite, content)
    if new_content != original_content:
        md_file.write_text(new_content)
        changed += 1

print(f"[INFO] Rewrote Docusaurus links in {changed} file(s)")
PYTHON

# Validate: no links point outside the skill directory
print_info "Validating links in generated skill files..."
python3 - "$SKILL_DIR" <<'PYTHON'
import sys
import re
from pathlib import Path

skill_dir = Path(sys.argv[1]).resolve()
errors = []

# Find all markdown links: [text](url) — exclude images too
link_pattern = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')

for md_file in skill_dir.rglob("*.md"):
    content = md_file.read_text()
    for match in link_pattern.finditer(content):
        url = match.group(2).split("#")[0].strip()  # strip anchors
        if not url:
            continue
        # Absolute URLs and anchors-only are fine
        if url.startswith(("http://", "https://", "mailto:", "ftp://")):
            continue
        # Resolve relative to the file's directory
        resolved = (md_file.parent / url).resolve()
        if not str(resolved).startswith(str(skill_dir)):
            errors.append(f"  {md_file.relative_to(skill_dir)}: '{url}' -> {resolved}")

if errors:
    print("ERROR: The following links point outside the skill directory.")
    print("All links must be absolute URLs or relative paths within the skill.")
    for e in errors:
        print(e)
    sys.exit(1)

print(f"[INFO] Link validation passed ({skill_dir})")
PYTHON

echo ""
print_info "Usage:"
echo "  - Agents can use Glob to find files: references/developer/api/*.md"
echo "  - Agents can use Grep to search content: pattern='disposition'"
echo "  - Agents can use Read to view full docs"
