#!/usr/bin/env python3
"""
Generates llms-full.txt by concatenating all documentation markdown files.
This file is used by LLMs to understand the full documentation.

Usage: generate-llms-full (after installing hindsight-dev)

Output: hindsight-docs/static/llms-full.txt (served at /llms-full.txt)
"""

import re
from datetime import datetime, timezone
from pathlib import Path

# Order matters - more important docs first
DOC_ORDER = [
    "developer/index.md",
    "developer/api/quickstart.md",
    "developer/api/main-methods.md",
    "developer/retain.md",
    "developer/retrieval.md",
    "developer/reflect.md",
    "developer/api/retain.md",
    "developer/api/recall.md",
    "developer/api/reflect.md",
    "developer/api/memory-banks.md",
    "developer/api/entities.md",
    "developer/api/documents.md",
    "developer/api/operations.md",
    "developer/installation.md",
    "developer/configuration.mdx",
    "developer/models.md",
    "developer/rag-vs-hindsight.md",
    "sdks/python.md",
    "sdks/nodejs.md",
    "sdks/cli.mdx",
    "sdks/mcp.md",
    "cookbook/index.mdx",
    "cookbook/recipes/quickstart.md",
    "cookbook/recipes/per-user-memory.md",
    "cookbook/recipes/support-agent-shared-knowledge.md",
    "cookbook/applications/openai-fitness-coach.md",
]


def get_docs_dir() -> Path:
    """Find the hindsight-docs directory relative to this script."""
    script_dir = Path(__file__).parent
    return script_dir.parent.parent / "hindsight-docs" / "docs"


def get_output_file() -> Path:
    """Get output file path."""
    script_dir = Path(__file__).parent
    return script_dir.parent.parent / "hindsight-docs" / "static" / "llms-full.txt"


def get_all_markdown_files(docs_dir: Path) -> list[str]:
    """Recursively find all markdown files."""
    files = []
    for path in docs_dir.rglob("*"):
        if path.suffix in (".md", ".mdx"):
            files.append(str(path.relative_to(docs_dir)))
    return files


def strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter (between --- markers)."""
    return re.sub(r"^---\n[\s\S]*?\n---\n", "", content)


def extract_section(code: str, section: str) -> str:
    """Return the lines between `[docs:section]` markers, as CodeSnippet.tsx does."""
    start = re.compile(rf"(?:#|//)\s*\[docs:{re.escape(section)}\]")
    end = re.compile(rf"(?:#|//)\s*\[/docs:{re.escape(section)}\]")
    lines: list[str] = []
    inside = False
    for line in code.split("\n"):
        if start.search(line):
            inside = True
        elif end.search(line):
            inside = False
        elif inside:
            lines.append(line)
    indent = min((len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0)
    return "\n".join(line[indent:] for line in lines).strip("\n")


def inline_code_snippets(content: str) -> str:
    """Replace <CodeSnippet> tags with the example code they render."""
    site_dir = get_docs_dir().parent
    sources = {
        var: (site_dir / path).read_text()
        for var, path in re.findall(r"^import\s+(\w+)\s+from\s+'!!raw-loader!@site/(.+?)';", content, re.MULTILINE)
    }

    def render(match: re.Match[str]) -> str:
        var, section, language = match.groups()
        code = extract_section(sources[var], section)
        if not code:
            raise ValueError(f"CodeSnippet section {section!r} not found in {var}")
        return f"```{language}\n{code}\n```"

    return re.sub(r'<CodeSnippet\s+code=\{(\w+)\}\s+section="([^"]+)"\s+language="([^"]+)"[^>]*/>', render, content)


def clean_markdown(content: str) -> str:
    """Clean markdown content for LLM consumption."""
    cleaned = inline_code_snippets(strip_frontmatter(content))

    # Remove the page's MDX imports (all from @site/ or @theme/), not import lines inside code examples
    cleaned = re.sub(r"^import\s+.*\s+from\s+'(?:!!raw-loader!)?@(?:site|theme)/.*$", "", cleaned, flags=re.MULTILINE)

    # Remove JSX components (like <RecipeCarousel ... />) and the Tabs wrappers
    # around the inlined snippets, keeping each tab's code
    cleaned = re.sub(r'<TabItem value="[^"]*" label="([^"]+)">', r"**\1**", cleaned)
    cleaned = re.sub(r"</?(Tabs|TabItem)>", "", cleaned)
    cleaned = re.sub(r"<[A-Z][a-zA-Z]*\s+[^>]*/>", "", cleaned)
    cleaned = re.sub(r"<[A-Z][a-zA-Z]*[^>]*>[\s\S]*?</[A-Z][a-zA-Z]*>", "", cleaned)

    # Remove empty lines at start
    cleaned = re.sub(r"^\n+", "", cleaned)

    return cleaned


def main():
    """Generate llms-full.txt."""
    print("Generating llms-full.txt...")

    docs_dir = get_docs_dir()
    output_file = get_output_file()

    # Get all markdown files
    all_files = get_all_markdown_files(docs_dir)

    # Create ordered list: prioritized files first, then remaining files
    ordered_files = []
    remaining_files = set(all_files)

    # Add prioritized files in order
    # Match on the name without extension, so a page renamed .md -> .mdx keeps its place
    for entry in DOC_ORDER:
        for file in (entry, str(Path(entry).with_suffix(".md")), str(Path(entry).with_suffix(".mdx"))):
            if file in remaining_files:
                ordered_files.append(file)
                remaining_files.discard(file)
                break

    # Add remaining files (sorted alphabetically)
    ordered_files.extend(sorted(remaining_files))

    # Build the output
    sections = []

    # Header
    timestamp = datetime.now(timezone.utc).isoformat()
    sections.append(f"""# Hindsight Documentation

> Agent Memory That Learns

This file contains the complete Hindsight documentation for LLM consumption.
Generated: {timestamp}

---
""")

    # Process each file
    for file in ordered_files:
        file_path = docs_dir / file

        if not file_path.exists():
            print(f"  Warning: {file} not found, skipping")
            continue

        content = file_path.read_text()
        cleaned_content = clean_markdown(content)

        if cleaned_content.strip():
            sections.append(f"\n## File: {file}\n")
            sections.append(cleaned_content)
            sections.append("\n---\n")
            print(f"  Added: {file}")

    # Write output
    output = "\n".join(sections)
    output_file.write_text(output)

    size_kb = output_file.stat().st_size / 1024

    print(f"\nGenerated: {output_file}")
    print(f"Size: {size_kb:.1f} KB")
    print(f"Files included: {len(ordered_files)}")


if __name__ == "__main__":
    main()
