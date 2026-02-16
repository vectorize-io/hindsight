#!/usr/bin/env python3
"""
Syncs content from the hindsight-cookbook repository.

- Clones the cookbook repo to a temp directory
- Converts notebooks/*.ipynb → docs/cookbook/recipes/*.md
- Converts applications/*/ directories (with README.md) → docs/cookbook/applications/*.md
- Updates sidebars.ts with the new entries

Usage: sync-cookbook (after installing hindsight-dev)

Conventions in cookbook repo:
- notebooks/*.ipynb → Recipes (use cases, tutorials)
- applications/*/ directories with README.md → Applications (complete apps)
- Notebook title extracted from first # heading in first markdown cell
- App title extracted from first # heading in README.md
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

COOKBOOK_REPO = "https://github.com/vectorize-io/hindsight-cookbook.git"
IGNORE_DIRS = {".git", "notebooks", "node_modules", "__pycache__", ".venv", "venv"}


def get_docs_dir() -> Path:
    """Find the hindsight-docs directory relative to this script."""
    # Navigate from hindsight-dev to hindsight-docs
    script_dir = Path(__file__).parent
    docs_dir = script_dir.parent.parent / "hindsight-docs" / "docs" / "cookbook"
    return docs_dir


def get_sidebars_file() -> Path:
    script_dir = Path(__file__).parent
    return script_dir.parent.parent / "hindsight-docs" / "sidebars.ts"


def slugify(filename: str) -> str:
    """Convert filename to slug. e.g., '01-quickstart.ipynb' → 'quickstart'"""
    slug = re.sub(r"\.ipynb$", "", filename)
    slug = re.sub(r"\.md$", "", slug)
    slug = re.sub(r"^\d+-", "", slug)
    return slug


def extract_title_from_notebook(notebook_path: Path) -> str:
    """Extract title from first markdown cell's # heading."""
    try:
        content = json.loads(notebook_path.read_text())
        for cell in content.get("cells", []):
            if cell.get("cell_type") == "markdown":
                source = cell.get("source", [])
                if isinstance(source, list):
                    source = "".join(source)
                match = re.search(r"^#\s+(.+)$", source, re.MULTILINE)
                if match:
                    return match.group(1).strip()
    except Exception as e:
        print(f"  Warning: Could not parse notebook {notebook_path}: {e}")

    # Fallback to filename
    slug = slugify(notebook_path.name)
    return " ".join(word.capitalize() for word in slug.split("-"))


def extract_description_from_notebook(notebook_path: Path) -> str | None:
    """Extract description from notebook metadata."""
    try:
        content = json.loads(notebook_path.read_text())
        metadata = content.get("metadata", {})
        description = metadata.get("description", "")
        if description:
            return description[:200]
    except Exception:
        pass
    return None


def extract_tags_from_notebook(notebook_path: Path) -> list[str]:
    """Extract tags from notebook metadata.

    Supports both array format and structured object format.
    """
    try:
        content = json.loads(notebook_path.read_text())
        metadata = content.get("metadata", {})
        tags = metadata.get("tags", [])

        # Array format: ["Python", "Client"]
        if isinstance(tags, list):
            return tags

        # Object format: { "language": "Python", "sdk": "Client", "topic": "Learning" }
        if isinstance(tags, dict):
            result = []
            for key in ["language", "sdk", "topic"]:
                if key in tags and tags[key]:
                    result.append(tags[key])
            return result
    except Exception:
        pass
    return []


def extract_description_from_readme(readme_path: Path) -> str | None:
    """Extract description from frontmatter in README."""
    try:
        content = readme_path.read_text()
        # Check for frontmatter
        if content.startswith("---"):
            end_idx = content.find("---", 3)
            if end_idx > 0:
                frontmatter = content[3:end_idx]
                # Look for description: line
                for line in frontmatter.split("\n"):
                    if line.strip().startswith("description:"):
                        desc = line.split("description:", 1)[1].strip()
                        # Remove quotes if present
                        desc = desc.strip('"').strip("'")
                        return desc[:200]
    except Exception:
        pass
    return None


def extract_tags_from_readme(readme_path: Path) -> list[str]:
    """Extract tags from frontmatter in README if present.

    Supports multiple formats:
    - Array: tags: ["Python", "Client"]
    - Structured YAML: tags:\n  language: "Python"\n  sdk: "Client"
    - Object literal: tags: { language: "Python", sdk: "Client" }
    """
    try:
        content = readme_path.read_text()
        # Check for frontmatter
        if content.startswith("---"):
            end_idx = content.find("---", 3)
            if end_idx > 0:
                frontmatter = content[3:end_idx]
                lines = frontmatter.split("\n")

                # Look for tags: line
                for i, line in enumerate(lines):
                    if line.strip().startswith("tags:"):
                        tags_str = line.split("tags:", 1)[1].strip()

                        # Inline array format: tags: ["Python", "Client"]
                        if tags_str.startswith("["):
                            tags_str = tags_str.strip("[]")
                            return [t.strip().strip('"').strip("'") for t in tags_str.split(",")]

                        # JavaScript object literal format: tags: { language: "Python", sdk: "Client", topic: "Learning" }
                        if tags_str.startswith("{"):
                            tags = []
                            # Extract the entire object literal (might span multiple lines)
                            obj_str = tags_str
                            if "}" not in obj_str:
                                # Multi-line object - collect remaining lines
                                for j in range(i + 1, len(lines)):
                                    obj_str += " " + lines[j].strip()
                                    if "}" in lines[j]:
                                        break

                            # Parse the object literal
                            obj_str = obj_str.strip("{}")
                            # Split by comma and extract key-value pairs
                            for pair in obj_str.split(","):
                                if ":" in pair:
                                    key, value = pair.split(":", 1)
                                    value = value.strip().strip('"').strip("'")
                                    if value:
                                        tags.append(value)
                            return tags

                        # Structured YAML format:
                        # tags:
                        #   language: "Python"
                        #   sdk: "Client"
                        if not tags_str or tags_str == "":
                            # Parse structured tags from following lines
                            tags = []
                            for j in range(i + 1, len(lines)):
                                next_line = lines[j].strip()
                                if not next_line or not next_line.startswith(("language:", "sdk:", "topic:")):
                                    break
                                # Extract value
                                if ":" in next_line:
                                    value = next_line.split(":", 1)[1].strip().strip('"').strip("'")
                                    if value:
                                        tags.append(value)
                            return tags
    except Exception:
        pass
    return []


def extract_title_from_readme(readme_path: Path) -> str | None:
    """Extract title from README's first # heading."""
    try:
        content = readme_path.read_text()
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
    except Exception as e:
        print(f"  Warning: Could not read {readme_path}: {e}")
    return None


def convert_notebook_to_markdown(notebook_path: Path) -> str:
    """Convert Jupyter notebook to markdown.

    Uses nbconvert with --no-input to exclude outputs (which often contain
    characters that break MDX parsing).
    """
    # Try nbconvert first
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "markdown",
                    "--TemplateExporter.exclude_output=True",  # Exclude cell outputs
                    str(notebook_path),
                    "--output-dir",
                    tmpdir,
                ],
                capture_output=True,
                check=True,
            )
            md_file = Path(tmpdir) / notebook_path.with_suffix(".md").name
            if md_file.exists():
                return md_file.read_text()
    except Exception as e:
        print(f"  Warning: nbconvert failed ({e}), using fallback parser")

    # Fallback: manual conversion
    return convert_notebook_manually(notebook_path)


def convert_notebook_manually(notebook_path: Path) -> str:
    """Manually convert notebook to markdown.

    Note: We skip cell outputs to avoid MDX parsing issues (outputs often contain
    characters like < and > that get interpreted as JSX tags).
    """
    content = json.loads(notebook_path.read_text())
    parts = []

    lang = content.get("metadata", {}).get("kernelspec", {}).get("language", "python")

    for cell in content.get("cells", []):
        source = cell.get("source", [])
        if isinstance(source, list):
            source = "".join(source)

        if cell.get("cell_type") == "markdown":
            parts.append(source)
        elif cell.get("cell_type") == "code":
            parts.append(f"```{lang}\n{source}\n```")
            # Skip outputs - they often contain characters that break MDX parsing

    return "\n\n".join(parts)


def process_notebooks(cookbook_dir: Path, recipes_dir: Path) -> list[dict]:
    """Process all notebooks and convert to recipe markdown files."""
    notebooks_dir = cookbook_dir / "notebooks"
    recipes = []

    if not notebooks_dir.exists():
        print("  No notebooks directory found")
        return recipes

    files = sorted(f for f in notebooks_dir.iterdir() if f.suffix == ".ipynb")
    print(f"  Found {len(files)} notebooks")

    for i, notebook_path in enumerate(files):
        slug = slugify(notebook_path.name)
        title = extract_title_from_notebook(notebook_path)
        description = extract_description_from_notebook(notebook_path)
        tags = extract_tags_from_notebook(notebook_path)

        print(f"  Processing: {notebook_path.name} → {slug}.md")

        # Convert notebook to markdown
        md_content = convert_notebook_to_markdown(notebook_path)

        # Strip any existing frontmatter from converted notebook
        md_content = strip_frontmatter(md_content)

        # Create recipe page with frontmatter
        notebook_url = f"https://github.com/vectorize-io/hindsight-cookbook/blob/main/notebooks/{notebook_path.name}"

        frontmatter = f"""---
sidebar_position: {i + 1}
---

"""

        callout = f"""
:::tip Run this notebook
This recipe is available as an interactive Jupyter notebook.
[**Open in GitHub →**]({notebook_url})
:::
"""

        # Insert callout after first heading
        first_heading_match = re.search(r"^(#\s+.+\n)", md_content, re.MULTILINE)
        if first_heading_match:
            idx = md_content.index(first_heading_match.group(0)) + len(first_heading_match.group(0))
            final_content = md_content[:idx] + "\n" + callout + "\n" + md_content[idx:]
        else:
            final_content = callout + "\n" + md_content

        output_path = recipes_dir / f"{slug}.md"
        output_path.write_text(frontmatter + final_content)

        recipes.append(
            {
                "slug": slug,
                "title": title,
                "description": description,
                "tags": tags,
                "id": f"cookbook/recipes/{slug}",
            }
        )

    return recipes


def strip_frontmatter(content: str) -> str:
    """Remove frontmatter from markdown content."""
    if content.startswith("---"):
        end_idx = content.find("---", 3)
        if end_idx > 0:
            return content[end_idx + 3 :].lstrip()
    return content


def process_applications(cookbook_dir: Path, apps_dir: Path) -> list[dict]:
    """Process application directories with README.md."""
    apps = []

    # Applications are now in the applications/ subdirectory
    applications_dir = cookbook_dir / "applications"
    if not applications_dir.exists():
        print("  No applications directory found")
        return apps

    for entry in sorted(applications_dir.iterdir()):
        if not entry.is_dir() or entry.name in IGNORE_DIRS:
            continue

        readme_path = entry / "README.md"
        if not readme_path.exists():
            continue

        slug = entry.name
        title = extract_title_from_readme(readme_path) or " ".join(word.capitalize() for word in slug.split("-"))
        description = extract_description_from_readme(readme_path)
        tags = extract_tags_from_readme(readme_path)

        print(f"  Processing app: {entry.name} → {slug}.md")

        # Read README content and strip existing frontmatter
        readme_content = readme_path.read_text()
        readme_content = strip_frontmatter(readme_content)

        # Create application page with frontmatter
        app_url = f"https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/{entry.name}"

        frontmatter = f"""---
sidebar_position: {len(apps) + 1}
---

"""

        callout = f"""
:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**]({app_url})
:::
"""

        # Insert callout after first heading
        first_heading_match = re.search(r"^(#\s+.+\n)", readme_content, re.MULTILINE)
        if first_heading_match:
            idx = readme_content.index(first_heading_match.group(0)) + len(first_heading_match.group(0))
            final_content = readme_content[:idx] + "\n" + callout + "\n" + readme_content[idx:]
        else:
            final_content = callout + "\n" + readme_content

        output_path = apps_dir / f"{slug}.md"
        output_path.write_text(frontmatter + final_content)

        apps.append(
            {
                "slug": slug,
                "title": title,
                "description": description,
                "tags": tags,
                "id": f"cookbook/applications/{slug}",
            }
        )

    return apps


def update_sidebars(recipes: list[dict], apps: list[dict], sidebars_file: Path):
    """Update sidebars.ts - keep it simple with just the index."""
    content = sidebars_file.read_text()

    # Simple sidebar with just the cookbook index
    new_cookbook_sidebar = """cookbookSidebar: [
    {
      type: 'doc',
      id: 'cookbook/index',
      label: 'Cookbook',
    },
  ]"""

    # Replace existing cookbookSidebar
    start = content.find("cookbookSidebar:")
    if start == -1:
        raise ValueError("cookbookSidebar not found in sidebars.ts")

    # Find the opening bracket
    bracket_start = content.find("[", start)
    if bracket_start == -1:
        raise ValueError("Could not find opening bracket for cookbookSidebar")

    # Find matching closing bracket by counting brackets
    depth = 0
    end = bracket_start
    for i, char in enumerate(content[bracket_start:], bracket_start):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    # Include trailing comma if present
    if end < len(content) and content[end] == ",":
        end += 1

    content = content[:start] + new_cookbook_sidebar + "," + content[end:]

    sidebars_file.write_text(content)
    print("\nUpdated sidebars.ts")


def clean_description(desc: str) -> str:
    """Clean description for display in carousel cards."""
    if not desc:
        return ""
    # Remove markdown formatting
    desc = re.sub(r"\*\*([^*]+)\*\*", r"\1", desc)  # Bold
    desc = re.sub(r"\*([^*]+)\*", r"\1", desc)  # Italic
    desc = re.sub(r"`([^`]+)`", r"\1", desc)  # Code
    desc = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", desc)  # Links
    desc = re.sub(r"^[-*]\s+", "", desc)  # List items
    desc = re.sub(r"\s+", " ", desc).strip()  # Normalize whitespace
    # Truncate at sentence boundary or max length
    if len(desc) > 120:
        # Try to cut at sentence
        period_idx = desc.rfind(".", 0, 120)
        if period_idx > 60:
            desc = desc[: period_idx + 1]
        else:
            desc = desc[:117] + "..."
    return desc


def convert_tags_to_structured(tags: list[str]) -> dict[str, str]:
    """Convert list of tags to structured format.

    New format has 2 tags:
    - sdk: Package name (detected from tag values)
    - topic: anything else (Learning, Quick Start, etc.)

    Supported languages:
    - Node.js: packages starting with '@vectorize-io'
    - Go: packages ending with '-go' or containing 'go-'
    - Python: everything else
    """
    structured = {}
    topic_tags = {"Learning", "Quick Start", "Recommendation", "Chat"}

    for tag in tags:
        # Check if it's a topic tag
        if tag in topic_tags:
            structured["topic"] = tag
        # Check if it's already a package name (contains @ or -)
        elif "@" in tag or (tag and not tag[0].isupper()):
            structured["sdk"] = tag
        else:
            # Legacy tag values - map to new format
            # For now, treat everything else as SDK/package identifier
            structured["sdk"] = tag

    return structured


def update_cookbook_index(recipes: list[dict], apps: list[dict], docs_dir: Path):
    """Update cookbook/index.mdx with recipe and app carousels."""
    # Build recipe items for the carousel with descriptions and tags
    recipe_items = []
    for r in recipes:
        title = r["title"].replace('"', '\\"')
        description = r.get("description", "")
        if description:
            description = clean_description(description).replace('"', '\\"')
        tags = r.get("tags", [])

        item = f'    {{\n      title: "{title}",\n      href: "/cookbook/recipes/{r["slug"]}"'
        if description:
            item += f',\n      description: "{description}"'
        if tags:
            # Convert tags list to structured format
            structured_tags = convert_tags_to_structured(tags)
            tags_parts = []
            if "language" in structured_tags:
                tags_parts.append(f'language: "{structured_tags["language"]}"')
            if "sdk" in structured_tags:
                tags_parts.append(f'sdk: "{structured_tags["sdk"]}"')
            if "topic" in structured_tags:
                tags_parts.append(f'topic: "{structured_tags["topic"]}"')
            if tags_parts:
                item += f",\n      tags: {{ {', '.join(tags_parts)} }}"
        item += "\n    }"
        recipe_items.append(item)

    recipes_json = ",\n".join(recipe_items)

    # Build app items for the carousel
    app_items = []
    for a in apps:
        title = a["title"].replace('"', '\\"')
        description = a.get("description", "")
        if description:
            description = clean_description(description).replace('"', '\\"')
        tags = a.get("tags", [])

        item = f'    {{\n      title: "{title}",\n      href: "/cookbook/applications/{a["slug"]}"'
        if description:
            item += f',\n      description: "{description}"'
        if tags:
            # Convert tags list to structured format
            structured_tags = convert_tags_to_structured(tags)
            tags_parts = []
            if "language" in structured_tags:
                tags_parts.append(f'language: "{structured_tags["language"]}"')
            if "sdk" in structured_tags:
                tags_parts.append(f'sdk: "{structured_tags["sdk"]}"')
            if "topic" in structured_tags:
                tags_parts.append(f'topic: "{structured_tags["topic"]}"')
            if tags_parts:
                item += f",\n      tags: {{ {', '.join(tags_parts)} }}"
        item += "\n    }"
        app_items.append(item)

    apps_json = ",\n".join(app_items)

    content = f"""---
sidebar_position: 1
hide_table_of_contents: true
pagination_next: null
pagination_prev: null
custom_edit_url: null
sidebar_class_name: hidden-sidebar
---

import RecipeCarousel from '@site/src/components/RecipeCarousel';

<div className="cookbook-page">

# Cookbook

Learn how to build with Hindsight through practical examples:

- **[Recipes](#recipes)** - Step-by-step guides and patterns for common use cases
- **[Applications](#applications)** - Complete, runnable applications demonstrating Hindsight integration

<RecipeCarousel
  title="Recipes"
  items={{[
{recipes_json}
  ]}}
/>

<RecipeCarousel
  title="Applications"
  items={{[
{apps_json}
  ]}}
/>

</div>
"""

    index_path = docs_dir / "index.mdx"
    index_path.write_text(content)

    # Remove old .md if exists
    old_index = docs_dir / "index.md"
    if old_index.exists():
        old_index.unlink()

    print("Updated cookbook/index.mdx")


def extract_existing_entries(docs_dir: Path) -> tuple[list[dict], list[dict]]:
    """Extract existing recipe and app entries before syncing.

    This allows us to preserve manually added entries that aren't in the cookbook repo.
    Returns entries with their content stored in memory.
    """
    existing_recipes = []
    existing_apps = []

    recipes_dir = docs_dir / "recipes"
    apps_dir = docs_dir / "applications"

    # Scan existing recipes
    if recipes_dir.exists():
        for md_file in recipes_dir.glob("*.md"):
            slug = md_file.stem
            # Read file content
            content = md_file.read_text()
            # Try to extract title from first heading
            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            title = (
                title_match.group(1).strip() if title_match else " ".join(word.capitalize() for word in slug.split("-"))
            )

            existing_recipes.append(
                {
                    "slug": slug,
                    "title": title,
                    "id": f"cookbook/recipes/{slug}",
                    "content": content,  # Store content in memory
                }
            )

    # Scan existing apps
    if apps_dir.exists():
        for md_file in apps_dir.glob("*.md"):
            slug = md_file.stem
            # Read file content
            content = md_file.read_text()
            # Try to extract title from first heading
            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            title = (
                title_match.group(1).strip() if title_match else " ".join(word.capitalize() for word in slug.split("-"))
            )

            existing_apps.append(
                {
                    "slug": slug,
                    "title": title,
                    "id": f"cookbook/applications/{slug}",
                    "content": content,  # Store content in memory
                }
            )

    return existing_recipes, existing_apps


def main():
    """Main entry point."""
    print("Syncing hindsight-cookbook...\n")

    docs_dir = get_docs_dir()
    sidebars_file = get_sidebars_file()
    recipes_dir = docs_dir / "recipes"
    apps_dir = docs_dir / "applications"

    # Extract existing entries before we delete anything
    print("Scanning for existing manual entries...")
    existing_recipes, existing_apps = extract_existing_entries(docs_dir)
    print(f"  Found {len(existing_recipes)} existing recipes, {len(existing_apps)} existing apps")

    # Create temp directory and clone
    with tempfile.TemporaryDirectory() as tmpdir:
        cookbook_dir = Path(tmpdir) / "cookbook"

        print(f"\nCloning {COOKBOOK_REPO}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", COOKBOOK_REPO, str(cookbook_dir)],
            capture_output=True,
            check=True,
        )
        print("Cloned successfully\n")

        # Clean and recreate output directories
        if recipes_dir.exists():
            shutil.rmtree(recipes_dir)
        if apps_dir.exists():
            shutil.rmtree(apps_dir)
        recipes_dir.mkdir(parents=True, exist_ok=True)
        apps_dir.mkdir(parents=True, exist_ok=True)

        # Process notebooks → Recipes
        print("Processing notebooks...")
        recipes = process_notebooks(cookbook_dir, recipes_dir)

        # Process app directories → Applications
        print("\nProcessing applications...")
        apps = process_applications(cookbook_dir, apps_dir)

        # Restore manually added entries that aren't in the cookbook repo
        print("\nRestoring manual entries...")
        synced_recipe_slugs = {r["slug"] for r in recipes}
        synced_app_slugs = {a["slug"] for a in apps}

        manual_recipes = []
        for entry in existing_recipes:
            if entry["slug"] not in synced_recipe_slugs:
                # This was a manual entry - restore it
                dest_path = recipes_dir / f"{entry['slug']}.md"
                dest_path.write_text(entry["content"])
                manual_recipes.append(
                    {
                        "slug": entry["slug"],
                        "title": entry["title"],
                        "id": entry["id"],
                    }
                )
                print(f"  Restored recipe: {entry['slug']}")

        manual_apps = []
        for entry in existing_apps:
            if entry["slug"] not in synced_app_slugs:
                # This was a manual entry - restore it
                dest_path = apps_dir / f"{entry['slug']}.md"
                dest_path.write_text(entry["content"])
                manual_apps.append(
                    {
                        "slug": entry["slug"],
                        "title": entry["title"],
                        "id": entry["id"],
                    }
                )
                print(f"  Restored app: {entry['slug']}")

        # Combine synced and manual entries
        all_recipes = recipes + manual_recipes
        all_apps = apps + manual_apps

        # Update sidebars.ts and index
        if all_recipes or all_apps:
            update_sidebars(all_recipes, all_apps, sidebars_file)
            update_cookbook_index(all_recipes, all_apps, docs_dir)

        print(
            f"\nDone! Generated {len(recipes)} recipes ({len(manual_recipes)} manual) and {len(apps)} apps ({len(manual_apps)} manual)"
        )


if __name__ == "__main__":
    main()
