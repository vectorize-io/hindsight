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
    """Extract first paragraph after title from notebook."""
    try:
        content = json.loads(notebook_path.read_text())
        for cell in content.get("cells", []):
            if cell.get("cell_type") == "markdown":
                source = cell.get("source", [])
                if isinstance(source, list):
                    source = "".join(source)

                lines = source.split("\n")
                found_title = False
                description = []

                for line in lines:
                    if line.startswith("#"):
                        found_title = True
                        continue
                    if found_title and line.strip():
                        if line.startswith("#"):
                            break
                        description.append(line.strip())
                        if line.strip().endswith("."):
                            break

                if description:
                    return " ".join(description)[:200]
    except Exception:
        pass
    return None


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

        print(f"  Processing: {notebook_path.name} → {slug}.md")

        # Convert notebook to markdown
        md_content = convert_notebook_to_markdown(notebook_path)

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
                "id": f"cookbook/recipes/{slug}",
            }
        )

    return recipes


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

        print(f"  Processing app: {entry.name} → {slug}.md")

        # Read README content
        readme_content = readme_path.read_text()

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
                "id": f"cookbook/applications/{slug}",
            }
        )

    return apps


def update_sidebars(recipes: list[dict], apps: list[dict], sidebars_file: Path):
    """Update sidebars.ts with new recipe and app entries."""
    content = sidebars_file.read_text()

    # Build recipe items
    recipe_item_list = []
    for r in recipes:
        label = r["title"].replace("'", "\\'")
        recipe_item_list.append(
            f"""        {{
          type: 'doc',
          id: '{r["id"]}',
          label: '{label}',
        }}"""
        )
    recipe_items = ",\n".join(recipe_item_list)

    # Build app items
    app_item_list = []
    for a in apps:
        label = a["title"].replace("'", "\\'")
        app_item_list.append(
            f"""        {{
          type: 'doc',
          id: '{a["id"]}',
          label: '{label}',
        }}"""
        )
    app_items = ",\n".join(app_item_list)

    new_cookbook_sidebar = f"""cookbookSidebar: [
    {{
      type: 'doc',
      id: 'cookbook/index',
      label: 'Overview',
    }},
    {{
      type: 'category',
      label: 'Recipes',
      collapsible: false,
      items: [
{recipe_items}
      ],
    }},
    {{
      type: 'category',
      label: 'Applications',
      collapsible: false,
      items: [
{app_items}
      ],
    }},
  ]"""

    # Replace existing cookbookSidebar - match the full sidebar array including nested structures
    # We need to match balanced brackets
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


def update_cookbook_index(recipes: list[dict], apps: list[dict], docs_dir: Path):
    """Update cookbook/index.mdx with recipe and app carousels."""
    # Build recipe items for the carousel
    recipe_items = []
    for r in recipes:
        title = r["title"].replace('"', '\\"')
        recipe_items.append(f'    {{ title: "{title}", href: "/cookbook/recipes/{r["slug"]}" }}')
    recipes_json = ",\n".join(recipe_items)

    # Build app items for the carousel
    app_items = []
    for a in apps:
        title = a["title"].replace('"', '\\"')
        app_items.append(f'    {{ title: "{title}", href: "/cookbook/applications/{a["slug"]}" }}')
    apps_json = ",\n".join(app_items)

    content = f"""---
sidebar_position: 1
---

import RecipeCarousel from '@site/src/components/RecipeCarousel';

# Cookbook

Practical patterns, recipes, and complete applications for building with Hindsight.

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
"""

    index_path = docs_dir / "index.mdx"
    index_path.write_text(content)

    # Remove old .md if exists
    old_index = docs_dir / "index.md"
    if old_index.exists():
        old_index.unlink()

    print("Updated cookbook/index.mdx")


def main():
    """Main entry point."""
    print("Syncing hindsight-cookbook...\n")

    docs_dir = get_docs_dir()
    sidebars_file = get_sidebars_file()
    recipes_dir = docs_dir / "recipes"
    apps_dir = docs_dir / "applications"

    # Create temp directory and clone
    with tempfile.TemporaryDirectory() as tmpdir:
        cookbook_dir = Path(tmpdir) / "cookbook"

        print(f"Cloning {COOKBOOK_REPO}...")
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

        # Update sidebars.ts and index
        if recipes or apps:
            update_sidebars(recipes, apps, sidebars_file)
            update_cookbook_index(recipes, apps, docs_dir)

        print(f"\nDone! Generated {len(recipes)} recipes and {len(apps)} applications")


if __name__ == "__main__":
    main()
