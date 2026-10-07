"""Turn the docs' animated figures into text for the docs skill.

A docs page shows a figure as `<Figure doc={name} />`, imported from
`@site/figures/<file>.json` (a Giotto diagram). The skill is plain markdown read by agents, so each
figure becomes its step-by-step narration (the `say` lines the figure plays), grouped by scene.
Without this an agent would see a component tag and learn nothing from it.

Usage: python3 docs_skill_figures.py <source page> <generated page>
The source page supplies the imports; the generated page is rewritten in place.

       python3 docs_skill_figures.py --check-all
fails if a figure no longer reads cleanly (CI runs this).
"""

import json
import re
import sys
from pathlib import Path

FIGURES = Path(__file__).resolve().parent.parent / "hindsight-docs" / "figures"
IMPORT = re.compile(r"^import (\w+) from '@site/figures/([\w-]+)\.json';\n", re.MULTILINE)
FIGURE_IMPORT = re.compile(r"^import Figure from '@site/src/components/Figure';\n", re.MULTILINE)
FIGURE = re.compile(r"<Figure doc=\{(\w+)\} />")


def figure_to_markdown(figure_file: Path) -> str:
    """The figure's title and, per scene, its narration as a numbered list."""
    doc = json.loads(figure_file.read_text())
    scenes = doc.get("scenes", [])
    if not scenes:
        raise SystemExit(f"{figure_file}: no scenes to narrate")
    title = doc.get("title") or figure_file.stem
    lines = [f"**Figure: {title}.** An animated diagram on the docs site; its narration, step by step:", ""]
    for scene in scenes:
        says = [beat["say"] for beat in scene.get("beats", []) if beat.get("say")]
        if not says:
            raise SystemExit(f"{figure_file}: scene {scene.get('label')!r} has no narration (`say`) to show")
        lines.append(f"- **{scene.get('label', '')}**")
        lines += [f"  {n}. {say}" for n, say in enumerate(says, 1)]
    return "\n".join(lines)


def render(source_page: str, generated: str) -> str:
    figures = {var: FIGURES / f"{name}.json" for var, name in IMPORT.findall(source_page)}
    generated = FIGURE_IMPORT.sub("", IMPORT.sub("", generated))
    return FIGURE.sub(lambda m: figure_to_markdown(figures[m.group(1)]), generated)


if __name__ == "__main__":
    if sys.argv[1:] == ["--check-all"]:
        for figure in sorted(FIGURES.glob("*.json")):
            figure_to_markdown(figure)
        print(f"docs_skill_figures: {len(list(FIGURES.glob('*.json')))} figures render")
    else:
        src, dest = Path(sys.argv[1]), Path(sys.argv[2])
        dest.write_text(render(src.read_text(), dest.read_text()))
