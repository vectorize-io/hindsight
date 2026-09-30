"""Turn the docs' interactive figures into text for the docs skill.

A docs page shows a figure as `<Flow {...name.props} />`, imported from
`@vectorize-io/interfig/figures/<file>`. The skill is plain markdown read by agents, so each
figure becomes its step-by-step narration (the `say` lines the figure plays), grouped by step.

An animated SVG figure (`![alt](/img/....svg)`) gets the same treatment: every SVG the figure
tooling writes carries its own spec in `<metadata>`, so the narration can be read straight out of
the image. Without this an agent would see a link to a picture and learn nothing from it.

Usage: python3 docs_skill_figures.py <source page> <generated page>
The source page supplies the imports; the generated page is rewritten in place.

       python3 docs_skill_figures.py --check-all
renders every figure and fails if one no longer reads cleanly (CI runs this).
"""

import json
import re
import sys
from pathlib import Path

FIGURES = Path(__file__).resolve().parent.parent / "hindsight-interfig" / "figures"
IMPORT = re.compile(r"^import (\w+) from '@vectorize-io/interfig/figures/([\w-]+)';\n", re.MULTILINE)
ANY_INTERFIG_IMPORT = re.compile(r"^import .* from '@vectorize-io/interfig[^']*';\n", re.MULTILINE)
FLOW = re.compile(r"<Flow \{\.\.\.(\w+)\.props\} />")
STRING = r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\""
STEP = re.compile(r"^\s*label: (?:" + STRING + r"),\s*\n(?:\s*caption:.*\n)?\s*flow:", re.MULTILINE)
FLOW_KEY = re.compile(r"^\s*flow:", re.MULTILINE)
SAY = re.compile(r"\bsay: (?:" + STRING + r")")
TITLE = re.compile(r"^\s*title: (?:" + STRING + r"),", re.MULTILINE)
STATIC = Path(__file__).resolve().parent.parent / "hindsight-docs" / "static"
IMAGE = re.compile(r"^!\[([^\]]*)\]\(([^)]+\.svg)\)$", re.MULTILINE)
SVG_SPEC = re.compile(r'<metadata id="figure-spec"><!\[CDATA\[(.*?)\]\]></metadata>', re.S)


def _text(match: re.Match[str]) -> str:
    raw = match.group(1) if match.group(1) is not None else match.group(2)
    return re.sub(r"\\(.)", r"\1", raw)


def figure_to_markdown(figure_file: Path) -> str:
    """The figure's title and, per step, its narration as a numbered list."""
    source = figure_file.read_text()
    title_match = TITLE.search(source)
    title = _text(title_match) if title_match else figure_file.stem
    steps = list(STEP.finditer(source))
    # The regexes follow the figures' prettier layout (`label`, optional `caption`, then `flow`). A step
    # written another way would be skipped and its narration merged into the previous one: fail instead.
    if not steps or len(steps) != len(FLOW_KEY.findall(source)):
        raise SystemExit(f"{figure_file}: found {len(steps)} steps but {len(FLOW_KEY.findall(source))} `flow:` keys")
    says_per_step = [
        [
            _text(m)
            for m in SAY.finditer(source, step.end(), steps[i + 1].start() if i + 1 < len(steps) else len(source))
        ]
        for i, step in enumerate(steps)
    ]
    if not all(says_per_step):
        raise SystemExit(f"{figure_file}: a step has no narration (`say`) to show")
    lines = [f"**Figure: {title}.** An animated diagram on the docs site; its narration, step by step:", ""]
    for step, says in zip(steps, says_per_step):
        lines.append(f"- **{_text(step)}**")
        lines += [f"  {n}. {say}" for n, say in enumerate(says, 1)]
    return "\n".join(lines)


def svg_to_markdown(spec: dict, alt: str) -> str:
    """The same narration, read from the spec an animated SVG carries."""
    steps = spec.get("props", {}).get("steps", [])
    lines = [f"**Figure: {alt}.** An animated diagram; its narration, step by step:", ""]
    for step in steps:
        says = [beat["say"] for beat in step.get("flow", []) if isinstance(beat, dict) and beat.get("say")]
        lines.append(f"- **{step.get('label', '')}**")
        lines += [f"  {n}. {say}" for n, say in enumerate(says, 1)]
    return "\n".join(lines)


def svg_figure(match: re.Match[str]) -> str:
    """An `![alt](/img/x.svg)` whose file carries a spec becomes its narration; anything else is left alone."""
    alt, src = match.group(1), match.group(2)
    if src.startswith(("http://", "https://")):
        return match.group(0)
    file = STATIC / src.lstrip("/")
    if not file.exists():
        return match.group(0)
    spec = SVG_SPEC.search(file.read_text())
    return svg_to_markdown(json.loads(spec.group(1)), alt or file.stem) if spec else match.group(0)


def render(source_page: str, generated: str) -> str:
    figures = {var: FIGURES / f"{name}.ts" for var, name in IMPORT.findall(source_page)}
    generated = ANY_INTERFIG_IMPORT.sub("", generated)
    generated = IMAGE.sub(svg_figure, generated)
    return FLOW.sub(lambda m: figure_to_markdown(figures[m.group(1)]), generated)


if __name__ == "__main__":
    if sys.argv[1:] == ["--check-all"]:
        for figure in sorted(FIGURES.glob("*.ts")):
            figure_to_markdown(figure)
        print(f"docs_skill_figures: {len(list(FIGURES.glob('*.ts')))} figures render")
    else:
        src, dest = Path(sys.argv[1]), Path(sys.argv[2])
        dest.write_text(render(src.read_text(), dest.read_text()))
