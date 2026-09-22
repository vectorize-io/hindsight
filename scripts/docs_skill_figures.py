"""Turn the docs' interactive figures into text for the docs skill.

A docs page shows a figure as `<Flow {...name.props} />`, imported from
`@vectorize-io/interfig/figures/<file>`. The skill is plain markdown read by agents, so each
figure becomes its step-by-step narration (the `say` lines the figure plays), grouped by step.

Usage: python3 docs_skill_figures.py <source page> <generated page>
The source page supplies the imports; the generated page is rewritten in place.
"""

import re
import sys
from pathlib import Path

FIGURES = Path(__file__).resolve().parent.parent / "hindsight-interfig" / "figures"
IMPORT = re.compile(r"^import (\w+) from '@vectorize-io/interfig/figures/([\w-]+)';\n", re.MULTILINE)
ANY_INTERFIG_IMPORT = re.compile(r"^import .* from '@vectorize-io/interfig[^']*';\n", re.MULTILINE)
FLOW = re.compile(r"<Flow \{\.\.\.(\w+)\.props\} />")
STRING = r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\""
STEP = re.compile(r"^\s*label: (?:" + STRING + r"),\s*\n(?:\s*caption:.*\n)?\s*flow:", re.MULTILINE)
SAY = re.compile(r"\bsay: (?:" + STRING + r")")
TITLE = re.compile(r"^\s*title: (?:" + STRING + r"),", re.MULTILINE)


def _text(match: re.Match[str]) -> str:
    raw = match.group(1) if match.group(1) is not None else match.group(2)
    return re.sub(r"\\(.)", r"\1", raw)


def figure_to_markdown(figure_file: Path) -> str:
    """The figure's title and, per step, its narration as a numbered list."""
    source = figure_file.read_text()
    title_match = TITLE.search(source)
    title = _text(title_match) if title_match else figure_file.stem
    steps = list(STEP.finditer(source))
    if not steps:
        # The regexes follow the figures' prettier layout; fail loudly rather than emit an empty walk-through.
        raise SystemExit(f"{figure_file}: no steps found — did the figure format change?")
    lines = [f"**Figure: {title}.** An animated diagram on the docs site; its narration, step by step:", ""]
    for i, step in enumerate(steps):
        end = steps[i + 1].start() if i + 1 < len(steps) else len(source)
        says = [_text(m) for m in SAY.finditer(source, step.end(), end)]
        lines.append(f"- **{_text(step)}**")
        lines += [f"  {n}. {say}" for n, say in enumerate(says, 1)]
    return "\n".join(lines)


def render(source_page: str, generated: str) -> str:
    figures = {var: FIGURES / f"{name}.ts" for var, name in IMPORT.findall(source_page)}
    generated = ANY_INTERFIG_IMPORT.sub("", generated)
    return FLOW.sub(lambda m: figure_to_markdown(figures[m.group(1)]), generated)


if __name__ == "__main__":
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.write_text(render(src.read_text(), dest.read_text()))
