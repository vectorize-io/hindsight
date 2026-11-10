"""Pages used to be called "Knowledge Pages" / the "Knowledge Base".

The word is reserved for an upcoming, unrelated feature (Knowledge Banks), so the
old feature name must not come back in what users read: docs, the control plane,
the CLI, the SDK wrappers, the integration READMEs and the published skills.

The old names survive only as deprecated aliases (removed in 2.x). A line that
names one passes when a line near it (within 8 lines) says so.
"""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SCANNED = [
    "hindsight-docs/docs",
    "hindsight-docs/docs-integrations",
    "hindsight-docs/src",
    "hindsight-control-plane/src",
    "hindsight-cli/src",
    "hindsight-clients/python/hindsight_client",
    "hindsight-clients/typescript/src",
    "hindsight-integrations/*/README.md",
    "skills",
]
# Changelogs describe what shipped at the time; generated API specs carry the deprecated routes.
EXCLUDED = [":!**/changelog/**", ":!**/openapi.json"]

# Case-sensitive on purpose: lowercase "knowledge base" is ordinary English.
OLD_NAME = re.compile(
    r"Knowledge ?Pages?|Knowledge ?Base|KnowledgeNode|knowledge[_-]?pages?|knowledge[_-]base|knowledge_node|knowledgePage|knowledgeBase"
)
MARKER = re.compile(
    r"deprecat|[Rr]emove(d)? in 2\.x|used to be|[Bb]efore the rename|include_?[Kk]nowledge_?[Bb]ase", re.I
)


def test_old_feature_name_appears_only_as_a_deprecated_alias():
    files = subprocess.run(
        ["git", "ls-files", "--", *SCANNED, *EXCLUDED], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    offenders = []
    for rel in files:
        try:
            lines = (REPO / rel).read_text().splitlines()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for i, line in enumerate(lines):
            if OLD_NAME.search(line) and not any(MARKER.search(l) for l in lines[max(i - 8, 0) : i + 9]):
                offenders.append(f"{rel}:{i + 1}: {line.strip()[:120]}")
    assert not offenders, "Old 'Knowledge Pages' name outside a deprecated alias:\n" + "\n".join(offenders)
