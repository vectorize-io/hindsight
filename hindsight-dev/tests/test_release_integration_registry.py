"""The release script and the changelog generator must know the same integrations.

`scripts/release-integration.sh` validates its argument against a bash `VALID_INTEGRATIONS`
array; the changelog step it later calls validates against `INTEGRATIONS` here. Nothing tied
the two together, so they drifted: `agno` sat in the bash array and not in the registry, which
means `release-integration.sh agno <v>` would bump the version, commit nothing, and die at the
changelog step — a half-finished release someone has to unpick by hand.

The failure is invisible until the first release of whichever integration was forgotten, and by
construction nobody writes a test for the one they forgot. So assert over the whole set.
"""

import re
from pathlib import Path

from hindsight_dev.generate_changelog import INTEGRATIONS

RELEASE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "release-integration.sh"


def _bash_valid_integrations() -> set[str]:
    """The names `release-integration.sh` accepts, read from its VALID_INTEGRATIONS array."""
    match = re.search(r"^VALID_INTEGRATIONS=\((.*?)\)$", RELEASE_SCRIPT.read_text(), re.M | re.S)
    assert match, f"VALID_INTEGRATIONS array not found in {RELEASE_SCRIPT}"
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def test_release_script_and_changelog_registry_agree():
    bash_names = _bash_valid_integrations()
    registry_names = set(INTEGRATIONS)

    missing_from_registry = sorted(bash_names - registry_names)
    assert not missing_from_registry, (
        "release-integration.sh accepts these, but the changelog generator does not know them, so "
        f"a release would fail after bumping the version: {missing_from_registry}. "
        "Add an IntegrationMeta for each in generate_changelog.py."
    )

    missing_from_script = sorted(registry_names - bash_names)
    assert not missing_from_script, (
        "These are in the changelog registry but release-integration.sh refuses them, so they "
        f"cannot be released at all: {missing_from_script}. Add them to VALID_INTEGRATIONS."
    )


def test_every_releasable_integration_has_a_directory():
    """A name in either list with no `hindsight-integrations/<name>/` is a typo: the release
    script exits on the missing directory, after it has already resolved the version."""
    root = RELEASE_SCRIPT.parent.parent / "hindsight-integrations"
    missing = sorted(name for name in _bash_valid_integrations() if not (root / name).is_dir())
    assert not missing, f"listed as releasable but no integration directory exists: {missing}"
