"""Pytest config — make the plugin root importable from tests.

Dify plugins use a flat layout (`provider/`, `tools/`) rather than a `src/`
package, so we add the plugin root to sys.path for tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))
