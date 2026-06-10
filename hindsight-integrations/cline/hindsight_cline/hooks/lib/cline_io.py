"""Cline hook I/O contract.

Cline runs each hook as a subprocess: it writes a JSON object to the hook's
stdin and reads a JSON object from stdout shaped as
``{"cancel": bool, "contextModification": str, "errorMessage": str}``.
``contextModification`` is how a hook injects text into the model's context.

We parse the structured stdin into a typed `HookInput` at the boundary rather
than passing raw dicts around.
"""

import json
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from .client import HindsightClient
from .config import HindsightClineConfig

RECALL_MIN_CHARS = 5


@dataclass
class HookInput:
    """Parsed Cline hook payload (only the fields we use)."""

    hook_name: str = ""
    task_id: str = ""
    prompt: str = ""  # UserPromptSubmit
    task: str = ""  # TaskStart / TaskComplete / TaskCancel
    workspace_roots: list = field(default_factory=list)
    model_slug: str = ""


def parse_hook_input(raw: dict[str, Any]) -> HookInput:
    """Build a HookInput from the raw stdin JSON, tolerating missing fields."""
    if not isinstance(raw, dict):
        raw = {}
    roots = raw.get("workspaceRoots")
    if not isinstance(roots, list):
        roots = []
    model = raw.get("model")
    model_slug = str(model.get("slug", "")) if isinstance(model, dict) else ""
    return HookInput(
        hook_name=str(raw.get("hookName", "")),
        task_id=str(raw.get("taskId", "")),
        prompt=str(raw.get("prompt", "") or ""),
        task=str(raw.get("task", "") or ""),
        workspace_roots=[str(r) for r in roots],
        model_slug=model_slug,
    )


def read_hook_input() -> HookInput:
    """Read and parse the hook payload from stdin (never raises)."""
    try:
        raw_text = sys.stdin.read()
    except Exception:
        raw_text = ""
    data = {}
    if raw_text.strip():
        try:
            data = json.loads(raw_text)
        except (ValueError, TypeError):
            data = {}
    return parse_hook_input(data)


def emit(cancel: bool = False, context_modification: str = "", error_message: str = "") -> None:
    """Write the Cline hook response to stdout."""
    print(
        json.dumps(
            {
                "cancel": cancel,
                "contextModification": context_modification,
                "errorMessage": error_message,
            }
        )
    )


def resolve_api_url(config: HindsightClineConfig) -> Optional[str]:
    """Resolve a reachable Hindsight server, or None.

    Lean v1: use the configured external URL, else a Hindsight server already
    running locally. We never auto-start a daemon — if nothing is reachable
    the hooks degrade to no-ops.
    """
    url = config.hindsight_api_url
    if url:
        return str(url).rstrip("/")
    port = config.api_port
    local = f"http://localhost:{port}"
    try:
        if HindsightClient(local).health_check(timeout=2):
            return local
    except Exception:
        pass
    return None
