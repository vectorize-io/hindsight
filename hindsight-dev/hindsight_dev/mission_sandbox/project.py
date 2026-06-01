"""Project model: a directory-based project with full history tracking.

Project structure:
    my-project/
        project.json        # metadata: bank_id, api_url, current mission
        history/
            0001_init.json
            0002_label.json
            0003_optimize.json
            0004_run.json
            ...
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ObservationSample:
    id: str
    text: str
    source_facts: list[str]
    label: str | None = None
    reason: str | None = None


# -- History step types --------------------------------------------------------


@dataclass
class InitStep:
    """Recorded when `init` ingests documents and runs first consolidation."""

    type: str = "init"
    timestamp: str = ""
    documents: list[str] = field(default_factory=list)
    observations: list[ObservationSample] = field(default_factory=list)


@dataclass
class LabelStep:
    """Recorded when `agent-label` scores observations."""

    type: str = "label"
    timestamp: str = ""
    instructions: str = ""
    model: str = ""
    observations: list[ObservationSample] = field(default_factory=list)


@dataclass
class OptimizeStep:
    """Recorded when `optimize` proposes a new mission."""

    type: str = "optimize"
    timestamp: str = ""
    model: str = ""
    previous_mission: str | None = None
    proposed_mission: str = ""
    good_count: int = 0
    bad_count: int = 0


@dataclass
class RunStep:
    """Recorded when `run` applies mission, re-consolidates, exports new observations."""

    type: str = "run"
    timestamp: str = ""
    mission_applied: str = ""
    observations: list[ObservationSample] = field(default_factory=list)


@dataclass
class PromoteStep:
    """Recorded when `promote` pushes mission to a production bank."""

    type: str = "promote"
    timestamp: str = ""
    target_bank: str = ""
    mission: str = ""
    backfill: bool = False


STEP_TYPES = {
    "init": InitStep,
    "label": LabelStep,
    "optimize": OptimizeStep,
    "run": RunStep,
    "promote": PromoteStep,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- Project -------------------------------------------------------------------


@dataclass
class Project:
    bank_id: str
    api_url: str
    mission: str | None = None
    created_at: str = ""
    observations: list[ObservationSample] = field(default_factory=list)

    _path: Path | None = field(default=None, repr=False)

    @classmethod
    def create(cls, path: Path, bank_id: str, api_url: str) -> Project:
        """Create a new project directory."""
        path.mkdir(parents=True, exist_ok=True)
        (path / "history").mkdir(exist_ok=True)
        proj = cls(bank_id=bank_id, api_url=api_url, created_at=_now(), _path=path)
        proj._save_meta()
        return proj

    @classmethod
    def load(cls, path: Path) -> Project:
        """Load an existing project."""
        meta_file = path / "project.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"No project found at {path} (missing project.json)")
        raw = json.loads(meta_file.read_text())
        observations = [ObservationSample(**o) for o in raw.get("observations", [])]
        proj = cls(
            bank_id=raw["bank_id"],
            api_url=raw["api_url"],
            mission=raw.get("mission"),
            created_at=raw.get("created_at", ""),
            observations=observations,
            _path=path,
        )
        return proj

    @property
    def path(self) -> Path:
        assert self._path is not None
        return self._path

    def _save_meta(self) -> None:
        """Write project.json."""
        data = {
            "bank_id": self.bank_id,
            "api_url": self.api_url,
            "mission": self.mission,
            "created_at": self.created_at,
            "observations": [asdict(o) for o in self.observations],
        }
        (self.path / "project.json").write_text(json.dumps(data, indent=2))

    def save(self) -> None:
        self._save_meta()

    # -- History ---------------------------------------------------------------

    def _next_step_number(self) -> int:
        history_dir = self.path / "history"
        existing = sorted(history_dir.glob("*.json"))
        if not existing:
            return 1
        last = existing[-1].stem  # e.g. "0003_optimize"
        return int(last.split("_")[0]) + 1

    def add_step(self, step: InitStep | LabelStep | OptimizeStep | RunStep | PromoteStep) -> Path:
        """Append a history step and save project metadata."""
        step.timestamp = _now()
        num = self._next_step_number()
        filename = f"{num:04d}_{step.type}.json"
        step_path = self.path / "history" / filename
        step_path.write_text(json.dumps(asdict(step), indent=2))
        self._save_meta()
        return step_path

    def list_steps(self) -> list[dict]:
        """Load all history steps in order."""
        history_dir = self.path / "history"
        steps = []
        for f in sorted(history_dir.glob("*.json")):
            raw = json.loads(f.read_text())
            raw["_file"] = f.name
            steps.append(raw)
        return steps
