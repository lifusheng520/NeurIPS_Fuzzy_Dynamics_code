"""Time-based directory conventions for isolated experiments."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


EXPERIMENT_TIME_PATTERN = re.compile(r"^\d{8}_\d{6}_\d{6}$")
EXPERIMENT_DIRECTORY_PREFIX = "fuzzy_dynamics_"


def generate_experiment_time() -> str:
    """Return a sortable timestamp with microsecond collision resistance."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def validate_experiment_time(value: str) -> str:
    """Validate the timestamp used as one experiment's directory name."""
    timestamp = value.strip()
    if not EXPERIMENT_TIME_PATTERN.fullmatch(timestamp):
        raise ValueError(
            "experiment time must use YYYYMMDD_HHMMSS_microseconds format."
        )
    return timestamp


def experiment_directory(results_root: str | Path, experiment_time: str) -> Path:
    """Return ``results/fuzzy_dynamics_<time>`` for one experiment."""
    timestamp = validate_experiment_time(experiment_time)
    return Path(results_root) / f"{EXPERIMENT_DIRECTORY_PREFIX}{timestamp}"


def infer_experiment_time_from_checkpoint(checkpoint: str | Path) -> str:
    """Infer time from ``.../fuzzy_dynamics_<time>/checkpoint/best.pt``."""
    experiment_name = Path(checkpoint).expanduser().parent.parent.name
    if not experiment_name.startswith(EXPERIMENT_DIRECTORY_PREFIX):
        raise ValueError("checkpoint is not inside a fuzzy_dynamics_<time> directory.")
    return validate_experiment_time(experiment_name[len(EXPERIMENT_DIRECTORY_PREFIX) :])


def latest_experiment_time(results_root: str | Path) -> str:
    """Return the newest result experiment containing ``checkpoint/best.pt``."""
    root = Path(results_root)

    def is_complete(path: Path) -> bool:
        manifest = path / "training" / "experiment_manifest.json"
        if not manifest.is_file():
            return False
        try:
            with manifest.open(encoding="utf-8") as handle:
                return json.load(handle).get("status") == "complete"
        except (OSError, json.JSONDecodeError, AttributeError):
            return False

    candidates = sorted(
        path.name[len(EXPERIMENT_DIRECTORY_PREFIX) :]
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith(EXPERIMENT_DIRECTORY_PREFIX)
        and EXPERIMENT_TIME_PATTERN.fullmatch(
            path.name[len(EXPERIMENT_DIRECTORY_PREFIX) :]
        )
        and (path / "checkpoint" / "best.pt").is_file()
        and is_complete(path)
    ) if root.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"No completed fuzzy-dynamics experiments under {root}.")
    return candidates[-1]
