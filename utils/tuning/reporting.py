import json
import math
from pathlib import Path
from typing import Any, Optional
import optuna
from common.types import PathLike
from models.base import _DragonModel


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class OptunaReporter:
    """Epoch callback that reports `metric` to the trial at step `record[step_key]`
    and raises optuna.TrialPruned when the pruner says so.

    Epochs that didn't record `metric` (e.g. evaluation only every N steps) are
    skipped. A non-finite metric stops training instead of being reported; the
    objective then marks the trial as failed. With `report=False` (multi-objective
    studies, where optuna has no trial.report) only that divergence check runs.
    Whether a reported value prunes is up to the study's pruner (NopPruner: never).
    """

    def __init__(self, trial: optuna.Trial, metric: str, step_key: str = 'epoch',
                 report: bool = True):
        self.trial = trial
        self.metric = metric
        self.step_key = step_key
        self.report = report

    def __call__(self, model: _DragonModel, record: dict[str, float]) -> Optional[bool]:
        if self.metric not in record:
            return None
        value = record[self.metric]
        if not _is_finite(value):
            return True
        if not self.report:
            return None
        if self.step_key not in record:
            raise KeyError(f"report step key '{self.step_key}' missing from history record, "
                           f"keys: {list(record)}")
        self.trial.report(float(value), int(record[self.step_key]))
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"pruned at {self.step_key}={record[self.step_key]} with {self.metric}={value}")
        return None


class JsonlLogger:
    """Epoch callback appending every history record as one JSON line, so the
    full curves (not just the reported metric) survive pruned and failed trials."""

    def __init__(self, path: PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, model: _DragonModel, record: dict[str, float]) -> None:
        with self.path.open('a') as f:
            f.write(json.dumps(record, default=float) + '\n')


def read_history(path: PathLike) -> list[dict[str, Any]]:
    """Load a JsonlLogger file back into a list of records."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
