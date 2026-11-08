import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union
import numpy as np
import optuna
from pydantic import BaseModel
from models.base import DataLike, EpochCallback, _DragonModel
from trainers import DragonTrainer, DragonTrainerConfig, TrainingDiverged
from trainers.base import OptimizerLike
from .config import Direction, TuneConfig
from .reporting import JsonlLogger, OptunaReporter

HISTORY_FILE = 'history.jsonl'
CONFIG_FILE = 'config.json'


@dataclass
class TrialSpec:
    """What `build` returns for one trial: either a ready `trainer` (e.g. a
    DragonRLTrainer, which brings its own data and callbacks), or the parts for
    a plain DragonTrainer (`model`, `optimizer`, `train_data`). Point the model's
    checkpoint_dir inside the trial dir `build` is given so trials don't
    overwrite each other.

    `callbacks` run before the tuner's own (logger, reporter), so they can add
    metrics to the record, e.g. an evaluation that runs every N steps.
    Anything here with a `close()` method (trainer, data, resources, callbacks)
    is closed when the trial ends. A trainer whose `config` is a pydantic model
    has it saved as config.json in the trial dir.
    """
    model: Optional[_DragonModel] = None
    optimizer: Optional[OptimizerLike] = None
    train_data: Optional[DataLike] = None
    trainer: Optional[DragonTrainer] = None
    val_data: Optional[DataLike] = None
    epochs: Optional[int] = None
    callbacks: Sequence[EpochCallback] = ()
    trainer_config: Optional[DragonTrainerConfig] = None
    # extra objects to close() at the end (e.g. envs owned by a callback)
    resources: Sequence[Any] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.trainer is None and (self.model is None or self.optimizer is None
                                     or self.train_data is None):
            raise ValueError("TrialSpec needs a trainer, or model + optimizer + train_data")

    def make_trainer(self) -> DragonTrainer:
        if self.trainer is not None:
            return self.trainer
        return DragonTrainer(self.model, self.optimizer, self.trainer_config)

    def closables(self) -> list[Any]:
        return [self.trainer, self.train_data, self.val_data, *self.resources, *self.callbacks]


def save_trainer_config(trainer: DragonTrainer, path: Path) -> Optional[Path]:
    """Write trainer.config as JSON (callables such as env factories are written
    as their repr, so the file documents the run even when it can't rebuild it)."""
    cfg = getattr(trainer, 'config', None)
    if not isinstance(cfg, BaseModel):
        return None
    path.write_text(cfg.model_dump_json(indent=2, fallback=repr))
    return path


type SuggestFn = Callable[[optuna.Trial], dict[str, Any]]
type BuildFn = Callable[[dict[str, Any], Path], TrialSpec]


def final_value(values: Sequence[float], direction: Direction, final: str, window: int) -> float:
    """Reduce a metric's recorded values to the trial's objective value."""
    if final == 'best':
        return float(max(values) if direction == 'maximize' else min(values))
    if final == 'mean_last':
        return float(np.mean(values[-window:]))
    return float(values[-1])


def objective_values(history: Sequence[dict[str, float]], cfg: TuneConfig) -> list[float]:
    out = []
    for metric, direction in zip(cfg.metrics, cfg.directions):
        values = [float(r[metric]) for r in history if metric in r]
        if not values:
            raise ValueError(
                f"metric '{metric}' never appeared in the trial history; pass val_data or a "
                f"callback that records it (keys seen: {sorted(history[-1]) if history else []})")
        out.append(final_value(values, direction, cfg.final, cfg.final_window))
    return out


class DragonObjective:
    """Optuna objective that trains one DragonTrainer run per trial.

    suggest(trial) -> params           the search space (trial.suggest_*)
    build(params, trial_dir) -> TrialSpec
                                       builds model / data from the params

    Each trial gets `cfg.study_dir / trial_NNNN` holding history.jsonl (every
    epoch record) plus whatever build puts there (checkpoints). Returns one
    float, or a tuple for multi-objective studies; a non-finite result fails
    the trial without stopping the study.
    """

    def __init__(self, cfg: TuneConfig, suggest: SuggestFn, build: BuildFn):
        self.cfg = cfg
        self.suggest = suggest
        self.build = build

    def trial_dir(self, trial: optuna.Trial) -> Path:
        return self.cfg.study_dir / f"trial_{trial.number:04d}"

    def callbacks(self, trial: optuna.Trial, trial_dir: Path) -> list[EpochCallback]:
        """Tuner callbacks appended after the spec's own. Override to add more."""
        return [
            JsonlLogger(trial_dir / HISTORY_FILE),
            OptunaReporter(trial, self.cfg.metrics[0], step_key=self.cfg.report_key,
                           report=not self.cfg.multi_objective),
        ]

    def __call__(self, trial: optuna.Trial) -> Union[float, tuple[float, ...]]:
        trial_dir = self.trial_dir(trial)
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / HISTORY_FILE).unlink(missing_ok=True)  # stale file from a deleted study
        trial.set_user_attr('trial_dir', str(trial_dir))

        params = self.suggest(trial)
        spec = self.build(params, trial_dir)
        try:
            trainer = spec.make_trainer()
            config_path = save_trainer_config(trainer, trial_dir / CONFIG_FILE)
            if config_path is not None:
                trial.set_user_attr('config_path', str(config_path))
            try:
                history = trainer.fit(spec.train_data, spec.val_data, epochs=spec.epochs,
                                      callbacks=[*spec.callbacks, *self.callbacks(trial, trial_dir)])
            except optuna.TrialPruned as e:
                trial.set_user_attr('prune_reason', str(e))  # shown in the dashboard
                raise
            except TrainingDiverged as e:
                trial.set_user_attr('prune_reason', f"diverged: {e}")
                raise optuna.TrialPruned(str(e)) from e
            if not history:
                raise ValueError("training produced no epochs")
            trial.set_user_attr('epochs', len(history))
            trial.set_user_attr('last_record', {k: float(v) for k, v in history[-1].items()})
            values = objective_values(history, self.cfg)
        finally:
            for obj in spec.closables():
                close = getattr(obj, 'close', None)
                if callable(close):
                    close()
        if not all(math.isfinite(v) for v in values):
            trial.set_user_attr('fail_reason', f"non-finite objective {values}")
            return float('nan')  # optuna marks the trial FAIL and carries on
        return values[0] if len(values) == 1 else tuple(values)
