"""Epoch callbacks for RL training: step-scheduled evaluation and a divergence guard.
DragonRLTrainer adds both by default; neither depends on optuna."""
import math
from collections import deque
from typing import Any, Optional
import numpy as np
from dragon_torch.models.base import DataLike, _DragonModel
from dragon_torch.trainers.base import TrainingDiverged

EVAL_STEP_KEY = 'eval_step'


class StepEvalCallback:
    """Evaluates `model` on `evaluator` (e.g. an EpisodeSource) every
    `every_steps` env steps, plus once when `total_steps` is reached.

    On epochs that evaluate it adds to the record:
      {metric}          the raw evaluation (mean episode return)
      {metric}_smooth   mean of the last `smooth_window` evaluations
      eval_step         the scheduled step this evaluation stands for: the
                        largest multiple of every_steps crossed, capped at
                        total_steps. Epoch sizes (n_envs x n_steps) differ between
                        runs, so global_step never lines up across runs; eval_step
                        does, which is what pruners and comparisons should use.
    Keep every_steps >= steps per epoch, or an epoch crossing two thresholds
    evaluates once and a scheduled step is skipped.

    state_dict / load_state_dict carry the smoothing window and schedule
    position, so a resumed run continues both (DragonRLTrainer registers it with
    the model's checkpoints).
    """

    def __init__(self, evaluator: DataLike, every_steps: int, total_steps: Optional[int] = None,
                 metric: str = 'val_return', smooth_window: int = 3,
                 step_key: str = 'global_step'):
        if every_steps < 1 or smooth_window < 1:
            raise ValueError("every_steps and smooth_window must be >= 1")
        self.evaluator = evaluator
        self.every_steps, self.total_steps = every_steps, total_steps
        self.metric, self.step_key = metric, step_key
        self.recent: deque[float] = deque(maxlen=smooth_window)
        self.last_eval_step = 0

    @property
    def smooth_metric(self) -> str:
        return f"{self.metric}_smooth"

    def due_step(self, step: int) -> Optional[int]:
        """The scheduled step to evaluate for at env step `step`, or None."""
        due = (step // self.every_steps) * self.every_steps
        if self.total_steps is not None and step >= self.total_steps:
            due = self.total_steps
        return due if due > self.last_eval_step else None

    def __call__(self, model: _DragonModel, record: dict[str, float]) -> None:
        if self.step_key not in record:
            raise KeyError(f"StepEvalCallback needs '{self.step_key}' in the history record "
                           f"(an RL source's metrics provide it), keys: {list(record)}")
        due = self.due_step(int(record[self.step_key]))
        if due is None:
            return
        value = float(model.evaluate(self.evaluator))
        self.last_eval_step = due
        self.recent.append(value)
        record[self.metric] = value
        record[self.smooth_metric] = float(np.mean(self.recent))
        record[EVAL_STEP_KEY] = due

    def state_dict(self) -> dict[str, Any]:
        return {'recent': list(self.recent), 'last_eval_step': self.last_eval_step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.recent.clear()
        self.recent.extend(state['recent'])
        self.last_eval_step = state['last_eval_step']

    def close(self) -> None:
        close = getattr(self.evaluator, 'close', None)
        if callable(close):
            close()


class DivergenceGuard:
    """Raises TrainingDiverged when a run has clearly blown up, instead of
    letting it burn its budget. Checks only the keys the record has, so it works
    for any loss (A2C records no approx_kl):
      - any of `finite_keys` is NaN / inf
      - approx_kl > max_kl for `patience` consecutive epochs
      - entropy < min_entropy after `entropy_after_steps` env steps (policy
        collapsed to deterministic before learning anything useful)
    """

    def __init__(self, max_kl: Optional[float] = 0.5, patience: int = 2,
                 min_entropy: Optional[float] = None, entropy_after_steps: int = 0,
                 finite_keys: tuple[str, ...] = ('train_loss', 'value_loss', 'policy_loss'),
                 step_key: str = 'global_step'):
        self.max_kl, self.patience = max_kl, max(1, patience)
        self.min_entropy, self.entropy_after_steps = min_entropy, entropy_after_steps
        self.finite_keys, self.step_key = finite_keys, step_key
        self._kl_strikes = 0

    def __call__(self, model: _DragonModel, record: dict[str, float]) -> None:
        step = record.get(self.step_key, record.get('epoch'))
        for key in self.finite_keys:
            if key in record and not math.isfinite(float(record[key])):
                raise TrainingDiverged(f"{key}={record[key]} at {self.step_key}={step}")
        if self.max_kl is not None and 'approx_kl' in record:
            self._kl_strikes = self._kl_strikes + 1 if record['approx_kl'] > self.max_kl else 0
            if self._kl_strikes >= self.patience:
                raise TrainingDiverged(f"approx_kl={record['approx_kl']:.3g} > {self.max_kl} for "
                                       f"{self.patience} epochs at {self.step_key}={step}")
        if (self.min_entropy is not None and 'entropy' in record
                and record.get(self.step_key, 0) >= self.entropy_after_steps
                and record['entropy'] < self.min_entropy):
            raise TrainingDiverged(f"entropy={record['entropy']:.3g} < {self.min_entropy} "
                                   f"at {self.step_key}={step}")
