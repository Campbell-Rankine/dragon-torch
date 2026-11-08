import math
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence, Union
import torch
from pydantic import BaseModel, Field
from dragon_torch.common.types import PathLike
from dragon_torch.models.base import _DragonModel, DataLike, EpochCallback

type OptimizerFactory = Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer]
type OptimizerLike = Union[torch.optim.Optimizer, OptimizerFactory]
# epoch record -> multiplier on each param group's base lr for the *next* epoch
type LRSchedule = Callable[[dict[str, float]], float]


class TrainingDiverged(RuntimeError):
    """Raised by callbacks (e.g. DivergenceGuard) when a run has blown up and
    should stop. The tuner turns it into a pruned trial."""


class DragonTrainerConfig(BaseModel):
    # epochs per fit() call; None defers to the model's own default (cfg.max_epochs)
    epochs: Optional[int] = Field(default=None, ge=1)
    # load the model's latest checkpoint (weights + optimizer) before the first fit
    resume: bool = False


def linear_schedule(total: float, key: str = 'epoch', final_frac: float = 0.0) -> LRSchedule:
    """Multiplier decaying linearly from 1 to `final_frac` as record[key] goes
    0 -> total, e.g. linear_schedule(total_timesteps, 'global_step') for PPO."""
    if total <= 0:
        raise ValueError("total must be > 0")

    def schedule(record: dict[str, float]) -> float:
        return max(final_frac, 1.0 - float(record[key]) / total * (1.0 - final_frac))
    return schedule


class DragonTrainer:
    """Drives any _DragonModel through its trainer-facing interface.

    The model owns *how* it learns (step / fit / evaluate / checkpoints); the
    trainer owns the run: the optimizer, the lr schedule, resuming, and the
    accumulated history. `optimizer` may be an instance or a factory such as
    `lambda params: torch.optim.Adam(params, lr=1e-3)`, which the trainer builds
    against model.parameters() (handy when the model is built later, e.g. by optuna).

    Every epoch record gets the `lr` it trained with. `lr_schedule(record) ->
    multiplier` then runs after the epoch and sets each param group's lr to
    base_lr * multiplier for the next one. It is a function of the record (epoch,
    global_step, ...) rather than a stateful torch scheduler, so a resumed run
    continues the schedule from its checkpointed position. The base lr is kept
    in each param group's 'initial_lr', which the optimizer checkpoint saves.
    """

    def __init__(self, model: _DragonModel, optimizer: OptimizerLike,
                 cfg: Optional[DragonTrainerConfig] = None,
                 lr_schedule: Optional[LRSchedule] = None):
        if not isinstance(model, _DragonModel):
            raise TypeError(f"model must be a _DragonModel, got {type(model)}")
        self.model = model
        self.optimizer = optimizer if isinstance(optimizer, torch.optim.Optimizer) \
            else optimizer(model.parameters())
        self.config = cfg if cfg is not None else DragonTrainerConfig()
        self.lr_schedule = lr_schedule
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.history: list[dict[str, float]] = []
        self._resume_pending = self.config.resume

    @property
    def higher_is_better(self) -> bool:
        """Whether a larger `evaluate` score is better (loss: False, return: True)."""
        return self.model.higher_is_better

    def resume(self, path: Optional[PathLike] = None) -> Optional[int]:
        """Load `path` (default: the model's latest checkpoint). Returns the
        restored epoch, or None if there was nothing to resume from."""
        path = path if path is not None else self.model.latest_checkpoint()
        self._resume_pending = False
        if path is None:
            return None
        return self.model.load_checkpoint(path, self.optimizer)

    def _lr_callback(self, model: _DragonModel, record: dict[str, float]) -> None:
        record.setdefault('lr', self.optimizer.param_groups[0]['lr'])
        if self.lr_schedule is None:
            return
        mult = float(self.lr_schedule(record))
        if not math.isfinite(mult) or mult < 0:
            raise ValueError(f"lr_schedule returned {mult}; expected a finite multiplier >= 0")
        for group in self.optimizer.param_groups:
            group['lr'] = group['initial_lr'] * mult

    def fit(self, train_data: DataLike, val_data: Optional[DataLike] = None,
            epochs: Optional[int] = None,
            callbacks: Sequence[EpochCallback] = ()) -> list[dict[str, float]]:
        """Train for `epochs` more epochs (default cfg.epochs, then the model's own).
        `callbacks` run after every epoch; see dragon_torch.models.base.EpochCallback."""
        if self._resume_pending:
            self.resume()
        callbacks = [self._lr_callback, *callbacks]
        history = self.model.fit(train_data, self.optimizer, val_data=val_data,
                                 epochs=epochs if epochs is not None else self.config.epochs,
                                 callbacks=callbacks)
        self.history.extend(history)
        return history

    def evaluate(self, data: DataLike) -> float:
        return self.model.evaluate(data)

    def save_checkpoint(self) -> Path:
        return self.model.save_checkpoint(self.optimizer)
