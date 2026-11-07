from pathlib import Path
from typing import Callable, Iterator, Optional, Union
import torch
from pydantic import BaseModel, Field
from common.types import PathLike
from models.base import _DragonModel, DataLike

type OptimizerFactory = Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer]
type OptimizerLike = Union[torch.optim.Optimizer, OptimizerFactory]


class DragonTrainerConfig(BaseModel):
    # epochs per fit() call; None defers to the model's own default (cfg.max_epochs)
    epochs: Optional[int] = Field(default=None, ge=1)
    # load the model's latest checkpoint (weights + optimizer) before the first fit
    resume: bool = False


class DragonTrainer:
    """Drives any _DragonModel through its trainer-facing interface.

    The model owns *how* it learns (step / fit / evaluate / checkpoints); the
    trainer owns the run: the optimizer, resuming, and the accumulated history.
    `optimizer` may be an instance or a factory such as
    `lambda params: torch.optim.Adam(params, lr=1e-3)`, which the trainer builds
    against model.parameters() (handy when the model is built later, e.g. by optuna).
    """

    def __init__(self, model: _DragonModel, optimizer: OptimizerLike,
                 cfg: Optional[DragonTrainerConfig] = None):
        if not isinstance(model, _DragonModel):
            raise TypeError(f"model must be a _DragonModel, got {type(model)}")
        self.model = model
        self.optimizer = optimizer if isinstance(optimizer, torch.optim.Optimizer) \
            else optimizer(model.parameters())
        self.config = cfg if cfg is not None else DragonTrainerConfig()
        self.history: list[dict[str, float]] = []
        self._resume_pending = self.config.resume

    def resume(self, path: Optional[PathLike] = None) -> Optional[int]:
        """Load `path` (default: the model's latest checkpoint). Returns the
        restored epoch, or None if there was nothing to resume from."""
        path = path if path is not None else self.model.latest_checkpoint()
        self._resume_pending = False
        if path is None:
            return None
        return self.model.load_checkpoint(path, self.optimizer)

    def fit(self, train_data: DataLike, val_data: Optional[DataLike] = None,
            epochs: Optional[int] = None) -> list[dict[str, float]]:
        """Train for `epochs` more epochs (default cfg.epochs, then the model's own)."""
        if self._resume_pending:
            self.resume()
        history = self.model.fit(train_data, self.optimizer, val_data=val_data,
                                 epochs=epochs if epochs is not None else self.config.epochs)
        self.history.extend(history)
        return history

    def evaluate(self, data: DataLike) -> float:
        return self.model.evaluate(data)

    def save_checkpoint(self) -> Path:
        return self.model.save_checkpoint(self.optimizer)
