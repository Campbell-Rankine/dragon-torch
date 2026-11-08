import torch
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from typing import Generic, Callable, ClassVar, Any, Iterable, Iterator, Optional, Protocol, Sequence, Union
from abc import abstractmethod, ABC
from torch.utils.data import DataLoader, IterableDataset
from pathlib import Path
from dragon_torch.common.types import PathLike
from dragon_torch.common.utils import build_path
from dragon_torch.modules import DragonModule, ModelDtype
from dragon_torch.loss import DragonLossFunction

type Batch = Any
type DataLike = Iterable[Batch]


class EpochCallback(Protocol):
    """Called by fit() after each epoch's history record is built.

    `record` is the live dict that goes into history, so a callback may add keys
    (e.g. an evaluation it ran) for later callbacks and the returned history.
    Return True to stop training after this epoch (a checkpoint is still saved).
    Exceptions propagate out of fit() unhandled, e.g. optuna.TrialPruned; no
    checkpoint is written for that epoch.
    """

    def __call__(self, model: "_DragonModel", record: dict[str, float]) -> Optional[bool]: ...


class _DragonModel(ABC, Generic[ModelDtype]):
    """The interface a DragonTrainer drives. Implement every abstract method to
    make a model trainable, e.g. a supervised DragonModel or an RL policy model
    whose `fit` collects rollouts and whose `evaluate` returns episode return."""
    kind: ClassVar[str]
    _registry: ClassVar[dict[str, type["_DragonModel"]]] = {}
    # direction of `evaluate`: False for a loss, True for e.g. an episode return.
    # Lets tuners / early stopping compare models without knowing what they are.
    higher_is_better: ClassVar[bool] = False
    # history key fit() stores evaluate(val_data) under
    eval_metric: ClassVar[str] = 'val_loss'

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        # only register a kind the subclass declares itself; an inherited kind
        # would overwrite the parent's registry entry
        _DragonModel._registry[cls.__dict__.get('kind', cls.__name__)] = cls

    # ------- torch.nn.Module passthroughs ------- #
    @abstractmethod
    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]: ...

    @abstractmethod
    def eval(self) -> "_DragonModel": ...

    @abstractmethod
    def to(self, device: Union[str, torch.device]) -> "_DragonModel": ...

    # ------- forward / loss ------- #
    @abstractmethod
    def forward(self, X: torch.Tensor, **kwargs) -> ModelDtype: ...

    @abstractmethod
    def loss(self, Y_hat: torch.Tensor, Y: Optional[torch.Tensor] = None,
             **kwargs) -> torch.Tensor: ...

    # ------- data ------- #
    @abstractmethod
    def _to_device(self, x: Any) -> Any: ...

    @abstractmethod
    def unpack_batch(self, batch: Batch) -> tuple[Any, Optional[torch.Tensor]]: ...

    @staticmethod
    @abstractmethod
    def iter_batches(data: DataLike) -> Iterator[Batch]: ...

    # ------- training ------- #
    @abstractmethod
    def _batch_loss(self, batch: Batch) -> torch.Tensor: ...

    @abstractmethod
    def step(self, batch: Batch, optimizer: torch.optim.Optimizer) -> float:
        """One optimisation step on one batch; returns the batch loss."""

    @abstractmethod
    def evaluate(self, data: DataLike) -> float: ...

    @abstractmethod
    def fit(self, train_data: DataLike, optimizer: torch.optim.Optimizer,
            val_data: Optional[DataLike] = None,
            epochs: Optional[int] = None,
            callbacks: Sequence[EpochCallback] = ()) -> list[dict[str, float]]: ...

    # ------- checkpoints ------- #
    @abstractmethod
    def checkpoint_path(self, epoch: Optional[int] = None) -> Path: ...

    @abstractmethod
    def latest_checkpoint(self) -> Optional[Path]: ...

    @abstractmethod
    def save_checkpoint(self, optimizer: Optional[torch.optim.Optimizer] = None) -> Path: ...

    @abstractmethod
    def load_checkpoint(self, path: PathLike,
                        optimizer: Optional[torch.optim.Optimizer] = None) -> int: ...

    def __call__(self, X: torch.Tensor, **kwargs) -> ModelDtype:
        return self.forward(X, **kwargs)

    def penalize(self, loss: torch.Tensor, penalty_fn: Callable, *args, **kwargs) -> torch.Tensor:
        result = penalty_fn(loss, *args, **kwargs)
        if not isinstance(result, torch.Tensor):
            raise ValueError(
                f"Expected penalty result to be type={torch.Tensor}, found: {type(result)}")
        return result


class DragonModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    checkpoint_dir: PathLike
    save_every: int = Field(ge=1)
    max_epochs: int = Field(ge=1)
    kind: str = 'torch-model'
    device: str = 'cpu'
    # keys used to pull (X, Y) out of dict batches, e.g. semasia's 'embeddings'/'labels'
    input_key: str = 'X'
    target_key: Optional[str] = 'Y'


def model_factory(module: DragonModule, loss_fn: Callable, cfg: DragonModelConfig, **kw) -> "DragonModel":
    return _DragonModel._registry[cfg.kind](module=module, loss_fn=loss_fn, cfg=cfg, **kw)


class DragonModel(_DragonModel[ModelDtype]):
    """Default supervised implementation: wraps a DragonModule + loss and trains
    it on DragonDatasets. Pass it to a DragonTrainer, or call `fit` directly.

    Every item a dataset yields is treated as one batch and split into (X, Y) by
    `unpack_batch`: dicts via cfg.input_key / cfg.target_key, (X, Y) tuples, or a
    bare X. Override `unpack_batch` for other layouts.

    Data sources can opt into these duck-typed hooks (all optional):
      set_epoch(epoch)        called before each training epoch (e.g. reshuffle)
      bind(model)             called once by fit / evaluate, before iterating, so
                              a source can use the model, e.g. a policy collecting
                              its own rollouts
      metrics() -> dict       merged into each epoch's history record (keys the
                              model already wrote, like 'train_loss', win)

    Subclasses can likewise return per-epoch values from `epoch_metrics()`
    (e.g. loss components accumulated in `step`); they win over source metrics.
      state_dict() / load_state_dict(state)
                              the training source's state is saved in checkpoints
                              (e.g. normalisation stats, RNG) and restored on load,
                              or on the next bind if no source is bound yet
    """
    kind = 'torch-model'

    def __init__(self, module: DragonModule, loss_fn: Callable, cfg: DragonModelConfig):
        if not isinstance(module, torch.nn.Module):
            raise TypeError(
                f"module must be a torch.nn.Module (DragonModule), got {type(module)}")
        self.config = cfg
        self.device = torch.device(cfg.device)
        self.torch_module = module.to(self.device)
        self.loss_fn = loss_fn if isinstance(
            loss_fn, DragonLossFunction) else DragonLossFunction(loss_fn)
        self.epoch = 0
        self.train_source: Optional[DataLike] = None
        # source state loaded from a checkpoint before any source was bound
        self._pending_source_state: Optional[dict] = None
        # other objects checkpointed alongside the model (see register_state)
        self._registered_state: dict[str, Any] = {}
        self._pending_registered_state: dict[str, dict] = {}

    # ------- torch.nn.Module passthroughs ------- #
    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return self.torch_module.parameters(recurse=recurse)

    def train(self, mode: bool = True) -> "DragonModel":
        self.torch_module.train(mode)
        return self

    def eval(self) -> "DragonModel":
        return self.train(False)

    def to(self, device: Union[str, torch.device]) -> "DragonModel":
        self.device = torch.device(device)
        self.torch_module.to(self.device)
        return self

    # ------- forward / loss ------- #
    def forward(self, X: torch.Tensor, **kw) -> ModelDtype:
        # call the module (not .forward) so torch hooks run
        return self.torch_module(X, **kw)

    def loss(self, Y_hat: torch.Tensor, Y: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        return self.loss_fn(Y_hat, Y, **kwargs)

    # ------- data ------- #
    def _to_device(self, x: Any) -> Any:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, dict):
            return {k: self._to_device(v) for k, v in x.items()}
        return x

    def unpack_batch(self, batch: Batch) -> tuple[Any, Optional[torch.Tensor]]:
        if isinstance(batch, dict):
            if self.config.input_key not in batch:
                raise KeyError(
                    f"Batch has no input_key='{self.config.input_key}', keys: {list(batch)}")
            X = batch[self.config.input_key]
            Y = batch.get(self.config.target_key) if self.config.target_key else None
        elif isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                X, Y = batch
            elif len(batch) == 1:
                X, Y = batch[0], None
            else:
                raise ValueError(
                    f"Expected (X,) or (X, Y) batch, got sequence of length {len(batch)}")
        else:
            X, Y = batch, None
        return self._to_device(X), self._to_device(Y)

    @staticmethod
    def iter_batches(data: DataLike) -> Iterator[Batch]:
        # DragonDataset inherits HF Dataset.__iter__, which breaks without HF
        # internals, so index map-style datasets rather than iterating them
        if isinstance(data, (DataLoader, IterableDataset)):
            yield from data
        elif hasattr(data, '__getitem__') and hasattr(data, '__len__'):
            for i in range(len(data)):
                yield data[i]
        else:
            yield from data

    def bind(self, data: DataLike, train: bool = False) -> DataLike:
        """Let `data` see this model (if it has `bind`). Binding a training source
        also makes its state part of checkpoints."""
        if hasattr(data, 'bind'):
            data.bind(self)
        if train and data is not self.train_source:
            self.train_source = data
            if self._pending_source_state is not None and hasattr(data, 'load_state_dict'):
                data.load_state_dict(self._pending_source_state)
                self._pending_source_state = None
        return data

    # ------- training ------- #
    def _batch_loss(self, batch: Batch) -> torch.Tensor:
        X, Y = self.unpack_batch(batch)
        loss = self.loss(self.forward(X), Y)
        if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
            raise ValueError(
                f"Loss must be a scalar tensor, got {type(loss)} with shape {getattr(loss, 'shape', None)}")
        return loss

    def step(self, batch: Batch, optimizer: torch.optim.Optimizer) -> float:
        self.train()
        optimizer.zero_grad()
        loss = self._batch_loss(batch)
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, data: DataLike) -> float:
        """Mean per-batch loss over data."""
        self.bind(data)
        self.eval()
        losses = [self._batch_loss(batch).item()
                  for batch in self.iter_batches(data)]
        if not losses:
            raise ValueError("evaluate() got no batches")
        return float(np.mean(losses))

    def fit(self, train_data: DataLike, optimizer: torch.optim.Optimizer,
            val_data: Optional[DataLike] = None, epochs: Optional[int] = None,
            callbacks: Sequence[EpochCallback] = ()) -> list[dict[str, float]]:
        """Train for `epochs` (default cfg.max_epochs) more epochs, checkpointing
        every cfg.save_every epochs and after the last one. Returns per-epoch history.

        `callbacks` run in order after each epoch (see EpochCallback); all of them
        run even if an earlier one asks to stop, then training stops early with a
        checkpoint."""
        epochs = self.config.max_epochs if epochs is None else epochs
        self.bind(train_data, train=True)
        history = []
        for _ in range(epochs):
            if hasattr(train_data, 'set_epoch'):  # e.g. Aggregator reshuffle
                train_data.set_epoch(self.epoch)
            losses = [self.step(batch, optimizer)
                      for batch in self.iter_batches(train_data)]
            if not losses:
                raise ValueError("fit() got no training batches")
            self.epoch += 1
            record = {'epoch': self.epoch, 'train_loss': float(np.mean(losses))}
            if val_data is not None:
                record[self.eval_metric] = self.evaluate(val_data)
            record.update({k: v for k, v in self.epoch_metrics().items()
                           if k not in record})
            if hasattr(train_data, 'metrics'):
                record.update({k: v for k, v in train_data.metrics().items()
                               if k not in record})
            history.append(record)
            stop = False
            for callback in callbacks:
                stop = bool(callback(self, record)) or stop
            if stop or self.epoch % self.config.save_every == 0 or len(history) == epochs:
                self.save_checkpoint(optimizer)
            if stop:
                break
        return history

    def epoch_metrics(self) -> dict[str, float]:
        """Extra values for the epoch that just finished; called once per epoch."""
        return {}

    # ------- checkpoints ------- #
    def checkpoint_path(self, epoch: Optional[int] = None) -> Path:
        epoch = self.epoch if epoch is None else epoch
        return build_path(self.config.checkpoint_dir) / f"{self.config.kind}-epoch-{epoch}.pt"

    def latest_checkpoint(self) -> Optional[Path]:
        """Highest-epoch checkpoint for this kind in checkpoint_dir, or None."""
        prefix = f"{self.config.kind}-epoch-"
        found = {}
        for path in build_path(self.config.checkpoint_dir).glob(f"{prefix}*.pt"):
            suffix = path.stem[len(prefix):]
            if suffix.isdigit():  # skip e.g. 'kind-epoch-1-backup.pt'
                found[int(suffix)] = path
        return found[max(found)] if found else None

    def save_checkpoint(self, optimizer: Optional[torch.optim.Optimizer] = None) -> Path:
        path = self.checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': self.epoch,
            'kind': self.config.kind,
            'module_state': self.torch_module.state_dict(),
            'optimizer_state': None if optimizer is None else optimizer.state_dict(),
            'extra_state': self.extra_state(),
        }, path)
        return path

    def load_checkpoint(self, path: PathLike, optimizer: Optional[torch.optim.Optimizer] = None) -> int:
        # weights_only=False: extra_state may hold source state (e.g. numpy RNG
        # / running stats) that the safe unpickler rejects. Only load your own files.
        state = torch.load(build_path(path), map_location=self.device, weights_only=False)
        self.torch_module.load_state_dict(state['module_state'])
        if optimizer is not None and state['optimizer_state'] is not None:
            optimizer.load_state_dict(state['optimizer_state'])
        self.epoch = state['epoch']
        self.load_extra_state(state.get('extra_state') or {})
        return self.epoch

    def register_state(self, name: str, obj: Any) -> None:
        """Checkpoint `obj` (anything with state_dict / load_state_dict, e.g. a
        trainer's eval callback) with the model under `name`. If a checkpoint
        with that name was already loaded, its state is applied now."""
        if not (hasattr(obj, 'state_dict') and hasattr(obj, 'load_state_dict')):
            raise TypeError(f"{type(obj)} needs state_dict() and load_state_dict()")
        self._registered_state[name] = obj
        if name in self._pending_registered_state:
            obj.load_state_dict(self._pending_registered_state.pop(name))

    def extra_state(self) -> dict[str, Any]:
        """Non-parameter state to checkpoint. Subclasses extend it, e.g.
        `return {**super().extra_state(), 'global_step': self.global_step}`."""
        state: dict[str, Any] = {}
        if self.train_source is not None and hasattr(self.train_source, 'state_dict'):
            state['train_source'] = self.train_source.state_dict()
        if self._registered_state:
            state['registered'] = {k: v.state_dict() for k, v in self._registered_state.items()}
        return state

    def load_extra_state(self, state: dict[str, Any]) -> None:
        for name, obj_state in (state.get('registered') or {}).items():
            if name in self._registered_state:
                self._registered_state[name].load_state_dict(obj_state)
            else:
                self._pending_registered_state[name] = obj_state
        source_state = state.get('train_source')
        if source_state is None:
            return
        if self.train_source is not None and hasattr(self.train_source, 'load_state_dict'):
            self.train_source.load_state_dict(source_state)
        else:  # e.g. DragonTrainer.resume() runs before fit() binds the source
            self._pending_source_state = source_state
