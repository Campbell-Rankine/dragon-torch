import torch
from pydantic import BaseModel
from typing import Generic, TypeVar, Callable, ClassVar, Any, Union
from abc import abstractmethod, ABC
from common.types import PathLike, Path
from modules import DragonModule, ModelDtype
from loss import DragonLossFunction


class _DragonModel(ABC):
    kind: ClassVar[str]
    _registry: ClassVar[dict[str, type["_DragonModel"]]] = {}

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if hasattr(cls, "kind"):
            _DragonModel._registry[cls.kind] = cls
        else:
            _DragonModel._registry[cls.__name__] = cls

    @abstractmethod
    def forward(self, X: torch.Tensor, **kwargs) -> ModelDtype: ...

    @abstractmethod
    def loss(self, Y_hat: torch.Tensor, Y: torch.Tensor = None,
             **kwargs) -> ModelDtype: ...

    def penalize(self, loss: torch.Tensor, penalty_fn: Callable, *args, **kwargs) -> ModelDtype:
        result = penalty_fn(loss, *args, **kwargs)
        if not isinstance(result, ModelDtype):
            raise ValueError(
                f"Expected penalty result to be type={ModelDtype}, found: {type(result)}")
        return result


class DragonModelConfig(BaseModel):
    checkpoint_dir: PathLike
    save_every: int
    max_epochs: int
    kind: str = 'torch-model'


class DragonModel(_DragonModel[ModelDtype]):
    kind = 'torch-model'

    def __init__(self, module: DragonModule, loss_fn: DragonLossFunction, cfg: DragonModelConfig):
        self.torch_module = module
        self.loss_fn = loss_fn
        self.config = cfg

    def forward(self, X: torch.Tensor, **kw) -> ModelDtype:
        return self.torch_module.forward(X, **kw)

    def loss(self, Y_hat: torch.Tensor, Y: torch.Tensor = None, **kwargs):
        return self.loss_fn(Y_hat, Y, **kwargs)
