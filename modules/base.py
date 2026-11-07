from abc import ABC, abstractmethod
import torch
from typing import Any, TypeVar, Generic, ClassVar

ModelDtype = TypeVar('ModelDtype', covariant=True)


class DragonModule(torch.nn.Module, ABC, Generic[ModelDtype]):
    """torch.nn.Module base so parameters, .to(), train()/eval() and state_dict work.
    Subclasses must call super().__init__() before assigning submodules."""
    kind: ClassVar[str]
    _registry: ClassVar[dict[str, type["DragonModule"]]] = {}

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        # only register a kind the subclass declares itself, not an inherited one
        DragonModule._registry[cls.__dict__.get('kind', cls.__name__)] = cls

    @abstractmethod
    def forward(self, X: torch.Tensor, **kwargs) -> ModelDtype: ...
