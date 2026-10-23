from abc import ABC, abstractmethod
import torch
from typing import Any, Union, TypeVar, Generic, ClassVar

ModelDtype = TypeVar('ModelDtype', covariant=True)


class DragonModule(ABC, Generic[ModelDtype]):
    kind: ClassVar[str]
    _registry: ClassVar[dict[str, type["DragonModule"]]] = {}

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if hasattr(cls, "kind"):
            DragonModule._registry[cls.kind] = cls
        else:
            DragonModule._registry[cls.__name__] = cls

    @abstractmethod
    def forward(self, X: torch.Tensor, **kwargs) -> ModelDtype: ...
