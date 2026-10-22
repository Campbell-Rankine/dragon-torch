import torch
from typing import Generic, TypeVar, Callable
from abc import abstractmethod
from templates.base_classes import DragonBaseClass

ModelDtype = TypeVar('ModelDtype', covariant=True)


class _DragonModel(DragonBaseClass, Generic[ModelDtype]):
    @abstractmethod
    def forward(self, X: torch.Tensor, **kwargs) -> ModelDtype: ...

    @abstractmethod
    def loss(self, Y_hat: torch.Tensor, Y: torch.Tensor,
             **kwargs) -> ModelDtype: ...

    def penalize(self, loss: torch.Tensor, penalty_fn: Callable, *args, **kwargs) -> ModelDtype:
        result = penalty_fn(loss, *args, **kwargs)
        if not isinstance(result, ModelDtype):
            raise ValueError(
                f"Expected penalty result to be type={ModelDtype}, found: {type(result)}")
        return result
