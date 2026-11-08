from typing import Any, Callable, ClassVar, Optional
import torch


class DragonLossFunction:
    """Callable loss wrapper: loss_fn(Y_hat, Y, **kw) -> scalar tensor.

    Wrap any callable (e.g. torch.nn.MSELoss() or F.cross_entropy), or subclass
    and override `compute` for custom / unsupervised losses where Y may be None.
    """
    kind: ClassVar[str]
    _registry: ClassVar[dict[str, type["DragonLossFunction"]]] = {}

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        # only register a kind the subclass declares itself, not an inherited one
        DragonLossFunction._registry[cls.__dict__.get('kind', cls.__name__)] = cls

    def __init__(self, fn: Optional[Callable[..., torch.Tensor]] = None):
        self.fn = fn

    def compute(self, Y_hat: torch.Tensor, Y: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if self.fn is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no loss fn: pass one in or override compute()")
        return self.fn(Y_hat, **kwargs) if Y is None else self.fn(Y_hat, Y, **kwargs)

    def __call__(self, Y_hat: torch.Tensor, Y: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        return self.compute(Y_hat, Y, **kwargs)
