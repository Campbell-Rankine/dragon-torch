"""Action-distribution heads, one per gymnasium action space type."""
import math
from typing import Sequence
import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical, Distribution, Independent, Normal
from gymnasium import spaces


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias: float = 0.0) -> nn.Linear:
    """Orthogonal init (the PPO-standard choice): sqrt(2) for hidden layers,
    0.01 for policy outputs, 1.0 for value outputs."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class MultiCategorical(Distribution):
    """Independent categoricals over a MultiDiscrete action; actions are [..., k]."""
    arg_constraints = {}

    def __init__(self, logits: torch.Tensor, nvec: Sequence[int]):
        self.dists = [Categorical(logits=l) for l in torch.split(logits, list(nvec), dim=-1)]
        super().__init__(batch_shape=logits.shape[:-1], event_shape=(len(nvec),), validate_args=False)

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        return torch.stack([d.sample(sample_shape) for d in self.dists], -1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return sum(d.log_prob(value[..., i]) for i, d in enumerate(self.dists))

    def entropy(self) -> torch.Tensor:
        return sum(d.entropy() for d in self.dists)

    @property
    def mode(self) -> torch.Tensor:
        return torch.stack([d.mode for d in self.dists], -1)


def deterministic_action(dist: Distribution) -> torch.Tensor:
    """Greedy action. Bernoulli.mode is NaN at p=0.5, so threshold it instead."""
    base = dist.base_dist if isinstance(dist, Independent) else dist
    if isinstance(base, Bernoulli):
        return (base.probs > 0.5).to(base.probs.dtype)
    return dist.mode


class DistHead(nn.Module):
    """features [B, in_features] -> torch Distribution with batch_shape [B]."""


class CategoricalHead(DistHead):
    def __init__(self, in_features: int, n: int):
        super().__init__()
        self.logits = layer_init(nn.Linear(in_features, n), std=0.01)

    def forward(self, features: torch.Tensor) -> Distribution:
        return Categorical(logits=self.logits(features))


class MultiCategoricalHead(DistHead):
    def __init__(self, in_features: int, nvec: Sequence[int]):
        super().__init__()
        self.nvec = [int(n) for n in nvec]
        self.logits = layer_init(nn.Linear(in_features, sum(self.nvec)), std=0.01)

    def forward(self, features: torch.Tensor) -> Distribution:
        return MultiCategorical(self.logits(features), self.nvec)


class BernoulliHead(DistHead):
    def __init__(self, in_features: int, n: int):
        super().__init__()
        self.logits = layer_init(nn.Linear(in_features, n), std=0.01)

    def forward(self, features: torch.Tensor) -> Distribution:
        return Independent(Bernoulli(logits=self.logits(features)), 1)


class GaussianHead(DistHead):
    """Diagonal Gaussian with a state-independent learned log-std. Samples are
    unbounded; action_to_env clips them to the Box when stepping the env."""

    def __init__(self, in_features: int, shape: Sequence[int], log_std_init: float = 0.0):
        super().__init__()
        self.shape = tuple(shape)
        dim = int(np.prod(self.shape))
        self.mean = layer_init(nn.Linear(in_features, dim), std=0.01)
        self.log_std = nn.Parameter(torch.full((dim,), log_std_init))

    def forward(self, features: torch.Tensor) -> Distribution:
        mean = self.mean(features)
        std = self.log_std.exp().expand_as(mean)
        shape = (*mean.shape[:-1], *self.shape)
        return Independent(Normal(mean.reshape(shape), std.reshape(shape)), len(self.shape))


def make_dist_head(action_space: spaces.Space, in_features: int) -> DistHead:
    if isinstance(action_space, spaces.Discrete):
        return CategoricalHead(in_features, int(action_space.n))
    if isinstance(action_space, spaces.MultiDiscrete):
        if action_space.nvec.ndim != 1:
            raise NotImplementedError("Only 1-D MultiDiscrete action spaces are supported")
        return MultiCategoricalHead(in_features, action_space.nvec)
    if isinstance(action_space, spaces.MultiBinary):
        return BernoulliHead(in_features, int(np.prod(action_space.shape)))
    if isinstance(action_space, spaces.Box):
        return GaussianHead(in_features, action_space.shape)
    raise NotImplementedError(f"No distribution head for {type(action_space).__name__}")
