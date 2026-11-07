from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence
import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution
from gymnasium import spaces
from modules.base import DragonModule
from modules.RL.heads import layer_init, make_dist_head


@dataclass
class PolicyOutput:
    dist: Distribution
    value: Optional[torch.Tensor] = None  # [B]; None for actor-only policies


class DragonPolicyModule(DragonModule[PolicyOutput]):
    """A DragonModule whose forward(obs) returns a PolicyOutput."""

    @abstractmethod
    def forward(self, X: Any, **kwargs) -> PolicyOutput: ...


class FlattenObs(nn.Module):
    """Box obs -> [B, prod(shape)] float; Discrete obs -> one-hot."""

    def __init__(self, observation_space: spaces.Space):
        super().__init__()
        if isinstance(observation_space, spaces.Box):
            self.n = None
            self.out_features = int(np.prod(observation_space.shape))
        elif isinstance(observation_space, spaces.Discrete):
            self.n = int(observation_space.n)
            self.start = int(observation_space.start)
            self.out_features = self.n
        else:
            raise TypeError(f"No default encoder for {type(observation_space).__name__} "
                            "observations; pass encoder= (with a features_dim attribute)")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.n is not None:
            return nn.functional.one_hot(obs.long() - self.start, self.n).float()
        return obs.reshape(obs.shape[0], -1).float()


def mlp(in_features: int, hidden: Sequence[int], activation: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    layers, size = [], in_features
    for h in hidden:
        layers += [layer_init(nn.Linear(size, h)), activation()]
        size = h
    return nn.Sequential(*layers)


class ActorCritic(DragonPolicyModule):
    """Actor-critic for on-policy algorithms (PPO, A2C).

    Default: separate MLP trunks for actor and critic over flattened obs (the
    stable choice for vector obs). Pass `encoder` (any nn.Module / DragonModule
    with `features_dim`, or give features_dim=) to share a feature extractor,
    e.g. a CNN for pixels; both heads then read its features.
    """
    kind = 'actor-critic'

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
                 hidden: Sequence[int] = (64, 64), activation: type[nn.Module] = nn.Tanh,
                 encoder: Optional[nn.Module] = None, features_dim: Optional[int] = None):
        super().__init__()
        if encoder is None:
            self.encoder = None
            flat = FlattenObs(observation_space)
            self.actor_trunk = nn.Sequential(flat, mlp(flat.out_features, hidden, activation))
            self.critic_trunk = nn.Sequential(FlattenObs(observation_space),
                                              mlp(flat.out_features, hidden, activation))
            features_dim = hidden[-1] if hidden else flat.out_features
        else:
            features_dim = features_dim if features_dim is not None else getattr(encoder, 'features_dim', None)
            if features_dim is None:
                raise ValueError("encoder needs a features_dim attribute, or pass features_dim=")
            self.encoder = encoder
        self.actor_head = make_dist_head(action_space, features_dim)
        self.critic_head = layer_init(nn.Linear(features_dim, 1), std=1.0)

    def forward(self, X: Any, **kw) -> PolicyOutput:
        if self.encoder is not None:
            actor_features = critic_features = self.encoder(X)
        else:
            actor_features, critic_features = self.actor_trunk(X), self.critic_trunk(X)
        return PolicyOutput(dist=self.actor_head(actor_features),
                            value=self.critic_head(critic_features).squeeze(-1))
