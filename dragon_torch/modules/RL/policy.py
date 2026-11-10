from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence
import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution
from gymnasium import spaces
from dragon_torch.common.torch.tree import Tree
from dragon_torch.modules.base import DragonModule, ModelDtype
from dragon_torch.modules.RL.heads import layer_init, make_dist_head


@dataclass
class PolicyOutput:
    dist: Distribution
    value: Optional[torch.Tensor] = None  # [B]; None for actor-only policies


class DragonPolicyModule(DragonModule[ModelDtype]):
    """Base of every RL policy network. Single-agent policies return one
    PolicyOutput; multi-agent policies return one per agent group."""

    @abstractmethod
    def forward(self, X: Any, **kwargs) -> ModelDtype: ...


class SingleAgentPolicyModule(DragonPolicyModule[PolicyOutput]):
    """forward(obs [B, ...]) -> PolicyOutput with batch shape [B]."""

    @abstractmethod
    def forward(self, X: Any, **kwargs) -> PolicyOutput: ...


class MultiAgentPolicyModule(DragonPolicyModule[dict[str, PolicyOutput]]):
    """forward({group: obs [B_g, ...]}) -> {group: PolicyOutput [B_g]}.

    Agents in a group share the group's weights, so each group's obs arrive
    flattened over (env, agent): B_g = batch x agents in the group. Groups
    absent from X (no live agents in a minibatch) are absent from the output."""

    @abstractmethod
    def forward(self, X: Mapping[str, Tree], **kwargs) -> dict[str, PolicyOutput]: ...


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


class ActorCritic(SingleAgentPolicyModule):
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


class MultiAgentActorCritic(MultiAgentPolicyModule):
    """One ActorCritic per agent group (weights shared within a group,
    independent across groups; each critic sees only its agent's obs).

        MultiAgentActorCritic({'red': (obs_space, act_space), 'blue': (...)}, hidden=(64, 64))

    `policies` maps group -> ActorCritic directly, for custom per-group networks.
    """
    kind = 'multi-agent-actor-critic'

    def __init__(self, spaces_by_group: Optional[Mapping[str, tuple[spaces.Space, spaces.Space]]] = None,
                 policies: Optional[Mapping[str, SingleAgentPolicyModule]] = None, **actor_critic_kw: Any):
        super().__init__()
        if (spaces_by_group is None) == (policies is None):
            raise ValueError("pass exactly one of spaces_by_group or policies")
        if policies is None:
            policies = {g: ActorCritic(obs, act, **actor_critic_kw)
                        for g, (obs, act) in spaces_by_group.items()}
        if not policies:
            raise ValueError("MultiAgentActorCritic needs at least one group")
        self.policies = nn.ModuleDict(dict(policies))

    @property
    def groups(self) -> list[str]:
        return list(self.policies.keys())

    def forward(self, X: Mapping[str, Tree], **kw) -> dict[str, PolicyOutput]:
        unknown = set(X) - set(self.policies)
        if unknown:
            raise KeyError(f"obs for unknown groups {sorted(unknown)}; module groups: {self.groups}")
        return {g: self.policies[g](obs, **kw) for g, obs in X.items()}
