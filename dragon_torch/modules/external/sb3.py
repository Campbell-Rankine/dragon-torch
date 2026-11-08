"""Use a DragonModule as the feature extractor of a stable-baselines3 policy.

    model = PPO('MlpPolicy', env, policy_kwargs=sb3_policy_kwargs(my_module, features_dim=64))

SB3 then owns training and checkpointing (DragonTrainer is not involved), which
makes it a quick baseline to check a DragonRLModel against. Notes:
  * SB3 preprocesses obs before the extractor: image Boxes are scaled to [0, 1],
    Discrete obs are one-hot encoded.
  * SB3 builds the extractor itself, passing the same module instance each time,
    so with share_features_extractor=False actor and critic still share weights.
"""
from typing import Any
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from dragon_torch.modules.base import DragonModule


class DragonFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space, module: DragonModule, features_dim: int):
        super().__init__(observation_space, features_dim)
        self.module = module

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.module(observations)


def sb3_policy_kwargs(module: DragonModule, features_dim: int, **extra: Any) -> dict[str, Any]:
    return {'features_extractor_class': DragonFeaturesExtractor,
            'features_extractor_kwargs': {'module': module, 'features_dim': features_dim},
            **extra}
