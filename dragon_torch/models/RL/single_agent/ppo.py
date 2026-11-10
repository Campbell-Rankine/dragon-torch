"""PPO as a model: a SingleAgentRLModel whose loss is built from its config."""
from typing import Callable, Optional
from dragon_torch.modules.RL.policy import SingleAgentPolicyModule
from ..base import DragonRLConfig
from ..loss_configs import PPOLossConfig
from .base import SingleAgentRLModel


class PPOConfig(PPOLossConfig, DragonRLConfig):
    """DragonRLConfig + the PPO loss settings."""
    kind: str = 'ppo-model'


class PPOModel(SingleAgentRLModel):
    """Proximal Policy Optimisation (Schulman et al. 2017) on a gymnasium env.

        model = PPOModel(ActorCritic(obs_space, act_space), PPOConfig(checkpoint_dir=..., clip_coef=0.1))

    The loss comes from `cfg` (pass `loss_fn` to replace it). Pair with a
    RolloutSource doing several update epochs of several minibatches."""
    kind = 'ppo-model'
    config: PPOConfig

    def __init__(self, module: SingleAgentPolicyModule, cfg: PPOConfig,
                 loss_fn: Optional[Callable] = None):
        if not isinstance(cfg, PPOConfig):
            raise TypeError(f"PPOModel needs a PPOConfig, got {type(cfg).__name__}")
        super().__init__(module, loss_fn if loss_fn is not None else cfg.build_loss(), cfg)
