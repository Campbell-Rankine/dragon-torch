"""PPO as a multi-agent model: a MultiAgentRLModel whose loss (PPOLoss
per agent group) is built from its config."""
from typing import Callable, Optional
from dragon_torch.loss.RL import MultiAgentLoss
from dragon_torch.modules.RL.policy import MultiAgentPolicyModule
from ..loss_configs import PPOLossConfig
from .base import MultiAgentRLConfig, MultiAgentRLModel


class MultiAgentPPOConfig(PPOLossConfig, MultiAgentRLConfig):
    """MultiAgentRLConfig + the PPO loss settings (the same fields as PPOConfig)."""
    kind: str = 'marl-ppo-model'


class MultiAgentPPOModel(MultiAgentRLModel):
    """Proximal Policy Optimisation on a multi-agent env: PPOLoss applied to each agent group
    (MultiAgentLoss), built from `cfg` (pass `loss_fn` to replace it)."""
    kind = 'marl-ppo-model'
    config: MultiAgentPPOConfig

    def __init__(self, module: MultiAgentPolicyModule, cfg: MultiAgentPPOConfig,
                 loss_fn: Optional[Callable] = None):
        if not isinstance(cfg, MultiAgentPPOConfig):
            raise TypeError(f"MultiAgentPPOModel needs a MultiAgentPPOConfig, got {type(cfg).__name__}")
        super().__init__(module, loss_fn if loss_fn is not None else MultiAgentLoss(cfg.build_loss()), cfg)
