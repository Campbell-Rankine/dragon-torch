"""A2C as a multi-agent model: a MultiAgentRLModel whose loss (A2CLoss
per agent group) is built from its config."""
from typing import Callable, Optional
from dragon_torch.loss.RL import MultiAgentLoss
from dragon_torch.modules.RL.policy import MultiAgentPolicyModule
from ..loss_configs import A2CLossConfig
from .base import MultiAgentRLConfig, MultiAgentRLModel


class MultiAgentA2CConfig(A2CLossConfig, MultiAgentRLConfig):
    """MultiAgentRLConfig + the A2C loss settings (the same fields as A2CConfig)."""
    kind: str = 'marl-a2c-model'


class MultiAgentA2CModel(MultiAgentRLModel):
    """Advantage actor-critic on a multi-agent env: A2CLoss applied to each agent group
    (MultiAgentLoss), built from `cfg` (pass `loss_fn` to replace it)."""
    kind = 'marl-a2c-model'
    config: MultiAgentA2CConfig

    def __init__(self, module: MultiAgentPolicyModule, cfg: MultiAgentA2CConfig,
                 loss_fn: Optional[Callable] = None):
        if not isinstance(cfg, MultiAgentA2CConfig):
            raise TypeError(f"MultiAgentA2CModel needs a MultiAgentA2CConfig, got {type(cfg).__name__}")
        super().__init__(module, loss_fn if loss_fn is not None else MultiAgentLoss(cfg.build_loss()), cfg)
