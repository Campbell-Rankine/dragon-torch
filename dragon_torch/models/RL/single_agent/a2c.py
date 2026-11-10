"""A2C as a model: a SingleAgentRLModel whose loss is built from its config."""
from typing import Callable, Optional
from dragon_torch.modules.RL.policy import SingleAgentPolicyModule
from ..base import DragonRLConfig
from ..loss_configs import A2CLossConfig
from .base import SingleAgentRLModel


class A2CConfig(A2CLossConfig, DragonRLConfig):
    """DragonRLConfig + the A2C loss settings."""
    kind: str = 'a2c-model'


class A2CModel(SingleAgentRLModel):
    """Advantage actor-critic on a gymnasium env.

        model = A2CModel(ActorCritic(obs_space, act_space), A2CConfig(checkpoint_dir=..., ent_coef=0.01))

    The loss comes from `cfg` (pass `loss_fn` to replace it). Pair with
    RolloutSource(update_epochs=1, n_minibatches=1)."""
    kind = 'a2c-model'
    config: A2CConfig

    def __init__(self, module: SingleAgentPolicyModule, cfg: A2CConfig,
                 loss_fn: Optional[Callable] = None):
        if not isinstance(cfg, A2CConfig):
            raise TypeError(f"A2CModel needs an A2CConfig, got {type(cfg).__name__}")
        super().__init__(module, loss_fn if loss_fn is not None else cfg.build_loss(), cfg)
