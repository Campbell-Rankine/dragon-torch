from typing import Optional
from dragon_torch.loss.RL import PPOLoss
from dragon_torch.modules.RL import DragonPolicyModule
from dragon_torch.trainers.RL.trainer import DragonRLTrainer
from .config import PPOTrainerConfig


class PPOTrainer(DragonRLTrainer):
    """Proximal Policy Optimisation over a DragonRLTrainer run."""
    config: PPOTrainerConfig

    def __init__(self, cfg: PPOTrainerConfig, module: Optional[DragonPolicyModule] = None):
        if not isinstance(cfg, PPOTrainerConfig):
            raise TypeError(f"PPOTrainer needs a PPOTrainerConfig, got {type(cfg).__name__}")
        super().__init__(cfg, module)

    def build_loss(self) -> PPOLoss:
        return PPOLoss(**self.config.ppo.model_dump())
