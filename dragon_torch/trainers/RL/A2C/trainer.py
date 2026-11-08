from typing import Optional
from dragon_torch.loss.RL import A2CLoss
from dragon_torch.modules.RL import DragonPolicyModule
from dragon_torch.trainers.RL.trainer import DragonRLTrainer
from .config import A2CTrainerConfig


class A2CTrainer(DragonRLTrainer):
    """Advantage actor-critic over a DragonRLTrainer run."""
    config: A2CTrainerConfig

    def __init__(self, cfg: A2CTrainerConfig, module: Optional[DragonPolicyModule] = None):
        if not isinstance(cfg, A2CTrainerConfig):
            raise TypeError(f"A2CTrainer needs an A2CTrainerConfig, got {type(cfg).__name__}")
        super().__init__(cfg, module)

    def build_loss(self) -> A2CLoss:
        return A2CLoss(**self.config.a2c.model_dump())
