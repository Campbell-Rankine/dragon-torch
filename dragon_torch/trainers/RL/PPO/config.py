from pydantic import BaseModel, Field
from dragon_torch.trainers.RL.config import DragonRLTrainerConfig, RolloutConfig


class PPOLossConfig(BaseModel):
    """PPOLoss settings (clipped surrogate, optional value clipping)."""
    clip_coef: float = Field(default=0.2, gt=0)
    vf_coef: float = Field(default=0.5, ge=0)
    ent_coef: float = Field(default=0.0, ge=0)
    clip_vloss: bool = True
    normalize_advantages: bool = True


class PPOTrainerConfig(DragonRLTrainerConfig):
    """DragonRLTrainerConfig + the PPO loss. Rollout defaults: 4 update epochs
    of 4 minibatches per rollout."""
    ppo: PPOLossConfig = Field(default_factory=PPOLossConfig)
    rollout: RolloutConfig = Field(default_factory=lambda: RolloutConfig(update_epochs=4,
                                                                         n_minibatches=4))
