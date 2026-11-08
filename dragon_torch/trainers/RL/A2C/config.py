from typing import Optional
from pydantic import BaseModel, Field
from dragon_torch.trainers.RL.config import DragonRLTrainerConfig, GuardConfig, OptimConfig, RolloutConfig


class A2CLossConfig(BaseModel):
    """A2CLoss settings."""
    vf_coef: float = Field(default=0.5, ge=0)
    ent_coef: float = Field(default=0.0, ge=0)
    normalize_advantages: bool = False


class A2CTrainerConfig(DragonRLTrainerConfig):
    """DragonRLTrainerConfig + the A2C loss. Defaults follow the usual A2C
    recipe: short 5-step rollouts, one full-batch update each, RMSprop at 7e-4.
    One update per rollout means approx_kl is always 0, so the default guard
    only checks for non-finite losses (max_kl=None)."""
    a2c: A2CLossConfig = Field(default_factory=A2CLossConfig)
    rollout: RolloutConfig = Field(default_factory=lambda: RolloutConfig(
        n_steps=5, gae_lambda=1.0, update_epochs=1, n_minibatches=1))
    optim: OptimConfig = Field(default_factory=lambda: OptimConfig(name='rmsprop', lr=7e-4))
    guard: Optional[GuardConfig] = Field(default_factory=lambda: GuardConfig(max_kl=None))
