from pydantic import BaseModel, Field
from trainers.RL.config import DragonRLTrainerConfig, OptimConfig, RolloutConfig


class A2CLossConfig(BaseModel):
    """A2CLoss settings."""
    vf_coef: float = Field(default=0.5, ge=0)
    ent_coef: float = Field(default=0.0, ge=0)
    normalize_advantages: bool = False


class A2CTrainerConfig(DragonRLTrainerConfig):
    """DragonRLTrainerConfig + the A2C loss. Defaults follow the usual A2C
    recipe: short 5-step rollouts, one full-batch update each, RMSprop at 7e-4."""
    a2c: A2CLossConfig = Field(default_factory=A2CLossConfig)
    rollout: RolloutConfig = Field(default_factory=lambda: RolloutConfig(
        n_steps=5, gae_lambda=1.0, update_epochs=1, n_minibatches=1))
    optim: OptimConfig = Field(default_factory=lambda: OptimConfig(name='rmsprop', lr=7e-4))
