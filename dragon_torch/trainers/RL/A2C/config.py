from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator
from dragon_torch.models.RL import A2CConfig, MultiAgentA2CConfig
from dragon_torch.trainers.RL.config import (GuardConfig, MultiAgentRLTrainerConfig, OptimConfig,
                                             RolloutConfig, SingleAgentRLTrainerConfig,
                                             upgrade_model_config)


class A2CDefaults(BaseModel):
    """The usual A2C recipe, shared by the single- and multi-agent configs:
    short 5-step rollouts, one full-batch update each, RMSprop at 7e-4. One
    update per rollout means approx_kl is always 0, so the default guard only
    checks for non-finite losses (max_kl=None)."""
    rollout: RolloutConfig = Field(default_factory=lambda: RolloutConfig(
        n_steps=5, gae_lambda=1.0, update_epochs=1, n_minibatches=1))
    optim: OptimConfig = Field(default_factory=lambda: OptimConfig(name='rmsprop', lr=7e-4))
    guard: Optional[GuardConfig] = Field(default_factory=lambda: GuardConfig(max_kl=None))


class A2CTrainerConfig(A2CDefaults, SingleAgentRLTrainerConfig):
    """SingleAgentRLTrainerConfig for A2CModel: `model` is an A2CConfig, so the
    loss settings are `model.vf_coef`, `model.ent_coef`, ... A plain
    DragonRLConfig is accepted and becomes an A2CConfig with default loss settings."""
    model: A2CConfig

    @field_validator('model', mode='before')
    @classmethod
    def _a2c_model(cls, v: Any) -> Any:
        return upgrade_model_config(v, A2CConfig)


class MultiAgentA2CTrainerConfig(A2CDefaults, MultiAgentRLTrainerConfig):
    """MultiAgentRLTrainerConfig for MultiAgentA2CModel: `model` is a
    MultiAgentA2CConfig, with the same loss settings as A2CTrainerConfig
    (`model.clip_coef`, ...), applied per agent group. A plain DragonRLConfig,
    or a single-agent A2CConfig (loss settings kept), is converted."""
    model: MultiAgentA2CConfig

    @field_validator('model', mode='before')
    @classmethod
    def _multi_agent_a2c_model(cls, v: Any) -> Any:
        return upgrade_model_config(v, MultiAgentA2CConfig)
