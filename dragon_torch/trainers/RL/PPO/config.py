from typing import Any
from pydantic import BaseModel, Field, field_validator
from dragon_torch.models.RL import PPOConfig, MultiAgentPPOConfig
from dragon_torch.trainers.RL.config import (MultiAgentRLTrainerConfig, RolloutConfig,
                                             SingleAgentRLTrainerConfig, upgrade_model_config)


class PPODefaults(BaseModel):
    """PPO's rollout defaults (4 update epochs of 4 minibatches per rollout),
    shared by the single- and multi-agent configs."""
    rollout: RolloutConfig = Field(default_factory=lambda: RolloutConfig(update_epochs=4,
                                                                         n_minibatches=4))


class PPOTrainerConfig(PPODefaults, SingleAgentRLTrainerConfig):
    """SingleAgentRLTrainerConfig for PPOModel: `model` is a PPOConfig, so the
    loss settings are `model.clip_coef`, `model.ent_coef`, ... A plain
    DragonRLConfig is accepted and becomes a PPOConfig with default loss settings."""
    model: PPOConfig

    @field_validator('model', mode='before')
    @classmethod
    def _ppo_model(cls, v: Any) -> Any:
        return upgrade_model_config(v, PPOConfig)


class MultiAgentPPOTrainerConfig(PPODefaults, MultiAgentRLTrainerConfig):
    """MultiAgentRLTrainerConfig for MultiAgentPPOModel: `model` is a
    MultiAgentPPOConfig, with the same loss settings as PPOTrainerConfig
    (`model.clip_coef`, ...), applied per agent group. A plain DragonRLConfig,
    or a single-agent PPOConfig (loss settings kept), is converted."""
    model: MultiAgentPPOConfig

    @field_validator('model', mode='before')
    @classmethod
    def _multi_agent_ppo_model(cls, v: Any) -> Any:
        return upgrade_model_config(v, MultiAgentPPOConfig)
