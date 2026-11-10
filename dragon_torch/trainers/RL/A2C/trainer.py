from dragon_torch.models.RL import MultiAgentA2CModel, A2CModel
from dragon_torch.trainers.RL.trainer import MultiAgentRLTrainer, SingleAgentRLTrainer
from .config import MultiAgentA2CTrainerConfig, A2CTrainerConfig


class A2CTrainer(SingleAgentRLTrainer):
    """A2C on a gymnasium env: trains a A2CModel, whose config (A2CConfig) holds
    the loss settings."""
    config: A2CTrainerConfig
    config_cls = A2CTrainerConfig
    model_cls = A2CModel


class MultiAgentA2CTrainer(MultiAgentRLTrainer):
    """A2C on a multi-agent env: trains a MultiAgentA2CModel (one network per
    agent group, A2CLoss per group), whose config (MultiAgentA2CConfig) holds
    the loss settings."""
    config: MultiAgentA2CTrainerConfig
    config_cls = MultiAgentA2CTrainerConfig
    model_cls = MultiAgentA2CModel
