from dragon_torch.models.RL import MultiAgentPPOModel, PPOModel
from dragon_torch.trainers.RL.trainer import MultiAgentRLTrainer, SingleAgentRLTrainer
from .config import MultiAgentPPOTrainerConfig, PPOTrainerConfig


class PPOTrainer(SingleAgentRLTrainer):
    """PPO on a gymnasium env: trains a PPOModel, whose config (PPOConfig) holds
    the loss settings."""
    config: PPOTrainerConfig
    config_cls = PPOTrainerConfig
    model_cls = PPOModel


class MultiAgentPPOTrainer(MultiAgentRLTrainer):
    """PPO on a multi-agent env: trains a MultiAgentPPOModel (one network per
    agent group, PPOLoss per group), whose config (MultiAgentPPOConfig) holds
    the loss settings."""
    config: MultiAgentPPOTrainerConfig
    config_cls = MultiAgentPPOTrainerConfig
    model_cls = MultiAgentPPOModel
