"""PPO search space, for PPOModel and MultiAgentPPOModel. Parameter names are
PPOTrainerConfig / MultiAgentPPOTrainerConfig paths: the loss settings are
model-config fields under `model.`.

    from dragon_torch.trainers.RL import PPOTrainer, PPOTrainerConfig
    from dragon_torch.utils.tuning.RL import RLTrialBuilder, rl_tune_config
    space = ppo_space().fix({'rollout.gamma': 0.99})     # or space_for(PPOModel)
    build = RLTrialBuilder(trainer_cls=PPOTrainer, base_config=PPOTrainerConfig(...))
    run_study(rl_tune_config('ppo-cartpole', PPOTrainer, ...), space.suggest, build)

n_steps / n_minibatches choices are powers of two, so n_steps x n_envs always
splits evenly into minibatches. Keep base_config.eval.every_steps >= the largest
n_steps x n_envs so every trial evaluates at every scheduled step.
"""
from ..space import CategoricalParam, FloatParam, IntParam, SearchSpace


def ppo_space() -> SearchSpace:
    return SearchSpace(params={
        'optim.lr': FloatParam(low=1e-4, high=3e-3, log=True),
        'optim.anneal_lr': CategoricalParam(choices=[False, True]),
        'rollout.n_steps': CategoricalParam(choices=[64, 128, 256, 512]),
        'rollout.n_minibatches': CategoricalParam(choices=[1, 2, 4, 8]),
        'rollout.update_epochs': IntParam(low=3, high=10),
        'rollout.gamma': CategoricalParam(choices=[0.95, 0.98, 0.99, 0.995, 0.999]),
        'rollout.gae_lambda': CategoricalParam(choices=[0.9, 0.95, 0.98, 1.0]),
        'rollout.normalize_obs': CategoricalParam(choices=[False, True]),
        'model.clip_coef': CategoricalParam(choices=[0.1, 0.2, 0.3]),
        'model.ent_coef': FloatParam(low=1e-6, high=1e-1, log=True),
        'model.vf_coef': FloatParam(low=0.25, high=1.0),
        'model.max_grad_norm': CategoricalParam(choices=[0.3, 0.5, 1.0, 2.0]),
        'policy.hidden': CategoricalParam(choices=['64x64', '128x128', '256x256']),
    })
