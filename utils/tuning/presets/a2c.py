"""A2C search space. Parameter names are A2CTrainerConfig paths. A2C does one
full-batch update per (short) rollout, so there are no minibatch / epoch knobs."""
from ..space import CategoricalParam, FloatParam, SearchSpace


def a2c_space() -> SearchSpace:
    return SearchSpace(params={
        'optim.lr': FloatParam(low=1e-4, high=3e-3, log=True),
        'optim.anneal_lr': CategoricalParam(choices=[False, True]),
        'rollout.n_steps': CategoricalParam(choices=[5, 8, 16, 32]),
        'rollout.gamma': CategoricalParam(choices=[0.95, 0.98, 0.99, 0.995, 0.999]),
        'rollout.gae_lambda': CategoricalParam(choices=[0.9, 0.95, 1.0]),
        'rollout.normalize_obs': CategoricalParam(choices=[False, True]),
        'a2c.ent_coef': FloatParam(low=1e-6, high=1e-1, log=True),
        'a2c.vf_coef': FloatParam(low=0.25, high=1.0),
        'a2c.normalize_advantages': CategoricalParam(choices=[False, True]),
        'model.max_grad_norm': CategoricalParam(choices=[0.3, 0.5, 1.0, 2.0]),
        'policy.hidden': CategoricalParam(choices=['64x64', '128x128', '256x256']),
    })
