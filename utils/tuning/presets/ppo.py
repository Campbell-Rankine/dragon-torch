"""PPO search space and trial builder for utils.tuning.

    space = PPOSpace(fixed={'gamma': 0.99})
    build = PPOBuilder(env=DragonEnvConfig(env_id='CartPole-v1', n_envs=8),
                       total_timesteps=100_000, eval_every_steps=10_000)
    run_study(rl_tune_config('ppo-cartpole', ...), space.suggest, build)

Set eval_every_steps >= the largest n_steps x n_envs in the space, so every
trial evaluates at every scheduled step (see StepEvalCallback).
"""
from pathlib import Path
from typing import Any, Optional
import optuna
import torch
from pydantic import BaseModel, ConfigDict, Field
from data.envs import DragonEnvConfig
from data.RL import EpisodeSource, RolloutSource
from loss.RL import PPOLoss
from models.RL import DragonRLConfig, DragonRLModel
from modules.RL import ActorCritic
from ..objective import TrialSpec
from ..RL import DivergenceGuard, StepEvalCallback

type Range = tuple[float, float]


class PPOSpace(BaseModel):
    """PPO hyperparameter ranges. Narrow any field, or pin a parameter with
    `fixed={'name': value}` (it is then not sampled and doesn't appear in
    trial.params). Categorical choices are powers of two so n_steps * n_envs
    always splits evenly into n_minibatches.
    """
    lr: Range = (1e-4, 3e-3)                          # log-uniform
    n_steps: list[int] = [64, 128, 256, 512]
    n_minibatches: list[int] = [1, 2, 4, 8]
    update_epochs: tuple[int, int] = (3, 10)
    gamma: list[float] = [0.95, 0.98, 0.99, 0.995, 0.999]
    gae_lambda: list[float] = [0.9, 0.95, 0.98, 1.0]
    clip_coef: list[float] = [0.1, 0.2, 0.3]
    ent_coef: Range = (1e-6, 1e-1)                    # log-uniform
    vf_coef: Range = (0.25, 1.0)
    max_grad_norm: list[float] = [0.3, 0.5, 1.0, 2.0]
    hidden: list[str] = ['64x64', '128x128', '256x256']
    normalize_obs: list[bool] = [False, True]
    fixed: dict[str, Any] = {}

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        fixed = self.fixed
        p: dict[str, Any] = dict(fixed)

        def pick(name, fn, *args, **kw):
            if name not in fixed:
                p[name] = fn(name, *args, **kw)

        pick('lr', trial.suggest_float, *self.lr, log=True)
        pick('n_steps', trial.suggest_categorical, self.n_steps)
        pick('n_minibatches', trial.suggest_categorical, self.n_minibatches)
        pick('update_epochs', trial.suggest_int, *self.update_epochs)
        pick('gamma', trial.suggest_categorical, self.gamma)
        pick('gae_lambda', trial.suggest_categorical, self.gae_lambda)
        pick('clip_coef', trial.suggest_categorical, self.clip_coef)
        pick('ent_coef', trial.suggest_float, *self.ent_coef, log=True)
        pick('vf_coef', trial.suggest_float, *self.vf_coef)
        pick('max_grad_norm', trial.suggest_categorical, self.max_grad_norm)
        pick('hidden', trial.suggest_categorical, self.hidden)
        pick('normalize_obs', trial.suggest_categorical, self.normalize_obs)
        return p


def parse_hidden(hidden: Any) -> tuple[int, ...]:
    """'64x64' -> (64, 64); sequences pass through."""
    if isinstance(hidden, str):
        return tuple(int(h) for h in hidden.lower().split('x') if h)
    return tuple(hidden)


class PPOBuilder(BaseModel):
    """build(params, trial_dir) for DragonObjective: a DragonRLModel + PPOLoss on
    `env`, trained for `total_timesteps`, evaluated every `eval_every_steps` on
    `eval_env` (default: `env` with one sub-env and seed + 10_000, so evaluation
    episodes never match training ones).

    Every trial uses the same seeds, so differences between trials come from the
    hyperparameters rather than luck; raise `eval_episodes` / `smooth_window` if
    the objective is still noisy.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: DragonEnvConfig
    total_timesteps: int = Field(ge=1)
    eval_every_steps: int = Field(ge=1)
    eval_env: Optional[DragonEnvConfig] = None
    eval_episodes: int = Field(default=10, ge=1)
    smooth_window: int = Field(default=3, ge=1)
    device: str = 'cpu'
    save_every: int = Field(default=1_000_000, ge=1)  # epochs; the final checkpoint is always saved
    # DivergenceGuard settings; max_kl=None disables the KL check
    max_kl: Optional[float] = 0.5
    kl_patience: int = 2
    min_entropy: Optional[float] = None

    def eval_env_config(self) -> DragonEnvConfig:
        if self.eval_env is not None:
            return self.eval_env
        return self.env.model_copy(update={'n_envs': 1, 'seed': self.env.seed + 10_000})

    def __call__(self, params: dict[str, Any], trial_dir: Path) -> TrialSpec:
        torch.manual_seed(self.env.seed)
        source = RolloutSource(self.env, n_steps=params['n_steps'], gamma=params['gamma'],
                               gae_lambda=params['gae_lambda'],
                               update_epochs=params['update_epochs'],
                               n_minibatches=params['n_minibatches'],
                               normalize_obs=params['normalize_obs'])
        evaluator = EpisodeSource(self.eval_env_config(), n_episodes=self.eval_episodes)
        try:
            col = source.collector
            module = ActorCritic(col.observation_space, col.action_space,
                                 hidden=parse_hidden(params['hidden']))
            loss = PPOLoss(clip_coef=params['clip_coef'], vf_coef=params['vf_coef'],
                           ent_coef=params['ent_coef'])
            cfg = DragonRLConfig(checkpoint_dir=trial_dir / 'ckpt', save_every=self.save_every,
                                 total_timesteps=self.total_timesteps,
                                 max_grad_norm=params['max_grad_norm'], device=self.device)
            model = DragonRLModel(module, loss, cfg)
        except BaseException:
            source.close()
            evaluator.close()
            raise
        lr = params['lr']
        return TrialSpec(
            model=model,
            optimizer=lambda ps: torch.optim.Adam(ps, lr=lr, eps=1e-5),
            train_data=source,
            callbacks=[
                StepEvalCallback(evaluator, self.eval_every_steps, self.total_timesteps,
                                 smooth_window=self.smooth_window),
                DivergenceGuard(max_kl=self.max_kl, patience=self.kl_patience,
                                min_entropy=self.min_entropy),
            ])
