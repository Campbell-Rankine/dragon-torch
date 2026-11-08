from abc import ABC, abstractmethod
from typing import ClassVar, Literal, Optional, Sequence
import torch
from dragon_torch.data.RL import EpisodeSource, RolloutSource
from dragon_torch.loss.RL import RLLoss
from dragon_torch.models.base import DataLike, EpochCallback
from dragon_torch.models.RL import DragonRLModel
from dragon_torch.modules.RL import ActorCritic, DragonPolicyModule
from dragon_torch.trainers.base import DragonTrainer, LRSchedule, OptimizerFactory, linear_schedule
from .callbacks import DivergenceGuard, StepEvalCallback
from .config import DragonRLTrainerConfig

type Direction = Literal['minimize', 'maximize']


class DragonRLTrainer(DragonTrainer, ABC):
    """A DragonTrainer that builds a whole on-policy RL run from one config:
    rollout source, evaluator, policy network, loss, model, optimizer, lr
    annealing and the default callbacks (StepEvalCallback, DivergenceGuard).

        with PPOTrainer(cfg) as trainer:
            history = trainer.fit()          # trains on its own rollout source
            score = trainer.evaluate()       # mean return on its own evaluator

    Subclasses implement `build_loss` (the algorithm); every other build_* hook
    can be overridden, e.g. build_module for a custom network (or pass `module=`).
    `objectives` is the default tuning objective, read by
    dragon_torch.utils.tuning.RL.rl_tune_config; the history holds every other metric too.
    """
    config: DragonRLTrainerConfig
    objectives: ClassVar[dict[str, Direction]] = {'val_return_smooth': 'maximize'}
    EVAL_STATE_KEY: ClassVar[str] = 'step_eval'

    def __init__(self, cfg: DragonRLTrainerConfig, module: Optional[DragonPolicyModule] = None):
        self.config = cfg
        self.source: Optional[RolloutSource] = None
        self.evaluator: Optional[EpisodeSource] = None
        self.eval_callback: Optional[StepEvalCallback] = None
        self.guard: Optional[DivergenceGuard] = None
        try:
            torch.manual_seed(cfg.seed)
            self.source = self.build_source()
            self.evaluator = self.build_evaluator()
            module = module if module is not None else self.build_module()
            model = self.build_model(module, self.build_loss())
            super().__init__(model, self.build_optimizer(), cfg, lr_schedule=self.build_lr_schedule())
            self.eval_callback = self.build_eval_callback()
            if self.eval_callback is not None:
                # checkpoint the smoothing window / schedule position with the model
                self.model.register_state(self.EVAL_STATE_KEY, self.eval_callback)
            self.guard = self.build_guard()
        except BaseException:
            self.close()
            raise

    # ------- build hooks (called in this order) ------- #
    def build_source(self) -> RolloutSource:
        r = self.config.rollout
        return RolloutSource(self.config.env, n_steps=r.n_steps, gamma=r.gamma,
                             gae_lambda=r.gae_lambda, update_epochs=r.update_epochs,
                             n_minibatches=r.n_minibatches, normalize_obs=r.normalize_obs,
                             normalize_reward=r.normalize_reward)

    def build_evaluator(self) -> Optional[EpisodeSource]:
        env = self.config.eval_env_config()
        if env is None:
            return None
        e = self.config.eval
        return EpisodeSource(env, n_episodes=e.n_episodes, deterministic=e.deterministic,
                             max_steps=e.max_steps)

    def build_module(self) -> DragonPolicyModule:
        col, p = self.source.collector, self.config.policy
        return ActorCritic(col.observation_space, col.action_space, hidden=p.hidden,
                           activation=p.activation_cls)

    @abstractmethod
    def build_loss(self) -> RLLoss:
        """The algorithm: e.g. PPOLoss or A2CLoss built from the config."""

    def build_model(self, module: DragonPolicyModule, loss: RLLoss) -> DragonRLModel:
        return DragonRLModel(module, loss, self.config.model)

    def build_optimizer(self) -> OptimizerFactory:
        return self.config.optim.factory()

    def build_lr_schedule(self) -> Optional[LRSchedule]:
        o = self.config.optim
        if not o.anneal_lr:
            return None
        return linear_schedule(self.config.model.total_timesteps, key='global_step',
                               final_frac=o.final_lr_frac)

    def build_eval_callback(self) -> Optional[StepEvalCallback]:
        if self.evaluator is None:
            return None
        e = self.config.eval
        return StepEvalCallback(self.evaluator, e.every_steps, self.config.model.total_timesteps,
                                metric=self.model.eval_metric, smooth_window=e.smooth_window)

    def build_guard(self) -> Optional[DivergenceGuard]:
        g = self.config.guard
        if g is None:
            return None
        return DivergenceGuard(max_kl=g.max_kl, patience=g.kl_patience,
                               min_entropy=g.min_entropy, entropy_after_steps=g.entropy_after_steps)

    def default_callbacks(self) -> list[EpochCallback]:
        """Run before caller callbacks, so those (e.g. a tuner) see val_return_smooth."""
        return [cb for cb in (self.eval_callback, self.guard) if cb is not None]

    # ------- running ------- #
    def fit(self, train_data: Optional[DataLike] = None, val_data: Optional[DataLike] = None,
            epochs: Optional[int] = None,
            callbacks: Sequence[EpochCallback] = ()) -> list[dict[str, float]]:
        """Train on the trainer's rollout source (or `train_data`) until
        model.total_timesteps, or for `epochs` epochs. Prefer the eval config
        over `val_data`, which runs a full evaluation every epoch."""
        return super().fit(train_data if train_data is not None else self.source, val_data,
                           epochs=epochs, callbacks=[*self.default_callbacks(), *callbacks])

    def evaluate(self, data: Optional[DataLike] = None) -> float:
        data = data if data is not None else self.evaluator
        if data is None:
            raise ValueError("no evaluator configured (cfg.eval is None); pass data")
        return super().evaluate(data)

    def close(self) -> None:
        for obj in (self.source, self.evaluator):
            if obj is not None:
                obj.close()

    def __enter__(self) -> "DragonRLTrainer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
