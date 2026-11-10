from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, Optional, Sequence
import torch
from dragon_torch.data.envs import check_env_config, check_multi_agent_env_config
from dragon_torch.data.RL import (EpisodeSource, MultiAgentEpisodeSource, MultiAgentRolloutSource,
                                  RolloutSource)
from dragon_torch.loss.RL import MultiAgentLoss, RLLoss
from dragon_torch.models.base import DataLike, EpochCallback
from dragon_torch.models.RL import DragonRLModel, MultiAgentRLModel, SingleAgentRLModel
from dragon_torch.modules.RL import (ActorCritic, DragonPolicyModule, MultiAgentActorCritic,
                                     MultiAgentPolicyModule, SingleAgentPolicyModule)
from dragon_torch.trainers.base import DragonTrainer, LRSchedule, OptimizerFactory, linear_schedule
from .callbacks import DivergenceGuard, StepEvalCallback
from .config import DragonRLTrainerConfig, MultiAgentRLTrainerConfig, SingleAgentRLTrainerConfig

type Direction = Literal['minimize', 'maximize']


class DragonRLTrainer(DragonTrainer, ABC):
    """A DragonTrainer that builds a whole on-policy RL run from one config:
    rollout source, evaluator, policy network, loss, model, optimizer, lr
    annealing and the default callbacks (StepEvalCallback, DivergenceGuard).

        with PPOTrainer(cfg) as trainer:
            history = trainer.fit()          # trains on its own rollout source
            score = trainer.evaluate()       # mean return on its own evaluator

    Two axes make a concrete trainer:
      the kind of env   SingleAgentRLTrainer / MultiAgentRLTrainer implement the
                        env-specific hooks (env check, source, evaluator,
                        module, model, and build_loss, which adapts the
                        algorithm's loss to the env)
      the algorithm     a model class (`model_cls`) whose config holds the loss
                        settings: PPOModel / MultiAgentPPOModel (PPOConfig /
                        MultiAgentPPOConfig), likewise for A2C. build_agent_loss
                        makes the single-agent loss from config.model
    so PPOTrainer is a SingleAgentRLTrainer with model_cls = PPOModel and
    MultiAgentPPOTrainer a MultiAgentRLTrainer with model_cls = MultiAgentPPOModel.
    Every build_* hook
    can be overridden, e.g. build_module for a custom network (or pass `module=`).
    `objectives` is the default tuning objective, read by
    dragon_torch.utils.tuning.RL.rl_tune_config; the history holds every other metric too.
    """
    config: DragonRLTrainerConfig
    # the config type this trainer accepts (checked on construction)
    config_cls: ClassVar[type[DragonRLTrainerConfig]] = DragonRLTrainerConfig
    objectives: ClassVar[dict[str, Direction]] = {'val_return_smooth': 'maximize'}
    EVAL_STATE_KEY: ClassVar[str] = 'step_eval'

    def __init__(self, cfg: DragonRLTrainerConfig, module: Optional[DragonPolicyModule] = None):
        if not isinstance(cfg, self.config_cls):
            raise TypeError(f"{type(self).__name__} needs a {self.config_cls.__name__}, "
                            f"got {type(cfg).__name__}")
        self.config = cfg
        self.source: Any = None
        self.evaluator: Optional[EpisodeSource] = None
        self.eval_callback: Optional[StepEvalCallback] = None
        self.guard: Optional[DivergenceGuard] = None
        try:
            self.check_envs()
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
    @abstractmethod
    def check_envs(self) -> None:
        """With config.validate_env, check the training env (and eval.env if
        it's set separately) before anything is built; raise EnvCheckError."""

    @abstractmethod
    def build_source(self) -> Any:
        """The training data source (collects rollouts with the bound model)."""

    @abstractmethod
    def build_evaluator(self) -> Optional[EpisodeSource]:
        """The evaluation source, or None without an eval config."""

    @abstractmethod
    def build_module(self) -> DragonPolicyModule:
        """The default policy network, built from config.policy."""

    def build_agent_loss(self) -> RLLoss:
        """The algorithm: a single-agent loss (e.g. PPOLoss), by default built
        by the model config (`config.model.build_loss()`, as PPOConfig does)."""
        build = getattr(self.config.model, 'build_loss', None)
        if build is None:
            raise NotImplementedError(
                f"{type(self.config.model).__name__} has no loss settings (build_loss); use an "
                "algorithm's model config (e.g. PPOConfig) or override build_agent_loss")
        return build()

    @abstractmethod
    def build_loss(self) -> RLLoss:
        """The loss the model trains on: build_agent_loss() adapted to the env kind."""

    @abstractmethod
    def build_model(self, module: DragonPolicyModule, loss: RLLoss) -> DragonRLModel: ...

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

    # shared by the env-kind subclasses' build_source / build_evaluator
    def _rollout_kwargs(self) -> dict[str, Any]:
        r = self.config.rollout
        return dict(n_steps=r.n_steps, gamma=r.gamma, gae_lambda=r.gae_lambda,
                    update_epochs=r.update_epochs, n_minibatches=r.n_minibatches,
                    normalize_obs=r.normalize_obs, normalize_reward=r.normalize_reward)

    def _episode_kwargs(self) -> dict[str, Any]:
        e = self.config.eval
        return dict(n_episodes=e.n_episodes, deterministic=e.deterministic, max_steps=e.max_steps)

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


class SingleAgentRLTrainer(DragonRLTrainer):
    """DragonRLTrainer on a gymnasium env: RolloutSource, EpisodeSource,
    ActorCritic and SingleAgentRLModel; the algorithm's loss is used as-is."""
    config: SingleAgentRLTrainerConfig
    config_cls = SingleAgentRLTrainerConfig
    # the model class build_model makes (algorithm trainers narrow it, e.g.
    # PPOModel); tuning presets are looked up by it (utils.tuning.presets.space_for)
    model_cls: ClassVar[type[SingleAgentRLModel]] = SingleAgentRLModel

    def check_envs(self) -> None:
        """With config.validate_env, run gymnasium's env checker on one sub-env
        of the training env as the agent sees it (wrappers applied), and on
        eval.env if it's set separately. Runs before any vector env is built,
        so a broken env or wrapper fails here with EnvCheckError, not mid-run."""
        if not self.config.validate_env:
            return
        check_env_config(self.config.env, 'env')
        if self.config.eval is not None and self.config.eval.env is not None:
            check_env_config(self.config.eval.env, 'eval.env')

    def build_source(self) -> RolloutSource:
        return RolloutSource(self.config.env, **self._rollout_kwargs())

    def build_evaluator(self) -> Optional[EpisodeSource]:
        env = self.config.eval_env_config()
        return None if env is None else EpisodeSource(env, **self._episode_kwargs())

    def build_module(self) -> SingleAgentPolicyModule:
        col, p = self.source.collector, self.config.policy
        return ActorCritic(col.observation_space, col.action_space, hidden=p.hidden,
                           activation=p.activation_cls)

    def build_loss(self) -> RLLoss:
        return self.build_agent_loss()

    def build_model(self, module: SingleAgentPolicyModule, loss: RLLoss) -> SingleAgentRLModel:
        return self.model_cls(module=module, loss_fn=loss, cfg=self.config.model)


class MultiAgentRLTrainer(DragonRLTrainer):
    """DragonRLTrainer on a multi-agent env (a PettingZoo-style ParallelEnv or
    a gymnasium env with per-agent Dict spaces): MultiAgentRolloutSource,
    MultiAgentEpisodeSource, one ActorCritic per agent group
    (MultiAgentActorCritic) and the algorithm's loss applied per group
    (MultiAgentLoss). Agents in a group share weights; each critic sees only
    its agent's own obs (independent learners with parameter sharing, e.g. IPPO)."""
    config: MultiAgentRLTrainerConfig
    config_cls = MultiAgentRLTrainerConfig
    # the model class build_model makes (algorithm trainers narrow it, e.g.
    # MultiAgentPPOModel); tuning presets are looked up by it
    model_cls: ClassVar[type[MultiAgentRLModel]] = MultiAgentRLModel

    def check_envs(self) -> None:
        """With config.validate_env, build one env of `env` (and eval.env if
        set) with wrappers applied, group its agents and run a short random
        rollout, raising EnvCheckError on bad spaces, obs or agent keys."""
        if not self.config.validate_env:
            return
        check_multi_agent_env_config(self.config.env, 'env')
        if self.config.eval is not None and self.config.eval.env is not None:
            check_multi_agent_env_config(self.config.eval.env, 'eval.env')

    def build_source(self) -> MultiAgentRolloutSource:
        return MultiAgentRolloutSource(self.config.env, **self._rollout_kwargs())

    def build_evaluator(self) -> Optional[MultiAgentEpisodeSource]:
        env = self.config.eval_env_config()
        return None if env is None else MultiAgentEpisodeSource(env, **self._episode_kwargs())

    def build_module(self) -> MultiAgentPolicyModule:
        p = self.config.policy
        return MultiAgentActorCritic(self.source.groups.spaces_by_group(), hidden=p.hidden,
                                     activation=p.activation_cls)

    def build_loss(self) -> MultiAgentLoss:
        return MultiAgentLoss(self.build_agent_loss())

    def build_model(self, module: MultiAgentPolicyModule, loss: RLLoss) -> MultiAgentRLModel:
        return self.model_cls(module=module, loss_fn=loss, cfg=self.config.model)
