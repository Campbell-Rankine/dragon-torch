"""Configs for DragonRLTrainer: one config describes a whole RL run.

    DragonRLTrainerConfig               sections shared by every RL run
      SingleAgentRLTrainerConfig        env: DragonEnvConfig (gymnasium)
      MultiAgentRLTrainerConfig         env: MultiAgentEnvConfig (ParallelEnv /
                                        gymnasium Dict-space multi-agent env)

      env      env config           what to train on
      model    DragonRLConfig       checkpoints, total_timesteps, grad clipping, device
                                    (algorithm configs narrow it, e.g. PPOConfig
                                    carries the PPO loss settings)
      rollout  RolloutConfig        RolloutSource settings (GAE, minibatching, normalisation)
      eval     EvalConfig | None    step-scheduled evaluation (StepEvalCallback)
      policy   PolicyConfig         default ActorCritic network (one per agent group
                                    for multi-agent runs)
      optim    OptimConfig          optimizer + linear lr annealing
      guard    GuardConfig | None   DivergenceGuard
      seed                          torch seed for network init

Algorithm configs exist for both kinds and take the algorithm's model config,
which holds the loss settings: PPOTrainerConfig -> PPOConfig,
MultiAgentPPOTrainerConfig -> MultiAgentPPOConfig, and likewise for A2C. The
loss settings are `model.clip_coef`, ... in every one of them.
`apply_overrides(cfg, {'rollout.n_steps': 256, 'optim.lr': 3e-4})` sets
nested fields by dotted path, which is how tuning parameters map onto a config.
"""
import warnings
from typing import Any, Literal, Mapping, Optional, TypeVar
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from dragon_torch.data.envs import DragonEnvConfig, MultiAgentEnvConfig
from dragon_torch.models.RL import DragonRLConfig, MultiAgentRLConfig
from dragon_torch.trainers.base import DragonTrainerConfig, OptimizerFactory

ACTIVATIONS: dict[str, type[torch.nn.Module]] = {
    'tanh': torch.nn.Tanh, 'relu': torch.nn.ReLU, 'gelu': torch.nn.GELU, 'elu': torch.nn.ELU,
}


class RolloutConfig(BaseModel):
    """RolloutSource settings. Each epoch collects n_steps x n_envs transitions,
    then makes update_epochs passes of n_minibatches minibatches over them."""
    n_steps: int = Field(default=128, ge=1)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    update_epochs: int = Field(default=4, ge=1)
    n_minibatches: int = Field(default=4, ge=1)
    normalize_obs: bool = False
    normalize_reward: bool = False


class EvalConfig(BaseModel):
    """Step-scheduled evaluation on a separate env. `env` defaults to the
    training env with `n_envs` sub-envs and seed + seed_offset, so evaluation
    episodes never replay training ones."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    every_steps: int = Field(ge=1)
    n_episodes: int = Field(default=10, ge=1)
    smooth_window: int = Field(default=3, ge=1)
    deterministic: bool = True
    env: Optional[DragonEnvConfig] = None
    n_envs: int = Field(default=1, ge=1)
    seed_offset: int = 10_000
    max_steps: int = Field(default=1_000_000, ge=1)  # vector steps per evaluation


class PolicyConfig(BaseModel):
    """Default ActorCritic network. `hidden` also accepts '64x64' strings (as
    tuning search spaces use)."""
    hidden: tuple[int, ...] = (64, 64)
    activation: Literal['tanh', 'relu', 'gelu', 'elu'] = 'tanh'

    @field_validator('hidden', mode='before')
    @classmethod
    def _parse_hidden(cls, v: Any) -> Any:
        if isinstance(v, str):
            return tuple(int(h) for h in v.lower().split('x') if h.strip())
        return v

    @property
    def activation_cls(self) -> type[torch.nn.Module]:
        return ACTIVATIONS[self.activation]


class OptimConfig(BaseModel):
    """Optimizer, plus optional linear lr annealing from lr to lr * final_lr_frac
    over model.total_timesteps (the usual PPO / A2C schedule)."""
    name: Literal['adam', 'adamw', 'rmsprop', 'sgd'] = 'adam'
    lr: float = Field(default=3e-4, gt=0)
    eps: float = Field(default=1e-5, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    alpha: float = Field(default=0.99, gt=0, lt=1)   # rmsprop smoothing
    momentum: float = Field(default=0.0, ge=0)       # sgd / rmsprop
    anneal_lr: bool = False
    final_lr_frac: float = Field(default=0.0, ge=0, le=1)

    def factory(self) -> OptimizerFactory:
        name, lr, eps, wd = self.name, self.lr, self.eps, self.weight_decay
        alpha, momentum = self.alpha, self.momentum

        def build(params):
            if name == 'adam':
                return torch.optim.Adam(params, lr=lr, eps=eps, weight_decay=wd)
            if name == 'adamw':
                return torch.optim.AdamW(params, lr=lr, eps=eps, weight_decay=wd)
            if name == 'rmsprop':
                return torch.optim.RMSprop(params, lr=lr, eps=eps, alpha=alpha,
                                           momentum=momentum, weight_decay=wd)
            return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
        return build


class GuardConfig(BaseModel):
    """DivergenceGuard settings; max_kl=None disables the KL check.

    approx_kl is measured before each minibatch's update, so the first minibatch
    of every rollout reads 0: with update_epochs x n_minibatches == 1 (A2C) the
    KL check can never fire. The non-finite-loss check works regardless."""
    max_kl: Optional[float] = Field(default=0.5, gt=0)
    kl_patience: int = Field(default=2, ge=1)
    min_entropy: Optional[float] = None
    entropy_after_steps: int = Field(default=0, ge=0)


class DragonRLTrainerConfig(DragonTrainerConfig):
    """Sections shared by single- and multi-agent runs. Build runs from
    SingleAgentRLTrainerConfig / MultiAgentRLTrainerConfig (or an algorithm's
    config), which fix the env type."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: Any
    model: DragonRLConfig
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    eval: Optional[EvalConfig] = None
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    guard: Optional[GuardConfig] = Field(default_factory=GuardConfig)
    seed: int = 0
    # run gymnasium's env checker on the wrapped env (and a custom eval.env)
    # before building anything; see DragonRLTrainer.check_envs
    validate_env: bool = False

    @property
    def steps_per_epoch(self) -> int:
        return self.rollout.n_steps * self.env.n_envs

    @model_validator(mode='after')
    def _check(self) -> "DragonRLTrainerConfig":
        spe, mb = self.steps_per_epoch, self.rollout.n_minibatches
        if spe % mb:
            raise ValueError(f"rollout.n_steps x env.n_envs = {spe} must be divisible by "
                             f"rollout.n_minibatches = {mb}")
        if self.optim.anneal_lr and self.model.total_timesteps is None:
            raise ValueError("optim.anneal_lr needs model.total_timesteps")
        updates = self.rollout.update_epochs * mb
        if self.guard is not None and self.guard.max_kl is not None and updates == 1:
            warnings.warn("guard.max_kl is set but rollout.update_epochs x n_minibatches == 1, so "
                          "approx_kl is always 0 and the KL check can't fire; set "
                          "guard.max_kl=None or use more minibatches / update epochs", stacklevel=2)
        if self.eval is not None and self.eval.every_steps < spe:
            warnings.warn(f"eval.every_steps={self.eval.every_steps} < steps per epoch ({spe}); "
                          "every epoch evaluates and scheduled steps get skipped", stacklevel=2)
        return self

    def eval_env_config(self) -> Any:
        if self.eval is None:
            return None
        if self.eval.env is not None:
            return self.eval.env
        return self.env.model_copy(update={'n_envs': self.eval.n_envs,
                                           'seed': self.env.seed + self.eval.seed_offset})


ModelConfigT = TypeVar('ModelConfigT', bound=DragonRLConfig)


def upgrade_model_config(value: Any, target: type[ModelConfigT]) -> Any:
    """For a config's `model` field: a DragonRLConfig of another type becomes
    a `target`, keeping the settings of every config class the two share:
    the DragonRLConfig ones (checkpoint_dir, total_timesteps, ...) always,
    and an algorithm's loss settings when both are that algorithm's (PPOConfig
    -> MultiAgentPPOConfig keeps clip_coef; A2CConfig -> PPOConfig doesn't).
    The rest take target's defaults; a `kind` left at its class default becomes
    target's default kind. Anything else (a dict, a `target` already) is
    returned for normal validation."""
    if not isinstance(value, DragonRLConfig) or isinstance(value, target):
        return value
    shared = {name for base in type(value).__mro__
              if isinstance(base, type) and issubclass(base, BaseModel) and issubclass(target, base)
              for name in base.model_fields}
    data = value.model_dump(include=shared)
    if data.get('kind') == type(value).model_fields['kind'].default:
        data.pop('kind')
    return target(**data)


class SingleAgentRLTrainerConfig(DragonRLTrainerConfig):
    """A run on a (vectorised) gymnasium env."""
    env: DragonEnvConfig


class MultiAgentRLTrainerConfig(DragonRLTrainerConfig):
    """A run on a multi-agent env. Agent groups, and how the team return is
    scored, are part of the env config (MultiAgentEnvConfig.groups /
    team_return). `model` accepts a plain DragonRLConfig too; it becomes a
    MultiAgentRLConfig (kind 'marl-model')."""
    env: MultiAgentEnvConfig
    model: MultiAgentRLConfig

    @field_validator('model', mode='before')
    @classmethod
    def _multi_agent_model(cls, v: Any) -> Any:
        return upgrade_model_config(v, MultiAgentRLConfig)


ConfigT = TypeVar('ConfigT', bound=BaseModel)


def apply_overrides(cfg: ConfigT, params: Mapping[str, Any]) -> ConfigT:
    """A copy of `cfg` with dotted-path values replaced. Paths walk config
    sections, list items by index and dict entries by key, e.g.
        {'rollout.n_steps': 256, 'env.wrappers.0.kwargs.cost': 0.05}
    Every config section on a changed path is re-validated (so cross-field
    validators run). Unknown fields, out-of-range indices and missing
    intermediate dict keys raise KeyError; a dict's last key may be new (so a
    wrapper kwarg with a default can be tuned). Nothing is modified in place."""
    if not isinstance(cfg, BaseModel):
        raise TypeError(f"apply_overrides needs a pydantic config, got {type(cfg).__name__}")
    return _override(cfg, dict(params), '')


def _split(params: dict[str, Any], prefix: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Group dotted paths by their first part: (whole replacements, nested)."""
    direct: dict[str, Any] = {}
    nested: dict[str, dict[str, Any]] = {}
    for key, value in params.items():
        head, _, rest = key.partition('.')
        if rest:
            nested.setdefault(head, {})[rest] = value
        else:
            direct[head] = value
    for head in direct.keys() & nested.keys():
        raise ValueError(f"'{prefix}{head}' is overridden both whole and by field")
    return direct, nested


def _override(obj: Any, params: dict[str, Any], prefix: str) -> Any:
    direct, nested = _split(params, prefix)

    if isinstance(obj, BaseModel):
        fields = type(obj).model_fields
        for head in (*direct, *nested):
            if head not in fields:
                raise KeyError(f"unknown config key '{prefix}{head}' for {type(obj).__name__} "
                               f"(fields: {sorted(fields)})")
        updates = dict(direct)
        for head, sub in nested.items():
            updates[head] = _override(getattr(obj, head), sub, f"{prefix}{head}.")
        return type(obj).model_validate({**dict(obj), **updates})

    if isinstance(obj, (list, tuple)):
        items = list(obj)
        for head, value in (*direct.items(), *((h, None) for h in nested)):
            if not head.lstrip('-').isdigit() or not -len(items) <= int(head) < len(items):
                raise KeyError(f"'{prefix}{head}': no item {head!r} in a list of {len(items)}")
        for head, value in direct.items():
            items[int(head)] = value
        for head, sub in nested.items():
            items[int(head)] = _override(items[int(head)], sub, f"{prefix}{head}.")
        return type(obj)(items)

    if isinstance(obj, dict):
        out = {**obj, **direct}
        for head, sub in nested.items():
            if head not in obj:
                raise KeyError(f"'{prefix}{head}' not found (keys: {sorted(obj)})")
            out[head] = _override(obj[head], sub, f"{prefix}{head}.")
        return out

    head = next(iter(params))
    where = prefix.rstrip('.') or 'config'
    raise KeyError(f"can't set '{prefix}{head}': '{where}' is {type(obj).__name__}, "
                   "not a config section, list or dict")
