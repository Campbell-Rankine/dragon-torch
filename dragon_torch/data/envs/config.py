import functools
import inspect
import warnings
from typing import Any, Callable, Literal, Optional, Union
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv, VectorEnv
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .registry import WrapperSpec
from .wrappers import RecordRawReward

type EnvFn = Callable[..., gym.Env]
type WrapperFn = Callable[[gym.Env], gym.Env]


class WrapperNormalizationWarning(UserWarning):
    """A normalising gymnasium wrapper is in DragonEnvConfig.wrappers. Silence
    it with warnings.filterwarnings('ignore', category=WrapperNormalizationWarning)
    if per-env normalisation is really what you want."""


# normalising wrapper -> the framework setting to use instead
_NORMALIZERS: dict[type[gym.Wrapper], str] = {
    gym.wrappers.NormalizeObservation: 'rollout.normalize_obs',
    gym.wrappers.NormalizeReward: 'rollout.normalize_reward',
}


def _wrapper_class(wrapper: Any) -> Optional[type]:
    """The class a wrappers entry builds, when it can be told without calling
    it: a spec's factory, a class, or functools.partial of one. Lambdas and
    other functions are opaque (None)."""
    target = wrapper.factory if isinstance(wrapper, WrapperSpec) else wrapper
    while isinstance(target, functools.partial):
        target = target.func
    return target if inspect.isclass(target) else None


def normalizing_wrappers(wrappers: list[Any]) -> list[tuple[str, str]]:
    """(wrapper class name, framework setting to use instead) for each entry
    that normalises observations or rewards."""
    found = []
    for wrapper in wrappers:
        cls = _wrapper_class(wrapper)
        for normalizer, setting in _NORMALIZERS.items():
            if cls is not None and issubclass(cls, normalizer):
                found.append((cls.__name__, setting))
    return found


class DragonEnvConfig(BaseModel):
    """How to build a (vectorised) env from a registered id or any callable.

    Wrappers are applied in order to each sub-env. Each is a WrapperSpec (a
    name + kwargs, which saves to JSON and can be tuned), its str shorthand, or
    any callable `env -> env`:
        [WrapperSpec(name='Discretizer', kwargs={'combos': COMBOS}),
         'FlattenObservation',
         lambda e: MyWrapper(e)]
    Configs whose env and wrappers are all data (`portable`) can be rebuilt
    exactly from their JSON; callables are saved only as their repr.
    Use vectorization='async' for envs that can't share a process, e.g.
    stable-retro (one emulator per process) or slow simulators.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    env_id: Optional[str] = None
    env_fn: Optional[EnvFn] = None
    env_kwargs: dict[str, Any] = {}
    wrappers: list[Union[WrapperSpec, WrapperFn]] = []
    n_envs: int = Field(default=1, ge=1)
    seed: int = 0
    vectorization: Literal['sync', 'async'] = 'sync'

    @model_validator(mode='after')
    def _one_source(self) -> "DragonEnvConfig":
        if (self.env_id is None) == (self.env_fn is None):
            raise ValueError("Set exactly one of env_id or env_fn")
        return self

    @model_validator(mode='after')
    def _warn_normalizing_wrappers(self) -> "DragonEnvConfig":
        # each sub-env (and the separate eval env) would keep its own running
        # stats, never checkpointed and never applied to info['final_obs'];
        # the collector's normalisers avoid all of that (see docs/envs.md)
        for name, setting in normalizing_wrappers(self.wrappers):
            warnings.warn(
                f"{name} in env.wrappers keeps separate, uncheckpointed stats in every sub-env "
                f"and in the eval env; use {setting}=True instead (docs/envs.md#normalisation)",
                WrapperNormalizationWarning, stacklevel=2)
        return self

    @property
    def shaped(self) -> bool:
        """Whether wrappers sit between the env and the agent, so the reward
        trained on may differ from the env's own (raw) reward."""
        return bool(self.wrappers)

    @property
    def portable(self) -> bool:
        """True if the config is pure data: rebuildable from its JSON alone."""
        return self.env_fn is None and all(isinstance(w, WrapperSpec) for w in self.wrappers)


def make_env_fn(cfg: DragonEnvConfig) -> Callable[[], gym.Env]:
    """A picklable thunk building one wrapped sub-env:
    base env -> RecordRawReward -> cfg.wrappers (in order)."""
    def _make() -> gym.Env:
        env = gym.make(cfg.env_id, **cfg.env_kwargs) if cfg.env_id is not None \
            else cfg.env_fn(**cfg.env_kwargs)
        env = RecordRawReward(env)
        for wrapper in cfg.wrappers:
            env = wrapper(env)
        return env
    return _make


def make_vec_env(cfg: DragonEnvConfig) -> VectorEnv:
    """Vectorised env using SAME_STEP autoreset.

    Gymnasium's default (NEXT_STEP) spends the step after an episode ends on a
    reset that ignores the action and returns reward 0; buffers would record it
    as a real transition. SAME_STEP returns the reset obs immediately and puts
    the true last obs in info['final_obs'], which EnvCollector substitutes.
    """
    cls = SyncVectorEnv if cfg.vectorization == 'sync' else AsyncVectorEnv
    return cls([make_env_fn(cfg) for _ in range(cfg.n_envs)],
               autoreset_mode=AutoresetMode.SAME_STEP)


def validate_env(env: Union[gym.Env, DragonEnvConfig], unwrapped: bool = True) -> None:
    """Run gymnasium's checker (API, spaces vs. returned values, seeding) on one
    sub-env. Raises on hard failures; warns on soft ones. By default checks the
    raw env (what gymnasium recommends); unwrapped=False checks it with your
    wrappers applied, e.g. to validate a custom wrapper's spaces."""
    from gymnasium.utils.env_checker import check_env
    built = make_env_fn(env)() if isinstance(env, DragonEnvConfig) else env
    try:
        with warnings.catch_warnings():
            if not unwrapped:  # checking the wrapped env is the point; skip gymnasium's nudge
                warnings.filterwarnings('ignore', message='.*different from the unwrapped version')
            # the close check rebuilds the env from env.spec, which fails for any
            # wrapper that doesn't record its constructor args; the framework
            # rebuilds envs from DragonEnvConfig instead, so skip it for wrapped envs
            check_env(built.unwrapped if unwrapped else built, skip_render_check=True,
                      skip_close_check=not unwrapped)
    finally:
        if built is not env:
            built.close()


class EnvCheckError(ValueError):
    """An env config failed gymnasium's env checker (see check_env_config)."""


def check_env_config(cfg: DragonEnvConfig, name: str = 'env') -> None:
    """Check one sub-env built from `cfg` with its wrappers applied, as the
    agent will see it. On failure, also checks the base env to say where the
    problem is, and raises EnvCheckError naming `name` (the config field)."""
    try:
        validate_env(cfg, unwrapped=False)
    except Exception as wrapped_error:
        where = ''
        if cfg.wrappers:
            try:
                validate_env(cfg, unwrapped=True)
                where = '; the base env passes, so the problem is in its wrappers'
            except Exception:
                where = '; the base env fails too'
        raise EnvCheckError(f"{name} failed gymnasium's env checker{where}: "
                            f"{type(wrapped_error).__name__}: {wrapped_error}") from wrapped_error


def register_dragon_env(env_id: str, entry_point: Union[str, EnvFn], **kwargs: Any) -> None:
    """Register an env so `DragonEnvConfig(env_id=...)` / `gym.make` can build it.
    Re-registering the same id replaces it (handy in notebooks)."""
    gym.envs.registration.registry.pop(env_id, None)
    gym.register(id=env_id, entry_point=entry_point, **kwargs)
