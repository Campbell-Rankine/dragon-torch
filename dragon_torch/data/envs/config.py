from typing import Any, Callable, Literal, Optional, Union
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv, VectorEnv
from pydantic import BaseModel, ConfigDict, Field, model_validator

type EnvFn = Callable[..., gym.Env]
type WrapperFn = Callable[[gym.Env], gym.Env]


class DragonEnvConfig(BaseModel):
    """How to build a (vectorised) env from a registered id or any callable.

    Wrappers are applied in order to each sub-env, e.g.
    `[gym.wrappers.FlattenObservation, lambda e: Discretizer(e, COMBOS)]`.
    Use vectorization='async' for envs that can't share a process, e.g.
    stable-retro (one emulator per process) or slow simulators.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    env_id: Optional[str] = None
    env_fn: Optional[EnvFn] = None
    env_kwargs: dict[str, Any] = {}
    wrappers: list[WrapperFn] = []
    n_envs: int = Field(default=1, ge=1)
    seed: int = 0
    vectorization: Literal['sync', 'async'] = 'sync'

    @model_validator(mode='after')
    def _one_source(self) -> "DragonEnvConfig":
        if (self.env_id is None) == (self.env_fn is None):
            raise ValueError("Set exactly one of env_id or env_fn")
        return self


def make_env_fn(cfg: DragonEnvConfig) -> Callable[[], gym.Env]:
    """A picklable thunk building one wrapped sub-env."""
    def _make() -> gym.Env:
        env = gym.make(cfg.env_id, **cfg.env_kwargs) if cfg.env_id is not None \
            else cfg.env_fn(**cfg.env_kwargs)
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
        check_env(built.unwrapped if unwrapped else built, skip_render_check=True)
    finally:
        if built is not env:
            built.close()


def register_dragon_env(env_id: str, entry_point: Union[str, EnvFn], **kwargs: Any) -> None:
    """Register an env so `DragonEnvConfig(env_id=...)` / `gym.make` can build it.
    Re-registering the same id replaces it (handy in notebooks)."""
    gym.envs.registration.registry.pop(env_id, None)
    gym.register(id=env_id, entry_point=entry_point, **kwargs)
