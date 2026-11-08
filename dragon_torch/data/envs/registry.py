"""Wrappers described as data, so env configs round-trip through JSON and
wrapper settings can be tuned.

    @register_wrapper('fuel_penalty')
    class FuelPenalty(gym.RewardWrapper):
        def __init__(self, env, cost: float = 0.01): ...

    DragonEnvConfig(env_id='CartPole-v1', wrappers=[
        WrapperSpec(name='fuel_penalty', kwargs={'cost': 0.05}),
        'FlattenObservation',                       # str shorthand for WrapperSpec(name=...)
        {'name': 'mypkg.envs:ClipObs', 'kwargs': {'low': -5.0}},
    ])

A spec's `name` resolves, in order, to:
  1. a wrapper registered with `register_wrapper`
  2. a gymnasium wrapper class, e.g. 'FlattenObservation', 'ClipReward'
  3. an import path 'package.module:attr', imported on demand, so a saved
     config can be rebuilt without first importing the module that registers it
"""
import importlib
import inspect
from typing import Any, Callable, Optional, TypeVar, Union, overload
import gymnasium as gym
from pydantic import BaseModel, model_validator

type WrapperFactory = Callable[..., gym.Env]  # (env, **kwargs) -> wrapped env
F = TypeVar('F', bound=WrapperFactory)

_REGISTRY: dict[str, WrapperFactory] = {}


@overload
def register_wrapper(name_or_factory: F) -> F: ...
@overload
def register_wrapper(name_or_factory: Optional[str] = None) -> Callable[[F], F]: ...


def register_wrapper(name_or_factory: Union[str, F, None] = None) -> Union[F, Callable[[F], F]]:
    """Register a wrapper class (or `factory(env, **kwargs)`) under a name for
    WrapperSpec. Use as `@register_wrapper` (registers the class name) or
    `@register_wrapper('name')`. Registering a name again replaces it, so
    re-running a notebook cell works."""
    def decorate(factory: F, name: Optional[str] = None) -> F:
        if not callable(factory):
            raise TypeError(f"can only register callables, got {type(factory)}")
        name = name or factory.__name__
        if ':' in name:
            raise ValueError(f"wrapper name {name!r} can't contain ':' (reserved for import paths)")
        _REGISTRY[name] = factory
        return factory

    if callable(name_or_factory):
        return decorate(name_or_factory)
    return lambda factory: decorate(factory, name_or_factory)


def unregister_wrapper(name: str) -> None:
    _REGISTRY.pop(name, None)


def registered_wrappers() -> dict[str, WrapperFactory]:
    return dict(_REGISTRY)


def resolve_wrapper(name: str) -> WrapperFactory:
    """The wrapper factory a WrapperSpec name refers to (see module docstring)."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    if ':' in name:
        module, _, attr = name.partition(':')
        try:
            obj = importlib.import_module(module)
            for part in attr.split('.'):
                obj = getattr(obj, part)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"can't import wrapper {name!r}: {e}") from e
        if not callable(obj):
            raise ValueError(f"wrapper {name!r} is not callable")
        return obj
    gym_wrapper = getattr(gym.wrappers, name, None)
    if inspect.isclass(gym_wrapper) and issubclass(gym_wrapper, gym.Wrapper):
        return gym_wrapper
    raise ValueError(f"unknown wrapper {name!r}: not registered ({sorted(_REGISTRY)}), not a "
                     "gymnasium.wrappers class, and not a 'module:attr' import path")


class WrapperSpec(BaseModel):
    """A wrapper as data: `name` (see module docstring) plus constructor `kwargs`.
    Calling a spec wraps an env, so specs and plain callables mix in
    DragonEnvConfig.wrappers. kwargs are checked against the wrapper's
    signature when the spec is created, so typos fail early; keep them
    JSON-friendly (numbers, strings, lists, dicts) so configs save cleanly."""
    name: str
    kwargs: dict[str, Any] = {}

    @model_validator(mode='before')
    @classmethod
    def _from_str(cls, data: Any) -> Any:
        return {'name': data} if isinstance(data, str) else data

    @model_validator(mode='after')
    def _check(self) -> "WrapperSpec":
        factory = resolve_wrapper(self.name)
        try:
            sig = inspect.signature(factory)
        except (TypeError, ValueError):  # builtins / C callables: can't check
            return self
        try:
            sig.bind(None, **self.kwargs)  # None stands in for the env
        except TypeError as e:
            raise ValueError(f"bad kwargs for wrapper {self.name!r} {sig}: {e}") from e
        return self

    @property
    def factory(self) -> WrapperFactory:
        return resolve_wrapper(self.name)

    def __call__(self, env: gym.Env) -> gym.Env:
        return self.factory(env, **self.kwargs)
