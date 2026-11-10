"""Search-space presets for the RL model classes. Parameter names are paths
in the matching trainer config, with the algorithm's loss settings under
`model.`. Single- and multi-agent configs share those paths, so one preset
serves both: ppo_space for PPOModel and MultiAgentPPOModel, a2c_space for
A2CModel and MultiAgentA2CModel.

    space = space_for(PPOModel)          # or a model, or a trainer (class or instance)
    space = space_for(MultiAgentPPOTrainer)
    space = ppo_space()                  # the same, by name

register_preset(MyModel, my_space) adds one for another model class; a
subclass of a registered model uses its parent's preset unless it has its own.
"""
from typing import Any, Callable
from dragon_torch.models.RL import A2CModel, MultiAgentA2CModel, MultiAgentPPOModel, PPOModel
from ..space import SearchSpace
from .a2c import a2c_space
from .ppo import ppo_space

PRESETS: dict[type, Callable[[], SearchSpace]] = {
    PPOModel: ppo_space, MultiAgentPPOModel: ppo_space,
    A2CModel: a2c_space, MultiAgentA2CModel: a2c_space,
}


def register_preset(model_cls: type, space_fn: Callable[[], SearchSpace]) -> None:
    """Make `space_fn` the preset for `model_cls` (and its subclasses)."""
    PRESETS[model_cls] = space_fn


def _model_class(target: Any) -> type:
    cls = target if isinstance(target, type) else type(target)
    # a trainer (class or instance) names the model it builds
    return getattr(cls, 'model_cls', cls)


def space_for(target: Any) -> SearchSpace:
    """A fresh copy of the preset for a model class, a model, or a trainer
    (via its `model_cls`). Raises KeyError when no preset covers it."""
    model_cls = _model_class(target)
    for cls in model_cls.__mro__:
        if cls in PRESETS:
            return PRESETS[cls]()
    raise KeyError(f"no search-space preset for {model_cls.__name__}; presets exist for "
                   f"{sorted(c.__name__ for c in PRESETS)} (see register_preset)")


__all__ = ['PRESETS', 'a2c_space', 'ppo_space', 'register_preset', 'space_for']
