from typing import Optional, Sequence, Union
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .registry import register_wrapper

RAW_REWARD_KEY = 'raw_reward'


class RecordRawReward(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Copies each step's reward into info['raw_reward'] before any outer
    wrapper changes it. make_env_fn puts it directly around the base env (inside
    every configured wrapper), so the collector can report the task's own
    return next to the shaped one the agent trains on. Not registered: the
    framework adds it itself."""

    def __init__(self, env: gym.Env):
        gym.utils.RecordConstructorArgs.__init__(self)
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info[RAW_REWARD_KEY] = float(reward)
        return obs, reward, terminated, truncated, info


@register_wrapper
class Discretizer(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """Expose a MultiBinary button env (e.g. stable-retro) as Discrete(len(combos)).

    Each combo lists the buttons pressed together, by name or index:
        Discretizer(env, [[], ['LEFT'], ['RIGHT'], ['RIGHT', 'B']])
    Button names default to `env.unwrapped.buttons` (what retro envs expose).
    """

    def __init__(self, env: gym.Env, combos: Sequence[Sequence[Union[str, int]]],
                 buttons: Optional[Sequence[str]] = None):
        gym.utils.RecordConstructorArgs.__init__(self, combos=combos, buttons=buttons)
        super().__init__(env)
        space = env.action_space
        if not isinstance(space, spaces.MultiBinary) or len(space.shape) != 1:
            raise TypeError(f"Discretizer needs a 1-D MultiBinary action space, got {space}")
        buttons = list(buttons) if buttons is not None else getattr(env.unwrapped, 'buttons', None)
        self._table = np.zeros((len(combos), space.shape[0]), dtype=space.dtype)
        for i, combo in enumerate(combos):
            for button in combo:
                if isinstance(button, str):
                    if buttons is None:
                        raise ValueError("Button names given but env has no `buttons`; pass buttons=")
                    button = buttons.index(button)
                self._table[i, button] = 1
        self.action_space = spaces.Discrete(len(combos))

    def action(self, action: int) -> np.ndarray:
        return self._table[int(action)].copy()
