# Environments

`DragonEnvConfig` describes how to build a vectorised Gymnasium environment:

- `env_id`: a registered env, built with `gym.make`.
- `env_fn`: any factory, such as an env class.
- `wrappers`: applied in order to each sub-env.
- `n_envs`, `seed` and `vectorization` (`sync` or `async`).

The trainers build the policy network from the spaces *after* wrapping, so a wrapper that changes the observation or action space needs no other changes.

```python
from dragon_torch.data.envs import DragonEnvConfig, WrapperSpec

env = DragonEnvConfig(
    env_id='CartPole-v1', n_envs=8,
    wrappers=[WrapperSpec(name='ClipReward', kwargs={'max_reward': 1.0}), 'FlattenObservation'])
```

## Wrappers as data

Each entry in `wrappers` can be:

| Entry | Example | Saves to JSON? |
|---|---|---|
| `WrapperSpec(name, kwargs)` | `WrapperSpec(name='ClipReward', kwargs={'max_reward': 1.0})` | ✅ |
| A string, short for `WrapperSpec(name=...)` | `'FlattenObservation'` | ✅ |
| A dict | `{'name': 'fuel_penalty', 'kwargs': {'cost': 0.05}}` | ✅ |
| Any callable `env -> env` | `lambda e: MyWrapper(e, 0.1)` | ❌ only as its `repr` |

Prefer specs. A config made only of data, which `DragonEnvConfig.portable` reports, gives you two things:

- A tuning trial's `config.json` rebuilds the exact env, wrappers included.
- Search spaces can tune wrapper settings (see below).

### Names

A spec's `name` resolves, in order, to:

1. **A wrapper you registered** with `register_wrapper`:
   ```python
   import gymnasium as gym
   from dragon_torch.data.envs import register_wrapper

   @register_wrapper('fuel_penalty')          # or bare @register_wrapper to use the class name
   class FuelPenalty(gym.RewardWrapper):
       def __init__(self, env, cost: float = 0.01):
           super().__init__(env)
           self.cost = cost

       def reward(self, reward):
           return reward - self.cost
   ```
   Registering a name again replaces the old entry, so re-running a notebook cell works. A registered name takes priority over a Gymnasium wrapper with the same name.
2. **A Gymnasium wrapper class**, for example `'FlattenObservation'`, `'ClipReward'`, `'FrameStackObservation'`, `'RescaleAction'` or `'TimeAwareObservation'`. Any class in `gymnasium.wrappers` works.
3. **An import path** `'package.module:attr'`, imported on demand. Use this for wrappers in your own packages when a saved config must rebuild without first importing the module that registers them.

`Discretizer`, the framework's wrapper for stable-retro button actions, is registered as `'Discretizer'`:

```python
WrapperSpec(name='Discretizer', kwargs={'combos': [[], ['LEFT'], ['RIGHT'], ['RIGHT', 'B']]})
```

### Checks

A spec checks its `kwargs` against the wrapper's signature as soon as it's created. Typos and missing required arguments therefore fail when you build the config, not partway through training:

```python
WrapperSpec(name='fuel_penalty', kwargs={'csot': 0.1})   # ValidationError: bad kwargs ...
WrapperSpec(name='FrameStackObservation')                 # ValidationError: missing stack_size
```

Keep `kwargs` JSON-friendly: numbers, strings, booleans, lists and dicts.

## Tuning wrapper settings

`apply_overrides` paths can reach list items by index and dict entries by key. A search space can therefore tune any wrapper argument:

```python
from dragon_torch.utils.tuning import FloatParam

space = ppo_space().narrow({
    'env.wrappers.0.kwargs.cost': FloatParam(low=0.0, high=0.1),
})
```

How these paths behave:

- **Indices** follow the order of `env.wrappers`, and negative indices count from the end.
- **The final key may be new.** A kwarg the spec doesn't set yet, such as one that relies on the wrapper's default, can still be tuned.
- **Earlier parts must exist.** Unknown fields, out-of-range indices and missing intermediate keys raise `KeyError`.
- **Values are checked.** The changed spec is re-validated, so a bad kwarg is still caught.

If a reward wrapper is part of the search, note that `episode_return` and `val_return` currently report the **shaped** reward. Trials with different shaping settings then aren't compared on the actual task. Separate tracking of the raw reward is the next stage of this work.
