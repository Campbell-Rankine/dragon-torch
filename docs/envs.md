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

Trials are scored on the env's raw reward, so tuning a reward-shaping wrapper compares trials on the actual task rather than on the shaping (see the next section).

## Raw and shaped rewards

Reward-shaping wrappers change what the agent trains on, not what the task is. The framework therefore tracks both:

- **Raw:** the reward the env itself returns, before any of `env.wrappers`.
- **Shaped:** the reward after all of `env.wrappers`, which is what the agent trains on.

Neither is affected by reward normalisation (`rollout.normalize_reward`), which only scales the training signal.

To capture the raw reward, `make_env_fn` puts a `RecordRawReward` wrapper directly around the base env, inside every configured wrapper. It copies each step's reward into `info['raw_reward']`. You don't need a special base class for your own wrappers: `gym.RewardWrapper`, `TransformReward`, `ClipReward` and custom `step` overrides are all captured.

### History keys

| Key | Meaning |
|---|---|
| `episode_return` | Mean **raw** return of training episodes finished this epoch |
| `episode_return_shaped` | Mean shaped return of the same episodes |
| `val_return` | What `model.evaluate()` scores: the **raw** return by default |
| `val_return_raw`, `val_return_shaped` | Both evaluation returns |
| `val_return_smooth` | The smoothed `val_return`; the default tuning objective |

The `_shaped` and `_raw` keys only appear when `env.wrappers` is non-empty. Without wrappers the two returns are identical, so the history stays uncluttered.

### Choosing what to optimise

`DragonRLConfig.eval_return` sets what `evaluate()` returns, and with it what the tuning objective optimises:

- `'raw'` (default) scores the task. Use it whenever wrappers shape the reward.
- `'shaped'` scores the shaped reward. Use it when a wrapper *defines* the task, for example a wrapper that turns a sparse goal signal into the actual objective you care about.

```python
model=DragonRLConfig(..., eval_return='shaped')
```

### Where "raw" is measured

"Raw" means the reward before the wrappers in `env.wrappers`. Anything the env does itself counts as raw:

- an env subclass that changes the reward,
- shaping inside an `env_fn` factory,
- wrappers that `gym.make` adds from the env's registration.

Put reward shaping in `env.wrappers` when you want it kept separate from the task.

If a wrapper drops `info`, so that `raw_reward` never reaches the collector, the collector falls back to the reward it actually received for those steps.
