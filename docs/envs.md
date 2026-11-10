# dragon_torch.data.envs

Build vectorised Gymnasium environments from configs, with wrappers that save to JSON and can be tuned. For multi-agent envs, see [Multi-agent RL](multi_agent.md).

| | |
|---|---|
| [`DragonEnvConfig`](#dragonenvconfig) | How to build a vectorised env |
| [`WrapperSpec`](#wrapperspec) | A wrapper described as data |
| [`register_wrapper`](#register_wrapper) | Register a wrapper by name |
| [`register_dragon_env`](#register_dragon_env) | Register an env id |
| [`make_env_fn`, `make_vec_env`](#make_env_fn) | Build one sub-env, or the vector env |
| [`validate_env`, `check_env_config`](#validate_env) | Run Gymnasium's env checker |
| [`Discretizer`](#discretizer) | Button combos as a `Discrete` action space |
| [`obs_to_tensor`, `action_to_env`](#obs_to_tensor) | Convert between env values and tensors |

## DragonEnvConfig

```python
class dragon_torch.data.envs.DragonEnvConfig(env_id=None, env_fn=None, env_kwargs={}, wrappers=[],
                                             n_envs=1, seed=0, vectorization='sync')
```

Describes a vectorised Gymnasium env. Trainers build the policy network from the spaces *after* wrapping, so a wrapper that changes a space needs no other changes.

**Parameters**
- **env_id** (*str, optional*) – a registered env, built with `gym.make`.
- **env_fn** (*callable, optional*) – any env factory, such as an env class. Set exactly one of `env_id` and `env_fn`.
- **env_kwargs** (*dict*) – keyword arguments for the env.
- **wrappers** (*list*) – applied in order to each sub-env. Each entry is a [`WrapperSpec`](#wrapperspec), a string or dict shorthand for one, or any callable `env -> env`.
- **n_envs** (*int*) – number of sub-envs. Default: `1`.
- **seed** (*int*) – reset seed. Default: `0`.
- **vectorization** (*str*) – `'sync'` or `'async'`. Use `'async'` for envs that can't share a process, such as stable-retro, or for slow simulators.

**Properties**
- **portable** – `True` when the config is pure data (`env_id` and `WrapperSpec`s only), so it rebuilds exactly from JSON. Callables are saved only as their `repr`.
- **shaped** – `True` when `wrappers` is non-empty, so the trained-on reward may differ from the env's own.

> **Warning**
> `NormalizeObservation` and `NormalizeReward` in `wrappers` raise a `WrapperNormalizationWarning`. Use `rollout.normalize_obs` / `normalize_reward` instead; see [Normalisation](#normalisation).

**Example**
```python
>>> env = DragonEnvConfig(env_id='CartPole-v1', n_envs=8,
...                       wrappers=[WrapperSpec(name='ClipReward', kwargs={'max_reward': 1.0}),
...                                 'FlattenObservation'])
```

## WrapperSpec

```python
class dragon_torch.data.envs.WrapperSpec(name, kwargs={})
```

A wrapper given by name and keyword arguments. `kwargs` are checked against the wrapper's signature when the spec is created, so typos fail when you build the config, not partway through training.

**Parameters**
- **name** (*str*) – resolved, in order, to:
  1. a wrapper registered with [`register_wrapper`](#register_wrapper),
  2. a class in `gymnasium.wrappers`, such as `'FlattenObservation'` or `'ClipReward'`,
  3. an import path `'package.module:attr'`, imported on demand.
- **kwargs** (*dict*) – constructor arguments. Keep them JSON-friendly.

In `wrappers`, the string `'FlattenObservation'` and the dict `{'name': ..., 'kwargs': ...}` are shorthands for a spec.

**Example**
```python
>>> WrapperSpec(name='Discretizer', kwargs={'combos': [[], ['LEFT'], ['RIGHT']]})
>>> WrapperSpec(name='FrameStackObservation')
ValidationError: ... missing stack_size
```

## register_wrapper

```python
dragon_torch.data.envs.register_wrapper(name=None)
```

Decorator that registers a wrapper class, or a `factory(env, **kwargs)`, for [`WrapperSpec`](#wrapperspec). Without `name`, the class name is used. Registering a name again replaces it, and registered names take priority over Gymnasium's. Related: `unregister_wrapper(name)`, `registered_wrappers()`, `resolve_wrapper(name)`.

Give the constructor keyword arguments with defaults, so specs can check them and search spaces can tune them.

**Example**
```python
>>> @register_wrapper('fuel_penalty')
... class FuelPenalty(gym.RewardWrapper):
...     def __init__(self, env, cost: float = 0.01):
...         super().__init__(env)
...         self.cost = cost
...     def reward(self, reward):
...         return reward - self.cost
>>> WrapperSpec(name='fuel_penalty', kwargs={'cost': 0.05})
```

## register_dragon_env

```python
dragon_torch.data.envs.register_dragon_env(env_id, entry_point, **kwargs)
```

Registers an env so `DragonEnvConfig(env_id=...)` can build it. `kwargs` go to `gym.register`; for example, `max_episode_steps` adds a `TimeLimit`. Registering an id again replaces it.

Prefer a registered `env_id` over `env_fn`: the config stays portable, and `env_kwargs` can be tuned (`'env.env_kwargs.masspole'`).

> **Note**
> Async sub-envs receive the registration from the process that builds the vector env. A tuning worker that reloads a saved `config.json` is a new process, though, so make the `register_dragon_env` call in a module that the worker imports. Alternatively, use an id of the form `'mypkg.envs:MyEnv-v0'`, which imports `mypkg.envs` on demand.

**Example**
```python
>>> class HeavyPoleCartPole(CartPoleEnv):
...     def __init__(self, masspole: float = 0.5, **kw):
...         super().__init__(**kw)
...         self.masspole = masspole
>>> register_dragon_env('HeavyPoleCartPole-v0', HeavyPoleCartPole, max_episode_steps=500)
>>> DragonEnvConfig(env_id='HeavyPoleCartPole-v0', env_kwargs={'masspole': 0.5})
```

## make_env_fn

```python
dragon_torch.data.envs.make_env_fn(cfg) -> Callable[[], gym.Env]
dragon_torch.data.envs.make_vec_env(cfg) -> gym.vector.VectorEnv
```

`make_env_fn` returns a picklable thunk that builds one sub-env as base env → `RecordRawReward` → `cfg.wrappers`. `make_vec_env` builds the vector env with `SAME_STEP` autoreset, so the step that ends an episode already returns the reset obs, and the true last obs is in `info['final_obs']`.

## validate_env

```python
dragon_torch.data.envs.validate_env(env, unwrapped=True)
dragon_torch.data.envs.check_env_config(cfg, name='env')
```

Run Gymnasium's env checker on one sub-env.

- `validate_env` takes an env or a config. By default it checks the base env; `unwrapped=False` checks it with wrappers applied.
- `check_env_config` checks the wrapped env. On failure it also checks the base env, then raises `EnvCheckError` naming the config field and saying whether the problem is in the base env or its wrappers.

Trainers run `check_env_config` before building anything when `validate_env=True`.

```
EnvCheckError: env failed gymnasium's env checker; the base env passes, so the problem is in its wrappers: ...
```

> **Note**
> Soft problems, such as infinite `Box` bounds, are warnings. The checker expects identical results under fixed seeds, so deliberately random wrappers fail it.

## Discretizer

```python
class dragon_torch.data.envs.Discretizer(env, combos, buttons=None)
```

Exposes a `MultiBinary` button env, such as stable-retro, as `Discrete(len(combos))`. It is registered as `'Discretizer'`.

**Parameters**
- **combos** (*list of lists*) – the buttons pressed together for each action, by name or index.
- **buttons** (*list of str, optional*) – button names. Default: `env.unwrapped.buttons`.

## obs_to_tensor

```python
dragon_torch.data.envs.obs_to_tensor(obs, device='cpu')
dragon_torch.data.envs.action_to_env(action, space)
```

- `obs_to_tensor` converts batched obs (arrays, dicts or tuples) to tensors. Floats become `float32` and bools become floats; integers keep their dtype.
- `action_to_env` converts a batched policy action to what the vector env accepts: `Box` actions are clipped to the bounds, and `Discrete` / `MultiDiscrete` actions are shifted by `start`.

## Notes

### Customising an env

| To change | Use |
|---|---|
| The reward the agent trains on (shaping, penalties) | A wrapper in `wrappers`. The env's own reward stays the task, see [raw and shaped rewards](#raw-and-shaped-rewards) |
| What the agent sees or how it acts | A wrapper in `wrappers` |
| The task itself: dynamics, termination, the reward to be measured on | An env subclass, registered with [`register_dragon_env`](#register_dragon_env) |
| An existing wrapper's behaviour | A wrapper subclass, registered with [`register_wrapper`](#register_wrapper) |

The default `ActorCritic` reads `Box` and `Discrete` observations. For `Dict` or `Tuple` observations, add `'FlattenObservation'` (after any wrapper that reads the structure) or pass a custom network.

### Raw and shaped rewards

- **Raw** is the env's own reward, before `wrappers`. `RecordRawReward`, placed directly around the base env, copies it into `info['raw_reward']`.
- **Shaped** is the reward after `wrappers`, which is what the agent trains on.

Neither is affected by `rollout.normalize_reward`. Anything the env does itself counts as raw, including wrappers that `gym.make` adds from the registration.

| History key | Meaning |
|---|---|
| `episode_return` | Mean raw return of the training episodes finished this epoch |
| `episode_return_shaped` | Their mean shaped return. Only present when `wrappers` is non-empty |
| `val_return` | What `evaluate()` scores: raw by default |
| `val_return_raw`, `val_return_shaped` | Both evaluation returns. Only present when `wrappers` is non-empty |

Set `DragonRLConfig(eval_return='shaped')` to evaluate, and so tune, on the shaped reward. Do this when a wrapper defines the task. If a wrapper drops `info`, the raw reward falls back to the reward the collector received.

### Normalisation

Normalise with `RolloutConfig(normalize_obs=True, normalize_reward=True)`, not with wrappers:

| | `rollout.normalize_*` | `Normalize*` wrappers |
|---|---|---|
| Statistics | One set, shared by all sub-envs | One set per sub-env |
| Evaluation | Uses the training stats, frozen | Starts fresh and keeps updating while it's scored |
| Checkpoints | Saved and restored | Lost |
| Logged returns | Unchanged | Shaped returns end up in normalised units |

`normalize_obs` needs a `Box` observation space. To keep per-env wrappers anyway, silence the warning with `warnings.filterwarnings('ignore', category=WrapperNormalizationWarning)`.

### Tuning wrapper settings

Config paths reach list items and dict keys, so a search space can tune wrapper arguments, for example `'env.wrappers.0.kwargs.cost'`.

- Indices follow the order of `wrappers`, and negative indices count from the end.
- The last key may be new, so a kwarg left at its default can still be tuned.
- Unknown fields and out-of-range indices raise `KeyError`.
- The changed spec is re-validated.
