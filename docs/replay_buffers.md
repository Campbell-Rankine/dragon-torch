# Write a replay buffer

This guide shows how to add a new replay buffer to `dragon_torch.modules.RL.replay` by subclassing `_ReplayBuffer`. As the example, it builds a proportional prioritized replay buffer ([Schaul et al., 2015](https://arxiv.org/abs/1511.05952)) step by step.

By the end, the buffer will:

- allocate its storage, including an extra `priorities` field
- accept steps from a single gymnasium env, a vector env or an `EnvCollector`
- sample with priorities and importance-sampling weights
- save and restore through `state_dict`

The finished class is about 70 lines, because `_ReplayBuffer` already handles allocation, input batching, circular writes, building from env spaces and checkpointing.

## What the base class gives you

| Provided by `_ReplayBuffer` | What it does |
|---|---|
| `__init__(action_dim, obs_dim, maxlength, reward_dim=1, action_dtype=np.float32, obs_dtype=np.float32, device='cpu', seed=None)` | Validates the backend and allocates every field in `_buffer_specs()` as `(maxlength, *shape)`. Creates the seeded generator `self.rng`. |
| `from_spaces(observation_space, action_space, maxlength, **kwargs)`, `from_env(env, maxlength, **kwargs)` | Builds the buffer from gymnasium spaces, an `Env`, a `VectorEnv` or an `EnvCollector`. Extra `kwargs` reach your `__init__`. |
| `_buffer_specs()` | Maps each field name to `(per-step shape, numpy dtype)`. Extend this to add fields. |
| `_batch(**fields)` | Turns a single-env or vector-env step into `(n, *shape)` numpy arrays. It checks each field's shape and casts it to the storage dtype. |
| `_store(batch)` | Writes a batch at `self.ptr`, wrapping around and evicting the oldest rows, and updates `ptr` and `size`. Returns the row indices it wrote. |
| `_ordered_idxs()` | The stored row indices, oldest first. |
| `add_transition(tr)` | Calls your `add` with the fields of a collector `Transition`. |
| `state_dict()`, `load_state_dict(state)` | Saves and restores the filled rows of every field, `ptr`, `size` and the generator state. |
| `validate_obs(obs)`, `to_np(x)` | Shape check for an observation batch, and conversion of any array or tensor to numpy. |

You implement four abstract methods: `add`, `sample`, `sample_obs` and `all`.

## 1. Declare the class

```python
from typing import Any, NamedTuple

import numpy as np
import torch

from dragon_torch.modules.RL.replay.base import _ReplayBuffer, BufferSpecs


class PrioritizedReplayBuffer(_ReplayBuffer[np.ndarray]):
    """Proportional prioritized replay (Schaul et al., 2015)."""
    kind = 'prioritized-numpy'
```

The type parameter sets the **backend**, which is what the storage arrays are made of. `_ReplayBuffer[np.ndarray]` stores numpy arrays and `_ReplayBuffer[torch.Tensor]` stores tensors on `device`. The class reads it into `PrioritizedReplayBuffer.backend` when it's defined, and any other type raises `InvalidBackendException` when you build the buffer.

`kind` is the buffer's name in the registry, `_BaseReplayBuffer._registry`. The `BaseReplayBuffer` pydantic field type uses it to validate buffers. Give every concrete buffer its own `kind`. If you leave it out, the class registers under its class name.

## 2. Add your own settings and storage

```python
    def __init__(self, *args, alpha: float = 0.6, beta: float = 0.4, eps: float = 1e-6, **kwargs):
        self.alpha, self.beta, self.eps = alpha, beta, eps
        self.max_priority = 1.0
        super().__init__(*args, **kwargs)

    def _buffer_specs(self) -> BufferSpecs:
        return {**super()._buffer_specs(), 'priorities': ((), np.dtype(np.float64))}
```

`_buffer_specs` maps each field to its per-step shape and dtype. The base `__init__` allocates every entry as an attribute, so `self.priorities` becomes a `(maxlength,)` float64 array. `_store`, `state_dict` and `load_state_dict` include it from then on with no extra code.

Keep the base fields by extending `super()._buffer_specs()` rather than replacing it.

> **Note**
> The base `__init__` calls `_buffer_specs()`. Set any attribute that `_buffer_specs` reads *before* calling `super().__init__()`.

## 3. Implement `add`

```python
    def add(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        batch = self._batch(obs=obs, next_obs=next_obs, actions=action, rewards=reward,
                            terminated=terminated, truncated=truncated)
        idxs = self._store(batch)
        self.priorities[idxs] = self.max_priority
```

`_batch` handles every input layout:

- one env (`obs` shaped `obs_dim`) or a vector env (`(n_envs, *obs_dim)`)
- numpy or torch input, including CUDA tensors and tensors that need gradients
- scalar Discrete actions

Its keyword names must match the spec names (`actions`, not `action`). It raises `ValueError` on a wrong shape before anything is written, so a bad step can't leave a partial row.

`_store` returns the rows it wrote, which is where per-row data goes. New transitions get the current max priority, so each one is sampled at least once.

## 4. Implement `sample`

```python
class PrioritizedSample(NamedTuple):
    obs: Any
    actions: Any
    rewards: Any
    next_obs: Any
    terminated: Any
    truncated: Any
    idxs: Any
    weights: Any
```

```python
    def _probs(self) -> np.ndarray:
        p = self.priorities[:self.size] ** self.alpha
        return p / p.sum()

    def sample(self, batch_size, replace=True, use_cer=False, to_tensor=False, device=None):
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        probs = self._probs()
        idxs = self.rng.choice(self.size, batch_size, replace=replace, p=probs)
        if use_cer and batch_size > 0:
            idxs[0] = (self.ptr - 1) % self.maxlength
        weights = (self.size * probs[idxs]) ** -self.beta
        weights = (weights / weights.max()).astype(np.float32)
        sample = PrioritizedSample(self.obs[idxs], self.actions[idxs], self.rewards[idxs],
                                   self.next_obs[idxs], self.terminated[idxs],
                                   self.truncated[idxs], idxs, weights)
        if to_tensor:
            device = self.device if device is None else device
            sample = PrioritizedSample(*(torch.as_tensor(x, device=device) for x in sample))
        return sample

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = np.abs(self.to_np(td_errors)) + self.eps
        self.priorities[self.to_np(idxs)] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max()))
```

Sampling follows these rules:

- **Filled rows:** the filled rows are always `0 .. size-1`, because the buffer only wraps once it's full. Index `[:self.size]`; you don't need `_ordered_idxs()` here.
- **Random draws:** use `self.rng` for all randomness, never `np.random`. That keeps runs repeatable for a given `seed`, and the generator state is checkpointed.
- **Return type:** return a `ReplaySample`, or a NamedTuple that starts with the same seven fields. Training code unpacks batches by field name, so extra fields like `weights` go at the end.
- **CER:** `use_cer` must put the newest row, `(ptr - 1) % maxlength`, in the batch.

`update_priorities` isn't part of the base interface. Add whatever methods your algorithm needs.

> **Note**
> `rng.choice(p=...)` is O(N) per batch. That's fine for a guide and for small buffers. Use a sum-tree for buffers of 10⁵ rows or more.

## 5. Implement `sample_obs` and `all`

```python
    def sample_obs(self, batch_size, replace=True, to_tensor=False, device=None):
        obs = self.sample(batch_size, replace).obs
        return torch.as_tensor(obs, device=device or self.device) if to_tensor else obs

    def all(self, truncate_to=None):
        idxs = self._ordered_idxs()
        if truncate_to is not None:
            idxs = self.rng.choice(idxs, min(truncate_to, self.size), replace=False)
        return PrioritizedSample(self.obs[idxs], self.actions[idxs], self.rewards[idxs],
                                 self.next_obs[idxs], self.terminated[idxs], self.truncated[idxs],
                                 idxs, np.ones(len(idxs), np.float32))
```

`all` returns every stored row oldest first, or a random subset of at most `truncate_to` rows.

## 6. Checkpoint the extra state

```python
    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), 'max_priority': self.max_priority}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.max_priority = state['max_priority']
```

The base `state_dict` already saves every spec field (here that includes `priorities`), plus `ptr`, `size` and `self.rng`. Extend it only for state that lives outside the specs, such as `max_priority`.

The saved arrays are CPU tensors, so the state loads with `torch.load(..., weights_only=True)`. Keep any extra entries to plain Python values or tensors so that stays true.

## 7. Use it

```python
>>> import gymnasium as gym
>>> env = gym.make('CartPole-v1')
>>> buffer = PrioritizedReplayBuffer.from_env(env, maxlength=1000, seed=0, alpha=0.6)
>>> obs, _ = env.reset(seed=0)
>>> for _ in range(200):
...     action = env.action_space.sample()
...     next_obs, reward, terminated, truncated, _ = env.step(action)
...     buffer.add(obs, action, reward, next_obs, terminated, truncated)
...     obs = env.reset()[0] if terminated or truncated else next_obs
>>> batch = buffer.sample(32, to_tensor=True)
>>> batch.obs.shape, batch.actions.shape
(torch.Size([32, 4]), torch.Size([32, 1]))
>>> buffer.update_priorities(batch.idxs, td_errors)   # td_errors from your critic
>>> restored = PrioritizedReplayBuffer.from_env(env, maxlength=1000, alpha=0.6)
>>> restored.load_state_dict(buffer.state_dict())
```

`from_env` passes `seed` and `alpha` through to `__init__`. With an `EnvCollector`, use `buffer.add_transition(collector.step(actor))`. The collector has already put the true final observation in `next_obs` where an episode ended.

## 8. Test it

Add `tests/test_RL/test_replay/test_replay_<kind>.py`. [`test_replay_uniform.py`](../tests/test_RL/test_replay/test_replay_uniform.py) is a template. Its `step()` helper gives every field a value derived from one tag, so a test can check the fields of each sampled row came from the same transition. At minimum, cover:

- single-env and vector-env `add`, oldest-first eviction, and a vector step that wraps around
- sampled fields lining up with each other and with `idxs`
- an empty buffer raising on `sample`
- the same `seed` giving the same batches
- `state_dict` round-tripping through `torch.save` and `torch.load(weights_only=True)`, including your extra state
- your own sampling rule (for this example, sample frequencies proportional to `p^α`)

## Shortcut: reuse the uniform buffer

If your buffer only changes *which* rows are drawn, subclass `_UniformReplayBuffer` instead. It already implements `add`, `sample`, `sample_obs`, `all` and tensor conversion on both backends. Override the hooks you need:

| Hook | Default |
|---|---|
| `_sample_idxs(batch_size, replace)` | Uniform draw from `self.rng`. Raises on an empty buffer. |
| `_gather(idxs)` | Indexes every field into a `ReplaySample`. |
| `_output(x, to_tensor, device)` | Converts a gathered batch to tensors when asked. |

The torch-backed `TorchUniformReplayBuffer` is an example: it overrides only `_gather` and `_output`.

## Notes

- **Torch backend.** Fields are tensors on `device`, and `_store` writes to them for you. Index them with a `torch.long` tensor on the same device. Keep drawing indices from `self.rng` so that seeding and checkpoints behave the same on both backends.
- **Terminated vs truncated.** Store and return both flags. Truncated steps still bootstrap from `next_obs`, and terminated steps don't.
- **Shapes.** Scalar rewards and both flags are `(N,)`. Discrete actions are `(N, 1)` int64. Float observations and actions are stored as float32, and integer observations, such as uint8 images, keep their dtype.
- **Writing to storage.** Write rows only through `_store`, so `ptr`, `size` and eviction stay consistent. Per-row extras, like `priorities` above, go in the rows `_store` returns.
