from abc import ABC, abstractmethod
from pydantic import AfterValidator, InstanceOf
import torch
import numpy as np
from typing import TYPE_CHECKING, TypeVar, Any, ClassVar, Annotated, Generic, NamedTuple, Self, get_args, get_origin
from dragon_torch.modules.RL.types import SUPPORTED_RL_BACKENDS
from dragon_torch.modules.RL.exceptions import InvalidBackendException
from dragon_torch.common.types import DeviceLike

if TYPE_CHECKING:
    from gymnasium import spaces

ReplayDtype = TypeVar('ReplayDtype', covariant=True)

# field name -> (per-transition shape, numpy dtype)
type BufferSpecs = dict[str, tuple[tuple[int, ...], np.dtype]]


class _BaseReplayBuffer(ABC, Generic[ReplayDtype]):
    kind: ClassVar[str]
    backend: ClassVar[type | None] = None
    _registry: ClassVar[dict[str, Any]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # only register a kind the subclass declares itself, not an inherited one
        _BaseReplayBuffer._registry[cls.__dict__.get('kind', cls.__name__)] = cls
        # `class X(_ReplayBuffer[np.ndarray])` sets the backend unless X sets it itself
        if 'backend' not in cls.__dict__:
            for base in getattr(cls, '__orig_bases__', ()):
                if isinstance(get_origin(base), type) and issubclass(get_origin(base), _BaseReplayBuffer):
                    args = get_args(base)
                    if args and isinstance(args[0], type):
                        cls.backend = args[0]


# Annotated validator
def validate_buffer(buf: _BaseReplayBuffer) -> _BaseReplayBuffer:
    if not (isinstance(buf.backend, type) and issubclass(buf.backend, SUPPORTED_RL_BACKENDS)):
        raise InvalidBackendException(buf.backend, SUPPORTED_RL_BACKENDS)
    if buf.kind not in _BaseReplayBuffer._registry:
        raise TypeError(f"{buf.kind} not found in {list(_BaseReplayBuffer._registry)}")
    return buf

# use as a pydantic field type: `buffer: BaseReplayBuffer`
BaseReplayBuffer = Annotated[InstanceOf[_BaseReplayBuffer], AfterValidator(validate_buffer)]

class _ReplayBuffer(_BaseReplayBuffer[ReplayDtype], ABC):
    """Circular storage shared by replay buffers.

    Concrete buffers pick a backend through the generic parameter, e.g.
    ``class Uniform(_ReplayBuffer[torch.Tensor])``, and add storage by extending
    ``_buffer_specs``. Every field is preallocated as ``(maxlength, *shape)``.
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int | tuple[int, ...],
        maxlength: int,
        reward_dim: int = 1,
        action_dtype: np.dtype | type = np.float32,
        obs_dtype: np.dtype | type = np.float32,
        device: DeviceLike | torch.device = 'cpu',
        seed: int | None = None,
    ):
        if not (isinstance(self.backend, type) and issubclass(self.backend, SUPPORTED_RL_BACKENDS)):
            raise InvalidBackendException(self.backend, SUPPORTED_RL_BACKENDS)
        if maxlength < 1:
            raise ValueError(f"maxlength must be positive, got {maxlength}")

        self.ptr, self.size = 0, 0
        self.maxlength = maxlength
        self.obs_dim = tuple(int(d) for d in np.atleast_1d(obs_dim))
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.action_dtype = np.dtype(action_dtype)
        self.obs_dtype = np.dtype(obs_dtype)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        self.specs = self._buffer_specs()
        for name, (shape, dtype) in self.specs.items():
            setattr(self, name, self._alloc(shape, dtype))

    def _buffer_specs(self) -> BufferSpecs:
        """Fields to allocate. Subclasses extend this, e.g. with priorities."""
        # scalar rewards are (N,) like the flags, so r + γ(1 − d)Q' can't broadcast to (B, B)
        reward_shape = () if self.reward_dim == 1 else (self.reward_dim,)
        return {
            'obs': (self.obs_dim, self.obs_dtype),
            'next_obs': (self.obs_dim, self.obs_dtype),
            'actions': ((self.action_dim,), self.action_dtype),
            'rewards': (reward_shape, np.dtype(np.float32)),
            # kept apart: truncated steps still bootstrap, terminated ones don't
            'terminated': ((), np.dtype(np.bool_)),
            'truncated': ((), np.dtype(np.bool_)),
        }

    def _alloc(self, shape: tuple[int, ...], dtype: np.dtype) -> ReplayDtype:
        shape = (self.maxlength, *shape)
        if issubclass(self.backend, torch.Tensor):
            return torch.zeros(shape, dtype=self.torch_dtype(dtype), device=self.device)
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def torch_dtype(dtype: np.dtype) -> torch.dtype:
        return torch.from_numpy(np.zeros(0, dtype=dtype)).dtype

    def __len__(self) -> int:
        return self.size

    # ------- construction from envs ------- #
    @classmethod
    def from_spaces(cls, observation_space: 'spaces.Space', action_space: 'spaces.Space',
                    maxlength: int, **kwargs) -> Self:
        """Size the buffer from a single env's spaces.

        Box, Discrete, MultiDiscrete and MultiBinary are supported. Float
        observations and actions are stored as float32; integer ones keep their
        dtype (e.g. uint8 images). Discrete actions are stored as ``(N, 1)`` int64.
        """
        obs_dim, obs_dtype = _space_layout(observation_space, 'observation')
        action_shape, action_dtype = _space_layout(action_space, 'action')
        if len(action_shape) != 1:
            raise ValueError(f"action space must be 1-D, got shape {action_space.shape}")
        kwargs.setdefault('obs_dtype', obs_dtype)
        kwargs.setdefault('action_dtype', action_dtype)
        return cls(action_dim=action_shape[0], obs_dim=obs_dim, maxlength=maxlength, **kwargs)

    @classmethod
    def from_env(cls, env: Any, maxlength: int, **kwargs) -> Self:
        """Size the buffer from a gymnasium ``Env``, ``VectorEnv`` or ``EnvCollector``.

        Vector envs are read through their ``single_*_space``.
        """
        obs_space = getattr(env, 'single_observation_space', None)
        action_space = getattr(env, 'single_action_space', None)
        obs_space = env.observation_space if obs_space is None else obs_space
        action_space = env.action_space if action_space is None else action_space
        return cls.from_spaces(obs_space, action_space, maxlength, **kwargs)

    # ------- writing ------- #
    def add_transition(self, transition: Any) -> None:
        """Add a ``dragon_torch.data.RL.Transition`` (or anything with the same fields)."""
        self.add(transition.obs, transition.action, transition.reward, transition.next_obs,
                 transition.terminated, transition.truncated)

    def _batch(self, **fields: Any) -> dict[str, np.ndarray]:
        """Turn one env step, or one vector-env step, into ``(n, *shape)`` numpy arrays.

        Keys are spec names. ``n`` comes from ``terminated``: a scalar means a
        single env. ``(1,)``-shaped fields (Discrete actions and observations)
        also accept a flat ``(n,)`` batch, or a scalar from a single env.
        """
        if 'terminated' not in fields:
            raise ValueError("a batch needs a 'terminated' field to know its size")
        terminated = self.to_np(fields['terminated'])
        n = 1 if terminated.ndim == 0 else terminated.shape[0]
        out = {}
        for name, value in fields.items():
            shape, dtype = self.specs[name]
            x = self.to_np(value)
            if x.shape == (n, *shape):
                pass
            elif (n == 1 and x.shape == shape) or (shape == (1,) and x.shape == (n,)) \
                    or (n == 1 and shape == (1,) and x.shape == ()):
                x = x.reshape(n, *shape)
            else:
                raise ValueError(f"{name}: got shape {x.shape}, expected ({n}, "
                                 f"{', '.join(map(str, shape))}) for a batch of {n}")
            out[name] = x.astype(dtype, copy=False)
        return out

    def _store(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        """Write a ``_batch`` at the write pointer, wrapping around. Returns the indices used."""
        n = next(iter(batch.values())).shape[0]
        if n > self.maxlength:  # only the newest maxlength rows survive
            batch = {k: v[n - self.maxlength:] for k, v in batch.items()}
            n = self.maxlength
        idxs = (self.ptr + np.arange(n)) % self.maxlength
        for name, rows in batch.items():
            storage = getattr(self, name)
            if isinstance(storage, torch.Tensor):
                storage[torch.as_tensor(idxs, device=storage.device)] = torch.as_tensor(
                    rows, dtype=storage.dtype, device=storage.device)
            else:
                storage[idxs] = rows
        self.ptr = int((self.ptr + n) % self.maxlength)
        self.size = min(self.size + n, self.maxlength)
        return idxs

    def _ordered_idxs(self) -> np.ndarray:
        """Indices of the stored transitions, oldest first."""
        if self.size < self.maxlength:
            return np.arange(self.size)
        return (self.ptr + np.arange(self.maxlength)) % self.maxlength

    # ------- checkpointing ------- #
    def state_dict(self) -> dict[str, Any]:
        """Snapshot of the stored transitions, write position and RNG.

        Only the filled rows are saved. Arrays are CPU tensors whatever the
        backend, so the state loads with ``torch.load(..., weights_only=True)``
        and can be restored into a buffer of either backend. Subclasses with
        extra state extend the returned dict.
        """
        def snapshot(storage: Any) -> torch.Tensor:
            rows = storage[:self.size]
            if isinstance(rows, torch.Tensor):
                return rows.detach().cpu().clone()
            return torch.from_numpy(np.array(rows, copy=True))

        return {
            'kind': self.kind,
            'maxlength': self.maxlength,
            'ptr': self.ptr,
            'size': self.size,
            'specs': {name: (tuple(shape), dtype.str) for name, (shape, dtype) in self.specs.items()},
            'data': {name: snapshot(getattr(self, name)) for name in self.specs},
            'rng': self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a ``state_dict`` into this buffer, keeping its backend and device.

        The buffer must have the same ``maxlength`` and field layout as the
        one that was saved.
        """
        if state['maxlength'] != self.maxlength:
            raise ValueError(f"state has maxlength {state['maxlength']}, buffer has {self.maxlength}")
        specs = {name: (tuple(shape), dtype.str) for name, (shape, dtype) in self.specs.items()}
        saved = {name: (tuple(shape), dtype) for name, (shape, dtype) in state['specs'].items()}
        if saved != specs:
            raise ValueError(f"state layout {saved} does not match buffer layout {specs}")
        size, ptr = int(state['size']), int(state['ptr'])
        if not (0 <= size <= self.maxlength and 0 <= ptr < self.maxlength):
            raise ValueError(f"invalid state: size={size}, ptr={ptr}, maxlength={self.maxlength}")
        for name, (shape, _) in self.specs.items():
            rows = state['data'][name]
            if tuple(rows.shape) != (size, *shape):
                raise ValueError(f"{name}: saved rows have shape {tuple(rows.shape)}, "
                                 f"expected {(size, *shape)}")

        for name in self.specs:
            storage, rows = getattr(self, name), state['data'][name]
            if isinstance(storage, torch.Tensor):
                storage.zero_()
                storage[:size] = torch.as_tensor(rows, dtype=storage.dtype, device=storage.device)
            else:
                storage.fill(0)
                storage[:size] = self.to_np(rows)
        self.ptr, self.size = ptr, size
        self.rng.bit_generator.state = state['rng']

    @abstractmethod
    def add(self, obs: ReplayDtype, action: ReplayDtype, reward: ReplayDtype, next_obs: ReplayDtype,
            terminated: ReplayDtype, truncated: ReplayDtype): ...

    @abstractmethod
    def sample(self, batch_size: int, replace: bool = True, use_cer: bool = False, to_tensor: bool = False,
               device: DeviceLike | None = None): ...

    @abstractmethod
    def sample_obs(self, batch_size: int, replace: bool = True, to_tensor: bool = False,
                   device: DeviceLike | None = None): ...

    @abstractmethod
    def all(self, truncate_to: int | None = None): ...

    def validate_obs(self, obs: ReplayDtype):
        """Check a batch of observations, shaped (n, *obs_dim)."""
        if not hasattr(obs, 'shape'):
            raise TypeError(f"observation must be an array, got {type(obs).__name__}")
        if tuple(obs.shape[1:]) != self.obs_dim:
            raise ValueError(f"observation shape: {tuple(obs.shape)} does not match expected "
                             f"shape: (n, {', '.join(map(str, self.obs_dim))})")

    def to_np(self, x: Any) -> np.ndarray:
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

class ReplaySample(NamedTuple, Generic[ReplayDtype]):
    obs: ReplayDtype
    actions: ReplayDtype
    rewards: ReplayDtype
    next_obs: ReplayDtype
    terminated: ReplayDtype
    truncated: ReplayDtype
    idxs: ReplayDtype


def _space_layout(space: 'spaces.Space', what: str) -> tuple[tuple[int, ...], np.dtype]:
    """Per-step storage shape and dtype for a gymnasium space."""
    from gymnasium import spaces
    if isinstance(space, spaces.Discrete):
        return (1,), np.dtype(np.int64)
    if isinstance(space, (spaces.Box, spaces.MultiDiscrete, spaces.MultiBinary)):
        dtype = np.dtype(space.dtype)
        if isinstance(space, spaces.MultiDiscrete):
            dtype = np.dtype(np.int64)
        elif np.issubdtype(dtype, np.floating):
            dtype = np.dtype(np.float32)
        return tuple(space.shape) or (1,), dtype
    raise NotImplementedError(
        f"No replay layout for a {type(space).__name__} {what} space; wrap the env so it uses "
        "Box / Discrete / MultiDiscrete / MultiBinary (e.g. gymnasium.wrappers.FlattenObservation)")
