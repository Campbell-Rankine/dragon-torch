from typing import Any, Generic, NamedTuple

import numpy as np
import torch

from dragon_torch.modules.RL.replay.base import _ReplayBuffer, BufferSpecs, DeviceLike, ReplayDtype


class PrioritizedSample(NamedTuple):
    obs: Any
    actions: Any
    rewards: Any
    next_obs: Any
    terminated: Any
    truncated: Any
    idxs: Any
    weights: Any


class _PrioritizedReplayBuffer(_ReplayBuffer[ReplayDtype], Generic[ReplayDtype]):
    """Proportional prioritized replay (Schaul et al., 2015).

    Row ``i`` is drawn with probability ``p_i^α / Σ p^α`` where ``p_i = |δ_i| + ε``.
    Each batch carries importance-sampling weights ``(N · P(i))^−β``, divided by
    the batch maximum. New rows get the largest priority seen so far. Indices
    are drawn from the seeded numpy RNG on both backends.

    Args:
        alpha: How strongly priorities skew sampling; ``0`` is uniform. Default: ``0.6``
        beta: Importance-sampling correction; anneal towards ``1``. Default: ``0.4``
        eps: Added to ``|δ|`` so no row's priority is zero. Default: ``1e-6``
    """

    def __init__(self, *args, alpha: float = 0.6, beta: float = 0.4, eps: float = 1e-6, **kwargs):
        if alpha < 0 or beta < 0 or eps <= 0:
            raise ValueError(f"need alpha >= 0, beta >= 0 and eps > 0; "
                             f"got alpha={alpha}, beta={beta}, eps={eps}")
        self.alpha, self.beta, self.eps = alpha, beta, eps
        self.max_priority = 1.0
        super().__init__(*args, **kwargs)

    def _buffer_specs(self) -> BufferSpecs:
        """Adds an extra "priorities" Array to the buffer"""
        return {**super()._buffer_specs(), 'priorities': ((), np.dtype(np.float64))}

    # ------- priorities ------- #
    def _set_priorities(self, idxs: np.ndarray, values: Any) -> None:
        if isinstance(self.priorities, torch.Tensor):
            self.priorities[torch.as_tensor(idxs, dtype=torch.long, device=self.device)] = \
                torch.as_tensor(values, dtype=self.priorities.dtype, device=self.device)
        else:
            self.priorities[idxs] = values

    def _probs(self) -> np.ndarray:
        p = self.to_np(self.priorities[:self.size]).astype(np.float64) ** self.alpha
        return p / p.sum()

    def update_priorities(self, idxs: Any, td_errors: Any) -> None:
        """Set ``p_i = |δ_i| + ε`` for the sampled rows ``idxs``."""
        idxs = self.to_np(idxs).astype(np.int64).reshape(-1)
        td_errors = self.to_np(td_errors).astype(np.float64).reshape(-1)
        if idxs.shape != td_errors.shape:
            raise ValueError(f"got {idxs.size} indices but {td_errors.size} TD errors")
        if not np.isfinite(td_errors).all():
            raise ValueError("TD errors must be finite")
        if idxs.size and (idxs.min() < 0 or idxs.max() >= self.size):
            raise IndexError(f"indices must be in [0, {self.size})")
        if idxs.size == 0:
            return
        priorities = np.abs(td_errors) + self.eps
        self._set_priorities(idxs, priorities)
        self.max_priority = max(self.max_priority, float(priorities.max()))

    # ------- adding ------- #
    def add(self, obs: Any, action: Any, reward: Any, next_obs: Any,
            terminated: Any, truncated: Any) -> None:
        """Add one env step, or one vector-env step, at the current max priority."""
        idxs = self._store(self._batch(obs=obs, next_obs=next_obs, actions=action, rewards=reward,
                                       terminated=terminated, truncated=truncated))
        self._set_priorities(idxs, self.max_priority)

    # ------- sampling ------- #
    def _sample_idxs(self, batch_size: int, replace: bool) -> tuple[np.ndarray, np.ndarray]:
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        probs = self._probs()
        return self.rng.choice(self.size, batch_size, replace=replace, p=probs), probs

    def _gather(self, idxs: np.ndarray, weights: np.ndarray) -> PrioritizedSample:
        return PrioritizedSample(self.obs[idxs], self.actions[idxs], self.rewards[idxs],
                                 self.next_obs[idxs], self.terminated[idxs], self.truncated[idxs],
                                 idxs, weights)

    def _output(self, x: Any, to_tensor: bool, device: DeviceLike | None) -> Any:
        if not to_tensor:
            return x
        device = self.device if device is None else device
        if isinstance(x, PrioritizedSample):
            return PrioritizedSample(*(torch.as_tensor(v, device=device) for v in x))
        return torch.as_tensor(x, device=device)

    def sample(self, batch_size: int, replace: bool = True, use_cer: bool = False,
               to_tensor: bool = False, device: DeviceLike | None = None) -> PrioritizedSample:
        """Sample by priority.

        Returns:
            A ``PrioritizedSample``: the ``ReplaySample`` fields followed by ``weights``
        """
        idxs, probs = self._sample_idxs(batch_size, replace)
        if use_cer and batch_size > 0:
            idxs[0] = (self.ptr - 1) % self.maxlength
        weights = (self.size * probs[idxs]) ** -self.beta
        weights = (weights / weights.max()).astype(np.float32) if batch_size > 0 \
            else weights.astype(np.float32)
        return self._output(self._gather(idxs, weights), to_tensor, device)

    def sample_obs(self, batch_size: int, replace: bool = True, to_tensor: bool = False,
                   device: DeviceLike | None = None) -> ReplayDtype:
        """Sample observations by priority."""
        idxs, _ = self._sample_idxs(batch_size, replace)
        return self._output(self._gather(idxs, None).obs, to_tensor, device)

    def all(self, truncate_to: int | None = None) -> PrioritizedSample:
        """Every stored row oldest first, or a random subset of at most ``truncate_to``.
        Weights are all ``1``."""
        idxs = self._ordered_idxs()
        if truncate_to is not None:
            idxs = self.rng.choice(idxs, min(truncate_to, self.size), replace=False)
        return self._output(self._gather(idxs, np.ones(len(idxs), np.float32)), False, None)

    # ------- checkpointing ------- #
    def state_dict(self) -> dict[str, Any]:
        # beta is annealed during training, so it's state, not just config
        return {**super().state_dict(), 'max_priority': self.max_priority, 'beta': self.beta}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.max_priority = float(state['max_priority'])
        self.beta = float(state['beta'])


class PrioritizedReplayBuffer(_PrioritizedReplayBuffer[np.ndarray]):
    """Prioritized Sampling Replay Buffer"""
    kind = 'prioritized-numpy'


class TorchPrioritizedReplayBuffer(_PrioritizedReplayBuffer[torch.Tensor]):
    """Prioritized replay stored as torch tensors on ``device``.

    Every output is a tensor on the buffer's device (``idxs`` and ``weights``
    included), so ``to_tensor`` has no effect and ``device`` only moves the batch.
    """
    kind = 'prioritized-torch'

    def _gather(self, idxs: np.ndarray, weights: np.ndarray | None) -> PrioritizedSample:
        t_idxs = torch.as_tensor(idxs, dtype=torch.long, device=self.device)
        t_weights = None if weights is None else torch.as_tensor(weights, device=self.device)
        return super()._gather(t_idxs, t_weights)

    def _output(self, x: Any, to_tensor: bool, device: DeviceLike | None) -> Any:
        return super()._output(x, True, device)
