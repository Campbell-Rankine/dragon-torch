from typing import Any, Generic
import numpy as np
import torch
from .base import _ReplayBuffer, DeviceLike, ReplaySample, ReplayDtype

class _UniformReplayBuffer(_ReplayBuffer[ReplayDtype], Generic[ReplayDtype]):
    """Uniform sampling shared by the numpy and torch buffers.

    Indices are always drawn from the seeded numpy RNG, so both backends draw
    the same batches for the same seed.
    """

    def add(self, obs: Any, action: Any, reward: Any, next_obs: Any,
            terminated: Any, truncated: Any) -> None:
        """Add one env step, or one vector-env step with a leading ``n_envs`` dim.

        Args:
            obs: Observation, ``obs_dim`` or ``(n_envs, *obs_dim)``; numpy or torch
            action: Action
            reward: Reward
            next_obs: True successor observation (the final obs where an episode ended)
            terminated: Episode ended in a terminal state; no bootstrapping
            truncated: Episode was cut short (e.g. time limit); still bootstraps
        """
        self._store(self._batch(obs=obs, next_obs=next_obs, actions=action, rewards=reward,
                                terminated=terminated, truncated=truncated))

    def sample(self, batch_size: int, replace: bool = True, use_cer: bool = False,
               to_tensor: bool = False, device: DeviceLike | None = None) -> ReplaySample:
        """Sample a batch of experiences from the buffer.

        Args:
            batch_size: Batch size
            replace: Whether to sample with replacement
            use_cer: Combined experience replay: the newest transition is always in the batch
            to_tensor: Whether to convert the data to PyTorch tensors
            device: Device for the tensors; defaults to the buffer's device

        Returns:
            A ``ReplaySample`` of (obs, actions, rewards, next_obs, terminated, truncated, idxs)
        """
        idxs = self._sample_idxs(batch_size, replace)
        if use_cer and batch_size > 0:
            idxs[0] = (self.ptr - 1) % self.maxlength
        return self._output(self._gather(idxs), to_tensor, device)

    def sample_obs(self, batch_size: int, replace: bool = True, to_tensor: bool = False,
                   device: DeviceLike | None = None) -> ReplayDtype:
        """Sample a batch of observations from the buffer.

        Args:
            batch_size: Batch size
            replace: Whether to sample with replacement
            to_tensor: Whether to convert the data to PyTorch tensors
            device: Device for the tensor; defaults to the buffer's device

        Returns:
            A batch of observations
        """
        idxs = self._sample_idxs(batch_size, replace)
        return self._output(self._gather(idxs).obs, to_tensor, device)

    def all(self, truncate_to: int | None = None) -> ReplaySample:
        """Get all the data in the buffer, oldest first.

        Args:
            truncate_to: If set, a random subset of at most this many transitions
                (without replacement) instead

        Returns:
            A ``ReplaySample`` of (obs, actions, rewards, next_obs, terminated, truncated, idxs)
        """
        idxs = self._ordered_idxs()
        if truncate_to is not None:
            idxs = self.rng.choice(idxs, min(truncate_to, self.size), replace=False)
        return self._gather(idxs)


    def _sample_idxs(self, batch_size: int, replace: bool) -> np.ndarray:
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        # positions 0..size-1 are always the filled ones: the buffer only wraps once full
        return self.rng.choice(self.size, batch_size, replace=replace)

    def _gather(self, idxs: np.ndarray) -> ReplaySample:
        return ReplaySample(self.obs[idxs], self.actions[idxs], self.rewards[idxs],
                            self.next_obs[idxs], self.terminated[idxs], self.truncated[idxs], idxs)

    def _output(self, x: Any, to_tensor: bool, device: DeviceLike | None) -> Any:
        """Convert a gathered array or ``ReplaySample`` for the caller."""
        if not to_tensor:
            return x
        device = self.device if device is None else device
        if isinstance(x, ReplaySample):
            return ReplaySample(*(torch.as_tensor(v, device=device) for v in x))
        return torch.as_tensor(x, device=device)


class UniformReplayBuffer(_UniformReplayBuffer[np.ndarray]):
    """Uniform Sampling Replay Buffer"""
    kind = 'uniform-numpy'


class TorchUniformReplayBuffer(_UniformReplayBuffer[torch.Tensor]):
    """Uniform replay stored as torch tensors on ``device``.

    Sampling returns tensors on the buffer's device (``idxs`` included), so
    ``to_tensor`` has no effect and ``device`` only moves the batch elsewhere.
    Indices are drawn from the same seeded numpy RNG as ``UniformReplayBuffer``.
    """
    kind = 'torch-uniform'

    def _gather(self, idxs: np.ndarray) -> ReplaySample:
        return super()._gather(torch.as_tensor(idxs, dtype=torch.long, device=self.device))

    def _output(self, x: Any, to_tensor: bool, device: DeviceLike | None) -> Any:
        return super()._output(x, True, device)
