"""Running normalisation owned by the collector (not by env wrappers), so the
same stats apply to info['final_obs'], are shared with evaluation, and are
checkpointed as tensors."""
from typing import Any
import numpy as np
import torch


class RunningMeanStd:
    """Parallel-algorithm running mean / variance over axis 0 of each batch."""

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float64)
        batch_mean, batch_var, n = x.mean(0), x.var(0), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + n
        # asarray: with shape () numpy arithmetic returns np.float64 scalars,
        # which torch.from_numpy (state_dict) rejects
        self.mean = np.asarray(self.mean + delta * n / total, np.float64)
        m2 = self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total
        self.var = np.asarray(m2 / total, np.float64)
        self.count = total

    def state_dict(self) -> dict[str, Any]:
        return {'mean': torch.from_numpy(self.mean.copy()),
                'var': torch.from_numpy(self.var.copy()), 'count': float(self.count)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state['mean'], np.float64)
        self.var = np.asarray(state['var'], np.float64)
        self.count = float(state['count'])


class ObsNormalizer:
    def __init__(self, shape: tuple[int, ...], clip: float = 10.0, epsilon: float = 1e-8):
        self.rms = RunningMeanStd(shape)
        self.clip, self.epsilon = clip, epsilon

    def update(self, obs: np.ndarray) -> None:
        self.rms.update(obs)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        out = (obs - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon)
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return self.rms.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rms.load_state_dict(state)


class RewardScaler:
    """Divide rewards by the std of the discounted return (as in gymnasium's
    NormalizeReward), keeping a per-env running return."""

    def __init__(self, n_envs: int, gamma: float, clip: float = 10.0, epsilon: float = 1e-8):
        self.rms = RunningMeanStd(())
        self.returns = np.zeros(n_envs, np.float64)
        self.gamma, self.clip, self.epsilon = gamma, clip, epsilon

    def __call__(self, rewards: np.ndarray, dones: np.ndarray, update: bool = True) -> np.ndarray:
        self.returns = self.returns * self.gamma + rewards
        if update:
            self.rms.update(self.returns)
        self.returns[dones] = 0.0
        scaled = rewards / np.sqrt(self.rms.var + self.epsilon)
        return np.clip(scaled, -self.clip, self.clip)

    def state_dict(self) -> dict[str, Any]:
        return self.rms.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rms.load_state_dict(state)
