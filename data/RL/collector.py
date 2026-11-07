"""Env interaction shared by every RL data source.

EnvCollector steps a vectorised env with any `Actor` and returns `Transition`s
that already carry the *true* successor obs (info['final_obs'] substituted on
episode end). On-policy sources (RolloutSource) and off-policy sources (a
replay source storing (obs, action, reward, next_obs, terminated)) both build
on it, so autoreset, normalisation, episode stats and step counting live here once.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable
import numpy as np
import torch
from gymnasium import spaces
from common.torch.tree import tree_map, tree_set
from data.envs.config import DragonEnvConfig, make_vec_env
from data.envs.normalize import ObsNormalizer, RewardScaler
from data.envs.spaces import action_to_env


@dataclass
class ActOutput:
    action: torch.Tensor  # batched, in the policy's own space (what buffers store)
    extras: dict[str, torch.Tensor] = field(default_factory=dict)  # e.g. log_prob, value


@runtime_checkable
class Actor(Protocol):
    """What sources need from a model. Off-policy models implement exploration
    (e.g. epsilon-greedy) inside `act`."""

    def act(self, obs: Any, deterministic: bool = False) -> ActOutput: ...


@runtime_checkable
class ValueActor(Actor, Protocol):
    """Actor with a state-value estimate, needed for GAE bootstrapping."""

    def value(self, obs: Any) -> torch.Tensor: ...


@dataclass
class Transition:
    """One vector step. obs / next_obs are (normalised) numpy batches."""
    obs: Any
    action: torch.Tensor
    reward: np.ndarray       # float32, scaled if reward normalisation is on
    next_obs: Any            # true successor: final obs where an episode ended
    terminated: np.ndarray
    truncated: np.ndarray
    extras: dict[str, torch.Tensor]

    @property
    def done(self) -> np.ndarray:
        return self.terminated | self.truncated


class EnvCollector:
    def __init__(self, env_cfg: DragonEnvConfig, normalize_obs: bool = False,
                 normalize_reward: bool = False, gamma: float = 0.99,
                 update_stats: bool = True):
        self.env_cfg = env_cfg
        self.env = make_vec_env(env_cfg)
        self.n_envs = self.env.num_envs
        self.observation_space = self.env.single_observation_space
        self.action_space = self.env.single_action_space
        if normalize_obs and not isinstance(self.observation_space, spaces.Box):
            raise TypeError(f"normalize_obs needs a Box observation space, got "
                            f"{self.observation_space}; normalise inside your encoder instead")
        self.obs_normalizer: Optional[ObsNormalizer] = \
            ObsNormalizer(self.observation_space.shape) if normalize_obs else None
        self.reward_scaler: Optional[RewardScaler] = \
            RewardScaler(self.n_envs, gamma) if normalize_reward else None
        # False for evaluation: use (shared) stats without changing them
        self.update_stats = update_stats
        self.global_step = 0
        self._obs: Any = None
        self._ep_return = np.zeros(self.n_envs)
        self._ep_length = np.zeros(self.n_envs, np.int64)
        self._finished: list[tuple[float, int]] = []
        self._clock = (time.perf_counter(), 0)

    # ------- obs ------- #
    def _normalize(self, raw: Any) -> Any:
        return raw if self.obs_normalizer is None else self.obs_normalizer.normalize(raw)

    def _observe(self, raw: Any) -> Any:
        if self.obs_normalizer is not None and self.update_stats:
            self.obs_normalizer.update(raw)
        return self._normalize(raw)

    def reset(self, seed: Optional[int] = None) -> Any:
        raw, _ = self.env.reset(seed=self.env_cfg.seed if seed is None else seed)
        self._obs = self._observe(raw)
        self._ep_return[:] = 0
        self._ep_length[:] = 0
        return self._obs

    @property
    def obs(self) -> Any:
        """Current (normalised) obs batch; resets the env on first use."""
        if self._obs is None:
            self.reset()
        return self._obs

    # ------- stepping ------- #
    def step(self, actor: Actor, deterministic: bool = False) -> Transition:
        obs = self.obs
        out = actor.act(obs, deterministic=deterministic)
        raw_next, reward, terminated, truncated, info = self.env.step(
            action_to_env(out.action, self.action_space))
        done = terminated | truncated

        next_policy_obs = self._observe(raw_next)  # reset obs where done
        next_obs = next_policy_obs
        if done.any():
            true_next = tree_map(np.copy, raw_next)
            for i in np.flatnonzero(done):
                tree_set(true_next, i, info['final_obs'][i])
            next_obs = self._normalize(true_next)

        self._ep_return += reward
        self._ep_length += 1
        for i in np.flatnonzero(done):
            self._finished.append((float(self._ep_return[i]), int(self._ep_length[i])))
            self._ep_return[i], self._ep_length[i] = 0, 0

        train_reward = reward if self.reward_scaler is None else \
            self.reward_scaler(reward, done, update=self.update_stats)
        self.global_step += self.n_envs
        self._obs = next_policy_obs
        return Transition(obs=obs, action=out.action, reward=np.asarray(train_reward, np.float32),
                          next_obs=next_obs, terminated=terminated, truncated=truncated,
                          extras=out.extras)

    # ------- stats / state ------- #
    def drain_episodes(self) -> list[tuple[float, int]]:
        """(raw return, length) of episodes finished since the last drain."""
        finished, self._finished = self._finished, []
        return finished

    def metrics(self) -> dict[str, float]:
        now, steps = time.perf_counter(), self.global_step
        then, steps_then = self._clock
        self._clock = (now, steps)
        record = {'global_step': steps, 'steps_per_sec': (steps - steps_then) / max(now - then, 1e-9)}
        episodes = self.drain_episodes()
        if episodes:
            record['episode_return'] = float(np.mean([r for r, _ in episodes]))
            record['episode_length'] = float(np.mean([l for _, l in episodes]))
            record['episodes'] = len(episodes)
        return record

    def state_dict(self) -> dict[str, Any]:
        # the env itself can't be checkpointed; a resumed run starts fresh episodes
        return {'global_step': self.global_step,
                'obs_normalizer': None if self.obs_normalizer is None else self.obs_normalizer.state_dict(),
                'reward_scaler': None if self.reward_scaler is None else self.reward_scaler.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.global_step = state['global_step']
        self._clock = (time.perf_counter(), self.global_step)
        for name in ('obs_normalizer', 'reward_scaler'):
            target = getattr(self, name)
            if target is not None and state.get(name) is not None:
                target.load_state_dict(state[name])

    def close(self) -> None:
        self.env.close()
