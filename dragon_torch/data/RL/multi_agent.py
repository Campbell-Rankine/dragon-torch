"""Multi-agent counterparts of EnvCollector, RolloutSource and EpisodeSource.

Everything is per agent group ({group: [n_envs, agents_in_group, ...]}, see
dragon_torch.data.envs.multi_agent), with an `alive` mask marking which slots
hold a real agent. global_step counts env steps (n_envs per vector step), as in
the single-agent sources, so total_timesteps means the same thing in both.
"""
import time
from dataclasses import dataclass
from typing import Any, Iterator, NamedTuple, Optional
import numpy as np
import torch
from gymnasium import spaces
from dragon_torch.common.torch.tree import tree_map, tree_stack
from dragon_torch.data.envs.multi_agent import AgentGroups, MultiAgentEnvConfig, MultiAgentVectorEnv
from dragon_torch.data.envs.normalize import ObsNormalizer, RewardScaler
from dragon_torch.data.envs.spaces import action_to_env, obs_to_tensor
from .collector import Actor, ValueActor
from .episodes import EpisodeSource
from .rollout import compute_gae

TARGET_KEYS = ('actions', 'log_prob', 'values', 'advantages', 'returns')


class MultiAgentEpisode(NamedTuple):
    """A finished episode of one env. raw / shaped returns are the team
    return (MultiAgentEnvConfig.team_return over agents); group_returns[g] is
    the mean raw return of g's agents that took part."""
    raw_return: float
    length: int
    shaped_return: float
    group_returns: dict[str, float]

    def record(self) -> dict[str, Any]:
        return {'return': self.raw_return, 'shaped_return': self.shaped_return,
                'length': self.length, 'group_returns': dict(self.group_returns)}


@dataclass
class MultiAgentTransition:
    """One vector step; every field (but env_done / extras) is {group: [N, n_g, ...]}.
    `alive` marks the agents that acted: only those slots are real transitions."""
    obs: dict[str, Any]
    alive: dict[str, np.ndarray]
    action: dict[str, torch.Tensor]
    reward: dict[str, np.ndarray]
    next_obs: dict[str, Any]
    terminated: dict[str, np.ndarray]
    truncated: dict[str, np.ndarray]
    env_done: np.ndarray
    extras: dict[str, dict[str, torch.Tensor]]

    @property
    def done(self) -> dict[str, np.ndarray]:
        return {g: self.terminated[g] | self.truncated[g] for g in self.terminated}


class MultiAgentCollector:
    """Steps a MultiAgentVectorEnv with a multi-agent Actor (e.g.
    MultiAgentRLModel). Obs normalisation and reward scaling are per group
    (shared by the group's agents, as its network is)."""

    def __init__(self, env_cfg: MultiAgentEnvConfig, normalize_obs: bool = False,
                 normalize_reward: bool = False, gamma: float = 0.99, update_stats: bool = True):
        self.env_cfg = env_cfg
        self.env = MultiAgentVectorEnv(env_cfg)
        self.groups: AgentGroups = self.env.groups
        self.n_envs = self.env.num_envs
        if normalize_obs:
            bad = {g: s for g, s in self.groups.observation_spaces.items() if not isinstance(s, spaces.Box)}
            if bad:
                raise TypeError(f"normalize_obs needs Box observation spaces; groups {sorted(bad)} "
                                "aren't; normalise inside your encoder instead")
        self.obs_normalizers: dict[str, ObsNormalizer] = {
            g: ObsNormalizer(s.shape) for g, s in self.groups.observation_spaces.items()
        } if normalize_obs else {}
        self.reward_scalers: dict[str, RewardScaler] = {
            g: RewardScaler(self.n_envs * self.groups.size(g), gamma) for g in self.groups.names
        } if normalize_reward else {}
        self.update_stats = update_stats
        self.global_step = 0
        self.shaped = env_cfg.shaped
        self._obs: Optional[dict[str, Any]] = None
        self._alive: dict[str, np.ndarray] = {}
        zeros = lambda: {g: np.zeros((self.n_envs, self.groups.size(g))) for g in self.groups.names}
        self._ep_return, self._ep_raw_return = zeros(), zeros()
        self._ep_joined = {g: np.zeros((self.n_envs, self.groups.size(g)), bool) for g in self.groups.names}
        self._ep_length = np.zeros(self.n_envs, np.int64)
        self._finished: list[MultiAgentEpisode] = []
        self._clock = (time.perf_counter(), 0)

    @property
    def observation_spaces(self) -> dict[str, spaces.Space]:
        return self.groups.observation_spaces

    @property
    def action_spaces(self) -> dict[str, spaces.Space]:
        return self.groups.action_spaces

    # ------- obs ------- #
    def _normalize(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {g: self.obs_normalizers[g].normalize(o) if g in self.obs_normalizers else o
                for g, o in obs.items()}

    def _observe(self, obs: dict[str, Any], alive: dict[str, np.ndarray]) -> dict[str, Any]:
        if self.update_stats:
            for g, norm in self.obs_normalizers.items():
                if alive[g].any():
                    norm.update(obs[g][alive[g]])
        return self._normalize(obs)

    def _reset_episode_stats(self, envs: Any = slice(None)) -> None:
        for g in self.groups.names:
            self._ep_return[g][envs] = 0
            self._ep_raw_return[g][envs] = 0
            self._ep_joined[g][envs] = False
        self._ep_length[envs] = 0

    def reset(self, seed: Optional[int] = None) -> dict[str, Any]:
        raw, alive = self.env.reset(seed=self.env_cfg.seed if seed is None else seed)
        self._obs, self._alive = self._observe(raw, alive), alive
        self._reset_episode_stats()
        return self._obs

    @property
    def obs(self) -> dict[str, Any]:
        """Current (normalised) obs; resets the env on first use."""
        if self._obs is None:
            self.reset()
        return self._obs

    @property
    def alive(self) -> dict[str, np.ndarray]:
        self.obs
        return self._alive

    # ------- stepping ------- #
    def _team(self, returns: dict[str, np.ndarray], joined: dict[str, np.ndarray], i: int) -> float:
        values = np.concatenate([returns[g][i][joined[g][i]] for g in self.groups.names])
        if values.size == 0:
            return 0.0
        return float(values.sum() if self.env_cfg.team_return == 'sum' else values.mean())

    def step(self, actor: Actor, deterministic: bool = False) -> MultiAgentTransition:
        obs, alive = self.obs, self.alive
        out = actor.act(obs, deterministic=deterministic)
        res = self.env.step({g: action_to_env(a, self.action_spaces[g]) for g, a in out.action.items()})

        for g in self.groups.names:
            self._ep_return[g] += res.reward[g] * alive[g]
            self._ep_raw_return[g] += res.raw_reward[g] * alive[g]
            self._ep_joined[g] |= alive[g]
        self._ep_length += 1
        for i in np.flatnonzero(res.env_done):
            group_returns = {g: float(self._ep_raw_return[g][i][self._ep_joined[g][i]].mean())
                             for g in self.groups.names if self._ep_joined[g][i].any()}
            self._finished.append(MultiAgentEpisode(
                raw_return=self._team(self._ep_raw_return, self._ep_joined, i),
                length=int(self._ep_length[i]),
                shaped_return=self._team(self._ep_return, self._ep_joined, i),
                group_returns=group_returns))
            self._reset_episode_stats(i)

        reward = {}
        for g in self.groups.names:
            r = res.reward[g] * alive[g]
            if g in self.reward_scalers:
                done = (res.terminated[g] | res.truncated[g]).reshape(-1)
                r = self.reward_scalers[g](r.reshape(-1), done, update=self.update_stats).reshape(r.shape)
            reward[g] = np.asarray(r, np.float32)

        self.global_step += self.n_envs
        self._obs, self._alive = self._observe(res.next_obs, res.next_alive), res.next_alive
        return MultiAgentTransition(
            obs=obs, alive=alive, action=out.action, reward=reward,
            next_obs=self._normalize(res.final_obs), terminated=res.terminated,
            truncated=res.truncated, env_done=res.env_done, extras=out.extras)

    # ------- stats / state ------- #
    def borrow_stats(self, other: Any) -> None:
        self.obs_normalizers = dict(getattr(other, 'obs_normalizers', {}) or {})

    def drain_episodes(self) -> list[MultiAgentEpisode]:
        finished, self._finished = self._finished, []
        return finished

    def metrics(self) -> dict[str, float]:
        now, steps = time.perf_counter(), self.global_step
        then, steps_then = self._clock
        self._clock = (now, steps)
        record = {'global_step': steps, 'steps_per_sec': (steps - steps_then) / max(now - then, 1e-9)}
        episodes = self.drain_episodes()
        if episodes:
            record['episode_return'] = float(np.mean([e.raw_return for e in episodes]))
            if self.shaped:
                record['episode_return_shaped'] = float(np.mean([e.shaped_return for e in episodes]))
            record['episode_length'] = float(np.mean([e.length for e in episodes]))
            record['episodes'] = len(episodes)
            for g in self.groups.names:
                values = [e.group_returns[g] for e in episodes if g in e.group_returns]
                if values:
                    record[f'episode_return/{g}'] = float(np.mean(values))
        return record

    def state_dict(self) -> dict[str, Any]:
        return {'global_step': self.global_step,
                'obs_normalizers': {g: n.state_dict() for g, n in self.obs_normalizers.items()},
                'reward_scalers': {g: s.state_dict() for g, s in self.reward_scalers.items()}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.global_step = state['global_step']
        self._clock = (time.perf_counter(), self.global_step)
        for name in ('obs_normalizers', 'reward_scalers'):
            for g, target in getattr(self, name).items():
                if g in (state.get(name) or {}):
                    target.load_state_dict(state[name][g])

    def close(self) -> None:
        self.env.close()


class MultiAgentRolloutSource:
    """On-policy multi-agent data source for MultiAgentRLModel.fit.

    Each iteration collects n_steps x n_envs env steps, computes GAE per agent
    (an agent's episode ends where it's done, even if its env goes on), then
    yields update_epochs x n_minibatches minibatches. Minibatches sample env
    steps, so a step's agents stay together; within each, a group's live agents
    are flattened into one batch:
        {'obs': {group: [B_g, ...]},
         'targets': {group: {'actions', 'log_prob', 'values', 'advantages', 'returns'}}}
    Groups with no live agents in a minibatch are left out.
    """

    def __init__(self, env_cfg: MultiAgentEnvConfig, n_steps: int = 128, gamma: float = 0.99,
                 gae_lambda: float = 0.95, update_epochs: int = 4, n_minibatches: int = 4,
                 normalize_obs: bool = False, normalize_reward: bool = False):
        self.collector = MultiAgentCollector(env_cfg, normalize_obs, normalize_reward, gamma)
        if n_steps * self.collector.n_envs < n_minibatches:
            raise ValueError("n_steps * n_envs must be >= n_minibatches")
        self.n_steps, self.gamma, self.gae_lambda = n_steps, gamma, gae_lambda
        self.update_epochs, self.n_minibatches = update_epochs, n_minibatches
        self.actor: Optional[ValueActor] = None
        self._rng = torch.Generator().manual_seed(env_cfg.seed)

    def bind(self, model: ValueActor) -> None:
        if not isinstance(model, ValueActor):
            raise TypeError(f"MultiAgentRolloutSource needs a model with act() and value(), "
                            f"got {type(model)}")
        self.actor = model

    @property
    def groups(self) -> AgentGroups:
        return self.collector.groups

    @property
    def steps_per_epoch(self) -> int:
        return self.n_steps * self.collector.n_envs

    @property
    def global_step(self) -> int:
        return self.collector.global_step

    # ------- collection ------- #
    def collect(self) -> dict[str, dict[str, Any]]:
        """One rollout: {group: {key: [T, N, n_g, ...]}} with 'valid' marking
        real (live-agent) transitions, plus advantages / returns."""
        if self.actor is None:
            raise RuntimeError("MultiAgentRolloutSource is unbound; fit()/evaluate() bind the model")
        names = self.groups.names
        keys = ('obs', 'actions', 'rewards', 'terminated', 'dones', 'log_prob', 'values', 'valid')
        buf = {g: {k: [] for k in keys} for g in names}
        truncation_values = []  # (t, {group: mask}, {group: V(final obs)})
        for t in range(self.n_steps):
            tr = self.collector.step(self.actor)
            missing = {'log_prob', 'value'} - set(tr.extras)
            if missing:
                raise KeyError(f"act() must return extras {sorted(missing)} for on-policy training")
            done = tr.done
            for g in names:
                b = buf[g]
                b['obs'].append(obs_to_tensor(tr.obs[g]))
                b['actions'].append(tr.action[g])
                b['rewards'].append(torch.as_tensor(tr.reward[g]))
                b['terminated'].append(torch.as_tensor(tr.terminated[g]))
                b['dones'].append(torch.as_tensor(done[g]))
                b['log_prob'].append(tr.extras['log_prob'][g])
                b['values'].append(tr.extras['value'][g])
                b['valid'].append(torch.as_tensor(tr.alive[g]))
            if any(tr.truncated[g].any() for g in names):
                masks = {g: torch.as_tensor(tr.truncated[g]) for g in names}
                truncation_values.append((t, masks, self.actor.value(tr.next_obs)))

        last_values = self.actor.value(self.collector.obs)
        out = {}
        for g in names:
            b = buf[g]
            values = torch.stack(b['values']).float()
            next_values = torch.cat([values[1:], last_values[g][None].float()])
            for t, masks, v in truncation_values:
                next_values[t] = torch.where(masks[g], v[g].float(), next_values[t])
            rewards = torch.stack(b['rewards'])
            advantages, returns = compute_gae(rewards, values, next_values, torch.stack(b['terminated']),
                                              torch.stack(b['dones']), self.gamma, self.gae_lambda)
            out[g] = {'obs': tree_stack(b['obs']), 'actions': torch.stack(b['actions']),
                      'log_prob': torch.stack(b['log_prob']), 'values': values,
                      'advantages': advantages, 'returns': returns, 'rewards': rewards,
                      'valid': torch.stack(b['valid'])}
        return out

    def __iter__(self) -> Iterator[dict[str, Any]]:
        data = self.collect()
        total = self.steps_per_epoch
        flat = lambda x: x.reshape(total, *x.shape[2:])  # [T, N, n_g, ...] -> [T*N, n_g, ...]
        obs = {g: tree_map(flat, d['obs']) for g, d in data.items()}
        targets = {g: {k: flat(d[k]) for k in TARGET_KEYS} for g, d in data.items()}
        valid = {g: flat(d['valid']) for g, d in data.items()}
        size = total // self.n_minibatches
        for _ in range(self.update_epochs):
            perm = torch.randperm(total, generator=self._rng)
            for i in range(self.n_minibatches):
                idx = perm[i * size:(i + 1) * size]
                batch_obs, batch_targets = {}, {}
                for g in data:
                    mask = valid[g][idx]
                    if not mask.any():
                        continue
                    pick = lambda x: x[idx][mask]
                    batch_obs[g] = tree_map(pick, obs[g])
                    batch_targets[g] = {k: pick(v) for k, v in targets[g].items()}
                if batch_obs:
                    yield {'obs': batch_obs, 'targets': batch_targets}

    # ------- hooks ------- #
    def metrics(self) -> dict[str, float]:
        return self.collector.metrics()

    def state_dict(self) -> dict[str, Any]:
        return {'collector': self.collector.state_dict(), 'rng': self._rng.get_state()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.collector.load_state_dict(state['collector'])
        self._rng.set_state(state['rng'])

    def close(self) -> None:
        self.collector.close()


class MultiAgentEpisodeSource(EpisodeSource):
    """EpisodeSource over a multi-agent env: yields {'return', 'shaped_return',
    'length', 'group_returns'} per finished env episode, 'return' being the
    team return (MultiAgentEnvConfig.team_return)."""

    def __init__(self, env_cfg: MultiAgentEnvConfig, n_episodes: int = 10,
                 deterministic: bool = True, max_steps: int = 1_000_000):
        super().__init__(env_cfg, n_episodes=n_episodes, deterministic=deterministic,
                         max_steps=max_steps)

    def make_collector(self, env_cfg: MultiAgentEnvConfig) -> MultiAgentCollector:
        return MultiAgentCollector(env_cfg, update_stats=False)

    @property
    def groups(self) -> AgentGroups:
        return self.collector.groups
