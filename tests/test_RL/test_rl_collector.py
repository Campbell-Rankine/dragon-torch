import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from dragon_torch.data.envs import DragonEnvConfig
from dragon_torch.data.RL import ActOutput, EnvCollector, EpisodeSource, RolloutSource, compute_gae


class CounterEnv(gym.Env):
    """obs = [t], reward 1 per step; terminates when t reaches terminate_at."""
    observation_space = spaces.Box(0, np.inf, (1,), np.float32)
    action_space = spaces.Discrete(2)

    def __init__(self, terminate_at=None):
        self.terminate_at = terminate_at

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([0.0], np.float32), {}

    def step(self, action):
        self.t += 1
        terminated = self.terminate_at is not None and self.t >= self.terminate_at
        return np.array([self.t], np.float32), 1.0, terminated, False, {}


def counter_cfg(n_envs=1, truncate_at=None, terminate_at=None):
    wrappers = [lambda e: gym.wrappers.TimeLimit(e, truncate_at)] if truncate_at else []
    return DragonEnvConfig(env_fn=CounterEnv, env_kwargs={'terminate_at': terminate_at},
                           wrappers=wrappers, n_envs=n_envs)


class CountingActor:
    """Always action 0; V(s) = t, so bootstrapping errors show up in advantages."""

    def act(self, obs, deterministic=False):
        obs = torch.as_tensor(obs)
        n = obs.shape[0]
        return ActOutput(torch.zeros(n, dtype=torch.long),
                         {'log_prob': torch.zeros(n), 'value': obs[:, 0].float()})

    def value(self, obs):
        return torch.as_tensor(obs)[:, 0].float()


class TestEnvCollector:
    def test_truncation_substitutes_final_obs(self):
        col = EnvCollector(counter_cfg(truncate_at=3))
        for _ in range(3):
            tr = col.step(CountingActor())
        assert tr.truncated.tolist() == [True] and tr.terminated.tolist() == [False]
        assert tr.next_obs.tolist() == [[3.0]]   # true successor
        assert col.obs.tolist() == [[0.0]]        # what the policy sees next
        assert col.drain_episodes() == [(3.0, 3)]
        assert col.global_step == 3

    def test_metrics(self):
        col = EnvCollector(counter_cfg(n_envs=2, truncate_at=2))
        for _ in range(4):
            col.step(CountingActor())
        m = col.metrics()
        assert m['global_step'] == 8 and m['episodes'] == 4 and m['episode_return'] == 2.0
        assert 'episode_return' not in col.metrics()  # drained

    def test_obs_normalization_and_state(self):
        col = EnvCollector(counter_cfg(truncate_at=5), normalize_obs=True)
        for _ in range(10):
            tr = col.step(CountingActor())
        assert tr.obs.dtype == np.float32
        other = EnvCollector(counter_cfg(truncate_at=5), normalize_obs=True)
        other.load_state_dict(col.state_dict())
        np.testing.assert_allclose(other.obs_normalizer.rms.mean, col.obs_normalizer.rms.mean)
        assert other.global_step == 10

    def test_normalize_obs_needs_box(self):
        with pytest.raises(TypeError, match='Box'):
            EnvCollector(DragonEnvConfig(env_id='FrozenLake-v1'), normalize_obs=True)


class TestGAE:
    def test_lambda(self):
        adv, ret = compute_gae(torch.ones(2, 1), torch.zeros(2, 1), torch.zeros(2, 1),
                               torch.zeros(2, 1, dtype=torch.bool),
                               torch.zeros(2, 1, dtype=torch.bool), gamma=1.0, gae_lambda=0.5)
        assert adv[:, 0].tolist() == [1.5, 1.0]
        assert torch.equal(ret, adv)

    def test_truncation_bootstraps_from_final_obs(self):
        # TD(0) with V(s)=t, gamma=0.5: adv_t = 1 + 0.5 * V(next) - V(s)
        src = RolloutSource(counter_cfg(truncate_at=3), n_steps=4, gamma=0.5, gae_lambda=0.0,
                            update_epochs=1, n_minibatches=1)
        src.bind(CountingActor())
        adv = src.collect()['advantages'][:, 0].tolist()
        # step 2 is truncated: bootstraps V(final obs=3)=3, not V(reset obs)=0
        assert adv == [1.5, 1.0, 0.5, 1.5]

    def test_termination_does_not_bootstrap(self):
        src = RolloutSource(counter_cfg(terminate_at=3), n_steps=3, gamma=0.5, gae_lambda=0.0,
                            update_epochs=1, n_minibatches=1)
        src.bind(CountingActor())
        assert src.collect()['advantages'][:, 0].tolist() == [1.5, 1.0, -1.0]


class TestRolloutSource:
    def test_minibatches(self):
        src = RolloutSource(counter_cfg(n_envs=2, truncate_at=4), n_steps=8,
                            update_epochs=3, n_minibatches=4)
        src.bind(CountingActor())
        batches = list(src)
        assert len(batches) == 12
        assert batches[0]['obs'].shape == (4, 1)
        assert set(batches[0]['targets']) == {'actions', 'log_prob', 'values', 'advantages', 'returns'}
        assert src.steps_per_epoch == 16 and src.global_step == 16

    def test_unbound(self):
        src = RolloutSource(counter_cfg(truncate_at=4), n_steps=2, n_minibatches=1)
        with pytest.raises(RuntimeError, match='unbound'):
            list(src)
        with pytest.raises(TypeError):
            src.bind(object())

    def test_missing_extras(self):
        class NoExtras(CountingActor):
            def act(self, obs, deterministic=False):
                return ActOutput(torch.zeros(len(obs), dtype=torch.long))

        src = RolloutSource(counter_cfg(truncate_at=4), n_steps=2, n_minibatches=1)
        src.bind(NoExtras())
        with pytest.raises(KeyError, match='log_prob'):
            list(src)


class TestEpisodeSource:
    def test_yields_n_episodes(self):
        src = EpisodeSource(counter_cfg(n_envs=2, truncate_at=3), n_episodes=3)
        src.bind(CountingActor())
        eps = list(src)
        assert eps == [{'return': 3.0, 'length': 3}] * 3
        assert list(src) == eps  # re-seeded / reset each iteration

    def test_never_ending_env_guard(self):
        src = EpisodeSource(counter_cfg(), n_episodes=1, max_steps=5)
        src.bind(CountingActor())
        with pytest.raises(RuntimeError, match='TimeLimit'):
            list(src)
