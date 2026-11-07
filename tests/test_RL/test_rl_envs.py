import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from pydantic import ValidationError
from data.envs import (Discretizer, DragonEnvConfig, RunningMeanStd, action_to_env, make_env_fn, make_vec_env,
                       obs_to_tensor, register_dragon_env, validate_env)
from modules.RL import FlattenObs, deterministic_action, make_dist_head

B = 5


class ButtonEnv(gym.Env):
    """MultiBinary(4) env recording the last button vector it received."""
    buttons = ['A', 'B', 'LEFT', 'RIGHT']
    observation_space = spaces.Box(0, 1, (1,), np.float32)
    action_space = spaces.MultiBinary(4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, np.float32), {}

    def step(self, action):
        self.last = action
        return np.zeros(1, np.float32), 0.0, False, False, {}


class TestEnvConfig:
    def test_needs_exactly_one_source(self):
        with pytest.raises(ValidationError):
            DragonEnvConfig()
        with pytest.raises(ValidationError):
            DragonEnvConfig(env_id='CartPole-v1', env_fn=ButtonEnv)

    def test_make_vec_env_same_step_autoreset(self):
        env = make_vec_env(DragonEnvConfig(env_id='CartPole-v1', n_envs=3))
        assert env.num_envs == 3
        assert env.metadata['autoreset_mode'] == gym.vector.AutoresetMode.SAME_STEP
        env.close()

    def test_env_fn_kwargs_and_wrappers(self):
        cfg = DragonEnvConfig(env_fn=gym.make, env_kwargs={'id': 'CartPole-v1'},
                              wrappers=[lambda e: gym.wrappers.TimeLimit(e, 7)], n_envs=2)
        single = make_env_fn(cfg)()
        assert isinstance(single, gym.wrappers.TimeLimit) and single._max_episode_steps == 7
        env = make_vec_env(cfg)
        assert env.num_envs == 2
        env.close()

    def test_validate_env(self):
        validate_env(DragonEnvConfig(env_id='CartPole-v1'))

    def test_register(self):
        register_dragon_env('DragonButtons-v0', ButtonEnv)
        register_dragon_env('DragonButtons-v0', ButtonEnv)  # re-register is fine
        assert isinstance(gym.make('DragonButtons-v0').unwrapped, ButtonEnv)


class TestSpaces:
    def test_obs_to_tensor_dtypes(self):
        out = obs_to_tensor({'f': np.zeros((2, 3)), 'img': np.zeros((2, 4), np.uint8),
                             'b': np.zeros(2, bool)})
        assert out['f'].dtype == torch.float32
        assert out['img'].dtype == torch.uint8
        assert out['b'].dtype == torch.float32

    @pytest.mark.parametrize('space', [
        spaces.Discrete(4, start=2),
        spaces.MultiDiscrete([3, 2, 5]),
        spaces.MultiBinary(3),
        spaces.Box(-1.0, 1.0, (2,), np.float32),
        spaces.Box(-1.0, 1.0, (2, 2), np.float32),
    ])
    def test_head_samples_are_valid_env_actions(self, space):
        head = make_dist_head(space, in_features=6)
        dist = head(torch.randn(B, 6))
        for action in (dist.sample(), deterministic_action(dist)):
            assert dist.log_prob(action).shape == (B,)
            env_actions = action_to_env(action * 10 if isinstance(space, spaces.Box) else action, space)
            assert all(space.contains(a) for a in env_actions)
        assert dist.entropy().shape == (B,)

    def test_bernoulli_deterministic_at_half(self):
        dist = make_dist_head(spaces.MultiBinary(3), 4)(torch.zeros(B, 4))
        assert not torch.isnan(deterministic_action(dist)).any()

    def test_unsupported_action_space(self):
        with pytest.raises(NotImplementedError):
            make_dist_head(spaces.Dict({'a': spaces.Discrete(2)}), 4)
        with pytest.raises(NotImplementedError):
            action_to_env(torch.zeros(1), spaces.Dict({'a': spaces.Discrete(2)}))

    def test_flatten_obs(self):
        assert FlattenObs(spaces.Box(0, 1, (2, 3)))(torch.zeros(B, 2, 3)).shape == (B, 6)
        onehot = FlattenObs(spaces.Discrete(4, start=1))(torch.tensor([1, 4]))
        assert onehot.tolist() == [[1, 0, 0, 0], [0, 0, 0, 1]]
        with pytest.raises(TypeError, match='encoder'):
            FlattenObs(spaces.Dict({'a': spaces.Discrete(2)}))


class TestDiscretizer:
    def test_maps_combos_by_name_and_index(self):
        env = Discretizer(ButtonEnv(), [[], ['LEFT'], ['RIGHT', 'B'], [0]])
        assert env.action_space == spaces.Discrete(4)
        env.reset()
        env.step(2)
        assert env.unwrapped.last.tolist() == [0, 1, 0, 1]
        env.step(3)
        assert env.unwrapped.last.tolist() == [1, 0, 0, 0]

    def test_rejects_non_multibinary(self):
        with pytest.raises(TypeError):
            Discretizer(gym.make('CartPole-v1'), [[0]])


class TestRunningMeanStd:
    def test_matches_numpy_over_batches(self):
        data = np.random.default_rng(0).normal(3.0, 2.0, size=(300, 2))
        rms = RunningMeanStd((2,), epsilon=0.0)
        for chunk in np.split(data, 6):
            rms.update(chunk)
        np.testing.assert_allclose(rms.mean, data.mean(0))
        np.testing.assert_allclose(rms.var, data.var(0))

    def test_state_round_trip(self):
        rms = RunningMeanStd((2,))
        rms.update(np.ones((4, 2)))
        other = RunningMeanStd((2,))
        other.load_state_dict(rms.state_dict())
        np.testing.assert_allclose(other.mean, rms.mean)
        assert other.count == rms.count
