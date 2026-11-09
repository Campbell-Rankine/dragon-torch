"""End to end: the ways docs/envs.md says to customise a prebuilt env all
train through the RL trainers, and wrapper-changed spaces reach the network."""
import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.envs.classic_control import CartPoleEnv
from dragon_torch.data.envs import (DragonEnvConfig, WrapperSpec, make_vec_env, obs_to_tensor,
                                    register_dragon_env, register_wrapper)
from dragon_torch.models.RL import DragonRLConfig
from dragon_torch.trainers.RL import EvalConfig, PPOTrainer, PPOTrainerConfig, RolloutConfig


class DoubleRewardCartPole(CartPoleEnv):
    """A subclassed prebuilt env that changes the task: 2 per step, not 1."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, 2.0 * reward, terminated, truncated, info


@register_wrapper('test_half_reward')
class HalfReward(gym.RewardWrapper):
    def reward(self, reward):
        return 0.5 * reward


class Corridor(gym.Env):
    """Dict observations (no default encoder) and MultiBinary button actions:
    usable only once wrappers flatten the obs and discretise the actions."""
    buttons = ['LEFT', 'RIGHT']
    observation_space = spaces.Dict({'pos': spaces.Box(0, 4, (1,), np.float32),
                                     'goal': spaces.Discrete(5)})
    action_space = spaces.MultiBinary(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos, self.goal, self.t = 0, int(self.np_random.integers(1, 5)), 0
        return self._obs(), {}

    def _obs(self):
        return {'pos': np.array([self.pos], np.float32), 'goal': self.goal}

    def step(self, action):
        self.pos = int(np.clip(self.pos + int(action[1]) - int(action[0]), 0, 4))
        self.t += 1
        reached = self.pos == self.goal
        return self._obs(), 1.0 if reached else -0.1, reached, self.t >= 20, {}


def trainer_config(tmp_path, env, **kw):
    return PPOTrainerConfig(
        env=env, model=DragonRLConfig(checkpoint_dir=tmp_path, save_every=1000, total_timesteps=128),
        rollout=RolloutConfig(n_steps=32, update_epochs=2, n_minibatches=2),
        eval=EvalConfig(every_steps=64, n_episodes=2), validate_env=True, **kw)


def test_subclassed_registered_env_trains(tmp_path):
    """Subclass + register_dragon_env: the subclass's reward is the task (raw);
    a wrapper on top only shapes it."""
    register_dragon_env('DragonDoubleCartPole-v0', DoubleRewardCartPole, max_episode_steps=50)
    env = DragonEnvConfig(env_id='DragonDoubleCartPole-v0', n_envs=2,
                          wrappers=[WrapperSpec(name='test_half_reward')])
    with PPOTrainer(trainer_config(tmp_path, env)) as trainer:
        assert isinstance(trainer.source.collector.env.envs[0].unwrapped, DoubleRewardCartPole)
        history = trainer.fit()
    episodes = [r for r in history if 'episode_return' in r]
    assert episodes
    for r in episodes:
        assert r['episode_return'] == pytest.approx(2.0 * r['episode_length'])
        assert r['episode_return_shaped'] == pytest.approx(r['episode_length'])
        assert r['episode_length'] <= 50  # the registration's TimeLimit applies
    assert any('val_return_smooth' in r for r in history)


def test_dict_obs_without_flatten_fails_clearly(tmp_path):
    env = DragonEnvConfig(env_fn=Corridor, wrappers=[
        WrapperSpec(name='Discretizer', kwargs={'combos': [[], ['LEFT'], ['RIGHT']]})])
    with pytest.raises(TypeError, match='No default encoder for Dict'):
        PPOTrainer(trainer_config(tmp_path, env))


def test_wrapper_spaces_reach_the_network(tmp_path):
    """FlattenObservation turns Dict obs into a 6-d Box (pos + one-hot goal) and
    Discretizer turns MultiBinary(2) into Discrete(3); the network is built
    from the wrapped spaces and the run trains."""
    env = DragonEnvConfig(env_fn=Corridor, n_envs=2, wrappers=[
        'FlattenObservation',
        WrapperSpec(name='Discretizer', kwargs={'combos': [[], ['LEFT'], ['RIGHT']]})])
    with PPOTrainer(trainer_config(tmp_path, env)) as trainer:
        col = trainer.source.collector
        assert isinstance(col.observation_space, spaces.Box) and col.observation_space.shape == (6,)
        assert col.action_space == spaces.Discrete(3)
        module = trainer.model.torch_module
        assert module.actor_trunk[0].out_features == 6
        action = module(obs_to_tensor(col.obs)).dist.sample()
        assert action.shape == (2,) and 0 <= int(action.min()) and int(action.max()) < 3
        history = trainer.fit()
    assert any('episode_return' in r for r in history)


def test_subclassed_gymnasium_wrapper(tmp_path):
    """Subclassing a gymnasium wrapper and registering it works like any other."""
    @register_wrapper('test_clip_and_bonus')
    class ClipAndBonus(gym.wrappers.ClipReward):
        def __init__(self, env, max_reward: float = 1.0, bonus: float = 0.0):
            super().__init__(env, max_reward=max_reward)
            self.bonus = bonus

        def reward(self, reward):
            return super().reward(reward) + self.bonus

    env = DragonEnvConfig(env_id='CartPole-v1', wrappers=[
        WrapperSpec(name='test_clip_and_bonus', kwargs={'max_reward': 0.5, 'bonus': 0.25})])
    with PPOTrainer(trainer_config(tmp_path, env)) as trainer:
        history = trainer.fit()
    episodes = [r for r in history if 'episode_return' in r]
    assert episodes
    for r in episodes:
        assert r['episode_return'] == pytest.approx(r['episode_length'])          # raw: CartPole's 1/step
        assert r['episode_return_shaped'] == pytest.approx(0.75 * r['episode_length'])  # clip 0.5 + 0.25


def test_registered_env_works_with_async_vectorization():
    """Async sub-envs start in fresh processes that never ran this
    registration; make_env_fn carries the EnvSpec so they don't need to."""
    register_dragon_env('DragonDoubleCartPoleAsync-v0', DoubleRewardCartPole, max_episode_steps=50)
    env = make_vec_env(DragonEnvConfig(env_id='DragonDoubleCartPoleAsync-v0', n_envs=2,
                                       vectorization='async'))
    try:
        env.reset(seed=0)
        _, reward, *_ = env.step(np.zeros(2, np.int64))
        assert reward.tolist() == [2.0, 2.0]
    finally:
        env.close()
