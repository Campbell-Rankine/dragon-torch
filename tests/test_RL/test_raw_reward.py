"""Raw (env) vs shaped (after wrappers) rewards: collection, evaluation, callbacks."""
import gymnasium as gym
import numpy as np
import pytest
from dragon_torch.data.envs import (RAW_REWARD_KEY, DragonEnvConfig, RecordRawReward, WrapperSpec,
                                    make_env_fn, register_wrapper)
from dragon_torch.data.RL import EnvCollector, EpisodeSource
from dragon_torch.models.RL import DragonRLConfig
from dragon_torch.trainers.RL import EvalConfig, PPOTrainer, PPOTrainerConfig, RolloutConfig
from test_rl_collector import CounterEnv, CountingActor


@register_wrapper('test_times_ten')
class TimesTen(gym.RewardWrapper):
    def reward(self, reward):
        return 10.0 * reward


class DropsInfo(gym.Wrapper):
    """A badly behaved wrapper that throws info away."""

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        return obs, reward + 1.0, terminated, truncated, {}


def cfg(wrappers=(), n_envs=1, terminate_at=3):
    return DragonEnvConfig(env_fn=CounterEnv, env_kwargs={'terminate_at': terminate_at},
                           wrappers=list(wrappers), n_envs=n_envs)


def run(col, steps):
    for _ in range(steps):
        col.step(CountingActor())
    return col.drain_episodes()


class TestRecordRawReward:
    def test_innermost(self):
        env = make_env_fn(cfg(['test_times_ten']))()
        assert isinstance(env.env, RecordRawReward)  # directly inside the configured wrappers
        env.reset()
        _, reward, *_, info = env.step(0)
        assert reward == 10.0 and info[RAW_REWARD_KEY] == 1.0
        env.close()


class TestCollector:
    def test_raw_and_shaped_returns(self):
        col = EnvCollector(cfg(['test_times_ten'], n_envs=2))
        episodes = run(col, 6)  # terminate_at=3: two episodes per env
        assert len(episodes) == 4
        assert all(e.raw_return == 3.0 and e.shaped_return == 30.0 and e.length == 3
                   for e in episodes)

    def test_last_step_raw_reward_comes_from_final_info(self):
        # every episode is one step long, so every raw reward arrives via final_info
        col = EnvCollector(cfg(['test_times_ten'], n_envs=3, terminate_at=1))
        assert all(e.raw_return == 1.0 and e.shaped_return == 10.0 for e in run(col, 2))

    def test_metrics_keys(self):
        col = EnvCollector(cfg(['test_times_ten']))
        for _ in range(3):
            col.step(CountingActor())
        m = col.metrics()
        assert m['episode_return'] == 3.0 and m['episode_return_shaped'] == 30.0

        plain = EnvCollector(cfg())
        for _ in range(3):
            plain.step(CountingActor())
        m = plain.metrics()
        assert m['episode_return'] == 3.0 and 'episode_return_shaped' not in m

    def test_falls_back_to_reward_when_info_is_dropped(self):
        col = EnvCollector(cfg([DropsInfo]))
        (episode,) = run(col, 3)
        assert episode.raw_return == episode.shaped_return == 6.0  # 3 x (1 + 1)

    def test_reward_normalisation_does_not_change_returns(self):
        col = EnvCollector(cfg(['test_times_ten']), normalize_reward=True)
        rewards = []
        for _ in range(3):
            rewards.append(float(col.step(CountingActor()).reward[0]))
        (episode,) = col.drain_episodes()
        assert episode.raw_return == 3.0 and episode.shaped_return == 30.0
        assert rewards != [10.0, 10.0, 10.0]  # the training reward was scaled

    def test_raw_rewards_helper(self):
        reward = np.array([5.0, 5.0, 5.0])
        info = {RAW_REWARD_KEY: np.array([1.0, 0.0, 0.0]), f'_{RAW_REWARD_KEY}': np.array([True, False, False]),
                'final_info': {RAW_REWARD_KEY: np.array([0.0, 2.0, 0.0]),
                               f'_{RAW_REWARD_KEY}': np.array([False, True, False])}}
        done = np.array([False, True, True])
        # env 0: live info; env 1: final_info; env 2: missing everywhere -> reward
        assert EnvCollector.raw_rewards(reward, info, done).tolist() == [1.0, 2.0, 5.0]


class TestEvaluation:
    def test_episode_source_yields_both(self):
        src = EpisodeSource(cfg(['test_times_ten']), n_episodes=2)
        src.bind(CountingActor())
        assert src.shaped
        assert list(src) == [{'return': 3.0, 'shaped_return': 30.0, 'length': 3}] * 2
        src.close()

    @pytest.fixture
    def trainer_cfg(self, tmp_path):
        def make(**model_kw):
            return PPOTrainerConfig(
                env=DragonEnvConfig(env_id='CartPole-v1', n_envs=2, wrappers=['test_times_ten']),
                model=DragonRLConfig(checkpoint_dir=tmp_path, save_every=1000, total_timesteps=64,
                                     **model_kw),
                rollout=RolloutConfig(n_steps=16, update_epochs=1, n_minibatches=2),
                eval=EvalConfig(every_steps=32, n_episodes=2))
        return make

    def test_evaluate_scores_raw_by_default(self, trainer_cfg):
        with PPOTrainer(trainer_cfg()) as trainer:
            score = trainer.evaluate()
            stats = trainer.model.last_eval
        assert score == stats['return'] == pytest.approx(stats['length'])  # 1 per step
        assert stats['shaped_return'] == pytest.approx(10 * stats['return'])

    def test_evaluate_shaped_option(self, trainer_cfg):
        with PPOTrainer(trainer_cfg(eval_return='shaped')) as trainer:
            score = trainer.evaluate()
            assert score == trainer.model.last_eval['shaped_return']

    def test_eval_callback_records_both(self, trainer_cfg):
        with PPOTrainer(trainer_cfg()) as trainer:
            history = trainer.fit()
        evals = [r for r in history if 'val_return' in r]
        assert len(evals) == 2
        for r in evals:
            assert r['val_return'] == r['val_return_raw']
            assert r['val_return_shaped'] == pytest.approx(10 * r['val_return_raw'])
        trained = [r for r in history if 'episode_return' in r]
        assert trained and all(r['episode_return_shaped'] == pytest.approx(10 * r['episode_return'])
                               for r in trained)

    def test_no_shaped_keys_without_wrappers(self, tmp_path):
        cfg = PPOTrainerConfig(
            env=DragonEnvConfig(env_id='CartPole-v1', n_envs=2),
            model=DragonRLConfig(checkpoint_dir=tmp_path, save_every=1000, total_timesteps=32),
            rollout=RolloutConfig(n_steps=16, update_epochs=1, n_minibatches=2),
            eval=EvalConfig(every_steps=32, n_episodes=1))
        with PPOTrainer(cfg) as trainer:
            history = trainer.fit()
        assert not any(k.endswith(('_shaped', '_raw')) for r in history for k in r)
