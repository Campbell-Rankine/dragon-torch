import json
import gymnasium as gym
import numpy as np
import optuna
import pytest
from gymnasium import spaces
from pydantic import ValidationError
from dragon_torch.data.envs import (DragonEnvConfig, WrapperSpec, make_env_fn, register_wrapper,
                                    registered_wrappers, resolve_wrapper, unregister_wrapper)
from dragon_torch.data.envs.wrappers import Discretizer
from dragon_torch.models.RL import DragonRLConfig
from dragon_torch.trainers.RL import EvalConfig, PPOTrainer, PPOTrainerConfig, RolloutConfig, apply_overrides
from dragon_torch.utils.tuning import FloatParam, SearchSpace, read_history, run_study
from dragon_torch.utils.tuning.RL import RLTrialBuilder, rl_tune_config

optuna.logging.set_verbosity(optuna.logging.WARNING)


@register_wrapper('test_scaled_reward')
class ScaledReward(gym.RewardWrapper):
    def __init__(self, env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


class Buttons(gym.Env):
    """MultiBinary(3) action env that records the last action it received."""
    observation_space = spaces.Box(-1, 1, (2,), np.float32)
    action_space = spaces.MultiBinary(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, np.float32), {}

    def step(self, action):
        self.last = action
        return np.zeros(2, np.float32), 0.0, False, False, {}


def cartpole(**kw):
    return DragonEnvConfig(env_id='CartPole-v1', **kw)


class TestRegistry:
    def test_decorator_forms_and_replace(self):
        @register_wrapper
        class NamedByClass(gym.Wrapper):
            pass

        @register_wrapper('test_custom_name')
        def factory(env):
            return env

        try:
            assert resolve_wrapper('NamedByClass') is NamedByClass
            assert resolve_wrapper('test_custom_name') is factory
            register_wrapper('test_custom_name')(NamedByClass)  # re-registering replaces
            assert registered_wrappers()['test_custom_name'] is NamedByClass
        finally:
            unregister_wrapper('NamedByClass')
            unregister_wrapper('test_custom_name')
        assert 'NamedByClass' not in registered_wrappers()

    def test_rejects_colon_names_and_non_callables(self):
        with pytest.raises(ValueError, match="':'"):
            register_wrapper('a:b')(ScaledReward)
        with pytest.raises(TypeError):
            register_wrapper('x')(42)

    def test_resolution_order(self):
        assert resolve_wrapper('test_scaled_reward') is ScaledReward          # registry
        assert resolve_wrapper('Discretizer') is Discretizer                  # built-in registration
        assert resolve_wrapper('ClipReward') is gym.wrappers.ClipReward       # gymnasium
        assert resolve_wrapper('dragon_torch.data.envs.wrappers:Discretizer') is Discretizer  # import path
        assert resolve_wrapper('gymnasium.wrappers:ClipReward') is gym.wrappers.ClipReward

    def test_registry_beats_gymnasium(self):
        register_wrapper('ClipReward')(ScaledReward)
        try:
            assert resolve_wrapper('ClipReward') is ScaledReward
        finally:
            unregister_wrapper('ClipReward')
        assert resolve_wrapper('ClipReward') is gym.wrappers.ClipReward

    @pytest.mark.parametrize('name', ['NoSuchWrapper', 'no_such_module:X',
                                      'gymnasium.wrappers:NoSuch', 'math:pi'])
    def test_unknown(self, name):
        with pytest.raises(ValueError):
            resolve_wrapper(name)


class TestWrapperSpec:
    def test_str_shorthand_and_call(self):
        spec = WrapperSpec.model_validate('FlattenObservation')
        assert spec == WrapperSpec(name='FlattenObservation')
        env = spec(gym.make('CartPole-v1'))
        assert isinstance(env, gym.wrappers.FlattenObservation)
        env.close()

    def test_kwargs_checked_against_signature(self):
        with pytest.raises(ValidationError, match='bad kwargs'):
            WrapperSpec(name='test_scaled_reward', kwargs={'scael': 2.0})       # typo
        with pytest.raises(ValidationError, match='bad kwargs'):
            WrapperSpec(name='FrameStackObservation')                          # missing stack_size
        WrapperSpec(name='FrameStackObservation', kwargs={'stack_size': 4})

    def test_unknown_name_fails_at_creation(self):
        with pytest.raises(ValidationError, match='unknown wrapper'):
            WrapperSpec(name='NoSuchWrapper')

    def test_discretizer_spec(self):
        spec = WrapperSpec(name='Discretizer', kwargs={'combos': [[], [0], [0, 2]]})
        env = spec(Buttons())
        assert env.action_space == spaces.Discrete(3)
        env.reset()
        env.step(2)
        assert env.unwrapped.last.tolist() == [1, 0, 1]


class TestEnvConfig:
    def test_mixes_specs_strings_dicts_and_callables(self):
        cfg = cartpole(wrappers=['FlattenObservation',
                                 {'name': 'test_scaled_reward', 'kwargs': {'scale': 0.5}},
                                 lambda e: gym.wrappers.ClipReward(e, max_reward=0.1)])
        assert [type(w) for w in cfg.wrappers[:2]] == [WrapperSpec, WrapperSpec]
        assert not cfg.portable
        env = make_env_fn(cfg)()
        env.reset(seed=0)
        assert env.step(0)[1] == pytest.approx(0.1)  # 1.0 * 0.5, then clipped to 0.1
        env.close()

    def test_portable_round_trip(self):
        cfg = cartpole(wrappers=['FlattenObservation', WrapperSpec(name='test_scaled_reward',
                                                                   kwargs={'scale': 2.0})])
        assert cfg.portable
        again = DragonEnvConfig.model_validate_json(cfg.model_dump_json())
        assert again == cfg
        env = make_env_fn(again)()
        env.reset(seed=0)
        assert env.step(0)[1] == 2.0
        env.close()

    def test_not_portable_with_env_fn(self):
        assert not DragonEnvConfig(env_fn=lambda: gym.make('CartPole-v1')).portable


class TestApplyOverridesPaths:
    @pytest.fixture
    def cfg(self, tmp_path):
        return PPOTrainerConfig(
            env=cartpole(wrappers=[WrapperSpec(name='test_scaled_reward', kwargs={'scale': 1.0}),
                                   'FlattenObservation']),
            model=DragonRLConfig(checkpoint_dir=tmp_path, save_every=1))

    def test_list_index_and_dict_key(self, cfg):
        out = apply_overrides(cfg, {'env.wrappers.0.kwargs.scale': 0.1, 'env.wrappers.-1.name': 'ClipAction'})
        assert out.env.wrappers[0].kwargs == {'scale': 0.1}
        assert out.env.wrappers[1].name == 'ClipAction'
        assert cfg.env.wrappers[0].kwargs == {'scale': 1.0}  # original untouched

    def test_new_leaf_dict_key_is_validated(self, cfg):
        spec_cfg = apply_overrides(cfg, {'env.wrappers.1': {'name': 'test_scaled_reward'}})
        out = apply_overrides(spec_cfg, {'env.wrappers.1.kwargs.scale': 3.0})  # key not set before
        assert out.env.wrappers[1].kwargs == {'scale': 3.0}
        with pytest.raises(ValidationError, match='bad kwargs'):
            apply_overrides(cfg, {'env.wrappers.0.kwargs.nope': 1})

    def test_tuple_items(self, cfg):
        assert apply_overrides(cfg, {'policy.hidden.0': 128}).policy.hidden == (128, 64)

    @pytest.mark.parametrize('key,match', [
        ('env.wrappers.5.name', 'no item'),
        ('env.wrappers.x.name', 'no item'),
        ('env.env_kwargs.missing.deep', 'not found'),
        ('rollout.n_steps.x', 'not a config section, list or dict'),
    ])
    def test_errors(self, cfg, key, match):
        with pytest.raises(KeyError, match=match):
            apply_overrides(cfg, {key: 1})

    def test_whole_and_field_conflict(self, cfg):
        with pytest.raises(ValueError, match='both whole and by field'):
            apply_overrides(cfg, {'env.wrappers.0': 'FlattenObservation',
                                  'env.wrappers.0.kwargs.scale': 2.0})


def test_tune_a_wrapper_setting(tmp_path):
    """A search space can tune wrapper kwargs, and each trial's config.json
    rebuilds the exact env, wrappers included."""
    base = PPOTrainerConfig(
        env=cartpole(n_envs=2, wrappers=[WrapperSpec(name='test_scaled_reward')]),
        model=DragonRLConfig(checkpoint_dir=tmp_path / 'unused', save_every=1000, total_timesteps=64),
        rollout=RolloutConfig(n_steps=16, update_epochs=1, n_minibatches=1),
        eval=EvalConfig(every_steps=32, n_episodes=1), guard=None)
    space = SearchSpace(params={'env.wrappers.0.kwargs.scale': FloatParam(low=0.1, high=0.2)})
    cfg = rl_tune_config('wrap', PPOTrainer, run_dir=tmp_path / 'runs', n_trials=2, seed=0)
    study = run_study(cfg, space.suggest, RLTrialBuilder(trainer_cls=PPOTrainer, base_config=base))
    for t in study.trials:
        saved = PPOTrainerConfig.model_validate_json(open(t.user_attrs['config_path']).read())
        assert saved.env.portable
        assert saved.env.wrappers[0].kwargs['scale'] == t.params['env.wrappers.0.kwargs.scale']
        # CartPole pays 1 per step: the raw return is the episode length, and the
        # shaped one (what the agent trained on) is scale x length
        history = read_history(f"{t.user_attrs['trial_dir']}/history.jsonl")
        episodes = [r for r in history if 'episode_return' in r]
        scale = saved.env.wrappers[0].kwargs['scale']
        assert episodes
        for r in episodes:
            assert r['episode_return'] == pytest.approx(r['episode_length'])
            assert r['episode_return_shaped'] == pytest.approx(scale * r['episode_length'])
        evals = [r for r in history if 'val_return' in r]
        assert evals and all(r['val_return'] == r['val_return_raw'] for r in evals)
        assert all(r['val_return_shaped'] == pytest.approx(scale * r['val_return_raw']) for r in evals)
        # the objective is the raw (task) return, not the shaping
        assert t.value == evals[-1]['val_return_smooth']
    assert json.loads((cfg.study_dir / 'best.json').read_text())['best']
