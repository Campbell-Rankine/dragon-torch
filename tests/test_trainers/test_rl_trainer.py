import json
import math
import pytest
import torch
from data.envs import DragonEnvConfig
from models.RL import DragonRLConfig
from modules.RL import ActorCritic
from trainers import TrainingDiverged
from trainers.RL import (EVAL_STEP_KEY, A2CTrainer, A2CTrainerConfig, DivergenceGuard,
                         DragonRLTrainer, EvalConfig, GuardConfig, OptimConfig, PPOTrainer,
                         PPOTrainerConfig, RolloutConfig, StepEvalCallback, apply_overrides)


def env(n_envs=4, seed=0):
    return DragonEnvConfig(env_id='CartPole-v1', n_envs=n_envs, seed=seed)


def ppo_cfg(tmp_path, **kw):
    kw.setdefault('env', env())
    kw.setdefault('model', DragonRLConfig(checkpoint_dir=tmp_path / 'ckpt', save_every=1000,
                                          total_timesteps=kw.pop('total_timesteps', 256)))
    kw.setdefault('rollout', RolloutConfig(n_steps=16, update_epochs=1, n_minibatches=2))
    kw.setdefault('eval', EvalConfig(every_steps=64, n_episodes=2))
    return PPOTrainerConfig(**kw)


def envs_of(trainer):
    return [s.collector.env for s in (trainer.source, trainer.evaluator) if s is not None]


class FakeModel:
    def __init__(self, values):
        self.values = iter(values)
        self.evaluated = 0

    def evaluate(self, data):
        self.evaluated += 1
        return next(self.values)


def run_epochs(callback, model, steps):
    records = [{'epoch': i + 1, 'global_step': s} for i, s in enumerate(steps)]
    for r in records:
        callback(model, r)
    return records


class TestConfig:
    def test_minibatch_divisibility(self, tmp_path):
        with pytest.raises(ValueError, match='divisible'):
            ppo_cfg(tmp_path, rollout=RolloutConfig(n_steps=10, n_minibatches=3))

    def test_anneal_needs_total_timesteps(self, tmp_path):
        with pytest.raises(ValueError, match='total_timesteps'):
            ppo_cfg(tmp_path, model=DragonRLConfig(checkpoint_dir=tmp_path, save_every=1),
                    optim=OptimConfig(anneal_lr=True))

    def test_eval_more_often_than_epochs_warns(self, tmp_path):
        with pytest.warns(UserWarning, match='every_steps'):
            ppo_cfg(tmp_path, eval=EvalConfig(every_steps=8))

    def test_policy_hidden_string(self, tmp_path):
        assert ppo_cfg(tmp_path, policy={'hidden': '32x16'}).policy.hidden == (32, 16)

    def test_eval_env_defaults(self, tmp_path):
        cfg = ppo_cfg(tmp_path, env=env(n_envs=4, seed=3))
        assert cfg.eval_env_config().n_envs == 1 and cfg.eval_env_config().seed == 10_003
        assert ppo_cfg(tmp_path, eval=None).eval_env_config() is None

    def test_algorithm_defaults(self, tmp_path):
        a2c = A2CTrainerConfig(env=env(), model=DragonRLConfig(checkpoint_dir=tmp_path, save_every=1))
        assert (a2c.rollout.update_epochs, a2c.rollout.n_minibatches) == (1, 1)
        assert a2c.optim.name == 'rmsprop'
        assert PPOTrainerConfig(env=env(), model=a2c.model).rollout.n_minibatches == 4


class TestApplyOverrides:
    def test_nested_and_revalidated(self, tmp_path):
        cfg = ppo_cfg(tmp_path)
        out = apply_overrides(cfg, {'rollout.n_steps': 8, 'ppo.clip_coef': 0.1,
                                    'policy.hidden': '8x8', 'seed': 5})
        assert type(out) is PPOTrainerConfig
        assert (out.rollout.n_steps, out.ppo.clip_coef, out.policy.hidden, out.seed) == \
            (8, 0.1, (8, 8), 5)
        assert out.rollout.update_epochs == 1  # untouched siblings kept
        assert cfg.rollout.n_steps == 16  # original unchanged

    def test_cross_field_validation_runs(self, tmp_path):
        with pytest.raises(ValueError, match='divisible'):
            apply_overrides(ppo_cfg(tmp_path), {'rollout.n_minibatches': 3})

    def test_errors(self, tmp_path):
        cfg = ppo_cfg(tmp_path)
        with pytest.raises(KeyError, match="'rollout.nope'"):
            apply_overrides(cfg, {'rollout.nope': 1})
        with pytest.raises(KeyError, match="'nope'"):
            apply_overrides(cfg, {'nope': 1})
        with pytest.raises(KeyError, match='not a config section'):
            apply_overrides(ppo_cfg(tmp_path, eval=None), {'eval.every_steps': 10})


class TestStepEvalCallback:
    def test_schedule_and_aligned_steps(self):
        cb = StepEvalCallback(None, every_steps=100, total_steps=250, smooth_window=2)
        model = FakeModel([1.0, 3.0, 5.0])
        evals = [r for r in run_epochs(cb, model, [60, 120, 180, 240, 300]) if 'val_return' in r]
        assert [r['global_step'] for r in evals] == [120, 240, 300]
        assert [r[EVAL_STEP_KEY] for r in evals] == [100, 200, 250]
        assert [r['val_return_smooth'] for r in evals] == [1.0, 2.0, 4.0]

    def test_crossing_two_thresholds_evaluates_once(self):
        records = run_epochs(StepEvalCallback(None, every_steps=10), FakeModel([7.0]), [35])
        assert records[0][EVAL_STEP_KEY] == 30

    def test_state_round_trip(self):
        cb = StepEvalCallback(None, every_steps=10, smooth_window=2)
        run_epochs(cb, FakeModel([1.0, 2.0, 3.0]), [10, 20, 30])
        other = StepEvalCallback(None, every_steps=10, smooth_window=2)
        other.load_state_dict(cb.state_dict())
        assert list(other.recent) == [2.0, 3.0] and other.due_step(30) is None

    def test_needs_step_key(self):
        with pytest.raises(KeyError, match='global_step'):
            StepEvalCallback(None, every_steps=10)(FakeModel([]), {'epoch': 1})


class TestDivergenceGuard:
    def test_kl_needs_patience(self):
        guard = DivergenceGuard(max_kl=0.1, patience=2)
        guard(None, {'approx_kl': 0.5, 'global_step': 1})
        guard(None, {'approx_kl': 0.01, 'global_step': 2})  # resets the streak
        guard(None, {'approx_kl': 0.5, 'global_step': 3})
        with pytest.raises(TrainingDiverged, match='approx_kl'):
            guard(None, {'approx_kl': 0.5, 'global_step': 4})

    def test_non_finite_loss(self):
        with pytest.raises(TrainingDiverged, match='train_loss'):
            DivergenceGuard()(None, {'train_loss': math.nan, 'global_step': 1})

    def test_entropy_collapse_after_warmup(self):
        guard = DivergenceGuard(max_kl=None, min_entropy=0.1, entropy_after_steps=100)
        guard(None, {'entropy': 0.01, 'global_step': 50})
        with pytest.raises(TrainingDiverged, match='entropy'):
            guard(None, {'entropy': 0.01, 'global_step': 100})

    def test_missing_keys_ignored(self):
        DivergenceGuard(min_entropy=0.1)(None, {'epoch': 1})  # e.g. A2C: no approx_kl


class TestPPOTrainer:
    def test_learns_cartpole(self, tmp_path):
        cfg = PPOTrainerConfig(
            env=env(), rollout=RolloutConfig(n_steps=128),
            model=DragonRLConfig(checkpoint_dir=tmp_path, save_every=1000, total_timesteps=25_000),
            eval=EvalConfig(every_steps=5_000, n_episodes=8),
            optim=OptimConfig(lr=2.5e-3, anneal_lr=True))
        with PPOTrainer(cfg) as trainer:
            before = trainer.evaluate()
            history = trainer.fit()
            after = trainer.evaluate()
        assert before < 50 and after > 200
        evals = [r for r in history if EVAL_STEP_KEY in r]
        assert [r[EVAL_STEP_KEY] for r in evals] == [5_000, 10_000, 15_000, 20_000, 25_000]
        assert history[0]['lr'] == 2.5e-3 and history[-1]['lr'] < 1e-4  # annealed

    def test_default_callbacks_before_callers(self, tmp_path):
        seen = []
        with PPOTrainer(ppo_cfg(tmp_path)) as trainer:
            trainer.fit(callbacks=[lambda m, r: seen.append('val_return_smooth' in r)])
        assert len(seen) == 4 and all(seen)  # 64 steps / epoch, eval every 64

    def test_eval_and_guard_optional(self, tmp_path):
        with PPOTrainer(ppo_cfg(tmp_path, eval=None, guard=None)) as trainer:
            assert trainer.default_callbacks() == []
            history = trainer.fit(epochs=1)
            with pytest.raises(ValueError, match='no evaluator'):
                trainer.evaluate()
        assert 'val_return' not in history[0]

    def test_guard_raises(self, tmp_path):
        cfg = ppo_cfg(tmp_path, guard=GuardConfig(max_kl=1e-9, kl_patience=1),
                      optim=OptimConfig(lr=0.5))
        with PPOTrainer(cfg) as trainer, pytest.raises(TrainingDiverged):
            trainer.fit()

    def test_custom_module(self, tmp_path):
        cfg = ppo_cfg(tmp_path)
        probe = PPOTrainer(cfg)
        col = probe.source.collector
        module = ActorCritic(col.observation_space, col.action_space, hidden=(8,))
        probe.close()
        with PPOTrainer(cfg, module=module) as trainer:
            assert trainer.model.torch_module is module

    def test_close(self, tmp_path):
        trainer = PPOTrainer(ppo_cfg(tmp_path))
        envs = envs_of(trainer)
        trainer.close()
        assert len(envs) == 2 and all(e.closed for e in envs)

    def test_failed_build_closes_envs(self, tmp_path, monkeypatch):
        opened = []

        def boom(self):
            opened.extend(envs_of(self))
            raise RuntimeError('boom')

        monkeypatch.setattr(PPOTrainer, 'build_loss', boom)
        with pytest.raises(RuntimeError, match='boom'):
            PPOTrainer(ppo_cfg(tmp_path))
        assert len(opened) == 2 and all(e.closed for e in opened)

    def test_rejects_wrong_config(self, tmp_path):
        a2c = A2CTrainerConfig(env=env(), model=DragonRLConfig(checkpoint_dir=tmp_path, save_every=1))
        with pytest.raises(TypeError, match='PPOTrainerConfig'):
            PPOTrainer(a2c)
        with pytest.raises(TypeError):
            DragonRLTrainer(a2c)  # abstract

    def test_resume_restores_eval_state_and_lr(self, tmp_path):
        def cfg(**kw):
            return ppo_cfg(tmp_path, total_timesteps=256, optim=OptimConfig(lr=1e-3, anneal_lr=True),
                           eval=EvalConfig(every_steps=64, n_episodes=1, smooth_window=3), **kw)

        with PPOTrainer(cfg()) as first:
            first.fit(epochs=2)
            first.save_checkpoint()
            recent = list(first.eval_callback.recent)
            lr = first.optimizer.param_groups[0]['lr']
        with PPOTrainer(cfg(resume=True)) as second:
            history = second.fit()
            assert history[0]['global_step'] == 192 and history[0]['lr'] == pytest.approx(lr)
            assert len(history) == 2
            smooth = [r['val_return_smooth'] for r in history if 'val_return_smooth' in r]
            expected = (recent[0] + recent[1] + history[0]['val_return']) / 3
            assert smooth[0] == pytest.approx(expected)
            assert second.optimizer.param_groups[0]['initial_lr'] == 1e-3


class TestA2CTrainer:
    def test_runs(self, tmp_path):
        cfg = A2CTrainerConfig(
            env=env(), model=DragonRLConfig(checkpoint_dir=tmp_path, save_every=1000,
                                            total_timesteps=400),
            eval=EvalConfig(every_steps=200, n_episodes=1))
        with A2CTrainer(cfg) as trainer:
            history = trainer.fit()
        assert history[-1]['global_step'] >= 400 and 'approx_kl' not in history[-1]
        assert [r[EVAL_STEP_KEY] for r in history if EVAL_STEP_KEY in r] == [200, 400]
        assert isinstance(trainer.optimizer, torch.optim.RMSprop)


def test_config_json_round_trip(tmp_path):
    cfg = ppo_cfg(tmp_path)
    again = PPOTrainerConfig.model_validate_json(cfg.model_dump_json())
    assert again == cfg
    assert json.loads(cfg.model_dump_json())['rollout']['n_steps'] == 16
