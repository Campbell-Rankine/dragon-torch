import json
import math
import optuna
import pytest
from data.envs import DragonEnvConfig
from utils.tuning import DragonObjective, read_history, run_study
from utils.tuning.presets import PPOBuilder, PPOSpace
from utils.tuning.presets.ppo import parse_hidden
from utils.tuning.RL import (EVAL_STEP_KEY, DivergenceGuard, StepEvalCallback, TrialDiverged,
                             rl_tune_config)

optuna.logging.set_verbosity(optuna.logging.WARNING)
COMPLETE, PRUNED = optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED


class FakeModel:
    def __init__(self, values):
        self.values = iter(values)
        self.evaluated = 0

    def evaluate(self, data):
        self.evaluated += 1
        return next(self.values)


class Closable:
    closed = False

    def close(self):
        self.closed = True


def run_epochs(callback, model, steps):
    records = [{'epoch': i + 1, 'global_step': s} for i, s in enumerate(steps)]
    for r in records:
        callback(model, r)
    return records


class TestStepEvalCallback:
    def test_schedule_and_aligned_steps(self):
        cb = StepEvalCallback(None, every_steps=100, total_steps=250, smooth_window=2)
        model = FakeModel([1.0, 3.0, 5.0])
        records = run_epochs(cb, model, [60, 120, 180, 240, 300])
        evals = [r for r in records if 'val_return' in r]
        # global_steps 120, 240, 300 evaluate, reported at the scheduled 100, 200, 250 (capped)
        assert [r['global_step'] for r in evals] == [120, 240, 300]
        assert [r[EVAL_STEP_KEY] for r in evals] == [100, 200, 250]
        assert [r['val_return'] for r in evals] == [1.0, 3.0, 5.0]
        assert [r['val_return_smooth'] for r in evals] == [1.0, 2.0, 4.0]
        assert model.evaluated == 3

    def test_epoch_crossing_two_thresholds_evaluates_once(self):
        cb = StepEvalCallback(None, every_steps=10)
        records = run_epochs(cb, FakeModel([7.0]), [35])
        assert records[0][EVAL_STEP_KEY] == 30

    def test_final_eval_not_repeated(self):
        cb = StepEvalCallback(None, every_steps=100, total_steps=200)
        model = FakeModel([1.0, 2.0])
        run_epochs(cb, model, [100, 200, 260])
        assert model.evaluated == 2

    def test_needs_step_key_and_closes_evaluator(self):
        evaluator = Closable()
        cb = StepEvalCallback(evaluator, every_steps=10)
        with pytest.raises(KeyError, match='global_step'):
            cb(FakeModel([]), {'epoch': 1})
        cb.close()
        assert evaluator.closed


class TestDivergenceGuard:
    def test_kl_needs_patience(self):
        guard = DivergenceGuard(max_kl=0.1, patience=2)
        guard(None, {'approx_kl': 0.5, 'global_step': 1})
        guard(None, {'approx_kl': 0.01, 'global_step': 2})  # resets the streak
        guard(None, {'approx_kl': 0.5, 'global_step': 3})
        with pytest.raises(TrialDiverged, match='approx_kl'):
            guard(None, {'approx_kl': 0.5, 'global_step': 4})

    def test_non_finite_loss(self):
        with pytest.raises(TrialDiverged, match='train_loss'):
            DivergenceGuard()(None, {'train_loss': math.nan, 'global_step': 1})

    def test_entropy_collapse_after_warmup(self):
        guard = DivergenceGuard(max_kl=None, min_entropy=0.1, entropy_after_steps=100)
        guard(None, {'entropy': 0.01, 'global_step': 50})
        with pytest.raises(TrialDiverged, match='entropy'):
            guard(None, {'entropy': 0.01, 'global_step': 100})

    def test_missing_keys_ignored(self):
        DivergenceGuard(min_entropy=0.1)(None, {'epoch': 1})  # e.g. A2C: no approx_kl

    def test_is_a_prune(self):
        assert issubclass(TrialDiverged, optuna.TrialPruned)


class TestRLTuneConfig:
    def test_defaults(self):
        cfg = rl_tune_config('rl')
        assert cfg.metrics == ['val_return_smooth'] and cfg.directions == ['maximize']
        assert cfg.report_key == EVAL_STEP_KEY and cfg.can_prune

    def test_multi_objective(self):
        cfg = rl_tune_config('rl', extra_metrics={'steps_per_sec': 'maximize'})
        assert cfg.metrics == ['val_return_smooth', 'steps_per_sec']
        assert cfg.directions == ['maximize', 'maximize'] and not cfg.can_prune


class TestPPOPreset:
    def test_space_and_fixed(self):
        params = PPOSpace(fixed={'gamma': 0.9, 'hidden': '32x32'}).suggest(
            optuna.create_study().ask())
        assert params['gamma'] == 0.9 and params['hidden'] == '32x32'
        study = optuna.create_study()
        trial = study.ask()
        PPOSpace(fixed={'gamma': 0.9}).suggest(trial)
        assert 'gamma' not in trial.params and 'lr' in trial.params

    def test_parse_hidden(self):
        assert parse_hidden('128x64') == (128, 64) and parse_hidden([8]) == (8,)


def small_space(**fixed):
    return PPOSpace(n_steps=[16, 32], n_minibatches=[1, 2], update_epochs=(1, 2),
                    hidden=['16x16'], fixed=fixed)


def builder(**kw):
    kw.setdefault('total_timesteps', 512)
    kw.setdefault('eval_every_steps', 128)  # >= 32 steps x 4 envs
    kw.setdefault('eval_episodes', 2)
    return PPOBuilder(env=DragonEnvConfig(env_id='CartPole-v1', n_envs=4), **kw)


class TestPPOStudy:
    def test_cartpole_study(self, tmp_path):
        cfg = rl_tune_config('ppo', storage=f"sqlite:///{tmp_path / 'o.db'}",
                             run_dir=tmp_path / 'runs', n_trials=3, seed=0, pruner='none')
        study = run_study(cfg, small_space().suggest, builder())
        assert [t.state for t in study.trials] == [COMPLETE] * 3
        for t in study.trials:
            # every trial reports at the same aligned steps, whatever its epoch size
            assert sorted(t.intermediate_values) == [128, 256, 384, 512]
            history = read_history(f"{t.user_attrs['trial_dir']}/history.jsonl")
            evals = [r for r in history if 'val_return_smooth' in r]
            assert t.value == evals[-1]['val_return_smooth']
            assert history[-1]['global_step'] >= 512
            assert {'approx_kl', 'entropy', 'steps_per_sec'} <= set(history[-1])
        assert json.loads((cfg.study_dir / 'best.json').read_text())['best']

    def test_pruned_trials_close_envs(self, tmp_path):
        class AlwaysPrune(optuna.pruners.BasePruner):
            def prune(self, study, trial):
                return True

        built = []
        build = builder()

        def tracking_build(params, trial_dir):
            spec = build(params, trial_dir)
            built.append(spec)
            return spec

        cfg = rl_tune_config('ppo', run_dir=tmp_path / 'runs')
        study = optuna.create_study(direction='maximize', pruner=AlwaysPrune())
        study.optimize(DragonObjective(cfg, small_space().suggest, tracking_build), n_trials=1)
        t = study.trials[0]
        assert t.state == PRUNED and t.last_step == 128 and 'prune_reason' in t.user_attrs
        spec = built[0]
        assert spec.train_data.collector.env.closed
        assert spec.callbacks[0].evaluator.collector.env.closed

    def test_divergence_prunes(self, tmp_path):
        cfg = rl_tune_config('ppo', run_dir=tmp_path / 'runs', n_trials=1)
        study = run_study(cfg, small_space(lr=0.5, update_epochs=2).suggest,
                          builder(max_kl=1e-6, kl_patience=1))
        t = study.trials[0]
        assert t.state == PRUNED and 'approx_kl' in t.user_attrs['prune_reason']

    def test_multi_objective(self, tmp_path):
        cfg = rl_tune_config('ppo-mo', run_dir=tmp_path / 'runs', n_trials=2,
                             extra_metrics={'steps_per_sec': 'maximize'})
        study = run_study(cfg, small_space().suggest, builder(total_timesteps=256))
        assert all(t.state == COMPLETE and len(t.values) == 2 for t in study.trials)
        assert study.best_trials
