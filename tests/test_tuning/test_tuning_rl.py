import json
import optuna
import pytest
from data.envs import DragonEnvConfig
from models.RL import DragonRLConfig
from trainers.RL import (EVAL_STEP_KEY, A2CTrainer, A2CTrainerConfig, EvalConfig, GuardConfig,
                         OptimConfig, PPOTrainer, PPOTrainerConfig)
from utils.tuning import (CategoricalParam, DragonObjective, FloatParam, IntParam, SearchSpace,
                          TrialSpec, read_history, run_study)
from utils.tuning.presets import a2c_space, ppo_space
from utils.tuning.RL import RLTrialBuilder, rl_tune_config

optuna.logging.set_verbosity(optuna.logging.WARNING)
COMPLETE, PRUNED = optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED


class AlwaysPrune(optuna.pruners.BasePruner):
    def prune(self, study, trial):
        return True


class TestSearchSpace:
    def test_suggest_fix_narrow_without(self):
        space = SearchSpace(params={'a.x': FloatParam(low=0.1, high=1.0, log=True),
                                    'a.n': IntParam(low=1, high=3),
                                    'b': CategoricalParam(choices=['64x64', '8x8'])})
        trial = optuna.create_study().ask()
        params = space.fix({'a.n': 7}).suggest(trial)
        assert params['a.n'] == 7 and 'a.n' not in trial.params
        assert 0.1 <= params['a.x'] <= 1.0 and params['b'] in ('64x64', '8x8')
        narrowed = space.narrow({'b': CategoricalParam(choices=['8x8'])})
        assert narrowed.suggest(optuna.create_study().ask())['b'] == '8x8'
        assert set(space.without('b').params) == {'a.x', 'a.n'}

    def test_validation_and_json(self):
        with pytest.raises(ValueError):
            FloatParam(low=0.0, high=1.0, log=True)
        space = ppo_space()
        assert SearchSpace.model_validate_json(space.model_dump_json()) == space

    def test_presets_match_their_configs(self, tmp_path):
        """Every preset key is a real config path."""
        model = DragonRLConfig(checkpoint_dir=tmp_path, save_every=1, total_timesteps=1000)
        for space, cfg_cls in ((ppo_space(), PPOTrainerConfig), (a2c_space(), A2CTrainerConfig)):
            builder = RLTrialBuilder(trainer_cls=PPOTrainer,
                                     base_config=cfg_cls(env=DragonEnvConfig(env_id='CartPole-v1'),
                                                         model=model))
            params = space.suggest(optuna.create_study().ask())
            cfg = builder.config_for(params, tmp_path / 't')
            assert cfg.rollout.n_steps == params['rollout.n_steps']
            assert cfg.model.checkpoint_dir == tmp_path / 't' / 'ckpt'


class TestRLTuneConfig:
    def test_defaults_from_trainer(self):
        cfg = rl_tune_config('rl', PPOTrainer)
        assert cfg.metrics == ['val_return_smooth'] and cfg.directions == ['maximize']
        assert cfg.report_key == EVAL_STEP_KEY and cfg.can_prune

    def test_multi_objective(self):
        cfg = rl_tune_config('rl', A2CTrainer, extra_metrics={'steps_per_sec': 'maximize'})
        assert cfg.metrics == ['val_return_smooth', 'steps_per_sec'] and not cfg.can_prune


def small_space(**fixed):
    space = ppo_space().narrow({
        'rollout.n_steps': CategoricalParam(choices=[16, 32]),
        'rollout.n_minibatches': CategoricalParam(choices=[1, 2]),
        'rollout.update_epochs': IntParam(low=1, high=2),
        'policy.hidden': CategoricalParam(choices=['16x16']),
    })
    return space.fix(fixed)


def builder(tmp_path, trainer_cls=PPOTrainer, cfg_cls=PPOTrainerConfig, total=512, **kw):
    base = cfg_cls(env=DragonEnvConfig(env_id='CartPole-v1', n_envs=4),
                   model=DragonRLConfig(checkpoint_dir=tmp_path / 'unused', save_every=1000,
                                        total_timesteps=total),
                   rollout=cfg_cls.model_fields['rollout'].default_factory().model_copy(
                       update={'n_steps': 16, 'n_minibatches': 1}),
                   eval=EvalConfig(every_steps=128, n_episodes=2), **kw)  # 128 >= 32 x 4
    return RLTrialBuilder(trainer_cls=trainer_cls, base_config=base)


class TestRLStudy:
    def test_ppo_study(self, tmp_path):
        cfg = rl_tune_config('ppo', PPOTrainer, storage=f"sqlite:///{tmp_path / 'o.db'}",
                             run_dir=tmp_path / 'runs', n_trials=3, seed=0, pruner='none')
        study = run_study(cfg, small_space().suggest, builder(tmp_path))
        assert [t.state for t in study.trials] == [COMPLETE] * 3
        for t in study.trials:
            # every trial reports at the same aligned steps, whatever its epoch size
            assert sorted(t.intermediate_values) == [128, 256, 384, 512]
            trial_dir = t.user_attrs['trial_dir']
            history = read_history(f"{trial_dir}/history.jsonl")
            assert t.value == [r for r in history if 'val_return_smooth' in r][-1]['val_return_smooth']
            assert {'approx_kl', 'entropy', 'steps_per_sec', 'lr'} <= set(history[-1])
            saved = PPOTrainerConfig.model_validate_json(open(t.user_attrs['config_path']).read())
            assert saved.rollout.n_steps == t.params['rollout.n_steps']
            assert str(saved.model.checkpoint_dir).startswith(trial_dir)
        best = json.loads((cfg.study_dir / 'best.json').read_text())['best'][0]
        assert best['config_path'].endswith('config.json')

    def test_pruned_trial_closes_trainer(self, tmp_path):
        trainers = []
        build = builder(tmp_path)

        def tracking_build(params, trial_dir):
            spec = build(params, trial_dir)
            trainers.append(spec.trainer)
            return spec

        cfg = rl_tune_config('ppo', PPOTrainer, run_dir=tmp_path / 'runs')
        study = optuna.create_study(direction='maximize', pruner=AlwaysPrune())
        study.optimize(DragonObjective(cfg, small_space().suggest, tracking_build), n_trials=1)
        t = study.trials[0]
        assert t.state == PRUNED and t.last_step == 128 and 'prune_reason' in t.user_attrs
        assert trainers[0].source.collector.env.closed
        assert trainers[0].evaluator.collector.env.closed

    def test_divergence_prunes(self, tmp_path):
        cfg = rl_tune_config('ppo', PPOTrainer, run_dir=tmp_path / 'runs', n_trials=1)
        study = run_study(cfg, small_space(**{'optim.lr': 0.5}).suggest,
                          builder(tmp_path, guard=GuardConfig(max_kl=1e-9, kl_patience=1)))
        t = study.trials[0]
        assert t.state == PRUNED and t.user_attrs['prune_reason'].startswith('diverged: approx_kl')

    def test_a2c_multi_objective(self, tmp_path):
        cfg = rl_tune_config('a2c-mo', A2CTrainer, run_dir=tmp_path / 'runs', n_trials=2,
                             extra_metrics={'steps_per_sec': 'maximize'})
        space = a2c_space().narrow({'rollout.n_steps': CategoricalParam(choices=[8, 16]),
                                    'policy.hidden': CategoricalParam(choices=['16x16'])})
        study = run_study(cfg, space.suggest,
                          builder(tmp_path, A2CTrainer, A2CTrainerConfig, total=256,
                                  optim=OptimConfig(name='rmsprop', lr=7e-4)))
        assert all(t.state == COMPLETE and len(t.values) == 2 for t in study.trials)
        assert study.best_trials

    def test_trial_spec_needs_trainer_or_parts(self):
        with pytest.raises(ValueError, match='trainer'):
            TrialSpec()
