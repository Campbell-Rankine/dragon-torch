# dragon_torch.utils.tuning

Optuna studies over any model or trainer. Each trial builds a run from sampled hyperparameters and trains it, reporting every epoch so weak trials are pruned early. Watch studies in `optuna-dashboard` or with the `dragon-tune` CLI. Needs the `tune` extra (`uv sync --extra tune`).

| | |
|---|---|
| [`TuneConfig`](#tuneconfig) | Study name, storage, objective, sampler, pruner, budget |
| [`run_study`](#run_study) | Run this worker's share of a study |
| [`TrialSpec`](#trialspec) | What a trial builds |
| [`SearchSpace`](#searchspace) | Parameters named by config path |
| [`rl_tune_config`, `RLTrialBuilder`](#rl_tune_config) | Tune a `DragonRLTrainer` |
| [`space_for`](#space_for) | Search-space presets per model class |
| [`dragon-tune`](#dragon-tune) | Dashboard, summary and CSV export |

**Example**
```python
>>> base = PPOTrainerConfig(
...     env=DragonEnvConfig(env_id='CartPole-v1', n_envs=8),
...     model=PPOConfig(checkpoint_dir='runs/unused', save_every=1_000_000, total_timesteps=100_000),
...     eval=EvalConfig(every_steps=10_000, n_episodes=10))
>>> cfg = rl_tune_config('ppo-cartpole', PPOTrainer, storage='journal:runs/optuna.log',
...                      n_trials=20, n_warmup_steps=20_000)
>>> space = space_for(PPOTrainer).fix({'rollout.gamma': 0.99})
>>> study = run_study(cfg, space.suggest, RLTrialBuilder(trainer_cls=PPOTrainer, base_config=base))
```

From the command line, `scripts/tune_rl.py --algo ppo --env CartPole-v1 --storage journal:runs/optuna.log` does the same. Start it in several terminals to run parallel workers.

## TuneConfig

```python
class dragon_torch.utils.tuning.TuneConfig(study_name, metrics, directions, storage=None, run_dir='runs',
                                           final='last', final_window=1, n_trials=None, timeout=None,
                                           n_jobs=1, sampler='tpe', seed=None, pruner='median',
                                           n_startup_trials=5, n_warmup_steps=0, report_key='epoch')
```

**Parameters**
- **study_name** (*str*) – shared by every worker in the study.
- **metrics**, **directions** (*list*) – history keys to optimise, each `'minimize'` or `'maximize'`. More than one makes a multi-objective study.
- **storage** (*str, optional*) – see [storage](#storage-and-workers). Default: in memory.
- **run_dir** (*path*) – trial directories go in `run_dir/<study_name>/trial_NNNN/`.
- **final** (*str*) – how a trial's history becomes its value: `'last'`, `'mean_last'` over `final_window` values, or `'best'`.
- **n_trials**, **timeout** – the budget, **per worker**.
- **n_jobs** (*int*) – threads in this process.
- **sampler** (*str*) – `'tpe'`, `'random'`, `'cmaes'` (single objective only) or `'nsga2'`.
- **seed** (*int, optional*) – sampler seed. Give each worker a different seed, or leave it as `None`.
- **pruner** (*str*) – `'median'`, `'hyperband'` or `'none'`. Multi-objective studies don't prune.
- **n_startup_trials**, **n_warmup_steps** (*int*) – how long before pruning starts. `n_warmup_steps` is in units of `report_key`.
- **report_key** (*str*) – the history key used as the pruning step.

`TuneConfig.for_model(model_cls, **kw)` takes the objective from the model class: its `eval_metric`, in the direction given by `higher_is_better`.

## run_study

```python
dragon_torch.utils.tuning.run_study(cfg, suggest, build) -> optuna.Study
```

Creates or joins the study and runs this worker's trials, then writes `run_dir/<study_name>/best.json`: the best trial, or the Pareto front for a multi-objective study.

**Parameters**
- **suggest** (*callable*) – `trial -> params`.
- **build** (*callable*) – `(params, trial_dir) -> TrialSpec`.

Related: `create_study(cfg)`, `load_study(name, storage)`, `best_summary(study)`, `read_history(path)`.

**Example**
```python
>>> def suggest(trial):
...     return {'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True)}
>>> def build(params, trial_dir):
...     cfg = DragonModelConfig(checkpoint_dir=trial_dir / 'ckpt', save_every=10, max_epochs=20)
...     return TrialSpec(model=DragonModel(MyModule(), torch.nn.MSELoss(), cfg),
...                      optimizer=lambda p: torch.optim.Adam(p, lr=params['lr']),
...                      train_data=train_ds, val_data=val_ds)
>>> run_study(TuneConfig.for_model(DragonModel, study_name='mlp'), suggest, build)
```

## TrialSpec

```python
class dragon_torch.utils.tuning.TrialSpec(model=None, optimizer=None, train_data=None, trainer=None,
                                          val_data=None, epochs=None, callbacks=(), trainer_config=None,
                                          resources=())
```

What `build` returns: either a ready `trainer`, or `model`, `optimizer` and `train_data`. Point checkpoints inside `trial_dir`.

- **callbacks** run before the tuner's own, so they can add metrics, for example an evaluation every N steps.
- Everything here with a `close()` method is closed when the trial ends, however it ends.

**Trial outcomes**

| Outcome | State |
|---|---|
| Finishes | `COMPLETE`, valued by `final` |
| Pruned, or `TrainingDiverged` | `PRUNED`, with the reason in the `prune_reason` attribute |
| Objective is NaN or inf | `FAIL`, and the study continues |
| Misconfigured (for example, the metric is never recorded) | Raises, and the study stops |

Each trial directory holds `history.jsonl`, written live with one line per epoch, plus `config.json` when the trainer has a pydantic config. The trial's attributes record `trial_dir`, `config_path`, `epochs` and `last_record`.

## SearchSpace

```python
class dragon_torch.utils.tuning.SearchSpace(params={}, fixed={})
class dragon_torch.utils.tuning.FloatParam(low, high, log=False, step=None)
class dragon_torch.utils.tuning.IntParam(low, high, log=False, step=1)
class dragon_torch.utils.tuning.CategoricalParam(choices)
```

Parameters are named by config path, such as `'optim.lr'`, so the sampled values apply directly with `apply_overrides`, and the dashboard shows those names.

**Methods**
- **suggest**(*trial*) → *dict* – sample every parameter. Pass it to `run_study` as `suggest`.
- **fix**(*values*) – pin values, which are then not sampled.
- **narrow**(*params*) – replace ranges.
- **without**(*\*names*) – drop parameters, so the base config's values are used.

## rl_tune_config

```python
dragon_torch.utils.tuning.RL.rl_tune_config(study_name, trainer_cls=DragonRLTrainer, extra_metrics=None, **kw)
class dragon_torch.utils.tuning.RL.RLTrialBuilder(trainer_cls, base_config)
```

`rl_tune_config` returns a `TuneConfig` that:

- maximises `trainer_cls.objectives`, which is the final `val_return_smooth`,
- reports at `eval_step`, so trials are compared at the same scheduled steps whatever their epoch size.

Set `n_warmup_steps` in env steps. `extra_metrics={'steps_per_sec': 'maximize'}` makes the study multi-objective.

`RLTrialBuilder` is a `build` function. It applies the sampled params to `base_config` with `apply_overrides`, sets `model.checkpoint_dir` to `trial_dir/ckpt`, and returns the trainer. Every trial shares `base_config`'s seeds.

> **Note**
> Keep `eval.every_steps` at least as large as the largest epoch in the search space. An epoch that crosses two scheduled steps evaluates once, and the trial skips a step.

To rebuild a trial: `PPOTrainer(PPOTrainerConfig.model_validate_json(open('.../config.json').read()))`. Callables in the config, such as `env_fn`, are saved only as their `repr`, so pass them in again.

## space_for

```python
dragon_torch.utils.tuning.presets.space_for(target) -> SearchSpace
dragon_torch.utils.tuning.presets.register_preset(model_cls, space_fn)
```

Returns a copy of the preset for a model class or model, or for a trainer (looked up through its `model_cls`). Subclasses inherit their parent's preset, and `KeyError` is raised when no preset covers the target. Loss settings are under `model.`, at the same paths in single- and multi-agent configs, so one preset serves both.

| Preset | Models | Parameters |
|---|---|---|
| `ppo_space()` | `PPOModel`, `MultiAgentPPOModel` | `optim.{lr, anneal_lr}`, `rollout.{n_steps, n_minibatches, update_epochs, gamma, gae_lambda, normalize_obs}`, `model.{clip_coef, ent_coef, vf_coef, max_grad_norm}`, `policy.hidden` |
| `a2c_space()` | `A2CModel`, `MultiAgentA2CModel` | `optim.{lr, anneal_lr}`, `rollout.{n_steps, gamma, gae_lambda, normalize_obs}`, `model.{ent_coef, vf_coef, normalize_advantages, max_grad_norm}`, `policy.hidden` |

The `n_steps` and `n_minibatches` choices are powers of two, so they always divide evenly.

## dragon-tune

```bash
dragon-tune dashboard --storage sqlite:///runs/optuna.db [--host 127.0.0.1] [--port 8080]
dragon-tune summary   --storage sqlite:///runs/optuna.db [--study NAME]
dragon-tune export    --storage sqlite:///runs/optuna.db --study NAME [--out trials.csv]
```

The dashboard plots the reported metric for each trial. Other metrics are in each trial's `history.jsonl`; load one with `read_history`.

## Notes

### Storage and workers

Every worker runs the same script with the same `study_name` and `storage`. The first worker creates the study, the others join it, and a restarted worker continues it.

| `storage` | Use for |
|---|---|
| `None` | One process (tests) |
| `'sqlite:///runs/optuna.db'` | One machine, a few workers |
| `'journal:runs/optuna.log'` | One machine, many processes |
| `'postgresql://...'` (any SQLAlchemy URL) | Several machines |
| `'redis://host:6379'` | Several machines, through Optuna's journal backend |

For CPU-bound training, prefer separate processes to `n_jobs > 1`, and call `torch.set_num_threads(1)` in each.

### Epoch callbacks

The tuner hooks into `fit(callbacks=...)`. After each epoch, a callback gets the model and that epoch's record. It can add keys to the record, return `True` to stop early (a checkpoint is still saved), or raise, for example `optuna.TrialPruned`.
