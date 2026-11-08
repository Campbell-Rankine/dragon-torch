# Tuning and monitoring with Optuna

`utils/tuning/` runs Optuna studies over any `_DragonModel`. Each trial builds a model from sampled hyperparameters, trains it with `DragonTrainer`, and reports per-epoch metrics so weak trials are pruned early. You can watch studies in `optuna-dashboard` or from the CLI.

```bash
uv sync --extra tune   # adds optuna-dashboard; optuna itself is a core dependency
```

## Pieces

| File | Contents |
|---|---|
| `config.py` | `TuneConfig`: study name, storage, objective metrics and directions, sampler, pruner and budget. `make_storage` turns a storage string into an Optuna storage |
| `objective.py` | `DragonObjective`, which runs one training run per trial; `TrialSpec`, which your `build` function returns; `objective_values` |
| `reporting.py` | Epoch callbacks: `OptunaReporter` (`trial.report` and pruning) and `JsonlLogger` (every epoch record goes to `history.jsonl`) |
| `study.py` | `run_study`, `create_study`, `load_study`, `best_summary` |
| `monitor.py` | CLI with `summary`, `export` (CSV) and `dashboard` |
| `space.py` | `SearchSpace` with `FloatParam`, `IntParam` and `CategoricalParam`. Parameters are named by config path, for example `optim.lr` |
| `RL.py` | `rl_tune_config` and `RLTrialBuilder`, for any `DragonRLTrainer`. Needs the `RL` extra |
| `presets/` | `ppo_space()` and `a2c_space()`: search spaces over the PPO and A2C trainer configs |

The tuner relies on the epoch-callback hook in `DragonModel.fit` / `DragonTrainer.fit` (`models.base.EpochCallback`). After each epoch, the hook gets the model and that epoch's history record. A callback can:

- add keys to the record,
- return `True` to stop early (a checkpoint is still saved),
- raise an exception such as `optuna.TrialPruned`, which propagates out of `fit`.

## Writing a study

You provide two functions:

- `suggest(trial) -> params` defines the search space.
- `build(params, trial_dir) -> TrialSpec` builds the model and data from those params.

```python
import torch
from models.base import DragonModel, DragonModelConfig
from utils.tuning import TrialSpec, TuneConfig, run_study

def suggest(trial):
    return {'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'hidden': trial.suggest_categorical('hidden', [32, 64, 128])}

def build(params, trial_dir):
    cfg = DragonModelConfig(checkpoint_dir=trial_dir / 'ckpt', save_every=10, max_epochs=20)
    return TrialSpec(
        model=DragonModel(MyModule(params['hidden']), torch.nn.MSELoss(), cfg),
        optimizer=lambda p: torch.optim.Adam(p, lr=params['lr']),
        train_data=train_ds, val_data=val_ds)

cfg = TuneConfig.for_model(DragonModel, study_name='mlp-sweep',
                           storage='sqlite:///runs/optuna.db', n_trials=50)
study = run_study(cfg, suggest, build)
```

`TuneConfig.for_model` reads the objective from the model class: it uses `eval_metric` (`val_loss`, or `val_return` for RL) in the direction set by `higher_is_better`. To set the objective yourself, pass `metrics` and `directions`.

### What each trial produces

Each trial gets a directory, `run_dir/<study_name>/trial_NNNN/`, containing:

- `history.jsonl`: one line per epoch with every recorded value. It's written as training runs, so pruned and failed trials keep their curves.
- whatever `build` puts there. Point `checkpoint_dir` inside the trial directory so trials don't overwrite each other.

The trial's user attributes in Optuna also store:

- `trial_dir`
- `epochs`
- `last_record`
- `fail_reason`, when the trial failed

When a worker finishes, it writes `run_dir/<study_name>/best.json`. For a single objective this is the best trial; for multiple objectives it's the Pareto front.

### How a trial ends

| Outcome | Trial state |
|---|---|
| Training finishes normally | `COMPLETE`. The objective is each metric reduced by `final`: `last`, `mean_last` over `final_window` values (smoothed), or `best` |
| The pruner stops it | `PRUNED` |
| A metric becomes NaN or inf | Training stops and the trial is `FAIL`. The study carries on |
| Misconfiguration, e.g. the metric is never recorded | Raises, and the study stops |

Anything in the `TrialSpec` that has a `close()` method (train data, val data, `resources`, `callbacks`) is closed when the trial ends, however it ends. A pruned trial's reason is saved in the `prune_reason` user attribute.

### Custom metrics

`TrialSpec.callbacks` run before the tuner's own callbacks, so a callback there can add a metric that the logger and the reporter will then see. For example, an evaluation that only runs every N steps. The reporter skips epochs that don't contain the metric. `report_key` sets which history key is used as the pruning step; it defaults to `epoch`.

## Multi-objective studies

```python
TuneConfig(study_name='tradeoff', metrics=['val_return', 'steps_per_sec'],
           directions=['maximize', 'maximize'], sampler='nsga2')   # or 'tpe'
```

The objective returns a tuple, and `best.json` and the `summary` command show the Pareto front.

Optuna can't prune multi-objective studies, because `trial.report` isn't supported there. The reporter therefore only runs its NaN/inf check, and the pruner is ignored. `cmaes` supports single-objective studies only.

## RL training runs

RL studies tune a `DragonRLTrainer` (`PPOTrainer` or `A2CTrainer`). [rl_trainers.md](rl_trainers.md) describes the trainers and their configs; this section covers the tuning side. To run a study directly:

```bash
uv run --extra RL --extra torch-cpu python scripts/tune_rl.py --algo ppo --env CartPole-v1 \
    --storage journal:runs/optuna.log --timesteps 100000 --eval-every 10000 --n-trials 20
```

You can run that command in several terminals at once; every worker joins the same study. Use `--algo a2c` for A2C. In code:

```python
from data.envs import DragonEnvConfig
from models.RL import DragonRLConfig
from trainers.RL import EvalConfig, PPOTrainer, PPOTrainerConfig
from utils.tuning import run_study
from utils.tuning.presets import ppo_space
from utils.tuning.RL import RLTrialBuilder, rl_tune_config

base = PPOTrainerConfig(
    env=DragonEnvConfig(env_id='CartPole-v1', n_envs=8),
    model=DragonRLConfig(checkpoint_dir='runs/unused', save_every=1_000_000, total_timesteps=100_000),
    eval=EvalConfig(every_steps=10_000, n_episodes=10, smooth_window=3))
cfg = rl_tune_config('ppo-cartpole', PPOTrainer, storage='journal:runs/optuna.log',
                     n_trials=20, n_warmup_steps=20_000)          # warmup in env steps
space = ppo_space().fix({'rollout.gamma': 0.99})
run_study(cfg, space.suggest, RLTrialBuilder(trainer_cls=PPOTrainer, base_config=base))
```

For each trial, `RLTrialBuilder` takes the sampled parameters, applies them to `base_config` with `apply_overrides`, and sets `model.checkpoint_dir` to `trial_dir/ckpt`. It then returns `TrialSpec(trainer=...)`. The objective trains the trainer, adds the tuner's callbacks after the trainer's own, and closes it at the end, including when the trial is pruned.

### Objective and pruning

`rl_tune_config` takes its objective from `trainer_cls.objectives`: the final `val_return_smooth`, maximised. It reports that value to the pruner at `eval_step`. The trainer's `StepEvalCallback` produces both values; see [rl_trainers.md](rl_trainers.md#evaluation).

The pruner compares trials at the **same reported step**, which is why it uses `eval_step`. An epoch is `n_envs × n_steps` env steps, and `n_steps` is a hyperparameter, so `global_step` values differ from trial to trial and would never match. The scheduled steps are the same for every trial.

- **Evaluation interval:** keep `eval.every_steps` at or above the largest epoch in the search space. An epoch that crosses two thresholds evaluates only once, and that trial skips a scheduled step.
- **Seeds:** every trial uses `base_config`'s seeds, so differences come from the hyperparameters rather than luck. If scores are still noisy, raise `eval.n_episodes` or `eval.smooth_window`.
- **Divergence:** the trainer's `DivergenceGuard` raises `TrainingDiverged`. The objective converts it to a pruned trial, and `prune_reason` starts with `diverged:`. This works in multi-objective studies too.

### Search spaces

A `SearchSpace` names its parameters by config path, so the values it samples apply directly onto the trainer config, and the dashboard shows names like `rollout.n_steps`.

| Preset | Parameters |
|---|---|
| `ppo_space()` | `optim.lr`, `optim.anneal_lr`, `rollout.{n_steps, n_minibatches, update_epochs, gamma, gae_lambda, normalize_obs}`, `ppo.{clip_coef, ent_coef, vf_coef}`, `model.max_grad_norm`, `policy.hidden` |
| `a2c_space()` | `optim.lr`, `optim.anneal_lr`, `rollout.{n_steps, gamma, gae_lambda, normalize_obs}`, `a2c.{ent_coef, vf_coef, normalize_advantages}`, `model.max_grad_norm`, `policy.hidden` |

To customise a preset:

- `.fix({...})` pins values; pinned parameters aren't sampled.
- `.narrow({...})` replaces ranges.
- `.without(...)` drops parameters, so the base config's values are used.

The `n_steps` and `n_minibatches` choices are powers of two, so they always divide evenly. The config validator rejects combinations that don't.

### What each trial saves

Besides `history.jsonl`, each trial directory holds:

- `ckpt/`
- `config.json`: the fully resolved trainer config.

`best.json` entries link to `config.json`. To rebuild the best run exactly:

```python
cfg = PPOTrainerConfig.model_validate_json(open('runs/ppo-cartpole/trial_0007/config.json').read())
with PPOTrainer(cfg) as trainer:
    trainer.fit()
```

Configs that contain callables, such as `env_fn` or wrappers, store those callables as their `repr`. Pass them in again when you rebuild.

Every trial's history includes `val_return`, `val_return_smooth`, `episode_return`, `approx_kl`, `clip_frac`, `entropy`, `policy_loss`, `value_loss`, `lr`, `steps_per_sec` and `global_step`.

### Multi-objective

To tune for two objectives, for example return versus throughput:

```python
rl_tune_config('ppo-mo', PPOTrainer, extra_metrics={'steps_per_sec': 'maximize'})   # or: scripts/tune_rl.py --multi-objective
```

## Storage and workers

Every worker runs the **same script with the same `study_name` and `storage`**. `create_study(load_if_exists=True)` lets the first worker create the study and the others join it. A restarted worker continues the same study.

| `storage` | Use for |
|---|---|
| `None` | In memory, a single process (tests, quick checks) |
| `sqlite:///runs/optuna.db` | One machine, a few workers |
| `journal:runs/optuna.log` | One machine, many processes. Uses a file lock and avoids SQLite lock contention |
| `postgresql://user:pw@host/db` (any SQLAlchemy URL) | Several machines |
| `redis://host:6379` | Several machines, through Optuna's journal backend |

Notes for parallel workers:

- `n_trials` and `timeout` apply **per worker**.
- Give each worker a different `seed`, or leave it as `None`. Workers that share a seed propose the same trials.
- `n_jobs > 1` runs threads in one process. For CPU-bound torch or env workloads, separate processes usually work better; set `torch.set_num_threads(1)` in each one.

## Monitoring

```bash
python -m utils.tuning.monitor dashboard --storage sqlite:///runs/optuna.db            # http://127.0.0.1:8080
python -m utils.tuning.monitor summary   --storage sqlite:///runs/optuna.db            # list studies
python -m utils.tuning.monitor summary   --storage sqlite:///runs/optuna.db --study mlp-sweep
python -m utils.tuning.monitor export    --storage sqlite:///runs/optuna.db --study mlp-sweep --out trials.csv
```

- **Dashboard:** shows intermediate values (the curve of the reported metric), parameter importance, slice and contour plots, and user attributes.
- **Other metrics:** curves for metrics other than the reported one are in each trial's `history.jsonl`. Load one with `utils.tuning.read_history`.
- **Remote access:** to reach the dashboard from another machine, pass `--host 0.0.0.0`, or tunnel the port over SSH.
