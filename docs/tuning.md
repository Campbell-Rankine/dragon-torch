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

Anything in the `TrialSpec` that has a `close()` method (train data, val data, `resources`) is closed when the trial ends, however it ends.

### Custom metrics

`TrialSpec.callbacks` run before the tuner's own callbacks, so a callback there can add a metric that the logger and the reporter will then see. For example, an evaluation that only runs every N steps. The reporter skips epochs that don't contain the metric. `report_key` sets which history key is used as the pruning step; it defaults to `epoch`.

## Multi-objective studies

```python
TuneConfig(study_name='tradeoff', metrics=['val_return', 'steps_per_sec'],
           directions=['maximize', 'maximize'], sampler='nsga2')   # or 'tpe'
```

The objective returns a tuple, and `best.json` and the `summary` command show the Pareto front.

Optuna can't prune multi-objective studies, because `trial.report` isn't supported there. The reporter therefore only runs its NaN/inf check, and the pruner is ignored. `cmaes` supports single-objective studies only.

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
