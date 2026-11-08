from pathlib import Path
from typing import Literal, Optional, Union
import optuna
from optuna.storages import BaseStorage, JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalRedisBackend
from pydantic import BaseModel, Field, model_validator
from common.types import PathLike
from models.base import _DragonModel

type Direction = Literal['minimize', 'maximize']


class TuneConfig(BaseModel):
    """Everything a study needs besides the search space and model builder.

    Objectives: `metrics[i]` is read from each epoch's history record and
    optimised in `directions[i]`. One metric is an ordinary study; several make
    a multi-objective study (optuna can't prune those, so pruning is skipped).
    Use `TuneConfig.for_model` to take the metric and direction from a model class.

    Storage (see `make_storage`), shared by every worker that runs the study:
      None                       in-memory, single process
      'sqlite:///runs/optuna.db' one machine, a few workers
      'journal:runs/optuna.log'  one machine, many processes (file lock, no DB)
      'redis://host:6379'        several machines, via optuna's journal backend
      'postgresql://...' etc.    several machines, any SQLAlchemy URL
    """
    study_name: str
    storage: Optional[str] = None
    # per-trial dirs (history.jsonl, checkpoints) go in run_dir/study_name/trial_NNNN
    run_dir: PathLike = Path('runs')

    metrics: list[str] = Field(min_length=1)
    directions: list[Direction] = Field(min_length=1)
    # how a trial's history becomes its objective value(s), per metric:
    #   last       value at the last epoch that recorded it
    #   mean_last  mean of the last `final_window` recorded values (smoothed)
    #   best       best recorded value in the metric's direction
    final: Literal['last', 'mean_last', 'best'] = 'last'
    final_window: int = Field(default=1, ge=1)

    n_trials: Optional[int] = Field(default=None, ge=1)  # per worker
    timeout: Optional[float] = Field(default=None, gt=0)  # seconds, per worker
    n_jobs: int = Field(default=1, ge=1)  # threads in this process

    sampler: Literal['tpe', 'random', 'cmaes', 'nsga2'] = 'tpe'
    # give each worker a different seed, or they will propose identical trials
    seed: Optional[int] = None
    pruner: Literal['median', 'hyperband', 'none'] = 'median'
    n_startup_trials: int = Field(default=5, ge=0)
    # in units of report_key, e.g. epochs, or env steps for RL
    n_warmup_steps: int = Field(default=0, ge=0)
    # history key used as the step for trial.report (must increase every report)
    report_key: str = 'epoch'

    @model_validator(mode='after')
    def _check(self) -> "TuneConfig":
        if len(self.metrics) != len(self.directions):
            raise ValueError(f"{len(self.metrics)} metrics but {len(self.directions)} directions")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError(f"duplicate metrics: {self.metrics}")
        if self.sampler == 'cmaes' and self.multi_objective:
            raise ValueError("cmaes supports one objective only; use tpe or nsga2")
        return self

    @classmethod
    def for_model(cls, model_cls: type[_DragonModel], **kw) -> "TuneConfig":
        """Single objective: the model's `eval_metric`, in its `higher_is_better` direction."""
        kw.setdefault('metrics', [model_cls.eval_metric])
        kw.setdefault('directions', ['maximize' if model_cls.higher_is_better else 'minimize'])
        return cls(**kw)

    @property
    def multi_objective(self) -> bool:
        return len(self.metrics) > 1

    @property
    def can_prune(self) -> bool:
        return self.pruner != 'none' and not self.multi_objective

    @property
    def study_dir(self) -> Path:
        return Path(self.run_dir) / self.study_name

    # ------- optuna objects ------- #
    def make_storage(self) -> Union[str, BaseStorage, None]:
        return make_storage(self.storage)

    def make_sampler(self) -> optuna.samplers.BaseSampler:
        if self.sampler == 'random':
            return optuna.samplers.RandomSampler(seed=self.seed)
        if self.sampler == 'cmaes':
            return optuna.samplers.CmaEsSampler(seed=self.seed, n_startup_trials=self.n_startup_trials)
        if self.sampler == 'nsga2':
            return optuna.samplers.NSGAIISampler(seed=self.seed)
        return optuna.samplers.TPESampler(seed=self.seed, n_startup_trials=self.n_startup_trials,
                                          multivariate=True)

    def make_pruner(self) -> optuna.pruners.BasePruner:
        if not self.can_prune:
            return optuna.pruners.NopPruner()
        if self.pruner == 'hyperband':
            return optuna.pruners.HyperbandPruner(min_resource=max(1, self.n_warmup_steps))
        return optuna.pruners.MedianPruner(n_startup_trials=self.n_startup_trials,
                                           n_warmup_steps=self.n_warmup_steps)


def make_storage(storage: Optional[str]) -> Union[str, BaseStorage, None]:
    """Turn a storage string into what optuna.create_study / load_study accept.
    'journal:<path>' and 'redis://...' become JournalStorage; other URLs pass
    through to optuna's RDBStorage; None stays in-memory."""
    if storage is None:
        return None
    if storage.startswith('journal:'):
        path = Path(storage.removeprefix('journal:'))
        path.parent.mkdir(parents=True, exist_ok=True)
        return JournalStorage(JournalFileBackend(str(path)))
    if storage.startswith('redis://'):
        return JournalStorage(JournalRedisBackend(storage))
    if storage.startswith('sqlite:///'):
        Path(storage.removeprefix('sqlite:///')).parent.mkdir(parents=True, exist_ok=True)
    return storage
