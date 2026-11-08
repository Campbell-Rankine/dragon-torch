import json
from typing import Any, Callable, Optional, Sequence
import optuna
from .config import TuneConfig, make_storage
from .objective import BuildFn, DragonObjective, SuggestFn

BEST_FILE = 'best.json'


def create_study(cfg: TuneConfig) -> optuna.Study:
    """Create the study, or join it if another worker already made it."""
    return optuna.create_study(
        storage=cfg.make_storage(), study_name=cfg.study_name, load_if_exists=True,
        directions=cfg.directions, sampler=cfg.make_sampler(), pruner=cfg.make_pruner())


def load_study(study_name: str, storage: Optional[str]) -> optuna.Study:
    return optuna.load_study(study_name=study_name, storage=make_storage(storage))


def best_summary(study: optuna.Study) -> dict[str, Any]:
    """Best trial (single objective) or the Pareto front (multi-objective)."""
    def row(t: optuna.trial.FrozenTrial) -> dict[str, Any]:
        return {'number': t.number, 'values': t.values, 'params': t.params,
                'trial_dir': t.user_attrs.get('trial_dir'),
                'config_path': t.user_attrs.get('config_path')}

    complete = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if not complete:
        return {'study': study.study_name, 'best': []}
    best = study.best_trials if len(study.directions) > 1 else [study.best_trial]
    return {'study': study.study_name,
            'directions': [d.name.lower() for d in study.directions],
            'best': [row(t) for t in best]}


def run_study(cfg: TuneConfig, suggest: SuggestFn, build: BuildFn,
              objective_cls: type[DragonObjective] = DragonObjective,
              callbacks: Optional[Sequence[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]]] = None,
              ) -> optuna.Study:
    """Run this worker's share of the study and write study_dir/best.json.

    Start the same script on any number of processes / machines with the same
    cfg (shared storage, different seeds) to run workers in parallel."""
    study = create_study(cfg)
    study.optimize(objective_cls(cfg, suggest, build), n_trials=cfg.n_trials,
                   timeout=cfg.timeout, n_jobs=cfg.n_jobs, gc_after_trial=True,
                   callbacks=callbacks)
    cfg.study_dir.mkdir(parents=True, exist_ok=True)
    (cfg.study_dir / BEST_FILE).write_text(json.dumps(best_summary(study), indent=2, default=str))
    return study
